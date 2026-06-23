"""Round-trip tests for the error persistence layer (errors.py).

Previously only reachable through ``ScrapeRun._store_error`` (which the
worker-conformance fakes stub out), so none of this had direct coverage:
classification, type-specific field extraction + traceback capture on
store, JSON-field parsing on fetch, and list/count filtering (type,
continuation join).
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    TransientException,
)
from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import (
    ErrorType,
    RequestStatus,
    SelectorType,
)
from jkent.driver.database_engine.errors import classify_error
from jkent.driver.database_engine.sql_manager import SQLManager


def _structural(url: str = "https://err.test/page") -> Exception:
    return HTMLStructuralAssumptionException(
        selector="//div[@class='case']",
        selector_type="xpath",
        description="case rows",
        expected_min=1,
        expected_max=20,
        actual_count=0,
        request_url=url,
    )


def _validation() -> Exception:
    return DataFormatAssumptionException(
        errors=[{"loc": ["docket"], "msg": "field required"}],
        failed_doc={"case_name": "Ant v. Bee"},
        model_name="CaseData",
        request_url="https://err.test/detail",
    )


def _raised(exc: Exception) -> Exception:
    """Raise and catch so the exception carries a real traceback."""
    try:
        raise exc
    except Exception as caught:
        return caught


class TestClassifyError:
    def test_taxonomy(self) -> None:
        assert classify_error(_structural()) == "structural"
        assert classify_error(_validation()) == "validation"
        assert classify_error(TransientException("flaky")) == "transient"
        assert (
            classify_error(
                HTTPResponseAssumptionException(503, [200], "https://x")
            )
            == "transient"  # subclass of TransientException
        )
        assert (
            classify_error(PersistentHTTPResponseException(404, "https://x"))
            == "persistent"
        )
        assert classify_error(ValueError("nope")) == "unknown"


async def test_structural_error_round_trip(sql_manager: SQLManager) -> None:
    error_id = await sql_manager.store_error(_raised(_structural()))
    record = await sql_manager.get_error(error_id)

    assert record is not None
    assert record.error_type == "structural"
    assert record.error_class.endswith("HTMLStructuralAssumptionException")
    assert record.selector == "//div[@class='case']"
    assert record.selector_type == "xpath"
    assert (record.expected_min, record.expected_max) == (1, 20)
    assert record.actual_count == 0
    assert record.request_url == "https://err.test/page"  # from the exc
    assert record.traceback is not None
    assert "HTMLStructuralAssumptionException" in record.traceback
    assert isinstance(record.created_at, datetime)


async def test_validation_error_round_trip(sql_manager: SQLManager) -> None:
    error_id = await sql_manager.store_error(_raised(_validation()))
    record = await sql_manager.get_error(error_id)

    assert record is not None
    assert record.error_type == "validation"
    assert record.model_name == "CaseData"
    # JSON fields come back parsed, not as strings.
    assert record.validation_errors == [
        {"loc": ["docket"], "msg": "field required"}
    ]
    assert record.failed_doc == {"case_name": "Ant v. Bee"}


async def test_validation_error_with_non_json_values(
    sql_manager: SQLManager,
) -> None:
    # Real Pydantic .errors() carry the raw failed `input` (any type), and
    # the failed_doc is raw scraped data. Non-JSON-native values here must
    # not blow up the error-logging path; they are coerced via default=str.
    exc = DataFormatAssumptionException(
        errors=[
            {
                "loc": ["filed"],
                "msg": "bad date",
                "input": datetime(2020, 1, 1),
            }
        ],
        failed_doc={"filed": datetime(2020, 1, 1)},
        model_name="CaseData",
        request_url="https://err.test/detail",
    )
    error_id = await sql_manager.store_error(_raised(exc))
    record = await sql_manager.get_error(error_id)

    assert record is not None
    assert record.error_type == "validation"
    # The datetime survives as its str() form, and the record JSON-round-trips.
    assert record.failed_doc == {"filed": "2020-01-01 00:00:00"}
    json.loads(record.model_dump_json())


async def test_transient_field_extraction(sql_manager: SQLManager) -> None:
    http_id = await sql_manager.store_error(
        _raised(HTTPResponseAssumptionException(503, [200], "https://x"))
    )
    http_record = await sql_manager.get_error(http_id)
    assert http_record is not None
    assert http_record.status_code == 503
    assert http_record.request_url == "https://x"  # taken from exc.url

    timeout_id = await sql_manager.store_error(
        _raised(RequestTimeoutException("https://slow", 30.0))
    )
    timeout_record = await sql_manager.get_error(timeout_id)
    assert timeout_record is not None
    assert timeout_record.timeout_seconds == 30.0


async def test_url_fallbacks(sql_manager: SQLManager) -> None:
    # A bare exception has no URL of its own.
    unknown_id = await sql_manager.store_error(_raised(ValueError("boom")))
    unknown = await sql_manager.get_error(unknown_id)
    assert unknown is not None
    assert unknown.request_url == "unknown"
    assert unknown.error_type == "unknown"

    # An explicit request_url wins over the fallback.
    explicit_id = await sql_manager.store_error(
        _raised(ValueError("boom")), request_url="https://explicit"
    )
    explicit = await sql_manager.get_error(explicit_id)
    assert explicit is not None
    assert explicit.request_url == "https://explicit"

    assert await sql_manager.get_error(99_999) is None


async def _insert_request(sql_manager: SQLManager, continuation: str) -> int:
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, queue_counter, "
                "method, url, continuation, current_location) "
                "VALUES (:status, 9, 1, :method, 'https://e', :c, '')"
            ),
            {
                "status": RequestStatus.COMPLETED.code,
                "method": HttpMethod.GET.code,
                "c": continuation,
            },
        )
        await session.commit()
        result = await session.execute(
            sa.text("SELECT id FROM requests ORDER BY id DESC LIMIT 1")
        )
        return result.scalar_one()


async def test_list_and_count_filters(sql_manager: SQLManager) -> None:
    request_id = await _insert_request(sql_manager, "parse_detail")

    structural_id = await sql_manager.store_error(
        _raised(_structural()), request_id=request_id
    )
    await sql_manager.store_error(_raised(_validation()))
    await sql_manager.store_error(_raised(TransientException("x")))

    assert await sql_manager.count_errors() == 3
    assert await sql_manager.count_errors(error_type="structural") == 1
    assert await sql_manager.count_errors(error_type="transient") == 1
    # The continuation filter mirrors list_errors (joins through requests).
    assert await sql_manager.count_errors(continuation="parse_detail") == 1
    assert await sql_manager.count_errors(continuation="no_such_step") == 0

    listed = await sql_manager.list_errors()
    assert {r.error_type for r in listed} == {
        "structural",
        "validation",
        "transient",
    }

    by_type = await sql_manager.list_errors(error_type="validation")
    assert [r.error_type for r in by_type] == ["validation"]

    # The continuation filter joins through the linked request row.
    by_continuation = await sql_manager.list_errors(
        continuation="parse_detail"
    )
    assert [r.id for r in by_continuation] == [structural_id]
    assert await sql_manager.list_errors(continuation="no_such_step") == []

    # Pagination.
    assert len(await sql_manager.list_errors(limit=2)) == 2
    assert len(await sql_manager.list_errors(limit=2, offset=2)) == 1


async def test_record_to_json_is_parseable(sql_manager: SQLManager) -> None:
    error_id = await sql_manager.store_error(_raised(_validation()))
    record = await sql_manager.get_error(error_id)
    assert record is not None

    payload = json.loads(record.model_dump_json())
    assert payload["id"] == error_id
    assert payload["error_type"] == "validation"
    assert payload["model_name"] == "CaseData"
    assert payload["validation_errors"] == [
        {"loc": ["docket"], "msg": "field required"}
    ]
    assert payload["created_at"]  # isoformat string


async def test_coded_columns_store_integers(sql_manager: SQLManager) -> None:
    """``error_type``/``selector_type`` land on disk as codes, not labels.

    The record-level assertions elsewhere in this file pass either way —
    members compare equal to their labels — so the storage format itself
    needs a raw read to be pinned, the same way the other coded columns are
    covered.
    """
    sf = sql_manager.session_factory
    error_id = await sql_manager.store_error(_raised(_structural()))

    async with sf() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT error_type, selector_type FROM errors "
                    "WHERE id = :id"
                ),
                {"id": error_id},
            )
        ).one()

    assert row == (ErrorType.STRUCTURAL.code, SelectorType.XPATH.code)
    assert row == (1, 2)


async def test_unknown_selector_grammar_is_rejected(
    sql_manager: SQLManager,
) -> None:
    """A selector grammar outside the vocabulary fails loudly.

    A caller that reaches this column with a grammar ``SelectorType`` does
    not know must raise rather than silently storing something the CHECK
    constraint or ``from_code`` cannot make sense of. A bare ``Selector``
    base instance, whose ``grammar`` is ``""``, is the way that happens.

    ``describe_error`` narrows the grammar to ``SelectorType`` as it builds
    the row, so this now raises ``ValueError`` at the call site rather than
    ``LookupError`` inside a SQLAlchemy ``StatementError`` at bind time. The
    row never reaches the database either way.
    """
    exc = HTMLStructuralAssumptionException(
        selector="??",
        selector_type="",  # a bare Selector base instance's grammar
        description="case rows",
        expected_min=1,
        expected_max=1,
        actual_count=0,
        request_url="https://err.test/page",
    )
    with pytest.raises(ValueError, match="not a valid SelectorType"):
        await sql_manager.store_error(_raised(exc))

    assert await sql_manager.count_errors() == 0
