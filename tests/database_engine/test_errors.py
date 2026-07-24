"""Round-trip tests for the error persistence layer (errors.py).

Previously only reachable through ``ScrapeRun._store_error`` (which the
worker-conformance fakes stub out), so none of this had direct coverage:
classification, type-specific field extraction + traceback capture on
store, JSON-field parsing on fetch, list/count filtering (type,
resolution, continuation join), and the resolve transition.
"""

from __future__ import annotations

import json
from datetime import datetime

import sqlalchemy as sa

from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    TransientException,
)
from jkent.driver.database_engine.errors import (
    classify_error,
    count_errors,
    get_error,
    list_errors,
    resolve_error,
    store_error,
)
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
    sf = sql_manager._session_factory
    error_id = await store_error(
        sf, _raised(_structural()), db_lock=sql_manager._lock
    )
    record = await get_error(sf, error_id)

    assert record is not None
    assert record.error_type == "structural"
    assert record.error_class.endswith("HTMLStructuralAssumptionException")
    assert record.selector == "//div[@class='case']"
    assert record.selector_type == "xpath"
    assert (record.expected_min, record.expected_max) == (1, 20)
    assert record.actual_count == 0
    assert record.request_url == "https://err.test/page"  # from the exc
    assert record.is_resolved is False
    assert record.traceback is not None
    assert "HTMLStructuralAssumptionException" in record.traceback
    assert isinstance(record.created_at, datetime)


async def test_validation_error_round_trip(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory
    error_id = await store_error(
        sf, _raised(_validation()), db_lock=sql_manager._lock
    )
    record = await get_error(sf, error_id)

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
    sf = sql_manager._session_factory
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
    error_id = await store_error(sf, _raised(exc), db_lock=sql_manager._lock)
    record = await get_error(sf, error_id)

    assert record is not None
    assert record.error_type == "validation"
    # The datetime survives as its str() form, and the record JSON-round-trips.
    assert record.failed_doc == {"filed": "2020-01-01 00:00:00"}
    json.loads(record.to_json())


async def test_transient_field_extraction(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory

    http_id = await store_error(
        sf,
        _raised(HTTPResponseAssumptionException(503, [200], "https://x")),
        db_lock=sql_manager._lock,
    )
    http_record = await get_error(sf, http_id)
    assert http_record is not None
    assert http_record.status_code == 503
    assert http_record.request_url == "https://x"  # taken from exc.url

    timeout_id = await store_error(
        sf,
        _raised(RequestTimeoutException("https://slow", 30.0)),
        db_lock=sql_manager._lock,
    )
    timeout_record = await get_error(sf, timeout_id)
    assert timeout_record is not None
    assert timeout_record.timeout_seconds == 30.0


async def test_url_fallbacks(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory

    # A bare exception has no URL of its own.
    unknown_id = await store_error(
        sf, _raised(ValueError("boom")), db_lock=sql_manager._lock
    )
    unknown = await get_error(sf, unknown_id)
    assert unknown is not None
    assert unknown.request_url == "unknown"
    assert unknown.error_type == "unknown"

    # An explicit request_url wins over the fallback.
    explicit_id = await store_error(
        sf,
        _raised(ValueError("boom")),
        request_url="https://explicit",
        db_lock=sql_manager._lock,
    )
    explicit = await get_error(sf, explicit_id)
    assert explicit is not None
    assert explicit.request_url == "https://explicit"

    assert await get_error(sf, 99_999) is None


async def _insert_request(sql_manager: SQLManager, continuation: str) -> int:
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, queue_counter, "
                "method, url, continuation, current_location) "
                "VALUES ('completed', 9, 1, 'GET', 'https://e', :c, '')"
            ),
            {"c": continuation},
        )
        await session.commit()
        result = await session.execute(
            sa.text("SELECT id FROM requests ORDER BY id DESC LIMIT 1")
        )
        return result.scalar_one()


async def test_list_and_count_filters(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory
    request_id = await _insert_request(sql_manager, "parse_detail")

    structural_id = await store_error(
        sf,
        _raised(_structural()),
        request_id=request_id,
        db_lock=sql_manager._lock,
    )
    await store_error(sf, _raised(_validation()), db_lock=sql_manager._lock)
    resolved_id = await store_error(
        sf, _raised(TransientException("x")), db_lock=sql_manager._lock
    )
    assert await resolve_error(sf, resolved_id, notes="handled")

    # unresolved_only (the default) hides the resolved transient.
    assert await count_errors(sf) == 2
    assert await count_errors(sf, unresolved_only=False) == 3
    assert await count_errors(sf, error_type="structural") == 1
    assert await count_errors(sf, error_type="transient") == 0
    # The continuation filter mirrors list_errors (joins through requests).
    assert await count_errors(sf, continuation="parse_detail") == 1
    assert await count_errors(sf, continuation="no_such_step") == 0
    assert (
        await count_errors(sf, error_type="transient", unresolved_only=False)
        == 1
    )

    listed = await list_errors(sf)
    assert {r.error_type for r in listed} == {"structural", "validation"}

    by_type = await list_errors(sf, error_type="validation")
    assert [r.error_type for r in by_type] == ["validation"]

    # The continuation filter joins through the linked request row.
    by_continuation = await list_errors(sf, continuation="parse_detail")
    assert [r.id for r in by_continuation] == [structural_id]
    assert await list_errors(sf, continuation="no_such_step") == []

    # Pagination.
    assert len(await list_errors(sf, unresolved_only=False, limit=2)) == 2
    assert (
        len(await list_errors(sf, unresolved_only=False, limit=2, offset=2))
        == 1
    )


async def test_resolve_error_transition(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory
    error_id = await store_error(
        sf, _raised(_structural()), db_lock=sql_manager._lock
    )

    assert await resolve_error(sf, error_id, notes="fixed selector") is True
    record = await get_error(sf, error_id)
    assert record is not None
    assert record.is_resolved is True
    assert record.resolution_notes == "fixed selector"
    assert isinstance(record.resolved_at, datetime)

    # Already resolved -> no-op; unknown id -> not found.
    assert await resolve_error(sf, error_id) is False
    assert await resolve_error(sf, 99_999) is False


async def test_record_to_json_is_parseable(sql_manager: SQLManager) -> None:
    sf = sql_manager._session_factory
    error_id = await store_error(
        sf, _raised(_validation()), db_lock=sql_manager._lock
    )
    record = await get_error(sf, error_id)
    assert record is not None

    payload = json.loads(record.to_json())
    assert payload["id"] == error_id
    assert payload["error_type"] == "validation"
    assert payload["model_name"] == "CaseData"
    assert payload["validation_errors"] == [
        {"loc": ["docket"], "msg": "field required"}
    ]
    assert payload["is_resolved"] is False
    assert payload["created_at"]  # isoformat string
