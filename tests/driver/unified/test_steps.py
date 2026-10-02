"""Tests for the unified driver's StepExecutor."""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from jkent.common.decorators import StepMetadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
)
from jkent.common.selector_observer import SelectorObserver, SelectorQuery
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.steps import StepExecutor
from tests.db_queries import fetch_requests, fetch_results

if TYPE_CHECKING:
    pass


class CaseData(BaseModel):
    """Minimal model for deferred-validation tests."""

    docket: str


class FakeScraper:
    """Minimal scraper exposing steps via get_step."""

    def __init__(
        self,
        yields_factory: Any,
    ) -> None:
        self._yields_factory = yields_factory
        self._step_metadata: dict[str, Any] = {}

    def get_step(self, name: str) -> Any:
        factory = self._yields_factory
        metadata = self._step_metadata.get(name)

        def step(
            response: Response,
        ) -> Generator[ScraperYield[Any], bool | None, None]:
            yield from factory(response)

        step.__name__ = name
        if metadata is not None:
            step._step_metadata = metadata  # type: ignore[attr-defined]
        return step


def _parent_context() -> Response:
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/listing"
        ),
        step="parse",
        current_location="https://example.com",
    )
    return Response(
        request=parent,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/listing",
    )


async def _seed_request(sql_manager: SQLManager) -> tuple[int, Request]:
    queue = RequestQueue(sql_manager)
    req = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/page"
        ),
        step="parse",
        current_location="https://example.com",
    )
    await queue.enqueue_request(req, _parent_context())
    dequeued = await queue.get_next_request()
    assert dequeued is not None
    request_id, restored, _, _ = dequeued
    assert isinstance(restored, Request)
    return request_id, restored


def _make_executor(
    sql_manager: SQLManager,
    scraper: FakeScraper,
    handled: list[Any] | None = None,
    invalid: list[DeferredValidation[Any]] | None = None,
) -> StepExecutor:
    queue = RequestQueue(sql_manager)
    storage = ResponseStorage(sql_manager)

    async def handle_data(data: Any) -> None:
        if handled is not None:
            handled.append(data)

    async def on_invalid_data(data: DeferredValidation[Any]) -> None:
        if invalid is not None:
            invalid.append(data)

    return StepExecutor(
        sql_manager,
        scraper,
        queue,
        storage,
        handle_data=handle_data,
        on_invalid_data=on_invalid_data if invalid is not None else None,
    )


async def test_mixed_yields_land_after_flush(sql_manager: SQLManager) -> None:
    """ParsedData / Request / None all persist after flush."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        yield ParsedData({"docket": "A-1"})
        yield None
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/child"
            ),
            step="parse",
            current_location="",
        )

    handled: list[Any] = []
    scraper = FakeScraper(yields)
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(request_id, response, request, "parse")

    # Result persisted.
    results = await fetch_results(sql_manager, request_id=request_id)
    assert len(results) == 1
    assert results[0].is_valid is True

    # handle_data fired post-flush.
    assert handled == [{"docket": "A-1"}]

    # Child request enqueued.
    pending = await fetch_requests(sql_manager, status="pending")
    child_urls = [r.url for r in pending]
    assert "https://example.com/child" in child_urls


async def test_deferred_invalid_row_is_flagged_and_kept_whole(
    sql_manager: SQLManager,
) -> None:
    """An invalid record lands in ``results`` with is_valid=0, errors, and data.

    Pins all three at once against the raw row: the flag is stored as SQLite
    0 (not a dropped/NULL column), the Pydantic error details survive as
    JSON, and the failed doc itself is still readable.
    """
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        # 'docket' is required and absent -> confirm() raises.
        yield ParsedData(
            DeferredValidation(CaseData, case_name="Smith v. Jones")
        )

    scraper = FakeScraper(yields)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(request_id, response, request, "parse")

    # The row is in results at all — unfiltered read, not the is_valid filter.
    rows = await fetch_results(sql_manager, request_id=request_id)
    assert len(rows) == 1
    row = rows[0]

    assert row.is_valid is False
    assert row.result_type == "dict"

    # Errors captured with Pydantic's structure intact.
    errors = json.loads(row.validation_errors_json or "[]")
    assert [e["type"] for e in errors] == ["missing"]
    assert [list(e["loc"]) for e in errors] == [["docket"]]

    # The failed doc is preserved, not discarded.
    assert json.loads(row.data_json) == {"case_name": "Smith v. Jones"}


@pytest.mark.parametrize(
    ("fields", "is_valid"),
    [
        pytest.param({"docket": "A-1"}, True, id="valid"),
        pytest.param({"case_name": "Smith v. Jones"}, False, id="invalid"),
    ],
)
async def test_deferred_validation_routes_to_its_callback(
    sql_manager: SQLManager, fields: dict[str, str], is_valid: bool
) -> None:
    """A confirmed record goes to handle_data; a failed one to on_invalid_data.

    Either way exactly one result row lands, flagged by the outcome.
    """
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
    )
    deferred = DeferredValidation(CaseData, **fields)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        yield ParsedData(deferred)

    handled: list[Any] = []
    invalid: list[DeferredValidation[CaseData]] = []
    executor = _make_executor(
        sql_manager, FakeScraper(yields), handled=handled, invalid=invalid
    )

    await executor.complete_request(request_id, response, request, "parse")

    rows = await fetch_results(sql_manager, request_id=request_id)
    assert [r.is_valid for r in rows] == [is_valid]
    assert json.loads(rows[0].data_json) == fields
    if is_valid:
        assert handled == [CaseData(**fields)]
        assert invalid == []
    else:
        assert handled == []
        assert invalid == [deferred]


async def test_deferred_invalid_with_non_json_native_value(
    sql_manager: SQLManager,
) -> None:
    """A failed_doc holding a date still stores as a validation failure.

    Regression: the failed_doc went through a bare ``json.dumps``, so any
    non-JSON-native scraped value raised TypeError out of the generator
    loop. That replaced the DataFormatAssumptionException the caller was
    trying to record, filing the error as unknown/builtins.TypeError and
    dropping message, model_name, and validation_errors_json.
    """
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        # A date where the model wants a str -> validation fails, and the
        # date lands in failed_doc.
        yield ParsedData(DeferredValidation(CaseData, docket=date(2026, 1, 2)))

    scraper = FakeScraper(yields)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(request_id, response, request, "parse")

    invalid = await fetch_results(
        sql_manager, request_id=request_id, is_valid=False
    )
    assert len(invalid) == 1
    assert invalid[0].validation_errors_json is not None
    # The date survives as its string form rather than blowing up the dump.
    assert "2026-01-02" in invalid[0].data_json


async def test_structural_error_propagates(
    sql_manager: SQLManager,
) -> None:
    """A raised HTMLStructuralAssumptionException propagates to the caller."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        raise HTMLStructuralAssumptionException(
            selector="//div",
            selector_type="xpath",
            description="rows",
            expected_min=1,
            expected_max=None,
            actual_count=0,
            request_url="https://example.com/page",
        )
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(request_id, response, request, "parse")


# --- Autowait (browser-free) ---------------------------------------------


class FakeAutowaitPage:
    """A browser-free stand-in satisfying the AutowaitPage Protocol."""

    def __init__(self, content: str, url: str) -> None:
        self._content = content
        self.url = url
        self.wait_calls: list[tuple[str, int]] = []

    async def wait_for_selector(self, selector: str, *, timeout: int) -> Any:
        self.wait_calls.append((selector, timeout))
        return object()

    async def content(self) -> str:
        return self._content


def _structural_exc(
    selector: str = "//div[@id='rows']",
) -> HTMLStructuralAssumptionException:
    """Build a structural exception the way the codebase constructs one."""
    return HTMLStructuralAssumptionException(
        selector=selector,
        selector_type="xpath",
        description="rows",
        expected_min=1,
        expected_max=None,
        actual_count=0,
        request_url="https://example.com/page",
    )


def _attach_step_metadata(
    scraper: FakeScraper,
    *,
    auto_await_timeout: int | None,
) -> None:
    """Attach StepMetadata so get_step_metadata(step) sees autowait config.

    ``get_step`` returns a bound method; attribute lookup falls through
    to the underlying function, so the metadata is set per-instance there.
    """
    metadata = StepMetadata(auto_await_timeout=auto_await_timeout)
    scraper._step_metadata["parse"] = metadata


async def test_autowait_waits_then_succeeds(sql_manager: SQLManager) -> None:
    """A first-attempt structural failure waits on the page, then succeeds."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _structural_exc()
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)

    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    # The loop waited on the failing selector, then the retry succeeded.
    assert len(page.wait_calls) == 1
    assert page.wait_calls[0][0] == "//div[@id='rows']"
    assert attempts["n"] == 2

    # The successful attempt's result was staged + flushed.
    results = await fetch_results(sql_manager, request_id=request_id)
    assert len(results) == 1
    assert handled == [{"docket": "A-1"}]


async def test_autowait_timeout_exhausted_reraises(
    sql_manager: SQLManager,
) -> None:
    """When the timeout elapses, the structural exception propagates."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        # Burn a little wall-clock so elapsed exceeds the tiny timeout before
        # the first exhausted-timeout check.
        time.sleep(0.01)
        raise _structural_exc()
        # Unreachable on purpose: the bare yield makes this a generator,
        # so the raise surfaces on first next() instead of at call time.
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    # 1ms timeout: the 10ms sleep above guarantees elapsed >= timeout.
    _attach_step_metadata(scraper, auto_await_timeout=1)
    page = FakeAutowaitPage(content="<html></html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    # Never waited: the loop bailed on the exhausted-timeout check.
    assert page.wait_calls == []


async def test_page_without_timeout_runs_normal_path(
    sql_manager: SQLManager,
) -> None:
    """A page with no auto_await_timeout takes the normal generator path."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=None)
    page = FakeAutowaitPage(content="<html></html>", url=response.url)

    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    # Normal path: no waits, result stored.
    assert page.wait_calls == []
    results = await fetch_results(sql_manager, request_id=request_id)
    assert len(results) == 1
    assert handled == [{"docket": "A-1"}]


async def test_autowait_composes_absolute_selector_from_observer(
    sql_manager: SQLManager,
) -> None:
    """A relative failing selector is composed to an absolute one via the observer."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _structural_exc(selector=".//td[@class='cell']")
        yield ParsedData({"docket": "A-1"})

    # Observer tree: //table/tbody//tr -> .//td[@class='cell'].
    parent = SelectorQuery(
        selector="//table",
        selector_type="xpath",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
    )
    child = SelectorQuery(
        selector=".//td[@class='cell']",
        selector_type="xpath",
        # Matches _structural_exc's description: real code passes the same
        # description to both the observer query and the raised exception.
        description="rows",
        match_count=0,
        expected_min=1,
        expected_max=None,
        parent=parent,
    )
    parent.children = [child]
    observer = SelectorObserver()
    observer.queries = [parent]
    # The step wrapper records the observer on the per-execution Response.
    response.observer = observer

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    # Composed: //table + .//td[...] -> //table//td[@class='cell'].
    assert page.wait_calls[0][0] == "//table//td[@class='cell']"


def _count_exc(
    *, expected_min: int, expected_max: int | None, actual_count: int
) -> HTMLStructuralAssumptionException:
    return HTMLStructuralAssumptionException(
        selector="//tr",
        selector_type="xpath",
        description="rows",
        expected_min=expected_min,
        expected_max=expected_max,
        actual_count=actual_count,
        request_url="https://example.com/page",
    )


def _page_response(request: Request) -> Response:
    return Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )


async def test_autowait_too_many_matches_raises_without_waiting(
    sql_manager: SQLManager,
) -> None:
    """An excess of matches can't be waited away, so it raises at once.

    ``wait_for_selector`` returns as soon as one match exists, so waiting on
    a selector that already over-matches returns immediately and the loop
    would spin until the budget ran out.
    """
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        raise _count_exc(expected_min=1, expected_max=1, actual_count=3)
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=2000)
    page = FakeAutowaitPage(content="<html></html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    started = time.monotonic()
    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    assert page.wait_calls == []
    assert attempts["n"] == 1
    assert time.monotonic() - started < 1.0


async def test_autowait_too_few_matches_waits_for_the_missing_ones(
    sql_manager: SQLManager,
) -> None:
    """With some matches present, the wait targets the ``expected_min``-th."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _count_exc(expected_min=3, expected_max=None, actual_count=1)
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert [selector for selector, _ in page.wait_calls] == ["//tr >> nth=2"]


@pytest.mark.parametrize(
    "dom",
    [
        "<html>same</html>",
        '<html><head><meta charset="windows-1252"></head>caf\xe9</html>',
    ],
    ids=["undeclared", "declares-windows-1252"],
)
async def test_autowait_same_failure_on_unchanged_snapshot_raises(
    sql_manager: SQLManager, dom: str
) -> None:
    """A retry that changes nothing is not retried again.

    The page says the selector is there, but the re-snapshot is identical and
    the step fails the same way; another round would only repeat it.
    """
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        raise _structural_exc()
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=2000)
    page = FakeAutowaitPage(content=dom, url=response.url)
    executor = _make_executor(sql_manager, scraper)

    stores: list[str] = []
    original_store = executor.storage.store_response

    async def counting_store(*args: Any, **kwargs: Any) -> int:
        stores.append(args[1].text)
        return await original_store(*args, **kwargs)

    executor.storage.store_response = counting_store  # type: ignore[method-assign]

    started = time.monotonic()
    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    assert time.monotonic() - started < 1.0
    assert attempts["n"] == 2
    assert len(page.wait_calls) == 2
    # The original response, then the one re-snapshot — not one per loop.
    assert stores == [
        "<html></html>",
        dom.replace("windows-1252", "utf-8"),
    ]


async def test_autowait_stored_snapshot_decodes_as_the_page(
    sql_manager: SQLManager,
) -> None:
    """The stored re-snapshot reads back as the DOM, whatever it declared."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _structural_exc()
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(
        content=(
            '<html><head><meta charset="windows-1252"></head>'
            "<body>caf\xe9</body></html>"
        ),
        url=response.url,
    )
    executor = _make_executor(sql_manager, scraper)
    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    stored = await executor.storage.load_preresolved_response(
        request_id, request
    )
    assert stored is not None
    assert "caf\xe9" in stored.text
