"""Tests for the unified driver's StepExecutor."""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest
from lxml import html
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from pydantic import BaseModel
from typing_extensions import override

from jkent.common.decorators import StepMetadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperConfigError,
)
from jkent.common.page_element import Form, FormField, PageElement
from jkent.common.selector_observer import SelectorObserver, SelectorQuery
from jkent.data_types import (
    FieldValue,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
    Selector,
)
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver import steps as steps_module
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


async def _seed_request(sql_manager: SQLManager) -> tuple[int, Request]:
    queue = RequestQueue(sql_manager)
    req = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/page"
        ),
        step="parse",
        current_location="https://example.com",
    )
    await queue.insert_root_request(req)
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
    *,
    captures_incidentals: bool = True,
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
        captures_incidentals=captures_incidentals,
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
        yield None  # pragma: no cover

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
    *,
    query: SelectorQuery | None = None,
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
        query=query,
    )


def _observed(exc: HTMLStructuralAssumptionException) -> SelectorObserver:
    """An observer holding the page-wide query ``exc`` came from.

    Attaches that query to ``exc``, as ``PageElement`` does, so raise this
    same ``exc`` from the step.
    """
    query = SelectorQuery(
        selector=exc.selector,
        selector_type=exc.selector_type,
        description=exc.description,
        match_count=exc.actual_count,
        expected_min=exc.expected_min,
        expected_max=exc.expected_max,
    )
    observer = SelectorObserver()
    observer.queries = [query]
    exc.query = query
    return observer


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

    exc = _structural_exc()
    response.observer = _observed(exc)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc
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
    assert page.wait_calls[0][0] == "xpath=//div[@id='rows']"
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
            raise _structural_exc(selector=".//td[@class='cell']", query=child)
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
    assert page.wait_calls[0][0] == "xpath=//table//td[@class='cell']"


async def test_autowait_checks_waitability_on_the_composed_selector(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent Playwright can't evaluate rules out the wait, not just a leaf.

    The leaf ``td.date`` is waitable, but the composed chain carries the
    parent's cssselect-only ``:contains()``, which the browser rejects, so
    autowait polls instead.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text == "<html>loaded</html>":
            yield ParsedData({"docket": "A-1"})
            return
        raise HTMLStructuralAssumptionException(
            selector="td.date",
            selector_type="css",
            description="date",
            expected_min=1,
            expected_max=None,
            actual_count=0,
            request_url=response.url,
            query=parent.children[0],
        )
        yield None  # type: ignore[unreachable]  # pragma: no cover

    parent = SelectorQuery(
        selector="tr:contains('Docket')",
        selector_type="css",
        description="row",
        match_count=1,
        expected_min=1,
        expected_max=None,
    )
    parent.children = [
        SelectorQuery(
            selector="td.date",
            selector_type="css",
            description="date",
            match_count=0,
            expected_min=1,
            expected_max=None,
            parent=parent,
        )
    ]
    observer = SelectorObserver()
    observer.queries = [parent]
    response.observer = observer

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=2)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert page.wait_calls == []
    assert page.reads == 3
    assert handled == [{"docket": "A-1"}]


async def test_autowait_polls_on_a_selector_that_selects_text(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``text()`` leaf can't be waited on, so the page is polled.

    The total's text fills in after load; raising at once would give up on
    exactly the late render autowait exists for.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc("//span[@id='total']/text()")
    response.observer = _observed(exc)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise exc
        yield ParsedData({"total": "12"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=2)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert page.wait_calls == []
    assert page.reads == 3
    assert handled == [{"total": "12"}]


async def test_autowait_resnapshot_keeps_the_speculation_outcome(
    sql_manager: SQLManager,
) -> None:
    """Re-storing the snapshot must not clear a probe's stored outcome."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc()
    response.observer = _observed(exc)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(
        request_id,
        response,
        request,
        "parse",
        page=page,
        speculation_outcome=SpeculationOutcome.HIT,
    )

    (row,) = [
        r for r in await fetch_requests(sql_manager) if r.id == request_id
    ]
    assert row.content_size_original == len(b"<html>fresh</html>")
    assert row.speculation_outcome == SpeculationOutcome.HIT


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
    exc = _count_exc(expected_min=3, expected_max=None, actual_count=1)
    response.observer = _observed(exc)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert [selector for selector, _ in page.wait_calls] == ["xpath=(//tr)[3]"]


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
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        # Like the page injector, observe each snapshot the step runs on.
        exc = _structural_exc()
        r.observer = _observed(exc)
        raise exc
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
    assert stores == ["<html></html>", dom]


async def test_autowait_stored_snapshot_decodes_as_the_page(
    sql_manager: SQLManager,
) -> None:
    """The stored re-snapshot reads back as the DOM, whatever it declared."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc()
    response.observer = _observed(exc)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc
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


async def test_unknown_yield_type_is_a_config_error(
    sql_manager: SQLManager,
) -> None:
    """A bare dict (ParsedData forgotten) fails loudly instead of vanishing."""
    request_id, request = await _seed_request(sql_manager)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        yield {"docket": "A-1"}  # type: ignore[misc]

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(ScraperConfigError, match="dict"):
        await executor.complete_request(
            request_id, _page_response(request), request, "parse"
        )
    assert await fetch_results(sql_manager, request_id=request_id) == []


class TimingOutAutowaitPage(FakeAutowaitPage):
    """A page whose waits always time out, the way a Playwright page does."""

    @override
    async def wait_for_selector(self, selector: str, *, timeout: int) -> Any:
        self.wait_calls.append((selector, timeout))
        raise PlaywrightTimeoutError(f"Timeout {timeout}ms exceeded.")


async def test_autowait_page_timeout_reraises_the_structural_error(
    sql_manager: SQLManager,
) -> None:
    """A wait timeout surfaces as the step's structural error, not a timeout."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc()
    response.observer = _observed(exc)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        raise exc
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = TimingOutAutowaitPage(content="<html></html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    assert len(page.wait_calls) == 1


async def test_autowait_unknown_grammar_polls_then_reraises_the_error(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grammar autowait can't parse is never waited on; the page is polled.

    The page never changes, so the step's own error is raised at the end.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        raise HTMLStructuralAssumptionException(
            selector=r"Docket No\. \d+",
            selector_type="regex",
            description="docket number",
            expected_min=1,
            expected_max=None,
            actual_count=0,
            request_url=response.url,
        )
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=200)
    page = FakeAutowaitPage(content="<html></html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException) as excinfo:
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    assert excinfo.value.selector_type == "regex"
    assert page.wait_calls == []


_SCOPED_CELL = ".//td[@class='status']"


def _scoped_under_rows(row_count: int) -> SelectorObserver:
    """An observer where ``_structural_exc``'s query ran under ``row_count`` rows."""
    parent = SelectorQuery(
        selector="//tr",
        selector_type="xpath",
        description="row",
        match_count=row_count,
        expected_min=1,
        expected_max=None,
        distinct_match_count=row_count,
    )
    parent.children = [
        SelectorQuery(
            selector=_SCOPED_CELL,
            selector_type="xpath",
            description="rows",
            match_count=0,
            expected_min=1,
            expected_max=None,
            parent=parent,
        )
    ]
    observer = SelectorObserver()
    observer.queries = [parent]
    return observer


def _scoped_exc(
    observer: SelectorObserver | None,
) -> HTMLStructuralAssumptionException:
    """The exception the scoped query of ``_scoped_under_rows`` raises."""
    assert observer is not None
    return _structural_exc(
        selector=_SCOPED_CELL, query=observer.queries[0].children[0]
    )


class ChangingAutowaitPage(FakeAutowaitPage):
    """A page whose DOM changes after ``unchanged_reads`` snapshots."""

    def __init__(self, url: str, unchanged_reads: int) -> None:
        super().__init__(content="<html></html>", url=url)
        self.reads = 0
        self._unchanged_reads = unchanged_reads

    @override
    async def content(self) -> str:
        self.reads += 1
        if self.reads > self._unchanged_reads:
            return "<html>loaded</html>"
        return "<html></html>"


async def test_autowait_ambiguous_scope_polls_instead_of_waiting(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A query under a multi-match parent polls the page for a change.

    Its composed selector would match the other rows at once, so it isn't
    waited on; the retry runs once the DOM actually changes.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    response.observer = _scoped_under_rows(3)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise _scoped_exc(response.observer)
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=3)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert page.wait_calls == []
    assert page.reads == 4
    assert handled == [{"docket": "A-1"}]


async def test_autowait_ambiguous_scope_on_a_static_page_waits_out_the_budget(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanging page is polled until the timeout, storing nothing new."""
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    response.observer = _scoped_under_rows(3)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        raise _scoped_exc(response.observer)
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=200)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=10**6)
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
    assert time.monotonic() - started >= 0.2
    assert page.wait_calls == []
    assert stores == ["<html></html>"]


async def test_autowait_single_match_parent_still_waits_on_the_selector(
    sql_manager: SQLManager,
) -> None:
    """A parent that matched once composes to an exact selector and waits."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    response.observer = _scoped_under_rows(1)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _scoped_exc(response.observer)
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert page.wait_calls[0][0] == "xpath=//tr//td[@class='status']"


async def test_autowait_mixed_grammar_scope_polls_instead_of_waiting(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chain that doesn't compose isn't waited on as a bare relative selector.

    ``.//td`` waited on page-wide matches nothing and burns the whole budget;
    polling retries as soon as the DOM changes.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    observer = _scoped_under_rows(1)
    parent = observer.queries[0]
    parent.selector, parent.selector_type = "tr.case", "css"
    response.observer = observer

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise _scoped_exc(response.observer)
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=2)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert page.wait_calls == []
    assert handled == [{"docket": "A-1"}]


class TickingAutowaitPage(FakeAutowaitPage):
    """A page whose DOM differs on every snapshot (a clock, a banner)."""

    def __init__(self, url: str) -> None:
        super().__init__(content="", url=url)
        self.reads = 0

    @override
    async def content(self) -> str:
        self.reads += 1
        return f"<html>{self.reads}</html>"


async def test_autowait_stores_only_the_last_snapshot(
    sql_manager: SQLManager,
) -> None:
    """Each retry runs on a fresh snapshot, but only the last is stored."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    attempts = {"n": 0}

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] < 4:
            exc = _structural_exc()
            r.observer = _observed(exc)
            raise exc
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = TickingAutowaitPage(url=response.url)
    executor = _make_executor(sql_manager, scraper)

    stores: list[str] = []
    original_store = executor.storage.store_response

    async def counting_store(*args: Any, **kwargs: Any) -> int:
        stores.append(args[1].text)
        return await original_store(*args, **kwargs)

    executor.storage.store_response = counting_store  # type: ignore[method-assign]

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert attempts["n"] == 4
    assert stores == ["<html></html>", "<html>3</html>"]


async def test_autowait_paces_rewaits_on_a_selector_the_page_matches(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matched selector the step still fails on doesn't busy-loop.

    The wait returns at once and the ticking DOM always differs, so only the
    recheck pause bounds how often the step re-runs.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_RECHECK_INTERVAL", 0.05)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        exc = _structural_exc()
        r.observer = _observed(exc)
        raise exc
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=200)
    page = TickingAutowaitPage(url=response.url)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    # 200ms at one re-check per 50ms, plus the first wait.
    assert 2 <= page.reads <= 6


async def test_autowait_without_an_observer_polls_instead_of_waiting(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecorded failure may have run scoped, so it isn't waited on.

    ``.//td`` searched page-wide can match some other cell at once; the wait
    would return, the DOM would be unchanged, and autowait would give up.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise _structural_exc(selector=_SCOPED_CELL)
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=2)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert page.wait_calls == []
    assert handled == [{"docket": "A-1"}]


_TWO_TABLES = (
    "<html><body>"
    "<table id='hdr'><tr><td class='date'>Filed</td></tr></table>"
    "<table id='body'><tr><td>{body}</td></tr></table>"
    "</body></html>"
)


async def test_autowait_waits_on_the_query_that_failed_not_its_twin(
    sql_manager: SQLManager,
) -> None:
    """A helper's query reused under two parents waits under the failing one.

    The header table's ``.//td[@class='date']`` succeeded first; composing
    from it would match at once and never wait for the body's cell.
    """
    request_id, request = await _seed_request(sql_manager)
    content = _TWO_TABLES.format(body="pending")
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=content.encode(),
        text=content,
        url="https://example.com/page",
    )

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        observer = SelectorObserver()
        r.observer = observer
        with observer:
            tree = PageElement(html.fromstring(r.text), r.url)
            for table_id in ("hdr", "body"):
                (table,) = tree.query(
                    Selector.XPath(f"//table[@id='{table_id}']"), "table"
                )
                table.query(Selector.XPath(".//td[@class='date']"), "date")
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(
        content=_TWO_TABLES.format(body="</td><td class='date'>Today"),
        url=response.url,
    )
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert [selector for selector, _ in page.wait_calls] == [
        "xpath=//table[@id='body']//td[@class='date']"
    ]
    assert handled == [{"docket": "A-1"}]


async def test_autowait_snapshot_store_failure_keeps_the_step_error(
    sql_manager: SQLManager,
) -> None:
    """A store that fails while the step's error propagates doesn't mask it."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        exc = _structural_exc()
        r.observer = _observed(exc)
        raise exc
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=2000)
    page = FakeAutowaitPage(content="<html>fresh</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    original_store = executor.storage.store_response

    async def failing_snapshot_store(*args: Any, **kwargs: Any) -> int:
        if args[1] is not response:
            raise RuntimeError("database is locked")
        return await original_store(*args, **kwargs)

    executor.storage.store_response = failing_snapshot_store  # type: ignore[method-assign]

    with pytest.raises(HTMLStructuralAssumptionException):
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )


@pytest.mark.parametrize("step_name", ["", "parse"])
async def test_unstored_response_completes_with_the_speculation_outcome(
    sql_manager: SQLManager, step_name: str
) -> None:
    """With ``store_response`` off, the completion carries the outcome."""
    request_id, request = await _seed_request(sql_manager)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        yield None

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        request_id,
        _page_response(request),
        request,
        step_name,
        store_response=False,
        speculation_outcome=SpeculationOutcome.HIT,
    )

    (row,) = [
        r for r in await fetch_requests(sql_manager) if r.id == request_id
    ]
    assert row.speculation_outcome == SpeculationOutcome.HIT


class NavigatingAutowaitPage(FakeAutowaitPage):
    """A page mid-navigation: Playwright can't evaluate anything on it."""

    @override
    async def wait_for_selector(self, selector: str, *, timeout: int) -> Any:
        self.wait_calls.append((selector, timeout))
        raise PlaywrightError(
            "Execution context was destroyed, most likely because of a "
            "navigation"
        )


async def test_autowait_page_error_reraises_the_structural_error(
    sql_manager: SQLManager,
) -> None:
    """A non-timeout Playwright error doesn't replace the step's own error."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc()
    response.observer = _observed(exc)

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        raise exc
        yield None  # type: ignore[unreachable]  # pragma: no cover

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=2000)
    page = NavigatingAutowaitPage(content="<html></html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)

    with pytest.raises(HTMLStructuralAssumptionException) as excinfo:
        await executor.complete_request(
            request_id, response, request, "parse", page=page
        )
    assert excinfo.value is exc
    assert len(page.wait_calls) == 1


class FlakyContentAutowaitPage(ChangingAutowaitPage):
    """A page whose first snapshot fails mid-navigation, then loads."""

    @override
    async def content(self) -> str:
        if self.reads == 0:
            self.reads += 1
            raise PlaywrightError("Execution context was destroyed")
        return await super().content()


async def test_autowait_poll_skips_a_snapshot_the_page_cannot_give(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed snapshot while polling is one lost tick, not the step's end."""
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 0.01)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise _structural_exc()
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FlakyContentAutowaitPage(url=response.url, unchanged_reads=1)
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert handled == [{"docket": "A-1"}]


async def test_autowait_snapshot_headers_describe_the_snapshot(
    sql_manager: SQLManager,
) -> None:
    """The stored snapshot is UTF-8 DOM, not the transport's encoded bytes."""
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    response.headers = {
        "Content-Type": "text/html; charset=windows-1252",
        "Content-Length": "18000",
        "Content-Encoding": "gzip",
        "X-Court": "kept",
    }
    exc = _structural_exc()
    response.observer = _observed(exc)

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(content="<html>caf\xe9</html>", url=response.url)
    executor = _make_executor(sql_manager, scraper)
    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    stored = await executor.storage.load_preresolved_response(
        request_id, request
    )
    assert stored is not None
    assert stored.headers == {
        "X-Court": "kept",
        "Content-Type": "text/html; charset=utf-8",
    }


def _counting_resolver() -> tuple[Any, dict[str, int]]:
    calls = {"count": 0}

    async def resolve() -> FieldValue:
        calls["count"] += 1
        return "solved"

    return resolve, calls


async def test_autowait_retries_do_not_rerun_deferred_field_resolvers(
    sql_manager: SQLManager,
) -> None:
    """A captcha resolver is paid for once, by the run whose yields commit.

    Each autowait retry re-runs the whole step; resolving at yield time
    would call the solver once per failed run as well.
    """
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)
    exc = _structural_exc()
    response.observer = _observed(exc)
    resolver, calls = _counting_resolver()
    form = Form(
        action="https://example.com/search",
        method="POST",
        fields=[FormField(name="captcha", field_type="text", value="")],
        selector=Selector.XPath("//form"),
    )

    attempts = {"n": 0}

    def yields(
        _response: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        attempts["n"] += 1
        yield form.submit(data={"captcha": resolver}, step="parse")
        if attempts["n"] < 3:
            raise exc

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = TickingAutowaitPage(url=response.url)
    executor = _make_executor(sql_manager, scraper)
    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert attempts["n"] == 3
    assert calls["count"] == 1
    assert len(await fetch_requests(sql_manager, status="pending")) == 1


async def test_autowait_first_poll_looks_without_waiting(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The step's first look at the live DOM doesn't wait out a poll tick.

    The step has only seen the transport's bytes; the browser's DOM is new
    to it, so it is read at once.
    """
    monkeypatch.setattr(steps_module, "AUTOWAIT_POLL_INTERVAL", 30)
    request_id, request = await _seed_request(sql_manager)
    response = _page_response(request)

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        if r.text != "<html>loaded</html>":
            raise _structural_exc()
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = ChangingAutowaitPage(url=response.url, unchanged_reads=0)
    executor = _make_executor(sql_manager, scraper)

    started = time.monotonic()
    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )
    assert time.monotonic() - started < 1
    assert page.reads == 1


async def test_autowait_waits_on_a_reused_selector_under_a_new_description(
    sql_manager: SQLManager,
) -> None:
    """The failing query is found however the step described it.

    The observer keeps the first description of a (parent, selector) pair,
    and the parent here is queried twice on one element (``match_count`` 2).
    Neither may push autowait off the targeted wait onto polling.
    """
    request_id, request = await _seed_request(sql_manager)
    content = "<html><div id='main'><span>pending</span></div></html>"
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=content.encode(),
        text=content,
        url="https://example.com/page",
    )

    def yields(
        r: Response,
    ) -> Generator[ScraperYield[Any], None, None]:
        observer = SelectorObserver()
        r.observer = observer
        with observer:
            tree = PageElement(html.fromstring(r.text), r.url)
            (main,) = tree.query(Selector.XPath("//div[@id='main']"), "main")
            main.query(Selector.XPath(".//td"), "cells", min_count=0)
            (main,) = tree.query(Selector.XPath("//div[@id='main']"), "main")
            main.query(Selector.XPath(".//td"), "cells (required)")
        yield ParsedData({"docket": "A-1"})

    scraper = FakeScraper(yields)
    _attach_step_metadata(scraper, auto_await_timeout=5000)
    page = FakeAutowaitPage(
        content="<html><div id='main'><table><tr><td>A-1</td></tr></table>"
        "</div></html>",
        url=response.url,
    )
    handled: list[Any] = []
    executor = _make_executor(sql_manager, scraper, handled=handled)

    await executor.complete_request(
        request_id, response, request, "parse", page=page
    )

    assert [selector for selector, _ in page.wait_calls] == [
        "xpath=//div[@id='main']//td"
    ]
    assert handled == [{"docket": "A-1"}]
