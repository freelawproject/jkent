"""Tests for the unified driver's ContinuationExecutor."""

from __future__ import annotations

import time
from collections.abc import Generator
from pathlib import Path
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
from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from tests.db_queries import fetch_requests, fetch_results

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class CaseData(BaseModel):
    """Minimal model for deferred-validation tests."""

    docket: str


class FakeScraper:
    """Minimal scraper exposing continuations via get_continuation."""

    def __init__(
        self,
        yields_factory: Any,
    ) -> None:
        self._yields_factory = yields_factory
        self._step_metadata: dict[str, Any] = {}

    def get_continuation(self, name: str) -> Any:
        factory = self._yields_factory
        metadata = self._step_metadata.get(name)

        def continuation(
            response: Response,
        ) -> Generator[ScraperYield, bool | None, None]:
            yield from factory(response)

        continuation.__name__ = name  # type: ignore
        if metadata is not None:
            continuation._step_metadata = metadata  # type: ignore[attr-defined]
        return continuation


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


def _parent_context() -> Response:
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/listing"
        ),
        continuation="parse",
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
        continuation="parse",
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
) -> ContinuationExecutor:
    queue = RequestQueue(sql_manager)
    storage = ResponseStorage(sql_manager)

    async def handle_data(data: Any) -> None:
        if handled is not None:
            handled.append(data)

    return ContinuationExecutor(
        sql_manager,
        scraper,
        queue,
        storage,
        handle_data=handle_data,
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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
        yield ParsedData({"docket": "A-1"})
        yield None
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/child"
            ),
            continuation="parse",
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


async def test_deferred_invalid_stored_as_invalid(
    sql_manager: SQLManager,
) -> None:
    """A ParsedData wrapping failed DeferredValidation stores is_valid=False."""
    request_id, request = await _seed_request(sql_manager)
    response = Response(
        request=request,
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
    )

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
        # Missing required 'docket' -> validation fails on confirm().
        yield ParsedData(DeferredValidation(CaseData, not_a_field="x"))

    scraper = FakeScraper(yields)
    executor = _make_executor(sql_manager, scraper)

    await executor.complete_request(request_id, response, request, "parse")

    invalid = await fetch_results(
        sql_manager, request_id=request_id, is_valid=False
    )
    assert len(invalid) == 1
    assert invalid[0].validation_errors_json is not None


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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
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
    """Attach StepMetadata so get_step_metadata(continuation) sees autowait config.

    ``get_continuation`` returns a bound method; attribute lookup falls through
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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
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

    def yields(_response: Response) -> Generator[ScraperYield, None, None]:
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
