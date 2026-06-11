"""ContinuationExecutor - run a scraper continuation and persist its yields.

Collaborators are injected explicitly:

* ``db``      - SQLManager, used only for the atomic ``StagedWrites.flush``.
* ``scraper`` - resolves continuation names and prep methods.
* ``queue``   - RequestQueue, stages enqueues (reusing its (de)serialization).
* ``storage`` - ResponseStorage, stores the response, serializes results,
  and marks the request completed when there is no continuation.

Speculation outcome tracking lives in the worker (its ``track_speculation``
callback), not here: this executor always stores a ``None``
``speculation_outcome`` alongside the response.
"""

from __future__ import annotations

import asyncio
import functools
import json as _json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Protocol

from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.decorators import get_step_metadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    ScraperConfigError,
    TransientException,
)
from jkent.contracts import ensure, require
from jkent.data_types import (
    EstimateData,
    HTTPRequestPrep,
    JSRequestPrep,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.database_engine.staging import StagedWrites

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from jkent.common.exceptions import ScraperAssumptionException
    from jkent.common.selector_observer import SelectorObserver, SelectorQuery
    from jkent.data_types import ScraperYield
    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.unified_driver.persistence import (
        RequestQueue,
        ResponseStorage,
    )

logger = logging.getLogger(__name__)


class AutowaitPage(Protocol):
    """Minimal page surface the autowait loop drives (a Playwright Page fits)."""

    url: str

    async def wait_for_selector(
        self, selector: str, *, timeout: int
    ) -> Any: ...

    async def content(self) -> str: ...


# XPath string literals have no escaping, so a regex can excise them
# exactly. Checks below run on the literal-free form: content inside
# quotes ("score: 5") must not trip structural checks like the EXSLT
# prefix scan.
_XPATH_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
# A trailing predicate like [1] or [@class='x'] (literals already
# stripped, so no brackets hide inside quotes). Applied repeatedly:
# //div/text()[1][2] → //div/text().
_TRAILING_PREDICATE = re.compile(r"\[[^\][]*\]$")


@require(
    lambda selector_type: selector_type in ("xpath", "css"),
    "selector_type is one of the two supported selector languages",
)
@ensure(
    lambda result, selector_type: selector_type != "css" or result is True,
    "CSS selectors always target elements, so they are always waitable",
)
def can_playwright_wait(selector: str, selector_type: str) -> bool:
    """Determine if a selector can be used with Playwright's wait_for_selector().

    Playwright's wait_for_selector() only works with selectors that target
    elements. It does not support XPath expressions that return text nodes,
    attributes, or use EXSLT functions, and it has no way to bind XPath
    variable references (``$var``).

    Args:
        selector: The selector string.
        selector_type: Type of selector ("xpath" or "css").

    Returns:
        True if Playwright can wait for this selector, False otherwise.

    Examples:
        >>> can_playwright_wait("//div[@class='content']", "xpath")
        True
        >>> can_playwright_wait("//div/@href", "xpath")
        False
        >>> can_playwright_wait("//div/text()", "xpath")
        False
        >>> can_playwright_wait("//div[@id=$section]", "xpath")
        False
        >>> can_playwright_wait("//a[text()='Price: $5']", "xpath")
        True
        >>> can_playwright_wait("div.content", "css")
        True
    """
    if selector_type == "css":
        # CSS selectors always target elements
        return True

    # XPath - check for non-element targeting. Structural checks run on
    # the selector with string literals excised, so quoted content
    # (e.g. [text()='score: 5']) can't trip them.
    selector = selector.strip()
    structural = _XPATH_STRING_LITERAL.sub("''", selector)

    # What the selector targets is its last step, ignoring trailing
    # predicates: //div/text()[1] still selects text nodes.
    last_step = structural
    while True:
        stripped = _TRAILING_PREDICATE.sub("", last_step)
        if stripped == last_step:
            break
        last_step = stripped

    # Check for text node selection
    if last_step.endswith("/text()"):
        return False

    # Check for attribute selection (last step is /@attribute_name)
    if last_step.split("/")[-1].startswith("@"):
        return False

    # XPath variable references ($var) can't be bound through Playwright.
    # Checked on the literal-free form so text like 'Price: $5' is fine.
    if "$" in structural:
        return False

    # Check for EXSLT functions (namespace prefixes)
    # Common EXSLT namespaces: re, str, math, set, dyn, exsl, func, date
    exslt_prefixes = [
        "re:",
        "str:",
        "math:",
        "set:",
        "dyn:",
        "exsl:",
        "func:",
        "date:",
    ]

    # Element-targeting XPath if no EXSLT prefixes found (checked on the
    # literal-free form so text content can't false-positive)
    return all(prefix not in structural for prefix in exslt_prefixes)


class ContinuationExecutor:
    """Store a response, run its continuation, and persist the yields atomically."""

    def __init__(
        self,
        db: SQLManager,
        scraper: Any,  # BaseScraper-like: needs get_continuation + prep attrs
        queue: RequestQueue,
        storage: ResponseStorage,
        *,
        prep_backoff_schedule: tuple[float, ...] = (1.0, 2.0, 4.0),
        provided_preps: dict[str, Callable[..., Any]] | None = None,
        handle_data: Callable[[Any], Awaitable[None]] | None = None,
        on_invalid_data: Callable[[DeferredValidation], Awaitable[None]]
        | None = None,
        on_structural_error: Callable[
            [ScraperAssumptionException], Awaitable[bool]
        ]
        | None = None,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
    ) -> None:
        self.db = db
        self.scraper = scraper
        self.queue = queue
        self.storage = storage
        self.prep_backoff_schedule = prep_backoff_schedule
        self._provided_preps = provided_preps or {}
        self._handle_data = handle_data
        self.on_invalid_data = on_invalid_data
        self.on_structural_error = on_structural_error
        self._on_progress = on_progress

    async def _emit_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> None:
        if self._on_progress is not None:
            await self._on_progress(event_type, data)

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        continuation_name: str,
        *,
        page: Any = None,
        store_response: bool = True,
    ) -> None:
        """Store response, run continuation, flush its yields, mark completed."""
        # Speculation outcome is tracked by the worker (track_speculation), not
        # here; the stored value is always None on this path.
        speculation_outcome: str | None = None

        if store_response:
            await self.storage._store_response(
                request_id, response, continuation_name, speculation_outcome
            )

        if not continuation_name:
            await self.storage._mark_request_completed(request_id)
            return

        continuation = self.scraper.get_continuation(continuation_name)
        staged = StagedWrites(request_id=request_id)

        # Autowait dispatch: only a live page carrying an auto_await_timeout
        # takes the retry loop; everything else runs the normal generator path.
        auto_await_timeout: int | None = None
        if page is not None:
            metadata = get_step_metadata(continuation)
            auto_await_timeout = (
                metadata.auto_await_timeout if metadata else None
            )

        if page is not None and auto_await_timeout:
            await self._process_generator_with_autowait(
                continuation,
                response,
                request,
                request_id,
                auto_await_timeout,
                page=page,
                staged=staged,
            )
        else:
            gen = continuation(response)
            await self._process_generator_with_storage(
                gen,
                response,
                request,
                continuation_name,
                request_id,
                staged,
                page=page,
            )

        emitted_events = await staged.flush(self.db)
        for event in emitted_events:
            await self._emit_progress("request_enqueued", event)

    async def _process_generator_with_storage(
        self,
        gen: Generator[ScraperYield, bool | None, None],
        response: Response,
        parent_request: Request,
        continuation_name: str,
        request_id: int,
        staged: StagedWrites,
        *,
        page: Any = None,
    ) -> None:
        """Process generator yields, buffering all DB writes in ``staged``."""
        try:
            for item in gen:
                match item:
                    case ParsedData():
                        raw_data = item.unwrap()
                        if isinstance(raw_data, DeferredValidation):
                            try:
                                validated_data = raw_data.confirm()
                                rt, dj, vej = (
                                    self.storage._serialize_result_for_storage(
                                        validated_data
                                    )
                                )
                                staged.stage_result(
                                    result_type=rt,
                                    data_json=dj,
                                    is_valid=True,
                                    validation_errors_json=vej,
                                )
                                if self._handle_data is not None:
                                    staged.stage_callback(
                                        functools.partial(
                                            self._handle_data, validated_data
                                        )
                                    )
                            except DataFormatAssumptionException as e:
                                rt, dj, vej = (
                                    self.storage._serialize_result_for_storage(
                                        e.failed_doc, e.errors
                                    )
                                )
                                staged.stage_result(
                                    result_type=rt,
                                    data_json=dj,
                                    is_valid=False,
                                    validation_errors_json=vej,
                                )
                                if self.on_invalid_data:
                                    staged.stage_callback(
                                        functools.partial(
                                            self.on_invalid_data, raw_data
                                        )
                                    )
                        else:
                            rt, dj, vej = (
                                self.storage._serialize_result_for_storage(
                                    raw_data
                                )
                            )
                            staged.stage_result(
                                result_type=rt,
                                data_json=dj,
                                is_valid=True,
                                validation_errors_json=vej,
                            )
                            if self._handle_data is not None:
                                staged.stage_callback(
                                    functools.partial(
                                        self._handle_data, raw_data
                                    )
                                )

                    case EstimateData():
                        types_json = _json.dumps(
                            [t.__name__ for t in item.expected_types]
                        )
                        staged.stage_estimate(
                            expected_types_json=types_json,
                            min_count=item.min_count,
                            max_count=item.max_count,
                        )

                    case Request() if (
                        not item.nonnavigating and not item.archive
                    ):
                        await self.queue._stage_enqueue_request(
                            item, response, request_id, staged
                        )

                    case Request():
                        await self.queue._stage_enqueue_request(
                            item, parent_request, request_id, staged
                        )

                    case JSRequestPrep() as wrapper:
                        modified = await self._run_prep(
                            wrapper,
                            response,
                            parent_request,
                            page=page,
                            kind="js",
                        )
                        is_nav = (
                            isinstance(modified, Request)
                            and not modified.nonnavigating
                            and not modified.archive
                        )
                        ctx = response if is_nav else parent_request
                        await self.queue._stage_enqueue_request(
                            modified, ctx, request_id, staged
                        )

                    case HTTPRequestPrep() as wrapper:
                        modified = await self._run_prep(
                            wrapper,
                            response,
                            parent_request,
                            page=page,
                            kind="http",
                        )
                        is_nav = (
                            isinstance(modified, Request)
                            and not modified.nonnavigating
                            and not modified.archive
                        )
                        ctx = response if is_nav else parent_request
                        await self.queue._stage_enqueue_request(
                            modified, ctx, request_id, staged
                        )

                    case None:
                        pass

        except HTMLStructuralAssumptionException as e:
            # NOTE: catches the PARENT ScraperAssumptionException scope
            # implicitly via HTMLStructuralAssumptionException.
            if self.on_structural_error:
                should_continue = await self.on_structural_error(e)
                if not should_continue:
                    return
            else:
                raise

    async def _process_generator_with_autowait(
        self,
        continuation: Callable[..., Any],
        response: Response,
        parent_request: Request,
        request_id: int,
        auto_await_timeout: int,
        *,
        page: AutowaitPage,
        staged: StagedWrites,
    ) -> None:
        """Run the continuation, waiting on the live page for missing selectors.

        On an HTMLStructuralAssumptionException it waits for the offending
        selector in the browser, re-snapshots the DOM, and retries until
        success or the ``auto_await_timeout`` (ms) elapses.
        """
        start_time = time.time()
        timeout_seconds = auto_await_timeout / 1000.0

        while True:
            try:
                # Reset the staged buffer so a failed prior attempt's partial
                # yields are discarded.
                staged.results.clear()
                staged.estimates.clear()
                staged.requests.clear()
                staged.seen_dedup_keys.clear()
                gen = continuation(response)
                await self._process_generator_with_storage(
                    gen,
                    response,
                    parent_request,
                    continuation.__name__,
                    request_id,
                    staged,
                )
                break  # success

            except HTMLStructuralAssumptionException as e:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(
                        f"Autowait timeout exhausted ({auto_await_timeout}ms) "
                        f"for request {request_id}"
                    )
                    raise

                if not self._is_playwright_compatible_selector(
                    e.selector, e.selector_type
                ):
                    logger.debug(
                        f"Selector not Playwright-compatible, skipping "
                        f"autowait: {e.selector}"
                    )
                    raise

                # The step wrapper records the observer on the per-execution
                # Response, never on the shared StepMetadata.
                observer = response.observer

                if observer and e.selector:
                    absolute_selector = self._compose_absolute_selector(
                        e.selector, observer
                    )
                else:
                    absolute_selector = e.selector

                logger.info(
                    f"Autowait: waiting for selector {absolute_selector}"
                )

                remaining_timeout = int((timeout_seconds - elapsed) * 1000)
                try:
                    await page.wait_for_selector(
                        absolute_selector, timeout=remaining_timeout
                    )
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"Autowait failed: selector {absolute_selector} not "
                        f"found within timeout"
                    )
                    raise e from None  # re-raise original exception

                # Re-snapshot DOM from the live page.
                html_content = await page.content()
                response = Response(
                    status_code=response.status_code,
                    url=page.url,
                    content=html_content.encode("utf-8"),
                    text=html_content,
                    headers=response.headers,
                    request=response.request,
                )

                await self.storage._store_response(
                    request_id, response, continuation.__name__
                )

                logger.info(
                    "Autowait: retrying step function with fresh DOM snapshot"
                )

    def _is_playwright_compatible_selector(
        self, selector: str, selector_type: str | None
    ) -> bool:
        """Whether a selector can be passed to Playwright wait_for_selector.

        Delegates to the module-level :func:`can_playwright_wait` above.
        ``selector_type`` comes from an
        ``HTMLStructuralAssumptionException`` and is always "xpath" or
        "css" in practice; None falls back to the stricter xpath rules.
        """
        return can_playwright_wait(selector, selector_type or "xpath")

    def _compose_absolute_selector(
        self, selector: str, observer: SelectorObserver
    ) -> str:
        """Compose an absolute selector from a relative one via the observer tree."""
        if selector.startswith("//") or selector.startswith("/"):
            return selector

        query = self._find_query_by_selector(observer.queries, selector)
        if not query:
            return selector

        path_parts: list[str] = []
        current: SelectorQuery | None = query
        while current:
            path_parts.append(current.selector)
            current = current.parent
        path_parts.reverse()

        if path_parts and path_parts[0].startswith("//"):
            result = path_parts[0]
            for part in path_parts[1:]:
                if part.startswith(".//"):
                    result = result + "//" + part[3:]
                elif part.startswith("./"):
                    result = result + "/" + part[2:]
                elif part.startswith("."):
                    result = result + "/" + part[1:]
                else:
                    result = result + "/" + part
            return result
        return "/".join(path_parts)

    def _find_query_by_selector(
        self, queries: list[SelectorQuery], selector: str
    ) -> SelectorQuery | None:
        """Find a SelectorQuery in the tree matching the given selector."""
        for query in queries:
            if query.selector == selector:
                return query
            found = self._find_query_by_selector(query.children, selector)
            if found:
                return found
        return None

    def _resolve_prep_method(self, prep_method: str) -> Callable[..., Any]:
        """Resolve a prep_method string to a callable on the scraper or provided table."""
        if prep_method.startswith("provided."):
            key = prep_method[len("provided.") :]
            cb = self._provided_preps.get(key)
            if cb is None:
                raise ScraperConfigError(
                    f"prep_method {prep_method!r} not provided by driver "
                    f"(register a RequestPrepProvider with provider_name="
                    f"{key!r} via Driver.open(request_preps=[...]))"
                )
            return cb
        cb = getattr(self.scraper, prep_method, None)
        if cb is None or not callable(cb):
            raise ScraperConfigError(
                f"prep_method {prep_method!r} not found on scraper"
            )
        return cb

    async def _run_prep(
        self,
        wrapper: Any,  # JSRequestPrep | HTTPRequestPrep
        response: Response,
        parent_request: Request,
        *,
        page: Any = None,
        kind: str,
    ) -> Request:
        """Invoke a prep callable with the transient-retry backoff loop."""
        if kind == "js" and page is None:
            raise ScraperConfigError(
                "JSRequestPrep yielded but driver has no live page "
                "(httpx driver?)"
            )

        cb = self._resolve_prep_method(wrapper.prep_method)
        delays = self.prep_backoff_schedule
        last_exc: TransientException | None = None
        for delay in (0.0, *delays):
            if delay:
                await asyncio.sleep(delay)
            try:
                if kind == "js":
                    return await cb(
                        response, wrapper.request, page, **wrapper.kwargs
                    )
                return await cb(response, wrapper.request, **wrapper.kwargs)
            except TransientException as e:
                last_exc = e
                continue

        raise TransientException(
            f"prep {wrapper.prep_method!r} exhausted retries: {last_exc}"
        ) from last_exc
