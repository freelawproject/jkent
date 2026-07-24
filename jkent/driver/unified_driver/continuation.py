"""ContinuationExecutor - run a scraper continuation and persist its yields.

Collaborators are injected explicitly:

* ``db``      - SQLManager, used only for the atomic ``StagedWrites.flush``.
* ``scraper`` - resolves continuation names.
* ``queue``   - RequestQueue, stages enqueues (reusing its (de)serialization).
* ``storage`` - ResponseStorage, stores the response, serializes results,
  and marks the request completed when there is no continuation.

Speculation outcome tracking lives in the worker (its ``track_speculation``
callback), not here: this executor always stores a ``None``
``speculation_outcome`` alongside the response.
"""

from __future__ import annotations

import functools
import json as _json
import logging
import re
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.decorators import get_step_metadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    IncidentalRequestAssumptionException,
)
from jkent.contracts import ensure, require
from jkent.data_types import (
    Multiple,
    ParsedData,
    Request,
    Response,
    Singular,
)
from jkent.driver.database_engine.sql_manager._types import PreresolvedResponse
from jkent.driver.database_engine.staging import StagedWrites

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

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
        scraper: Any,  # BaseScraper-like: needs get_continuation
        queue: RequestQueue,
        storage: ResponseStorage,
        *,
        handle_data: Callable[[Any], Awaitable[None]] | None = None,
        on_invalid_data: Callable[[DeferredValidation], Awaitable[None]]
        | None = None,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
    ) -> None:
        self.db = db
        self.scraper = scraper
        self.queue = queue
        self.storage = storage
        self._handle_data = handle_data
        self.on_invalid_data = on_invalid_data
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
            await self.storage.store_response(
                request_id, response, continuation_name, speculation_outcome
            )

        if not continuation_name:
            await self.storage.mark_request_completed(request_id)
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
    ) -> None:
        """Process generator yields, buffering all DB writes in ``staged``.

        A scraper assumption violation (e.g. HTMLStructuralAssumptionException)
        propagates to the caller; the autowait loop relies on that to drive its
        wait-and-retry, and otherwise it is the driver's terminal error.
        """
        for item in gen:
            match item:
                case ParsedData():
                    raw_data = item.unwrap()
                    if isinstance(raw_data, DeferredValidation):
                        try:
                            validated_data = raw_data.confirm()
                            self._stage_valid_result(staged, validated_data)
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
                        self._stage_valid_result(staged, raw_data)

                case Request():
                    if item.incidental is not None:
                        await self._stage_preresolved_incidentals(
                            item, response, parent_request, request_id, staged
                        )
                    else:
                        await self.queue._stage_enqueue_request(
                            item,
                            self._enqueue_ctx(item, response, parent_request),
                            request_id,
                            staged,
                        )

                case None:
                    pass

    def _stage_valid_result(self, staged: StagedWrites, data: Any) -> None:
        """Serialize a valid result, stage it, and stage its on-data callback."""
        rt, dj, vej = self.storage._serialize_result_for_storage(data)
        staged.stage_result(
            result_type=rt,
            data_json=dj,
            is_valid=True,
            validation_errors_json=vej,
        )
        if self._handle_data is not None:
            staged.stage_callback(functools.partial(self._handle_data, data))

    @staticmethod
    def _enqueue_ctx(
        req: Request, response: Response, parent_request: Request
    ) -> Response | Request:
        """Pick the URL-resolution context for an enqueued request.

        A navigating request resolves against the response it came from;
        a nonnavigating/archive request resolves against its parent.
        """
        if not req.nonnavigating and not req.archive:
            return response
        return parent_request

    async def _stage_preresolved_incidentals(
        self,
        item: Request,
        response: Response,
        parent_request: Request,
        parent_request_id: int,
        staged: StagedWrites,
    ) -> None:
        """Promote captured incidentals into pre-resolved child requests.

        Matches ``item.incidental`` against the sub-requests the *parent*
        navigation captured (``parent_request_id``), enforces the spec's
        cardinality, and stages one pre-resolved request per match — its
        promoted response stored in the same flush transaction. The cardinality
        error surfaces here, attributed to the parent navigation, so an
        over/under-broad spec fails fast at enqueue rather than as a phantom
        request later.
        """
        spec = item.incidental
        assert spec is not None
        records = await self.db.get_incidental_requests(parent_request_id)

        matched = []
        for rec in records:
            headers: dict[str, str] = (
                _json.loads(rec.headers_json) if rec.headers_json else {}
            )
            body: bytes | None = None
            if spec.body_contains is not None and rec.storage_id is not None:
                storage = await self.db.get_incidental_request_storage(
                    rec.storage_id
                )
                body = storage.get("body") if storage else None
            if spec.matches(
                url=rec.url,
                method=rec.method,
                headers=headers,
                body=body,
                resource_type=rec.resource_type,
            ):
                matched.append(rec)

        self._enforce_incidental_cardinality(spec, matched, response.url)

        for index, rec in enumerate(matched):
            preresolved = await self._build_preresolved_response(rec)
            child = self._prepare_incidental_child(item, spec, rec, index)
            await self.queue._stage_enqueue_request(
                child,
                self._enqueue_ctx(child, response, parent_request),
                parent_request_id,
                staged,
                preresolved_response=preresolved,
            )

    @staticmethod
    def _enforce_incidental_cardinality(
        spec: Singular | Multiple,
        matched: list[Any],
        request_url: str,
    ) -> None:
        """Raise if the match count violates the spec's cardinality."""
        if isinstance(spec, Singular) and len(matched) != 1:
            raise IncidentalRequestAssumptionException(
                f"Singular incidental expected exactly 1 match, "
                f"found {len(matched)}",
                request_url,
                match_spec=repr(spec),
                candidate_urls=[r.url for r in matched],
            )
        if isinstance(spec, Multiple) and not matched:
            raise IncidentalRequestAssumptionException(
                "Multiple incidental expected at least 1 match, found 0",
                request_url,
                match_spec=repr(spec),
            )

    async def _build_preresolved_response(
        self, rec: Any
    ) -> PreresolvedResponse:
        """Build a PreresolvedResponse from a captured incidental's storage.

        Copies the already-compressed response body verbatim (no
        re-compression); status/headers/url come from the incidental record
        and its storage row.
        """
        storage = (
            await self.db.get_incidental_request_storage(rec.storage_id)
            if rec.storage_id is not None
            else None
        )
        status_code = rec.status_code if rec.status_code is not None else 200
        headers_json: str | None = None
        content_compressed: bytes | None = None
        content_size_original: int | None = None
        content_size_compressed: int | None = None
        compression_dict_id: int | None = None
        if storage is not None:
            if storage.get("status_code") is not None:
                status_code = storage["status_code"]
            headers_json = storage.get("response_headers_json")
            content_compressed = storage.get("content_compressed")
            content_size_original = storage.get("content_size_original")
            content_size_compressed = storage.get("content_size_compressed")
            compression_dict_id = storage.get("compression_dict_id")
        return PreresolvedResponse(
            status_code=status_code,
            headers_json=headers_json,
            url=rec.url,
            content_compressed=content_compressed,
            content_size_original=content_size_original,
            content_size_compressed=content_size_compressed,
            compression_dict_id=compression_dict_id,
        )

    @staticmethod
    def _prepare_incidental_child(
        item: Request,
        spec: Singular | Multiple,
        rec: Any,
        index: int,
    ) -> Request:
        """Build the child request to enqueue for one promoted incidental.

        A promoted response can't be re-fetched standalone (the browser minted
        it once), so the child is non-reseedable. For ``Multiple`` matches the
        yielded request's single dedup key would collapse all matches into one
        row, so it is suffixed per match to keep them distinct.
        """
        dedup_key = item.deduplication_key
        if isinstance(spec, Multiple) and isinstance(dedup_key, str):
            dedup_key = f"{dedup_key}:{index}:{rec.url}"
        return replace(
            item,
            incidental=None,
            reseedable=False,
            deduplication_key=dedup_key,
        )

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
                # yields (and its deferred on_data/on_invalid_data callbacks)
                # are discarded before the retry.
                staged.reset()
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
                absolute_selector = e.selector

                if observer:
                    query = self._find_failing_query(observer, e)
                    composed = (
                        observer.compose_absolute_selector(query)
                        if query is not None
                        else None
                    )
                    if composed is not None:
                        absolute_selector = composed

                logger.info(
                    f"Autowait: waiting for selector {absolute_selector}"
                )

                # Floor at 1ms: Playwright treats timeout=0 as "wait forever",
                # so a sub-millisecond remaining budget must not round down to 0.
                remaining_timeout = max(
                    1, int((timeout_seconds - elapsed) * 1000)
                )
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

                await self.storage.store_response(
                    request_id, response, continuation.__name__
                )

                logger.info(
                    "Autowait: retrying step function with fresh DOM snapshot"
                )

    def _is_playwright_compatible_selector(
        self, selector: str, selector_type: str
    ) -> bool:
        """Whether a selector can be passed to Playwright wait_for_selector.

        Delegates to the module-level :func:`can_playwright_wait` above.
        ``selector_type`` comes from an
        ``HTMLStructuralAssumptionException`` and is always "xpath" or "css".
        """
        return can_playwright_wait(selector, selector_type)

    def _find_failing_query(
        self,
        observer: SelectorObserver,
        exc: HTMLStructuralAssumptionException,
    ) -> SelectorQuery | None:
        """Locate the observer query that raised this structural exception.

        Matched on the exception's distinguishing fields (selector, type, and
        description) rather than the selector string alone, so a selector
        reused under different parents resolves to the node that actually
        failed. Returns None if no recorded query matches, in which case the
        caller falls back to the raw (relative) selector.
        """

        def walk(queries: list[SelectorQuery]) -> SelectorQuery | None:
            for query in queries:
                if (
                    query.selector == exc.selector
                    and query.selector_type == exc.selector_type
                    and query.description == exc.description
                ):
                    return query
                found = walk(query.children)
                if found is not None:
                    return found
            return None

        return walk(observer.queries)
