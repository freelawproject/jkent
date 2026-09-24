"""StepExecutor - run a scraper step and persist its yields.

Collaborators are injected explicitly:

* ``db``      - SQLManager, used only for the atomic ``StagedWrites.flush``.
* ``scraper`` - resolves step names.
* ``queue``   - RequestQueue, stages enqueues (reusing its (de)serialization).
* ``storage`` - ResponseStorage, stores the response, serializes results,
  and marks the request completed when there is no step.

Speculation outcome tracking lives in the worker (its ``track_speculation``
callback), not here: the executor stores whatever ``speculation_outcome`` the
worker hands it alongside the response.
"""

from __future__ import annotations

import functools
import json as _json
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.decorator_metadata import get_step_metadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    IncidentalRequestAssumptionException,
    ScraperConfigError,
)
from jkent.common.response import utf8_document
from jkent.common.selectors import Selector
from jkent.contracts import require
from jkent.data_types import (
    Multiple,
    ParsedData,
    Request,
    Response,
    Singular,
)
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.database_engine.sql_manager import (
    CompressedPayload,
    StoredResponse,
)
from jkent.driver.database_engine.staging import StagedWrites
from jkent.driver.database_engine.storage import serialize_result
from jkent.driver.unified_driver.persistence import no_progress

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from jkent.common.selector_observer import SelectorObserver, SelectorQuery
    from jkent.data_types import ScraperYield
    from jkent.driver.database_engine.sql_manager import (
        IncidentalRequestRecord,
        SQLManager,
    )
    from jkent.driver.unified_driver.persistence import (
        ProgressCallback,
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


@require(
    lambda selector_type: selector_type in ("xpath", "css"),  # pyrefly: ignore[implicit-any-lambda]
    "selector_type is one of the two supported selector languages",
)
def can_playwright_wait(selector: str, selector_type: str) -> bool:
    """Whether ``wait_for_selector()`` can wait on this selector.

    A thin reader over :meth:`Selector.can_playwright_wait` for the callers
    that hold a ``(value, grammar)`` pair rather than a
    :class:`~jkent.common.selectors.Selector`; the grammar-specific rules
    live on the grammar.

    Args:
        selector: The selector string.
        selector_type: Type of selector ("xpath" or "css").

    Returns:
        True if Playwright can wait for this selector, False otherwise.
    """
    return Selector.of(selector, selector_type).can_playwright_wait()


class StepExecutor:
    """Store a response, run its step, and persist the yields atomically."""

    def __init__(
        self,
        db: SQLManager,
        scraper: Any,  # BaseScraper-like: needs get_step
        queue: RequestQueue,
        storage: ResponseStorage,
        *,
        handle_data: Callable[[Any], Awaitable[None]] | None = None,
        on_invalid_data: Callable[[DeferredValidation[Any]], Awaitable[None]]
        | None = None,
        on_progress: ProgressCallback | None = None,
        captures_incidentals: bool = True,
    ) -> None:
        self.db = db
        self.scraper = scraper
        self.queue = queue
        self.storage = storage
        self._handle_data = handle_data
        self.on_invalid_data = on_invalid_data
        self._on_progress = on_progress or no_progress
        # Read off the run's transport (``Transport.captures_incidentals``)
        # so an ``incidental=`` step meets a transport-shaped error instead
        # of a cardinality violation against zero captures.
        self._captures_incidentals = captures_incidentals

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        *,
        page: Any = None,
        store_response: bool = True,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> None:
        """Store response, run step, flush its yields, mark completed.

        ``speculation_outcome`` is what the worker's speculation tracker
        recorded for a probe (``hit``), stored with the response.
        """
        if store_response:
            await self.storage.store_response(
                request_id, response, step_name, speculation_outcome
            )

        if not step_name:
            await self.storage.mark_request_completed(request_id)
            return

        step = self.scraper.get_step(step_name)
        staged = StagedWrites(request_id=request_id)

        # Autowait dispatch: only a live page carrying an auto_await_timeout
        # takes the retry loop; everything else runs the normal generator path.
        auto_await_timeout: int | None = None
        if page is not None:
            metadata = get_step_metadata(step)
            auto_await_timeout = (
                metadata.auto_await_timeout if metadata else None
            )

        if page is not None and auto_await_timeout:
            await self._process_generator_with_autowait(
                step,
                response,
                request,
                request_id,
                auto_await_timeout,
                page=page,
                staged=staged,
            )
        else:
            gen = step(response)
            await self._process_generator_with_storage(
                gen,
                response,
                request,
                step_name,
                request_id,
                staged,
            )

        emitted_events = await staged.flush(self.db)
        for event in emitted_events:
            await self._on_progress("request_enqueued", event)

    async def _process_generator_with_storage(
        self,
        gen: Generator[ScraperYield[Any], bool | None, None],
        response: Response,
        parent_request: Request,
        step_name: str,
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
                            staged.stage_result(
                                serialize_result(e.failed_doc, e.errors)
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
        staged.stage_result(serialize_result(data))
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
        if not self._captures_incidentals:
            # No captures will ever exist on this transport, so the
            # cardinality check below would fail with "found 0" — a site
            # diagnosis for what is really a wiring mistake. Say so instead.
            raise ScraperConfigError(
                f"Step yields incidental={spec!r}, but the run's transport "
                "does not capture incidental sub-requests. Incidentals are "
                "recorded by a browser transport only — declare a browser "
                "driver requirement (e.g. DriverRequirement.JS_EVAL) on the "
                "scraper."
            )
        records = await self.db.get_incidental_requests(parent_request_id)

        matched = []
        for rec in records:
            headers: dict[str, str] = (
                _json.loads(rec.headers_json) if rec.headers_json else {}
            )
            body: bytes | None = None
            if spec.body_contains is not None:
                body = await self.db.get_incidental_request_body(rec.id)
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
        self, rec: IncidentalRequestRecord
    ) -> StoredResponse:
        """Build the response to store on a promoted incidental's request.

        Copies the already-compressed response body verbatim (no
        re-compression). Status and url come from the capture row, which
        records what *this* fetch saw; the response headers come from the
        shared storage row, so they are the first capture's (see
        ``replace_incidental_requests``). A capture with a status but no body
        (an excluded resource type) is stored with no body.

        Raises:
            IncidentalRequestAssumptionException: the capture never got a
                response — the browser aborted it or it failed — so there is
                nothing to hand the child step, and promoting it as an empty
                success would blame the child for the parent's fetch.
        """
        if rec.status_code is None:
            raise IncidentalRequestAssumptionException(
                f"matched incidental {rec.url} got no response"
                + (f" ({rec.failure_reason})" if rec.failure_reason else ""),
                rec.url,
            )
        storage = (
            await self.db.get_incidental_request_storage(rec.storage_id)
            if rec.storage_id is not None
            else None
        )
        if storage is None:
            return StoredResponse(
                response_status_code=rec.status_code,
                response_url=rec.url,
            )
        return StoredResponse(
            response_status_code=rec.status_code,
            response_headers_json=storage.response_headers_json,
            response_url=rec.url,
            **CompressedPayload.model_validate(storage).model_dump(),
        )

    @staticmethod
    def _prepare_incidental_child(
        item: Request,
        spec: Singular | Multiple,
        rec: IncidentalRequestRecord,
        index: int,
    ) -> Request:
        """Build the child request to enqueue for one promoted incidental.

        A promoted response can't be re-fetched standalone (the browser minted
        it once), so the child is non-reseedable. For ``Multiple`` matches the
        yielded request's single dedup key would collapse all matches into one
        row, so it is suffixed per match to keep them distinct.
        """
        dedup_key = item.deduplication_key
        effective_key = item.effective_deduplication_key
        if isinstance(spec, Multiple) and effective_key is not None:
            dedup_key = f"{effective_key}:{index}:{rec.url}"
        return replace(
            item,
            incidental=None,
            reseedable=False,
            deduplication_key=dedup_key,
        )

    async def _process_generator_with_autowait(
        self,
        step: Callable[..., Any],
        response: Response,
        parent_request: Request,
        request_id: int,
        auto_await_timeout: int,
        *,
        page: AutowaitPage,
        staged: StagedWrites,
    ) -> None:
        """Run the step, waiting on the live page for missing selectors.

        On an HTMLStructuralAssumptionException it waits for the offending
        selector in the browser, re-snapshots the DOM, and retries until
        success or the ``auto_await_timeout`` (ms) elapses.

        ``wait_for_selector`` returns on the first match, so waiting can only
        cure too few matches: the wait targets the ``expected_min``-th match,
        too many matches raise at once, and so does a wait that leaves the
        DOM unchanged (the step would fail the same way on it again), so
        every stored re-snapshot is a changed one.
        """
        start_time = time.time()
        timeout_seconds = auto_await_timeout / 1000.0

        while True:
            try:
                # Reset the staged buffer so a failed prior attempt's
                # partial yields (and its deferred on_data/on_invalid_data
                # callbacks) are discarded before the retry.
                staged.reset()
                gen = step(response)
                await self._process_generator_with_storage(
                    gen,
                    response,
                    parent_request,
                    step.__name__,
                    request_id,
                    staged,
                )
                break  # success

            except HTMLStructuralAssumptionException as e:
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(
                        f"Autowait timeout exhausted "
                        f"({auto_await_timeout}ms) for request {request_id}"
                    )
                    raise

                if (
                    e.expected_max is not None
                    and e.actual_count > e.expected_max
                ):
                    logger.debug(
                        f"Selector over-matches ({e.actual_count} > "
                        f"{e.expected_max}), skipping autowait: "
                        f"{e.selector}"
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

                # The step wrapper records the observer on the
                # per-execution Response, never on the shared StepMetadata.
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

                # Some matches exist already, so a plain wait would
                # return at once; wait for the last one still missing.
                if e.actual_count > 0:
                    absolute_selector = (
                        f"{absolute_selector} >> nth={e.expected_min - 1}"
                    )

                logger.info(
                    f"Autowait: waiting for selector {absolute_selector}"
                )

                # Floor at 1ms: Playwright treats timeout=0 as "wait
                # forever", so a sub-millisecond remaining budget must not
                # round down to 0.
                remaining_timeout = max(
                    1, int((timeout_seconds - elapsed) * 1000)
                )
                try:
                    await page.wait_for_selector(
                        absolute_selector, timeout=remaining_timeout
                    )
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"Autowait failed: selector {absolute_selector} "
                        f"not found within timeout"
                    )
                    raise e from None  # re-raise original exception

                # Re-snapshot DOM from the live page.
                content = utf8_document(await page.content())
                if content == response.content:
                    logger.warning(
                        f"Autowait: selector {absolute_selector} matched "
                        f"but the DOM is unchanged"
                    )
                    raise e from None
                response = Response(
                    status_code=response.status_code,
                    url=page.url,
                    content=content,
                    headers=response.headers,
                    request=response.request,
                )

                await self.storage.store_response(
                    request_id, response, step.__name__
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
