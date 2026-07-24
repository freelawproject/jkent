"""Concrete worker: the per-worker execution loop for a unified-driver run.

It takes its collaborators explicitly: it leases a handle from the
``Transport``, pulls rows from the ``RequestQueue``, gates on the
``RateLimiter``, resolves via the transport, and hands the response to the
``ContinuationExecutor``. Retries/skips/marks go through ``ResponseStorage``;
step completions are reported to a ``Compactor``.

Transport recovery is opaque here: a dead resource
arrives as a ``TransientException`` and is retried like any other; the rebuild
happens inside the next ``transport.acquire``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from jkent import observability as obs
from jkent.common.decorators import get_step_metadata
from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    RequestFailedHalt,
    RequestFailedSkip,
    SpeculationHTTPFailure,
    TransientException,
)
from jkent.data_types import ArchiveResponse, DriverRequirement, Response
from jkent.driver.unified_driver.transport import QueuedRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from jkent.data_types import ArchiveDecision, BaseScraper, Request
    from jkent.driver.unified_driver.circuit_breaker import CircuitBreaker
    from jkent.driver.unified_driver.continuation import ContinuationExecutor
    from jkent.driver.unified_driver.orchestration import Compactor
    from jkent.driver.unified_driver.persistence import (
        RequestQueue,
        ResponseStorage,
    )
    from jkent.driver.unified_driver.rate_limiter import RateLimiter
    from jkent.driver.unified_driver.transport import (
        ArchiveStream,
        AwaitCondition,
        Transport,
    )

logger = logging.getLogger(__name__)


class PoolWorker:
    """A single unit of execution within a run.

    Leases a per-worker handle from the transport, then loops: pull the next
    request from the queue, resolve it via the transport, persist the result,
    and route failures by the exception taxonomy (transient → retry,
    persistent → fail, halt/skip → propagate). Exits on the stop event or a
    drained queue — nothing pending and nothing in flight anywhere in the
    pool — releasing its handle on the way out. The pool is pinned at spawn
    time and never re-grown, so retirement is deliberately conservative.

    A worker observes transport failure at point of use — it is the one
    holding the handle when the resource dies — and recovers by asking the
    transport to restart, then renewing its lease (see
    :class:`~jkent.driver.unified_driver.lifecycle.Recoverable`).

    **Extension points (stable).** Subclasses replacing the request lifecycle
    (jent's ``ReplayWorker`` overrides :meth:`_handle_one` to route every
    miss/error through a replay miss policy) build on these members; treat a
    rename or signature change as a cross-repo breaking change
    (``tests/driver/unified/test_worker_extension_surface.py`` pins them):

    * hooks: ``_handle_one``, ``_execute_preresolved``
    * helpers: ``_continuation_name``, ``_await_conditions``,
      ``_record_for_compactor``, ``_store_error_for``, ``_request_url``
    * collaborators: ``_transport``, ``_storage``, ``_continuation`` (and
      ``ContinuationExecutor.complete_request``), ``_track_speculation``,
      ``_circuit_breaker``, plus ``worker_id``
    """

    # How long an idle worker sleeps before re-checking the queue while a
    # sibling still holds a request in flight (whose continuation may enqueue
    # children). Local SQLite reads make this poll effectively free.
    IN_FLIGHT_POLL_INTERVAL_S = 0.5

    def __init__(
        self,
        worker_id: int,
        *,
        queue: RequestQueue,
        transport: Transport[Any],
        rate_limiter: RateLimiter,
        continuation: ContinuationExecutor,
        storage: ResponseStorage,
        stop_event: asyncio.Event,
        scraper: BaseScraper[Any],
        archive_handler: Any,
        compactor_for: Callable[[str], Compactor | None] | None = None,
        store_error: Callable[..., Awaitable[Any]] | None = None,
        track_speculation: Callable[[Request, Response], Awaitable[None]]
        | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._transport = transport
        self._rate_limiter = rate_limiter
        # Optional like store_error/track_speculation: None means no breaker
        # (direct construction in tests/minimal hosts). ScrapeRun always
        # wires one — a single instance shared pool-wide, so the failure
        # count is a run-level signal. A per-worker default here would both
        # fracture that signal and stall transient-heavy standalone workers.
        self._circuit_breaker = circuit_breaker
        self._continuation = continuation
        self._storage = storage
        self._stop_event = stop_event
        self._scraper = scraper
        self._archive_handler = archive_handler
        self._compactor_for = compactor_for
        self._store_error = store_error
        self._track_speculation = track_speculation

    @property
    def _strictly_serial(self) -> bool:
        """Whether the scraper requires strictly-serial processing."""
        return (
            DriverRequirement.STRICTLY_SERIAL
            in self._scraper.driver_requirements
        )

    async def run(self) -> None:
        """Process requests until shutdown or the queue is durably empty."""
        # Bind the scraper label for this worker task's whole lifetime so even
        # the dequeue path's DB-lock waits are attributed. The contextvar is
        # per-task (each worker is its own asyncio.Task), so this does not bleed
        # across workers.
        with obs.labeled(scraper=self._scraper.__class__.__name__):
            try:
                await self._run_loop()
            finally:
                await self._transport.release(self.worker_id)

    async def _run_loop(self) -> None:
        """The dequeue/handle loop, run inside the scraper label scope."""
        while not self._stop_event.is_set():
            result = await self._queue.get_next_request()
            if result is None:
                # No request is ready right now. That can mean the queue
                # is truly drained, OR that the only remaining work is
                # retries still in their backoff window (pending rows with
                # a future started_at, which the dequeue skips), OR that a
                # sibling worker holds a request in flight whose continuation
                # may yet enqueue children. The pool is pinned — a retired
                # worker is never replaced — so only retire when none of
                # those can produce work: nothing pending now or later AND
                # nothing in flight. Otherwise sleep and re-check.
                delay = await self._queue.seconds_until_next_pending()
                if delay is None:
                    if self._queue.in_flight_count == 0:
                        return  # drained: nothing pending, nothing in flight
                    delay = self.IN_FLIGHT_POLL_INTERVAL_S
                idle_start = time.monotonic()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=delay
                    )
                # The pinned pool's utilization signal: time this worker sat
                # with nothing dequeuable. Compare against
                # request.duration{phase=total} to size num_workers.
                obs.instruments().worker_idle.record(
                    time.monotonic() - idle_start, obs.current_labels()
                )
                continue
            request_id, request, parent_request_id, preresolved = result
            try:
                await self._handle_one(
                    request_id, request, parent_request_id, preresolved
                )
            finally:
                # Balance get_next_request's in-flight claim on every exit —
                # including halt/cancel — so idle siblings never wait on a
                # request nobody is handling.
                self._queue.request_done()

    async def _handle_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
        preresolved: bool = False,
    ) -> None:
        """Lease, gate, resolve, persist, and route failures for one request."""

        # Compute the target step up front so it labels the whole request span
        # (and every phase/metric under it), including failure paths.
        continuation_name = self._continuation_name(request)
        outcome = obs.Outcome.OK
        with (
            obs.labeled(step=continuation_name),
            obs.request_span(
                scraper=self._scraper.__class__.__name__,
                step=continuation_name,
            ) as span,
        ):
            try:
                if preresolved:
                    await self._execute_preresolved(
                        request_id, request, continuation_name
                    )
                else:
                    await self._execute_one(
                        request_id,
                        request,
                        parent_request_id,
                        continuation_name,
                    )
            except RequestFailedHalt:
                outcome = obs.Outcome.HALT
                raise  # propagate, stops the run
            except RequestFailedSkip:
                outcome = obs.Outcome.SKIP
                await self._storage.mark_request_failed(
                    request_id, "Skipped by on_transient_exception callback"
                )
            except TransientException as e:
                outcome = obs.Outcome.TRANSIENT
                await self._handle_transient(request_id, request, e)
            except SpeculationHTTPFailure as e:
                outcome = obs.Outcome.SPECULATION_HTTP
                # A persistent status on a probe is still the server
                # answering — availability evidence for the breaker.
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                await self._handle_speculation_http(request_id, request, e)
            except PersistentHTTPResponseException as e:
                outcome = obs.Outcome.PERSISTENT_HTTP
                # The server answered; a persistent status is a routing
                # verdict about this request, not server distress — the
                # breaker counts it as availability, not failure.
                if self._circuit_breaker is not None:
                    self._circuit_breaker.record_success()
                # Classifier said this status is persistent: no retry. Persist
                # the observed response (body/headers travel on the exception)
                # so the failure is inspectable from the run db.
                logger.warning(
                    "Worker %d persistent HTTP %s on request %d: %s",
                    self.worker_id,
                    e.status_code,
                    request_id,
                    e.url,
                )
                error_response = self._response_from_http_error(e, request)
                if error_response is not None:
                    await self._storage.store_response(
                        request_id, error_response, continuation_name
                    )
                await self._storage.mark_request_failed(request_id, str(e))
                await self._store_error_for(e, request_id, e.url)
            except Exception as e:
                outcome = obs.Outcome.ERROR
                logger.exception(
                    "Worker %d error processing request %d",
                    self.worker_id,
                    request_id,
                )
                await self._storage.mark_request_failed(request_id, str(e))
                await self._store_error_for(
                    e, request_id, self._request_url(request)
                )
            finally:
                span.set_attribute("jkent.outcome", outcome)

    async def _execute_preresolved(
        self,
        request_id: int,
        request: Request,
        continuation_name: str,
    ) -> None:
        """Run a pre-resolved request's continuation without the transport.

        The response was stored at enqueue time (promoted from a captured
        incidental sub-request), so there is no lease, no rate-limit gate, and
        no network I/O: load the stored response and run the continuation.
        ``store_response=False`` because the row already holds it.
        """
        response = await self._storage.load_preresolved_response(
            request_id, request
        )
        if response is None:
            # preresolved implies a stored response; a missing one is a bug in
            # the enqueue path, not a retryable condition.
            raise RuntimeError(
                f"pre-resolved request {request_id} has no stored response"
            )
        with obs.phase(obs.Phase.CONTINUATION):
            await self._continuation.complete_request(
                request_id,
                response,
                request,
                continuation_name,
                page=None,
                store_response=False,
            )

    async def _handle_transient(
        self, request_id: int, request: Request, e: TransientException
    ) -> None:
        """Route a transient failure: persist debug snapshot, retry or fail."""
        # Every transient — resolve, acquire, timeout, any transport — funnels
        # through here, so this is the breaker's single failure-count site.
        if self._circuit_breaker is not None:
            self._circuit_breaker.record_failure()
        # A transport may attach a partial-DOM snapshot taken before a
        # timeout (e.g. PlaywrightTransport's ResolveTimeout). Failing that,
        # a transient HTTP error carries the observed headers/body on the
        # exception itself. Persist either for debugging before the retry;
        # the next attempt (or an eventual success) overwrites it.
        debug_response = getattr(e, "debug_response", None)
        if debug_response is None:
            debug_response = self._response_from_http_error(e, request)
        if debug_response is not None:
            await self._storage.store_response(
                request_id,
                debug_response,
                self._continuation_name(request),
            )
        retry_delay = await self._storage.handle_retry(request_id, e)
        if retry_delay is None:
            # Max backoff exceeded (or no retry state): give up — mark
            # failed and store the error.
            await self._storage.mark_request_failed(request_id, str(e))
            await self._store_error_for(
                e, request_id, self._request_url(request)
            )
            return
        # A retry rate that climbs with worker count is server pushback —
        # the counter-signal against raising num_workers.
        obs.instruments().request_retries.add(1, obs.current_labels())
        if self._strictly_serial:
            # Strict serialization: idle until the just-scheduled retry is
            # ready rather than pulling other pending work. Stop-event-aware
            # so a shutdown during the wait stays prompt.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=retry_delay
                )

    async def _handle_speculation_http(
        self, request_id: int, request: Request, e: SpeculationHTTPFailure
    ) -> None:
        """Route a persistent-HTTP result on a speculative probe."""
        track_speculation = self._track_speculation
        if not request.is_speculative or track_speculation is None:
            # SpeculationHTTPFailure only makes sense for a speculative
            # probe with a tracker wired up. If it ever reaches a
            # non-speculative request (or one with no tracker), do NOT
            # silently mark it completed — that would record a persistent
            # HTTP failure as a success and drop it. Treat it as a failure.
            logger.warning(
                "Worker %d got SpeculationHTTPFailure on non-speculative "
                "request %d (HTTP %s): %s",
                self.worker_id,
                request_id,
                e.status_code,
                e.url,
            )
            await self._storage.mark_request_failed(request_id, str(e))
            await self._store_error_for(e, request_id, e.url)
        else:
            # Persistent HTTP on a speculative probe: record it as a
            # speculation outcome (not an error), then mark complete. No
            # retry, no continuation, no error row.
            logger.info(
                "Worker %d speculation probe HTTP %s on request %d: %s",
                self.worker_id,
                e.status_code,
                request_id,
                e.url,
            )
            synthetic = Response(
                status_code=e.status_code,
                headers={},
                content=b"",
                text="",
                url=e.url,
                request=request,
            )
            await track_speculation(request, synthetic)
            await self._storage.mark_request_completed(request_id)

    async def _execute_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
        continuation_name: str,
    ) -> None:
        """The success path: lease, gate, resolve, persist, run continuation.

        Failures propagate to :meth:`_handle_one`, which routes them by the
        exception taxonomy.
        """
        # Lease at the top of each attempt; a poisoned handle is rebuilt
        # here. acquire can raise TransientException — same handling as
        # resolve below.
        handle = await self._transport.acquire(self.worker_id)

        queued = QueuedRequest(
            request=request,
            request_id=request_id,
            parent_request_id=parent_request_id,
        )
        is_archive = request.archive

        # Archive pre-check BEFORE gating: a skipped download does no
        # network I/O, so it must not consume a rate-limiter token.
        archive_decision: ArchiveDecision | None = None
        skip_download = False
        if is_archive:
            archive_decision = await self._archive_should_download(request)
            skip_download = not archive_decision.download

        # Gate outside the timed region (and skip it for a skipped
        # download). Circuit breaker first: an open circuit must not claim a
        # rate-limiter slot, and when it closes the limiter re-spaces the
        # released workers. gate itself no-ops on bypass / replay.
        if not skip_download:
            breaker = self._circuit_breaker
            if breaker is not None:
                with obs.phase(obs.Phase.CIRCUIT_BREAKER_GATE):
                    await breaker.gate()
            with obs.phase(obs.Phase.RATE_LIMITER_GATE):
                await self._rate_limiter.gate(request)

        # Re-stamp the persisted start after the gate so a DB-derived
        # duration reflects the execute region, not time spent waiting for
        # a rate-limiter token (started_at was stamped at dequeue).
        await self._queue.restamp_request_start(request_id)

        with obs.phase(obs.Phase.TRANSPORT_RESOLVE):
            if is_archive:
                response = await self._resolve_archive(
                    handle,
                    queued,
                    archive_decision,
                    skip_download=skip_download,
                )
            else:
                response = await self._transport.resolve(
                    handle,
                    queued,
                    await_conditions=self._await_conditions(continuation_name),
                )

        # The server answered: reset the breaker's consecutive-failure count
        # (and close the circuit if a straggler/probe just proved recovery).
        # A skipped archive download did no network I/O, so it is no evidence.
        if not skip_download and self._circuit_breaker is not None:
            self._circuit_breaker.record_success()

        # Track speculation outcome for @speculate requests before the
        # continuation runs (on the success path).
        if request.is_speculative and self._track_speculation is not None:
            await self._track_speculation(request, response)

        # Persist + run continuation + mark complete. A Playwright
        # WorkerPage handle exposes a live ``.page`` (for autowait);
        # HTTP/replay noop handles do not, so this is None for them —
        # a soft duck-typed capability, no protocol change.
        with obs.phase(obs.Phase.CONTINUATION):
            await self._continuation.complete_request(
                request_id,
                response,
                request,
                continuation_name,
                page=getattr(handle, "page", None),
            )

        # Count toward the step's compactor — but only for requests that
        # store a compressible response body. Archive requests persist file
        # metadata (no body), so counting them would trip the compactor into
        # training a compression dict over zero responses (ValueError).
        if not is_archive:
            await self._record_for_compactor(continuation_name)

    async def _resolve_archive(
        self,
        handle: Any,
        queued: QueuedRequest,
        decision: ArchiveDecision | None,
        *,
        skip_download: bool,
    ) -> Response:
        """Resolve an archive request into an ``ArchiveResponse``.

        On a skip decision, returns a synthetic response pointing at the
        existing file with no network I/O. Otherwise streams the body via the
        transport, saves it through the archive handler, and releases the
        transport-side backing with ``finish_archiving``.
        """
        request = queued.request
        if skip_download:
            assert decision is not None
            return ArchiveResponse(
                status_code=200,
                headers={},
                content=b"",
                text="",
                url=request.request.url,
                request=request,
                file_url=decision.file_url,
            )

        dedup_key = (
            request.deduplication_key
            if isinstance(request.deduplication_key, str)
            else None
        )
        stream: ArchiveStream = await self._transport.resolve_archive(
            handle, queued, decision=decision
        )
        try:
            file_url = await self._archive_handler.save_stream(
                url=request.request.url,
                deduplication_key=dedup_key,
                expected_type=request.expected_type,
                hash_header_value=None,
                chunks=aiter(stream),
            )
        finally:
            await self._transport.finish_archiving(stream)

        return ArchiveResponse(
            status_code=stream.status_code,
            headers=dict(stream.headers),
            content=b"",
            text="",
            url=request.request.url,
            request=request,
            file_url=file_url,
        )

    async def _archive_should_download(
        self, request: Request
    ) -> ArchiveDecision:
        """Consult the archive handler's ``should_download`` for ``request``."""
        dedup_key = (
            request.deduplication_key
            if isinstance(request.deduplication_key, str)
            else None
        )
        return await self._archive_handler.should_download(
            url=request.request.url,
            deduplication_key=dedup_key,
            expected_type=request.expected_type,
            hash_header_value=None,
        )

    def _continuation_name(self, request: Request) -> str:
        """Resolve the request's continuation to its method name."""
        continuation = request.continuation
        if isinstance(continuation, str):
            return continuation
        return continuation.__name__

    def _await_conditions(
        self, continuation_name: str
    ) -> Sequence[AwaitCondition]:
        """Derive resolve await-conditions from the target step's await_list."""
        if not continuation_name:
            return ()
        step = self._scraper.get_continuation(continuation_name)
        metadata = get_step_metadata(step)
        if metadata is None:
            return ()
        return tuple(metadata.await_list)

    async def _record_for_compactor(self, continuation_name: str) -> None:
        """Count one completed request toward its step's compactor, if any."""
        if self._compactor_for is None or not continuation_name:
            return
        compactor = self._compactor_for(continuation_name)
        if compactor is not None:
            await compactor.record_request()

    async def _store_error_for(
        self, exc: Exception, request_id: int, request_url: str | None
    ) -> None:
        """Store an error via the injected sink, if one was provided."""
        if self._store_error is None:
            return
        await self._store_error(
            exc, request_id=request_id, request_url=request_url
        )

    def _request_url(self, request: Request) -> str | None:
        """Best-effort URL for error reporting."""
        try:
            return request.request.url
        except AttributeError:
            return None

    def _response_from_http_error(
        self, e: Exception, request: Request
    ) -> Response | None:
        """Rebuild the observed response from an HTTP-error exception.

        ``classify_and_raise`` attaches the status/headers/body it classified
        on to the exception; reassemble them into a ``Response`` so the failed
        exchange can be stored like a successful one. Returns None when the
        exception carries no response payload (headers and body both absent —
        e.g. a plain network-level transient), so callers skip the store.
        """
        status_code = getattr(e, "status_code", None)
        headers = getattr(e, "headers", None)
        body = getattr(e, "body", None)
        if status_code is None or (headers is None and body is None):
            return None
        content = body or b""
        return Response(
            status_code=status_code,
            headers=dict(headers) if headers else {},
            content=content,
            text=content.decode("utf-8", errors="replace"),
            url=getattr(e, "url", None) or self._request_url(request) or "",
            request=request,
        )
