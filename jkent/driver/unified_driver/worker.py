"""Concrete worker: the per-worker execution loop for a unified-driver run.

It takes its collaborators explicitly: it leases a handle from the
``Transport``, pulls rows from the ``RequestQueue``, gates on the
``RateLimiter``, resolves via the transport, and hands the response to the
``ContinuationExecutor``. Retries/skips/marks go through ``ResponseStorage``;
duration is reported to the monitor and step completions to a ``Compactor``.

Transport recovery is opaque here (see ``worker_contract.md``): a dead resource
arrives as a ``TransientException`` and is retried like any other; the rebuild
happens inside the next ``transport.acquire``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

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

    from jkent.data_types import ArchiveDecision, Request
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
    """A runnable :class:`~jkent.driver.unified_driver.orchestration.Worker`.

    Leases a transport handle, drains the queue one request at a time, and
    routes failures by the exception taxonomy. Exits on the stop event or a
    durably empty queue, releasing its handle on the way out.
    """

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
        scraper: Any,
        archive_handler: Any,
        on_request_duration: Callable[[float], None] | None = None,
        compactor_for: Callable[[str], Compactor | None] | None = None,
        store_error: Callable[..., Awaitable[Any]] | None = None,
        track_speculation: Callable[[Request, Response], Awaitable[None]]
        | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._transport = transport
        self._rate_limiter = rate_limiter
        self._continuation = continuation
        self._storage = storage
        self._stop_event = stop_event
        self._scraper = scraper
        self._archive_handler = archive_handler
        self._on_request_duration = on_request_duration
        self._compactor_for = compactor_for
        self._store_error = store_error
        self._track_speculation = track_speculation

    @property
    def _strictly_serial(self) -> bool:
        """Whether the scraper requires strictly-serial processing."""
        return DriverRequirement.STRICTLY_SERIAL in getattr(
            self._scraper, "driver_requirements", []
        )

    async def run(self) -> None:
        """Process requests until shutdown or the queue is durably empty."""
        try:
            while not self._stop_event.is_set():
                result = await self._queue._get_next_request()
                if result is None:
                    return  # durably idle
                request_id, request, parent_request_id = result
                await self._handle_one(request_id, request, parent_request_id)
        finally:
            await self._transport.release(self.worker_id)

    async def _handle_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
    ) -> None:
        """Lease, gate, resolve, persist, and route failures for one request."""
        try:
            # Lease at the top of each attempt; a poisoned handle is rebuilt
            # here. acquire can raise TransientException — same handling as
            # resolve below.
            handle = await self._transport.acquire(self.worker_id)

            queued = QueuedRequest(
                request=request,
                request_id=request_id,
                parent_request_id=parent_request_id,
            )
            continuation_name = self._continuation_name(request)

            # Archive pre-check BEFORE gating: a skipped download does no
            # network I/O, so it must not consume a rate-limiter token.
            archive_decision: ArchiveDecision | None = None
            skip_download = False
            if getattr(request, "archive", False):
                archive_decision = await self._archive_should_download(request)
                skip_download = not archive_decision.download

            # Gate outside the timed region (and skip it for a skipped
            # download). gate itself no-ops on bypass / replay.
            if not skip_download:
                await self._rate_limiter.gate(request)

            # Time only the execute region.
            started = time.monotonic()
            if getattr(request, "archive", False):
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
            duration_s = time.monotonic() - started

            # Track speculation outcome for @speculate requests before the
            # continuation runs (on the success path).
            if (
                getattr(request, "is_speculative", False)
                and self._track_speculation is not None
            ):
                await self._track_speculation(request, response)

            # Persist + run continuation + mark complete. A Playwright
            # WorkerPage handle exposes a live ``.page`` (for autowait /
            # JSRequestPrep); HTTP/replay noop handles do not, so this is
            # None for them — a soft duck-typed capability, no protocol change.
            await self._continuation.complete_request(
                request_id,
                response,
                request,
                continuation_name,
                page=getattr(handle, "page", None),
            )

            # Report duration to the monitor and count toward the step's
            # compactor — but only for requests that store a compressible
            # response body. Archive requests persist file metadata (no body),
            # so counting them would trip the compactor into training a
            # compression dict over zero responses (ValueError).
            if self._on_request_duration is not None:
                self._on_request_duration(duration_s)
            if not getattr(request, "archive", False):
                await self._record_for_compactor(continuation_name)

        except RequestFailedHalt:
            raise  # propagate, stops the run
        except RequestFailedSkip:
            await self._storage._mark_request_failed(
                request_id, "Skipped by on_transient_exception callback"
            )
        except TransientException as e:
            # A transport may attach a partial-DOM snapshot taken before a
            # timeout (e.g. PlaywrightTransport's ResolveTimeout). Persist it
            # for debugging before the retry; the next attempt overwrites it.
            debug_response = getattr(e, "debug_response", None)
            if debug_response is not None:
                await self._storage._store_response(
                    request_id,
                    debug_response,
                    self._continuation_name(request),
                )
            retry_delay = await self._storage._handle_retry(request_id, e)
            if retry_delay is None:
                # Max backoff exceeded (or no retry state): give up — mark
                # failed and store the error.
                await self._storage._mark_request_failed(request_id, str(e))
                await self._store_error_for(
                    e, request_id, self._request_url(request)
                )
            elif self._strictly_serial:
                # Strict serialization: idle until the just-scheduled retry is
                # ready rather than pulling other pending work. Stop-event-aware
                # so a shutdown during the wait stays prompt.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=retry_delay
                    )
        except SpeculationHTTPFailure as e:
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
            track_speculation = self._track_speculation
            if (
                getattr(request, "is_speculative", False)
                and track_speculation is not None
            ):
                synthetic = Response(
                    status_code=e.status_code,
                    headers={},
                    content=b"",
                    text="",
                    url=e.url,
                    request=request,
                )
                await track_speculation(request, synthetic)
            await self._storage._mark_request_completed(request_id)
        except PersistentHTTPResponseException as e:
            # Classifier said this status is persistent: no retry.
            logger.warning(
                "Worker %d persistent HTTP %s on request %d: %s",
                self.worker_id,
                e.status_code,
                request_id,
                e.url,
            )
            await self._storage._mark_request_failed(request_id, str(e))
            await self._store_error_for(e, request_id, e.url)
        except Exception as e:
            logger.exception(
                "Worker %d error processing request %d",
                self.worker_id,
                request_id,
            )
            await self._storage._mark_request_failed(request_id, str(e))
            await self._store_error_for(
                e, request_id, self._request_url(request)
            )

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
                expected_type=getattr(request, "expected_type", None),
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
            expected_type=getattr(request, "expected_type", None),
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
        try:
            step = self._scraper.get_continuation(continuation_name)
        except AttributeError:
            return ()
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
