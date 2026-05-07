"""WorkerMixin - Worker management and request processing loop."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING, Any

from pyrate_limiter import Limiter

from kent.common.exceptions import (
    PersistentHTTPResponseException,
    RequestFailedHalt,
    RequestFailedSkip,
    SpeculationHTTPFailure,
    TransientException,
)
from kent.data_types import (
    BaseRequest,
    BaseScraper,
    Request,
    Response,
    ScraperYield,
)
from kent.driver.persistent_driver._staging import StagedWrites
from kent.driver.persistent_driver.sql_manager import SQLManager
from kent.driver.sync_driver import SpeculationState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from pyrate_limiter import Rate

    from kent.common.deferred_validation import DeferredValidation
    from kent.common.exceptions import (
        ScraperAssumptionException,
    )
    from kent.data_types import ArchiveResponse
    from kent.driver.archive_handler import (
        AsyncArchiveHandler,
        AsyncStreamingArchiveHandler,
    )

logger = logging.getLogger(__name__)


class WorkerMixin:
    """Worker lifecycle, dynamic scaling, and request processing.

    Provides the main worker loop (_db_worker), worker scaling (_worker_monitor),
    and request processing (_process_regular_request, _process_generator_with_storage).
    """

    db: SQLManager
    scraper: BaseScraper
    stop_event: asyncio.Event
    max_workers: int
    num_workers: int
    request_manager: Any
    rate_limiter: Limiter | None
    archive_handler: AsyncArchiveHandler | AsyncStreamingArchiveHandler
    _rates: list[Rate] | None
    _worker_tasks: dict[int, asyncio.Task[None]]
    _next_worker_id: int
    _speculation_state: dict[str, SpeculationState]
    # Callback attrs — defined on AsyncDriver, annotated here for mypy
    on_progress: Callable[..., Awaitable[None]] | None
    on_invalid_data: Callable[[DeferredValidation], Awaitable[None]] | None
    on_structural_error: (
        Callable[[ScraperAssumptionException], Awaitable[bool]] | None
    )

    if TYPE_CHECKING:

        async def _emit_progress(
            self, event_type: str, data: dict[str, Any]
        ) -> None: ...

        # Provided by QueueMixin
        async def _get_next_request(
            self,
        ) -> tuple[int, BaseRequest, int | None] | None: ...

        async def enqueue_request(
            self,
            new_request: BaseRequest,
            context: Response | BaseRequest,
            parent_request_id: int | None = None,
        ) -> None: ...

        async def _stage_enqueue_request(
            self,
            new_request: BaseRequest,
            context: Response | BaseRequest,
            parent_request_id: int | None,
            staged: StagedWrites,
        ) -> None: ...

        # Provided by StorageMixin
        async def _mark_request_completed(self, request_id: int) -> None: ...

        async def _mark_request_failed(
            self, request_id: int, error_message: str
        ) -> None: ...

        async def _handle_retry(
            self, request_id: int, error: Exception
        ) -> bool: ...

        async def _store_response(
            self,
            request_id: int,
            response: Response,
            continuation: str,
            speculation_outcome: str | None = None,
        ) -> int: ...

        async def _store_result(
            self,
            request_id: int,
            data: Any,
            is_valid: bool = True,
            validation_errors: list[dict[str, Any]] | None = None,
        ) -> int: ...

        @staticmethod
        def _serialize_result_for_storage(
            data: Any,
            validation_errors: list[dict[str, Any]] | None = None,
        ) -> tuple[str, str, str | None]: ...

        # Provided by SpeculationMixin
        async def _track_speculation_outcome(
            self, request: BaseRequest, response: Response
        ) -> None: ...

        # Provided by AsyncDriver
        async def resolve_request(self, request: BaseRequest) -> Response: ...

        async def resolve_archive_request(
            self,
            request: Request,
            archive_decision: Any = None,
        ) -> ArchiveResponse: ...

        async def handle_data(self, data: Any) -> None: ...

        # Provided by PlaywrightDriver (optional — only called when page is set)
        async def _process_generator_with_autowait(
            self,
            continuation: Any,
            response: Response,
            parent_request: BaseRequest,
            request_id: int,
            auto_await_timeout: int,
            page: Any = None,
            staged: StagedWrites | None = None,
        ) -> None: ...

    # --- Worker Management ---

    @property
    def active_worker_count(self) -> int:
        """Number of currently active workers."""
        return sum(1 for t in self._worker_tasks.values() if not t.done())

    def _spawn_worker(self) -> int:
        """Spawn a new worker and return its ID.

        Returns:
            The worker ID of the newly spawned worker.
        """
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        task = asyncio.create_task(self._db_worker(worker_id))
        self._worker_tasks[worker_id] = task

        # Clean up when worker exits
        def on_worker_done(
            _: asyncio.Task[None], wid: int = worker_id
        ) -> None:
            self._worker_tasks.pop(wid, None)

        task.add_done_callback(on_worker_done)

        logger.info(
            f"Spawned worker {worker_id}, total active: {self.active_worker_count}"
        )
        return worker_id

    async def _worker_monitor(self) -> None:
        """Monitor task that dynamically scales workers and manages compression.

        Each 60 s cycle performs two checks:

        **Worker scaling** — adds a worker if:
        - There are pending requests
        - active_worker_count < workers_needed (based on rate limit headroom)
        - active_worker_count < max_workers

        workers_needed is ``ceil(max_allowed_rate * avg_request_duration)``.
        If a single request takes 2 s on average and the rate limit allows
        5 req/s, then 10 workers are needed to saturate the rate limit.

        **Compression dict training** — for any continuation with 1000+
        responses that lack a compression dictionary, trains a new zstd
        dictionary from 1000 sample responses and recompresses all
        existing responses for that continuation.

        Exits when:
        - stop_event is set, OR
        - active_worker_count == 0 and no pending requests
        """
        from kent.driver.persistent_driver.scoped_session import (
            clear_scope,
            set_scope,
        )

        set_scope("monitor")
        logger.info(
            f"Worker monitor started (max_workers={self.max_workers}, "
            f"poll_interval=60s)"
        )

        while not self.stop_event.is_set():
            # Wait 60 seconds between checks
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=60.0)
                # If we get here, stop_event was set
                break
            except asyncio.TimeoutError:
                # Normal timeout - proceed with check
                pass

            # Check exit condition: no workers and no pending work
            active_count = self.active_worker_count
            pending_count = await self.db.count_pending_requests()

            if active_count == 0 and pending_count == 0:
                logger.info(
                    "Worker monitor exiting: no workers and no pending requests"
                )
                break

            # --- Worker scaling ---
            if pending_count == 0:
                logger.debug(
                    f"Worker monitor: no pending requests "
                    f"(active_workers={active_count})"
                )
            elif active_count >= self.max_workers:
                logger.debug(
                    f"Worker monitor: at max workers "
                    f"({active_count}/{self.max_workers})"
                )
            else:
                # Determine how many workers are needed to saturate the
                # configured rate limit given observed request durations.
                if self._rates:
                    # Most restrictive rate expressed as requests/second.
                    max_rate_per_sec: float | None = min(
                        r.limit / (r.interval / 1000) for r in self._rates
                    )
                else:
                    max_rate_per_sec = None

                avg_duration_s = (
                    await self.db.avg_completed_request_duration_s()
                )

                if max_rate_per_sec is None:
                    # No rate limits — scale to max_workers if pending.
                    workers_needed = self.max_workers
                elif avg_duration_s is None:
                    # No timing data yet — be conservative.
                    workers_needed = active_count + 1
                else:
                    import math

                    workers_needed = max(
                        math.ceil(max_rate_per_sec * avg_duration_s), 1
                    )

                if active_count < workers_needed:
                    new_worker_id = self._spawn_worker()
                    logger.info(
                        f"Worker monitor: scaled up to "
                        f"{self.active_worker_count} workers "
                        f"(workers_needed={workers_needed}, "
                        f"avg_duration={avg_duration_s}, "
                        f"max_rate={max_rate_per_sec}, "
                        f"pending={pending_count})"
                    )

                    await self._emit_progress(
                        "worker_scaled",
                        {
                            "worker_id": new_worker_id,
                            "active_workers": self.active_worker_count,
                            "max_rate_per_sec": max_rate_per_sec,
                            "avg_duration_s": avg_duration_s,
                            "workers_needed": workers_needed,
                            "pending_requests": pending_count,
                        },
                    )
                else:
                    logger.debug(
                        f"Worker monitor: no scale-up needed "
                        f"(active={active_count}, needed={workers_needed}, "
                        f"max_workers={self.max_workers}, "
                        f"avg_duration={avg_duration_s}, "
                        f"max_rate={max_rate_per_sec})"
                    )

            # --- Auto-train compression dictionaries ---
            try:
                continuations = (
                    await self.db.continuations_needing_compression_dict()
                )
                for cont in continuations:
                    from kent.driver.persistent_driver.compression import (
                        recompress_responses,
                        train_compression_dict,
                    )

                    logger.info(
                        f"Worker monitor: training compression dict "
                        f"for continuation '{cont}'"
                    )
                    dict_id = await train_compression_dict(
                        self.db._session_factory,
                        cont,
                        sample_limit=1000,
                        db_lock=self.db._lock,
                    )
                    count, orig, compressed = await recompress_responses(
                        self.db._session_factory,
                        cont,
                        dict_id=dict_id,
                        db_lock=self.db._lock,
                    )
                    logger.info(
                        f"Worker monitor: trained dict {dict_id} and "
                        f"recompressed {count} responses for '{cont}' "
                        f"({orig} -> {compressed} bytes)"
                    )

                    await self._emit_progress(
                        "compression_dict_trained",
                        {
                            "continuation": cont,
                            "dict_id": dict_id,
                            "recompressed_count": count,
                            "original_bytes": orig,
                            "compressed_bytes": compressed,
                        },
                    )
            except Exception:
                logger.exception(
                    "Worker monitor: error during compression dict training"
                )

        # Clean up all scoped sessions (monitor + any leaked worker sessions)
        await self.db._session_factory.remove_all()
        clear_scope()
        logger.info("Worker monitor stopped")

    # --- Request Processing ---

    async def _db_worker(self, worker_id: int) -> None:
        """Worker that processes requests from the database queue.

        Handles regular requests (navigating, non-navigating, and archive).
        Speculative requests are handled via the new @speculate decorator pattern.

        Args:
            worker_id: Identifier for this worker.
        """
        import time as time_module

        from kent.driver.persistent_driver.scoped_session import (
            clear_scope,
            set_scope,
        )

        scope_key = f"worker-{worker_id}"
        set_scope(scope_key)
        logger.info(f"[W{worker_id}] Worker started (scope={scope_key})")
        requests_processed = 0

        while True:
            loop_start = time_module.time()

            # Check for graceful shutdown
            if self.stop_event.is_set():
                logger.info(
                    f"[W{worker_id}] Exiting: stop_event set (processed {requests_processed} requests)"
                )
                break

            # Get next request from DB
            result = await self._get_next_request()

            if result is None:
                # No immediately available requests - check for scheduled retries
                retry_delay = await self.db.get_next_scheduled_retry_delay()

                if retry_delay is not None and retry_delay > 0:
                    # There are scheduled retries - wait for the next one
                    # Add a small buffer and cap at a reasonable max wait
                    wait_time = min(retry_delay + 0.1, 60.0)
                    logger.info(
                        f"[W{worker_id}] Waiting {wait_time:.1f}s for scheduled retry"
                    )
                    await asyncio.sleep(wait_time)

                    # Check for shutdown after waiting
                    if self.stop_event.is_set():
                        break

                    # Try again after waiting
                    result = await self._get_next_request()
                    if result is None:
                        # Still nothing - continue loop to check again
                        continue
                else:
                    # No scheduled retries - poll for new work
                    # Other workers may still be processing and generating new requests
                    # Poll at moderate rate (100ms) to balance responsiveness and DB load
                    consecutive_empty = 0
                    max_polls = 100  # 10 seconds max polling

                    for poll_attempt in range(max_polls):
                        # Wait before retry (100ms gives good balance)
                        await asyncio.sleep(0.1)

                        # Check for shutdown
                        if self.stop_event.is_set():
                            logger.info(
                                f"[W{worker_id}] Stop event during polling"
                            )
                            break

                        # Try to get work - this is the only DB call per iteration
                        result = await self._get_next_request()
                        if result is not None:
                            logger.info(
                                f"[W{worker_id}] Found work after {poll_attempt + 1} polls"
                            )
                            break

                        # Check exit condition periodically (every 0.5s)
                        if poll_attempt % 5 == 4:
                            in_progress_count = (
                                await self.db.count_in_progress()
                            )
                            pending_count = (
                                await self.db.count_pending_requests()
                            )

                            if in_progress_count == 0 and pending_count == 0:
                                consecutive_empty += 1
                                if (
                                    consecutive_empty >= 6
                                ):  # ~3 seconds of true idle
                                    logger.info(
                                        f"[W{worker_id}] Exiting: idle (processed {requests_processed})"
                                    )
                                    break
                            else:
                                consecutive_empty = 0

                            if poll_attempt % 20 == 19:
                                logger.info(
                                    f"[W{worker_id}] Polling... in_progress={in_progress_count}, pending={pending_count}"
                                )

                    if result is None:
                        logger.info(
                            f"[W{worker_id}] Exiting: queue empty after polling (processed {requests_processed} requests)"
                        )
                        break

            request_id, request, parent_request_id = result
            logger.debug(f"[W{worker_id}] Dequeued request {request_id}")

            try:
                await self._emit_progress(
                    "request_started",
                    {
                        "request_id": request_id,
                        "url": request.request.url,
                        "continuation": request.continuation,
                    },
                )

                # Get continuation name
                continuation_name = (
                    request.continuation
                    if isinstance(request.continuation, str)
                    else request.continuation.__name__
                )

                # Decide whether to skip the rate limiter for this request.
                bypass = getattr(request, "bypass_rate_limit", False)
                archive_decision = None
                if getattr(request, "archive", False):
                    # Pre-check archive handler so skipped downloads
                    # don't consume a rate-limiter token.
                    dedup_key = (
                        request.deduplication_key
                        if isinstance(request.deduplication_key, str)
                        else None
                    )
                    archive_decision = (
                        await self.archive_handler.should_download(
                            url=request.request.url,
                            deduplication_key=dedup_key,
                            expected_type=getattr(
                                request, "expected_type", None
                            ),
                            hash_header_value=None,
                        )
                    )
                    if not archive_decision.download:
                        bypass = True

                if self.rate_limiter and not bypass:
                    await self.rate_limiter.try_acquire_async(
                        name="request", weight=1
                    )
                    # Re-stamp start time so duration excludes rate limiter wait
                    await self.db.restamp_request_start(request_id)

                # Process the request
                req_start = time_module.time()
                await self._process_regular_request(
                    request_id,
                    request,  # type: ignore[arg-type]
                    continuation_name,
                    parent_request_id=parent_request_id,
                    worker_id=worker_id,
                    archive_decision=archive_decision,
                )
                req_time = time_module.time() - req_start
                loop_time = time_module.time() - loop_start
                requests_processed += 1
                logger.info(
                    f"[W{worker_id}] Completed request {request_id} in {req_time * 1000:.1f}ms (loop={loop_time * 1000:.1f}ms, total={requests_processed})"
                )

            except RequestFailedHalt:
                # User callback requested halt - propagate up
                raise

            except RequestFailedSkip:
                # User callback requested skip - mark as failed and continue
                await self._mark_request_failed(
                    request_id, "Skipped by on_transient_exception callback"
                )
                await self._emit_progress(
                    "request_skipped",
                    {
                        "request_id": request_id,
                        "url": request.request.url,
                        "reason": "callback_requested_skip",
                    },
                )
                continue

            except TransientException as e:
                should_retry = await self._handle_retry(request_id, e)
                if should_retry:
                    # Log at warning level without full traceback for transient errors
                    logger.warning(
                        f"Worker {worker_id} transient error on request "
                        f"{request_id}: {type(e).__name__}: {e}"
                    )
                    await self._emit_progress(
                        "request_retry_scheduled",
                        {
                            "request_id": request_id,
                            "url": request.request.url,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    continue  # Don't store as error, will be retried
                else:
                    # Max backoff exceeded - log the full traceback and mark failed
                    logger.exception(
                        f"Worker {worker_id} transient error exceeded max "
                        f"backoff for request {request_id}"
                    )

                    # Mark as failed and store error
                    await self._mark_request_failed(request_id, str(e))

                    from kent.driver.persistent_driver.errors import (
                        store_error,
                    )

                    await store_error(
                        self.db._session_factory,
                        e,
                        request_id=request_id,
                        request_url=request.request.url,
                        db_lock=self.db._lock,
                    )

                    await self._emit_progress(
                        "request_failed",
                        {
                            "request_id": request_id,
                            "url": request.request.url,
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "reason": "max_backoff_exceeded",
                        },
                    )

            except SpeculationHTTPFailure as e:
                # Persistent HTTP on a speculative probe — record as a
                # speculation outcome, not an error. No retries, no
                # continuation, no errors-table row.
                from kent.data_types import Response as _Response

                logger.info(
                    f"Worker {worker_id} speculation probe returned "
                    f"HTTP {e.status_code} on request {request_id}: {e.url}"
                )
                synthetic = _Response(
                    status_code=e.status_code,
                    headers={},
                    content=b"",
                    text="",
                    url=e.url,
                    request=request,
                )
                if request.is_speculative and self._speculation_state:
                    await self._track_speculation_outcome(request, synthetic)
                await self._mark_request_completed(request_id)
                await self._emit_progress(
                    "request_completed",
                    {
                        "request_id": request_id,
                        "url": e.url,
                        "reason": "speculation_failure",
                        "status_code": e.status_code,
                    },
                )

            except PersistentHTTPResponseException as e:
                # Classifier said this status is persistent — don't retry,
                # don't bury the operator in traceback output.
                logger.warning(
                    f"Worker {worker_id} persistent HTTP {e.status_code} on "
                    f"request {request_id}: {e.url}"
                )
                await self._mark_request_failed(request_id, str(e))
                from kent.driver.persistent_driver.errors import (
                    store_error,
                )

                await store_error(
                    self.db._session_factory,
                    e,
                    request_id=request_id,
                    request_url=e.url,
                    db_lock=self.db._lock,
                )

                await self._emit_progress(
                    "request_failed",
                    {
                        "request_id": request_id,
                        "url": e.url,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "reason": "persistent_http_error",
                    },
                )

            except Exception as e:
                # Non-transient error - log full traceback
                logger.exception(
                    f"Worker {worker_id} error processing request {request_id}"
                )

                # Non-transient error or max backoff exceeded - mark as failed
                await self._mark_request_failed(request_id, str(e))

                # Store error in database for tracking
                from kent.driver.persistent_driver.errors import (
                    store_error,
                )

                await store_error(
                    self.db._session_factory,
                    e,
                    request_id=request_id,
                    request_url=request.request.url,
                    db_lock=self.db._lock,
                )

                await self._emit_progress(
                    "request_failed",
                    {
                        "request_id": request_id,
                        "url": request.request.url,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

        # Worker exiting normally — clean up scoped session
        await self.db._session_factory.remove(scope_key)
        clear_scope()

    async def _complete_request(
        self,
        request_id: int,
        response: Response,
        request: BaseRequest,
        continuation_name: str,
        *,
        speculation_outcome: str | None = None,
        page: Any = None,
        store_response: bool = True,
    ) -> None:
        """Store response, run continuation, and mark request completed.

        This is the shared tail of every ``_process_regular_request``
        implementation — both the HTTP driver and the Playwright driver
        converge here after obtaining a response.

        Args:
            request_id: Database ID of the request.
            response: The response to store and pass to the continuation.
            request: The original request (used as parent context for
                any new requests yielded by the continuation).
            continuation_name: Name of the continuation method.
            speculation_outcome: Optional speculation outcome to record.
            page: Playwright page for autowait retry (Playwright driver only).
            store_response: If False, skip storing the response (caller
                already stored it, e.g. for incidental request tracking).
        """
        if store_response:
            await self._store_response(
                request_id, response, continuation_name, speculation_outcome
            )

        if not continuation_name:
            await self._mark_request_completed(request_id)
            return

        continuation = self.scraper.get_continuation(continuation_name)
        staged = StagedWrites(request_id=request_id)

        # Check for autowait (Playwright driver only)
        auto_await_timeout: int | None = None
        if page is not None:
            from kent.common.decorators import get_step_metadata

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

    async def _process_regular_request(
        self,
        request_id: int,
        request: Request,
        continuation_name: str,
        parent_request_id: int | None = None,
        worker_id: int = 0,
        archive_decision: Any = None,
    ) -> None:
        """Process a regular (non-speculative, non-resume) request.

        Args:
            request_id: Database ID of the request.
            request: The request to process.
            continuation_name: Name of the continuation method.
            parent_request_id: Parent request ID for tab forking (Playwright).
            worker_id: Identifier of the calling worker (used by Playwright driver).
            archive_decision: Pre-computed ArchiveDecision from the worker loop.
                Passed through to ``resolve_archive_request`` to avoid a
                redundant ``should_download()`` call.
        """
        logger.info(f"Request {request_id}: starting HTTP fetch")
        response: Response = (
            await self.resolve_archive_request(
                request, archive_decision=archive_decision
            )
            if request.archive
            else await self.resolve_request(request)
        )
        logger.info(
            f"Request {request_id}: HTTP fetch complete, status={response.status_code}"
        )

        # Track speculation outcome for @speculate requests
        if request.is_speculative and self._speculation_state:
            await self._track_speculation_outcome(request, response)

        await self._complete_request(
            request_id, response, request, continuation_name
        )

        await self._emit_progress(
            "request_completed",
            {
                "request_id": request_id,
                "url": request.request.url,
            },
        )

    async def _process_generator_with_storage(
        self,
        gen: Generator[ScraperYield, bool | None, None],
        response: Response,
        parent_request: BaseRequest,
        continuation_name: str,
        request_id: int,
        staged: StagedWrites,
    ) -> None:
        """Process generator with DB storage.

        Uses simple iteration (for item in gen). All DB writes derived from
        yields are buffered in ``staged`` and flushed by the caller after
        the generator finishes without exception.

        Args:
            gen: The generator from the continuation method.
            response: The Response that triggered this continuation.
            parent_request: The request that initiated this continuation.
            continuation_name: Name of the continuation method.
            request_id: Database ID for result storage.
            staged: Buffer to receive staged writes for atomic flush.
        """
        from kent.common.deferred_validation import (
            DeferredValidation,
        )
        from kent.common.exceptions import (
            DataFormatAssumptionException,
            HTMLStructuralAssumptionException,
        )
        from kent.data_types import EstimateData, ParsedData

        try:
            for item in gen:
                match item:
                    case ParsedData():
                        raw_data = item.unwrap()
                        # Handle deferred validation
                        if isinstance(raw_data, DeferredValidation):
                            try:
                                validated_data = raw_data.confirm()
                                rt, dj, vej = (
                                    self._serialize_result_for_storage(
                                        validated_data
                                    )
                                )
                                staged.stage_result(
                                    result_type=rt,
                                    data_json=dj,
                                    is_valid=True,
                                    validation_errors_json=vej,
                                )
                                staged.stage_callback(
                                    functools.partial(
                                        self.handle_data, validated_data
                                    )
                                )
                            except DataFormatAssumptionException as e:
                                rt, dj, vej = (
                                    self._serialize_result_for_storage(
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
                            rt, dj, vej = self._serialize_result_for_storage(
                                raw_data
                            )
                            staged.stage_result(
                                result_type=rt,
                                data_json=dj,
                                is_valid=True,
                                validation_errors_json=vej,
                            )
                            staged.stage_callback(
                                functools.partial(self.handle_data, raw_data)
                            )

                    case EstimateData():
                        import json as _json

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
                        await self._stage_enqueue_request(
                            item, response, request_id, staged
                        )

                    case Request():
                        await self._stage_enqueue_request(
                            item, parent_request, request_id, staged
                        )

                    case None:
                        pass

        except HTMLStructuralAssumptionException as e:
            if self.on_structural_error:
                should_continue = await self.on_structural_error(e)
                if not should_continue:
                    return
            else:
                raise
