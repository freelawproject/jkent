"""Asynchronous driver implementation.

This module contains the async driver that processes scraper generators
using multiple concurrent workers.

The AsyncDriver closely mirrors SyncDriver with three key differences:

1. Factors out the main run loop to a worker method for concurrency
2. Uses an async-compatible priority queue (asyncio.PriorityQueue)
3. Takes num_workers argument to control concurrency
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Generator
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Generic, TypeVar

from typing_extensions import assert_never

from kent.common.deferred_validation import (
    DeferredValidation,
)
from kent.common.exceptions import (
    DataFormatAssumptionException,
    RequestFailedHalt,
    RequestFailedSkip,
    ScraperAssumptionException,
    TransientException,
)
from kent.common.h11_patch import lenient_te_for
from kent.common.request_manager import (
    AsyncRequestManager,
)
from kent.data_types import (
    ArchiveDecision,
    ArchiveResponse,
    BaseRequest,
    BaseScraper,
    EstimateData,
    HttpMethod,
    HTTPRequestParams,
    HTTPRequestPrep,
    JSRequestPrep,
    ParsedData,
    Request,
    Response,
    ScraperYield,
    SkipDeduplicationCheck,
)
from kent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
)
from kent.driver.archive_handler import (
    AsyncArchiveHandler,
    AsyncStreamingArchiveHandler,
    LocalAsyncArchiveHandler,
)
from kent.driver.callbacks import log_and_validate_invalid_data

__all__ = ["AsyncDriver", "log_and_validate_invalid_data"]

logger = logging.getLogger(__name__)

ScraperReturnDatatype = TypeVar("ScraperReturnDatatype")


class AsyncDriver(AsyncSpeculationSupport, Generic[ScraperReturnDatatype]):
    """Asynchronous driver for running scrapers with multiple workers.

    This driver closely mirrors SyncDriver with three key differences:
    - Uses asyncio.PriorityQueue for async-compatible priority queue
    - Factors out the main loop to _worker() for concurrent execution
    - Takes num_workers to control the number of concurrent workers

    Example usage:
        from tests.utils import collect_results

        callback, results = collect_results()
        driver = AsyncDriver(scraper, on_data=callback, num_workers=4)
        await driver.run()
        # Results are now in the results list
    """

    def __init__(
        self,
        scraper: BaseScraper[ScraperReturnDatatype],
        storage_dir: Path | None = None,
        request_manager: AsyncRequestManager | None = None,
        on_data: Callable[
            [ScraperReturnDatatype],
            Awaitable[None],
        ]
        | None = None,
        on_structural_error: Callable[
            [ScraperAssumptionException], Awaitable[bool]
        ]
        | None = None,
        on_invalid_data: Callable[[DeferredValidation], Awaitable[None]]
        | None = None,
        on_transient_exception: Callable[[TransientException], Awaitable[bool]]
        | None = None,
        archive_handler: AsyncArchiveHandler
        | AsyncStreamingArchiveHandler
        | None = None,
        on_run_start: Callable[[str], Awaitable[None]] | None = None,
        on_run_complete: Callable[
            [str, str, Exception | None], Awaitable[None]
        ]
        | None = None,
        duplicate_check: Callable[[str], Awaitable[bool]] | None = None,
        stop_event: asyncio.Event | None = None,
        num_workers: int = 1,
        proxy: str | None = None,
    ) -> None:
        """Initialize the driver.

        Args:
            scraper: Scraper instance with continuation methods.
            storage_dir: Directory for storing downloaded files. If None, uses system temp directory.
            request_manager: AsyncRequestManager for handling HTTP requests.
            on_data: Optional async callback invoked when ParsedData is yielded and validated. Useful
                for persistence, logging, or other side effects. The callback receives the
                unwrapped data from ParsedData.
            on_structural_error: Optional async callback invoked when a ScraperAssumptionException
                (e.g. HTMLStructuralAssumptionException, or a DataFormatAssumptionException re-raised
                from handle_data when on_invalid_data is unset) is raised during scraping. The
                callback receives the exception and should return True to continue scraping or False
                to stop. If not provided, exceptions propagate normally and stop the scraper.
            on_invalid_data: Optional async callback invoked when data fails validation. If not
                provided, invalid data is sent to on_data callback (if present), otherwise validation
                exceptions propagate normally.
            on_transient_exception: Optional async callback invoked when TransientException is raised
                during HTTP requests. The callback receives the exception and should return True
                to continue scraping or False to stop. If not provided, exceptions propagate
                normally and stop the scraper.
            archive_handler: Handler for archive requests. Controls whether files are
                downloaded and how they are saved. If not provided, uses LocalAsyncArchiveHandler.
            on_run_start: Optional async callback invoked when the scraper run starts. Receives
                scraper_name (str).
            on_run_complete: Optional async callback invoked when the scraper run completes. Receives
                scraper_name (str), status ("completed" | "error")
                and error (Exception | None).
            duplicate_check: Optional async callback invoked before enqueuing a request. Receives the
                deduplication_key (str) and should return True to enqueue the request or False to
                skip it. If not provided, all requests are enqueued (no deduplication).
            stop_event: Optional asyncio.Event for graceful shutdown. When set, workers
                will stop processing after completing their current request.
            num_workers: Number of concurrent workers to process requests. Defaults to 1.
            proxy: Optional proxy URL for HTTP requests (e.g.
                ``"socks5://user:pass@host:1080"``). Ignored when
                ``request_manager`` is also provided.
        """
        self.scraper = scraper
        # Use asyncio.PriorityQueue for async-compatible priority queue
        # Each entry is (priority, counter, request) for stable FIFO ordering
        self.request_queue: asyncio.PriorityQueue[
            tuple[int, int, BaseRequest]
        ] = asyncio.PriorityQueue()
        self._queue_counter = 0  # For FIFO tie-breaking within same priority
        self._queue_lock = asyncio.Lock()  # Protect counter increments
        self.storage_dir = (
            storage_dir or Path(gettempdir()) / "juriscraper_files"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Set up request manager - either use provided one or create default
        if request_manager is not None:
            self.request_manager = request_manager
            self._owns_request_manager = False
        else:
            self.request_manager = AsyncRequestManager(
                ssl_context=scraper.get_ssl_context(),
                rates=scraper.rate_limits,
                proxy=proxy,
                scraper=scraper,
            )
            self._owns_request_manager = True

        self.seed_params: list[dict[str, dict[str, Any]]] | None = None

        self.on_data = on_data
        self.on_structural_error = on_structural_error
        self.on_invalid_data = on_invalid_data
        self.on_transient_exception = on_transient_exception
        self.archive_handler: (
            AsyncArchiveHandler | AsyncStreamingArchiveHandler
        ) = archive_handler or LocalAsyncArchiveHandler(self.storage_dir)
        self.on_run_start = on_run_start
        self.on_run_complete = on_run_complete
        self.duplicate_check = duplicate_check
        self.stop_event = stop_event
        self.num_workers = num_workers

        # Speculation state - populated by _discover_speculate_functions
        self._speculation_state: dict[str, SpeculationState] = {}
        # Lock for speculation state updates from concurrent workers
        self._speculation_lock = asyncio.Lock()

    async def _enqueue_speculative(self, request: BaseRequest) -> None:
        async with self._queue_lock:
            await self.request_queue.put(
                (request.priority, self._queue_counter, request)
            )
            self._queue_counter += 1

    async def run(self) -> None:
        """Run the scraper starting from the scraper's entry point.

        Data is passed to the on_data callback as it is yielded. If you need to
        collect results, use a callback that appends to a list (see
        tests/design/utils.py::collect_results for a helper function).
        """
        # Must wrap the entire body: asyncio.Task snapshots context at
        # create time, so the contextvar has to be set before workers spawn.
        with lenient_te_for(self.scraper):
            # Fire on_run_start callback
            scraper_name = self.scraper.__class__.__name__
            if self.on_run_start:
                await self.on_run_start(scraper_name)

            status = "completed"
            error: Exception | None = None

            try:
                # Check for early stop before doing any work
                if self.stop_event and self.stop_event.is_set():
                    return

                # Initialize priority queue with entry requests.
                self.request_queue = asyncio.PriorityQueue()
                self._queue_counter = 0
                for entry_request in self._get_entry_requests():
                    await self.request_queue.put(
                        (
                            entry_request.priority,
                            self._queue_counter,
                            entry_request,
                        )
                    )
                    self._queue_counter += 1

                # Discover and seed speculative requests
                self._speculation_state = self._discover_speculate_functions()
                if self._speculation_state:
                    await self._seed_speculative_queue()

                # Start workers
                workers = [
                    asyncio.create_task(self._worker(i))
                    for i in range(self.num_workers)
                ]

                # Wait for all items in the queue to be processed
                # Use wait_for with periodic checks for stop_event
                while True:
                    if self.stop_event and self.stop_event.is_set():
                        # Stop requested - cancel workers and drain queue
                        for worker in workers:
                            worker.cancel()
                        # Drain the queue to prevent join() from blocking
                        while not self.request_queue.empty():
                            try:
                                self.request_queue.get_nowait()
                                self.request_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        break

                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self.request_queue.join()),
                            timeout=0.1,
                        )
                        # join() completed - all work is done
                        break
                    except TimeoutError:
                        # Check stop_event and continue waiting
                        continue

                # Cancel workers (they're waiting on the queue)
                for worker in workers:
                    worker.cancel()

                # Wait for workers to finish cancellation
                await asyncio.gather(*workers, return_exceptions=True)

            except Exception as e:
                # Capture error for on_run_complete
                status = "error"
                error = e
                raise
            finally:
                # Close request manager if we own it
                if self._owns_request_manager:
                    await self.request_manager.close()

                # Fire on_run_complete callback
                if self.on_run_complete:
                    await self.on_run_complete(
                        scraper_name,
                        status,
                        error,
                    )

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes requests from the queue.

        Args:
            worker_id: Identifier for this worker (for debugging).
        """
        while True:
            # Check for graceful shutdown before getting next request
            if self.stop_event and self.stop_event.is_set():
                break

            # Get next request from queue (blocks until available)
            try:
                _priority, _counter, request = await self.request_queue.get()
            except asyncio.CancelledError:
                # Worker was cancelled (normal shutdown)
                break

            try:
                # Use match/case for exhaustive request type handling
                match request:
                    case Request():
                        # Normal request flow
                        # Wrap request resolution to catch transient exceptions
                        try:
                            response: Response = (
                                await self.resolve_archive_request(request)
                                if request.archive
                                else await self.resolve_request(request)
                            )
                        except TransientException as e:
                            # Handle transient errors via callback
                            if self.on_transient_exception:
                                should_continue = (
                                    await self.on_transient_exception(e)
                                )
                                if not should_continue:
                                    break
                                continue
                            else:
                                raise
                        except RequestFailedHalt:
                            raise
                        except RequestFailedSkip:
                            # Skip this request silently and continue to next
                            continue

                        # Track speculation outcome if this is a speculative request
                        if request.is_speculative:
                            await self._track_speculation_outcome(
                                request, response
                            )

                        # Handle Callable continuations (convert to string)
                        continuation_name = (
                            request.continuation
                            if isinstance(request.continuation, str)
                            else getattr(
                                request.continuation,
                                "__name__",
                                str(request.continuation),
                            )
                        )

                        continuation_method = self.scraper.get_continuation(
                            continuation_name
                        )

                        # Process the generator
                        gen = continuation_method(response)
                        await self._process_generator(gen, response, request)

                    case _:
                        # Exhaustive match - should never reach here
                        assert_never(request)  # type: ignore[arg-type]
            finally:
                # Always mark task as done to allow join() to complete
                self.request_queue.task_done()

    async def enqueue_request(
        self, new_request: BaseRequest, context: Response | BaseRequest
    ) -> None:
        """Enqueue a new request, resolving it from the given context.

        Check for duplicates using duplicate_check callback before enqueuing.

        For navigating Request yields: context is the Response
        For non-navigating Request yields: context is the originating request
        For archive Request yields: context is the Response

        Args:
            new_request: The new request to enqueue.
            context: Response or originating request for URL resolution.
        """
        # Use the request's resolve_from method with the appropriate context
        resolved_request = new_request.resolve_from(context)  # type: ignore

        # Check for duplicates before enqueuing
        dedup_key = resolved_request.deduplication_key
        match dedup_key:
            case None:
                pass
            case SkipDeduplicationCheck():
                pass
            case str():
                if self.duplicate_check and not await self.duplicate_check(
                    dedup_key
                ):
                    return

        # Push onto queue with priority and counter for stable ordering
        async with self._queue_lock:
            await self.request_queue.put(
                (
                    resolved_request.priority,
                    self._queue_counter,
                    resolved_request,
                )
            )
            self._queue_counter += 1

    async def resolve_request(self, request: BaseRequest) -> Response:
        """Fetch a BaseRequest and return the Response.

        Delegates to the request manager for HTTP handling.

        Args:
            request: The BaseRequest to fetch.

        Returns:
            Response containing the HTTP response data.

        Raises:
            HTMLResponseAssumptionException: If server returns 5xx status code.
            httpx.TimeoutException: If request times out (for retry handling).
        """
        # Simply delegate to request manager - exception handling is done
        # by the driver's worker (LocalDevDriver._db_worker handles retries)
        response = await self.request_manager.resolve_request(request)
        return response

    async def resolve_archive_request(
        self,
        request: Request,
        archive_decision: ArchiveDecision | None = None,
    ) -> ArchiveResponse:
        """Fetch an archive Request, download the file, and return an ArchiveResponse.

        Uses the archive_handler to decide whether to download. If the request
        has an archive_hash_header, a HEAD request is issued first to extract
        the header value for the handler's decision.

        Args:
            request: The archive Request to fetch (must have archive=True).
            archive_decision: Pre-computed decision from the archive handler.
                When provided, skips the ``should_download()`` call.  The
                caller is responsible for having already consulted the handler.

        Returns:
            ArchiveResponse containing the HTTP response data and local file path.
        """
        dedup_key = (
            request.deduplication_key
            if isinstance(request.deduplication_key, str)
            else None
        )

        if archive_decision is None:
            # Extract hash header value via HEAD if requested
            hash_header_value = None
            if request.archive_hash_header:
                try:
                    head_request = BaseRequest(
                        request=HTTPRequestParams(
                            method=HttpMethod.HEAD,
                            url=request.request.url,
                        ),
                        continuation="",
                    )
                    head_response = await self.resolve_request(head_request)
                    hash_header_value = head_response.headers.get(
                        request.archive_hash_header
                    )
                except Exception:
                    pass

            archive_decision = await self.archive_handler.should_download(
                url=request.request.url,
                deduplication_key=dedup_key,
                expected_type=request.expected_type,
                hash_header_value=hash_header_value,
            )

        if not archive_decision.download:
            logger.info(
                "resolve_archive_request: skip download url=%s file_url=%s",
                request.request.url,
                archive_decision.file_url,
            )
            return ArchiveResponse(
                status_code=200,
                headers={},
                content=b"",
                text="",
                url=request.request.url,
                request=request,
                file_url=archive_decision.file_url,
            )

        if hasattr(self.archive_handler, "save_stream"):
            logger.info(
                "resolve_archive_request: streaming branch url=%s",
                request.request.url,
            )
            async with self.request_manager.stream_request(request) as stream:
                file_url = await self.archive_handler.save_stream(
                    url=request.request.url,
                    deduplication_key=dedup_key,
                    expected_type=request.expected_type,
                    hash_header_value=None,
                    chunks=stream.aiter_bytes(),
                )
                logger.info(
                    "resolve_archive_request: streaming done url=%s "
                    "file_url=%s",
                    request.request.url,
                    file_url,
                )
                return ArchiveResponse(
                    status_code=stream.status_code,
                    headers=dict(stream.headers),
                    content=b"",
                    text="",
                    url=request.request.url,
                    request=request,
                    file_url=file_url,
                )

        logger.info(
            "resolve_archive_request: buffered branch url=%s",
            request.request.url,
        )
        http_response = await self.resolve_request(request)

        file_url = await self.archive_handler.save(
            url=request.request.url,
            deduplication_key=dedup_key,
            expected_type=request.expected_type,
            hash_header_value=None,
            content=http_response.content,
        )

        return ArchiveResponse(
            status_code=http_response.status_code,
            headers=dict(http_response.headers),
            content=http_response.content,
            text=http_response.text,
            url=request.request.url,
            request=request,
            file_url=file_url,
        )

    async def handle_data(self, data: ScraperReturnDatatype) -> None:
        # Validate deferred data if present
        if isinstance(data, DeferredValidation):
            try:
                validated_data: ScraperReturnDatatype = (
                    data.confirm()
                )  # ty: ignore[invalid-assignment]
                # Increment data counter on successful validation
                # Validation succeeded - send to on_data callback
                if self.on_data:
                    await self.on_data(validated_data)
            except DataFormatAssumptionException:
                # Validation failed - use callback hierarchy
                if self.on_invalid_data:
                    await self.on_invalid_data(data)
                else:
                    # No callbacks - re-raise the exception
                    raise
        else:
            # Increment data counter for non-validated data
            # Not deferred validation - invoke callback if provided
            if self.on_data:
                await self.on_data(data)

    async def _process_generator(
        self,
        gen: Generator[ScraperYield, bool | None, None],
        response: Response,
        parent_request: BaseRequest,
    ) -> None:
        """Process generator yields, enqueueing requests and handling data.

        Per-step atomicity: yields are buffered as deferred actions and only
        drained on successful iteration. If the step raises an unhandled
        exception mid-iteration, the buffer is dropped — no on_data /
        on_invalid_data callbacks fire, no requests enqueue.

        Args:
            gen: The generator from the continuation method.
            response: The Response that triggered this continuation.
            parent_request: The request that initiated this continuation.
        """
        import functools

        deferred: list[Callable[[], Awaitable[None]]] = []

        try:
            for item in gen:
                match item:
                    case ParsedData():
                        deferred.append(
                            functools.partial(self.handle_data, item.unwrap())
                        )
                    case EstimateData():
                        pass
                    case Request() if (
                        not item.nonnavigating and not item.archive
                    ):
                        deferred.append(
                            functools.partial(
                                self.enqueue_request, item, response
                            )
                        )
                    case Request():
                        deferred.append(
                            functools.partial(
                                self.enqueue_request, item, parent_request
                            )
                        )
                    case JSRequestPrep() | HTTPRequestPrep():
                        # Preps are only supported by the persistent driver
                        # path; the in-memory async driver doesn't have the
                        # staging machinery to host them.
                        from kent.common.exceptions import ScraperConfigError

                        raise ScraperConfigError(
                            f"{type(item).__name__} is not supported by "
                            f"the in-memory AsyncDriver; use the persistent "
                            f"driver (or a subclass thereof)"
                        )
                    case None:
                        pass
                    case _:
                        assert_never(item)

            # Step iterated to completion — drain the buffer. Drain happens
            # inside the same try block so a DataFormatAssumptionException
            # raised by handle_data still routes through on_structural_error,
            # matching the pre-staging behavior.
            for cb in deferred:
                await cb()
        except ScraperAssumptionException as e:
            # Handle structural errors via callback. Catches the parent class
            # so that DataFormatAssumptionException re-raised from handle_data
            # (when on_invalid_data is unset) falls back to on_structural_error.
            if self.on_structural_error:
                should_continue = await self.on_structural_error(e)
                if not should_continue:
                    return
            else:
                raise
