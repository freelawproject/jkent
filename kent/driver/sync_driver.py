"""Synchronous driver implementation.

This module contains the sync driver that processes scraper generators.
It evolves across the 29 steps of the design documentation.

- Step 1: A simple function that runs a scraper generator and collects results.
- Step 2: A class-based driver that handles Request, fetches pages,
  and calls continuation methods by name.
- Step 3: Tracks current_location and handles non-navigating requests.
- Step 4: Handles archive requests to download and save files locally.
- Step 5: No driver changes - accumulated_data flows through requests automatically.
- Step 7: Adds on_data callback for side effects (persistence, logging) when data yielded.
- Step 9: Adds on_invalid_data callback for handling validation failures.
- Step 10: Adds on_transient_exception callback for handling transient errors.
- Step 13: Adds archive_handler for customizing file archival behavior.
- Step 14: Adds on_run_start and on_run_complete lifecycle hooks for tracking scraper runs.
- Step 15: Replaces list queue with heapq priority queue for memory optimization.
- Step 16: Adds deduplication_key field to requests and duplicate_check callback for preventing duplicate requests.
"""

from __future__ import annotations

import heapq
import threading
from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Generic, TypeVar

from typing_extensions import assert_never

from kent.common.deferred_validation import (
    DeferredValidation,
)
from kent.common.exceptions import (
    DataFormatAssumptionException,
    ScraperAssumptionException,
    TransientException,
)
from kent.common.h11_patch import lenient_te_for
from kent.common.request_manager import (
    SyncRequestManager,
)
from kent.data_types import (
    ArchiveResponse,
    BaseRequest,
    BaseScraper,
    EstimateData,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
    SkipDeduplicationCheck,
)
from kent.driver._speculation_support import (
    SpeculationState,
    SyncSpeculationSupport,
)
from kent.driver.archive_handler import (
    LocalSyncArchiveHandler,
    SyncArchiveHandler,
    SyncStreamingArchiveHandler,
)
from kent.driver.callbacks import log_and_validate_invalid_data

__all__ = [
    "SpeculationState",
    "SyncDriver",
    "log_and_validate_invalid_data",
]

# =============================================================================
# Step 2: Class-based Driver with HTTP Support
# =============================================================================
# Step 3: current_location tracking and non-navigating request support
# Step 4: Archive request handling for file downloads
# Step 9: Data validation with on_invalid_data callback


ScraperReturnDatatype = TypeVar("ScraperReturnDatatype")


class SyncDriver(SyncSpeculationSupport, Generic[ScraperReturnDatatype]):
    """Synchronous driver for running scrapers.

    This Step 4 driver:
    - Maintains a request queue of Request objects
    - Fetches URLs using httpx
    - Looks up continuation methods by name
    - Each request carries its own current_location and ancestry
    - Uses exhaustive pattern matching for scraper yields
    - Handles archive requests (Request with archive=True) to download and save files locally

    Example usage:
        from tests.utils import collect_results

        callback, results = collect_results()
        driver = SyncDriver(scraper, on_data=callback)
        driver.run()
        # Results are now in the results list
    """

    def __init__(
        self,
        scraper: BaseScraper[ScraperReturnDatatype],
        storage_dir: Path | None = None,
        request_manager: SyncRequestManager | None = None,
        on_data: Callable[
            [ScraperReturnDatatype],
            None,
        ]
        | None = None,
        on_structural_error: Callable[[ScraperAssumptionException], bool]
        | None = None,
        on_invalid_data: Callable[[DeferredValidation], None] | None = None,
        on_transient_exception: Callable[[TransientException], bool]
        | None = None,
        archive_handler: SyncArchiveHandler
        | SyncStreamingArchiveHandler
        | None = None,
        on_run_start: Callable[[str], None] | None = None,
        on_run_complete: Callable[[str, str, Exception | None], None]
        | None = None,
        duplicate_check: Callable[[str], bool] | None = None,
        stop_event: threading.Event | None = None,
        proxy: str | None = None,
    ) -> None:
        """Initialize the driver.

        Args:
            scraper: Scraper instance with continuation methods.
            storage_dir: Directory for storing downloaded files. If None, uses system temp directory.
            request_manager: SyncRequestManager for handling HTTP requests.
            on_data: Optional callback invoked when ParsedData is yielded and validated. Useful for
                persistence, logging, or other side effects. The callback receives the
                unwrapped data from ParsedData.
            on_structural_error: Optional callback invoked when a ScraperAssumptionException
                (e.g. HTMLStructuralAssumptionException, or a DataFormatAssumptionException re-raised
                from handle_data when on_invalid_data is unset) is raised during scraping. The
                callback receives the exception and should return True to continue scraping or False
                to stop. If not provided, exceptions propagate normally and stop the scraper.
            on_invalid_data: Optional callback invoked when data fails validation. If not provided,
                invalid data is sent to on_data callback (if present), otherwise validation
                exceptions propagate normally.
            on_transient_exception: Optional callback invoked when TransientException is raised
                during HTTP requests. The callback receives the exception and should return True
                to continue scraping or False to stop. If not provided, exceptions propagate
                normally and stop the scraper.
            archive_handler: Handler for archive requests. Controls whether files are
                downloaded and how they are saved. If not provided, uses LocalSyncArchiveHandler.
            on_run_start: Optional callback invoked when the scraper run starts. Receives scraper_name (str).
            on_run_complete: Optional callback invoked when the scraper run completes. Receives
                scraper_name (str), status ("completed" | "error"),
                and error (Exception | None).
            duplicate_check: Optional callback invoked before enqueuing a request. Receives the
                deduplication_key (str) and should return True to enqueue the request or False to
                skip it. If not provided, all requests are enqueued (no deduplication).
            stop_event: Optional threading.Event for graceful shutdown. When set, the driver
                will stop processing after completing the current request.
            proxy: Optional proxy URL for HTTP requests (e.g.
                ``"socks5://user:pass@host:1080"``). Ignored when
                ``request_manager`` is also provided.
        """
        self.scraper = scraper
        # Step 15: Use heapq for priority queue (min heap)
        # Each entry is (priority, counter, request) for stable FIFO ordering
        self.request_queue: list[tuple[int, int, BaseRequest]] = []
        self._queue_counter = 0  # For FIFO tie-breaking within same priority
        # Step 16: Track seen deduplication keys for default duplicate checking
        self._seen_keys: set[str] = set()
        self.storage_dir = (
            storage_dir or Path(gettempdir()) / "juriscraper_files"
        )
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Set up request manager - either use provided one or create default
        if request_manager is not None:
            self.request_manager = request_manager
            self._owns_request_manager = False
        else:
            self.request_manager = SyncRequestManager(
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
            SyncArchiveHandler | SyncStreamingArchiveHandler
        ) = archive_handler or LocalSyncArchiveHandler(self.storage_dir)
        self.on_run_start = on_run_start
        self.on_run_complete = on_run_complete
        self.duplicate_check = duplicate_check
        self.stop_event = stop_event

        # Speculation state - populated by _discover_speculate_functions
        self._speculation_state: dict[str, SpeculationState] = {}

    def _enqueue_speculative(self, request: BaseRequest) -> None:
        heapq.heappush(
            self.request_queue,
            (request.priority, self._queue_counter, request),
        )
        self._queue_counter += 1

    def run(self) -> None:
        """Run the scraper starting from the scraper's entry point.

        Data is passed to the on_data callback as it is yielded. If you need to
        collect results, use a callback that appends to a list (see
        tests/design/utils.py::collect_results for a helper function).
        """
        with lenient_te_for(self.scraper):
            # Step 14: Fire on_run_start callback
            scraper_name = self.scraper.__class__.__name__
            if self.on_run_start:
                self.on_run_start(scraper_name)

            status = "completed"
            error: Exception | None = None

            try:
                # Initialize priority queue with entry requests.
                self.request_queue = []
                for entry_request in self._get_entry_requests():
                    heapq.heappush(
                        self.request_queue,
                        (
                            entry_request.priority,
                            self._queue_counter,
                            entry_request,
                        ),
                    )
                    self._queue_counter += 1

                # Discover and seed speculative requests
                self._speculation_state = self._discover_speculate_functions()
                if self._speculation_state:
                    self._seed_speculative_queue()

                while self.request_queue:
                    # Check for graceful shutdown before processing next request
                    if self.stop_event and self.stop_event.is_set():
                        break

                    # Step 15: Pop from heap (lowest priority first)
                    _priority, _counter, request = heapq.heappop(
                        self.request_queue
                    )

                    # Use match/case for exhaustive request type handling
                    match request:
                        case Request():
                            # Normal request flow
                            # Step 10: Wrap request resolution to catch transient exceptions
                            try:
                                response: Response = (
                                    self.resolve_archive_request(request)
                                    if request.archive
                                    else self.resolve_request(request)
                                )
                            except TransientException as e:
                                # Step 10: Handle transient errors via callback
                                if self.on_transient_exception:
                                    should_continue = (
                                        self.on_transient_exception(e)
                                    )
                                    if not should_continue:
                                        return
                                    continue
                                else:
                                    raise

                            # Track speculation outcome if this is a speculative request
                            if request.is_speculative:
                                self._track_speculation_outcome(
                                    request, response
                                )

                            # Handle Callable continuations (convert to string)
                            continuation_name = (
                                request.continuation
                                if isinstance(request.continuation, str)
                                else request.continuation.__name__
                            )

                            continuation_method = (
                                self.scraper.get_continuation(
                                    continuation_name
                                )
                            )

                            # Process the generator
                            gen = continuation_method(response)
                            self._process_generator(gen, response, request)

                        case _:
                            # Exhaustive match - should never reach here
                            assert_never(request)  # type: ignore[arg-type]

            except Exception as e:
                # Step 14: Capture error for on_run_complete
                status = "error"
                error = e
                raise
            finally:
                # Close request manager if we own it
                if self._owns_request_manager:
                    self.request_manager.close()

                # Step 14: Fire on_run_complete callback
                if self.on_run_complete:
                    self.on_run_complete(
                        scraper_name,
                        status,
                        error,
                    )

    def enqueue_request(
        self, new_request: BaseRequest, context: Response | BaseRequest
    ) -> None:
        """Enqueue a new request, resolving it from the given context.

        Step 16: Check for duplicates using duplicate_check callback before enqueuing.

        For navigating Request yields: context is the Response
        For non-navigating Request yields: context is the originating request
        For archive Request yields: context is the Response

        Args:
            new_request: The new request to enqueue.
            context: Response or originating request for URL resolution.
        """
        # Use the request's resolve_from method with the appropriate context
        resolved_request = new_request.resolve_from(context)  # type: ignore

        # Step 16: Check for duplicates before enqueuing
        dedup_key = resolved_request.deduplication_key

        match dedup_key:
            case None:
                pass
            case SkipDeduplicationCheck():
                pass
            case str():
                if self.duplicate_check and not self.duplicate_check(
                    dedup_key
                ):
                    return

        # Step 15: Push onto heap with priority and counter for stable ordering
        heapq.heappush(
            self.request_queue,
            (resolved_request.priority, self._queue_counter, resolved_request),
        )
        self._queue_counter += 1

    def resolve_request(self, request: BaseRequest) -> Response:
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
        return self.request_manager.resolve_request(request)

    def resolve_archive_request(self, request: Request) -> ArchiveResponse:
        """Fetch an archive Request, download the file, and return an ArchiveResponse.

        Uses the archive_handler to decide whether to download. If the request
        has an archive_hash_header, a HEAD request is issued first to extract
        the header value for the handler's decision.

        Args:
            request: The archive Request to fetch (must have archive=True).

        Returns:
            ArchiveResponse containing the HTTP response data and local file path.
        """
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
                head_response = self.resolve_request(head_request)
                hash_header_value = head_response.headers.get(
                    request.archive_hash_header
                )
            except Exception:
                pass

        dedup_key = (
            request.deduplication_key
            if isinstance(request.deduplication_key, str)
            else None
        )

        decision = self.archive_handler.should_download(
            url=request.request.url,
            deduplication_key=dedup_key,
            expected_type=request.expected_type,
            hash_header_value=hash_header_value,
        )

        if not decision.download:
            return ArchiveResponse(
                status_code=200,
                headers={},
                content=b"",
                text="",
                url=request.request.url,
                request=request,
                file_url=decision.file_url,
            )

        if hasattr(self.archive_handler, "save_stream"):
            with self.request_manager.stream_request(request) as stream:
                file_url = self.archive_handler.save_stream(
                    url=request.request.url,
                    deduplication_key=dedup_key,
                    expected_type=request.expected_type,
                    hash_header_value=hash_header_value,
                    chunks=stream.iter_bytes(),
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

        http_response = self.resolve_request(request)

        file_url = self.archive_handler.save(
            url=request.request.url,
            deduplication_key=dedup_key,
            expected_type=request.expected_type,
            hash_header_value=hash_header_value,
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

    def handle_data(self, data: ScraperReturnDatatype) -> None:
        # Step 9: Validate deferred data if present
        if isinstance(data, DeferredValidation):
            try:
                validated_data: ScraperReturnDatatype = (
                    data.confirm()
                )  # ty: ignore[invalid-assignment]
                # Validation succeeded - send to on_data callback
                if self.on_data:
                    self.on_data(validated_data)
            except DataFormatAssumptionException:
                # Validation failed - use callback hierarchy
                if self.on_invalid_data:
                    self.on_invalid_data(data)
                else:
                    # No callbacks - re-raise the exception
                    raise
        else:
            # Step 7: Not deferred validation - invoke callback if provided
            if self.on_data:
                self.on_data(data)

    def _process_generator(
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

        deferred: list[Callable[[], None]] = []

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
                    case None:
                        pass
                    case _:
                        assert_never(item)

            # Drain inside the same try block so a DataFormatAssumptionException
            # raised by handle_data still routes through on_structural_error.
            for cb in deferred:
                cb()
        except ScraperAssumptionException as e:
            # Step 8: Handle structural errors via callback
            if self.on_structural_error:
                should_continue = self.on_structural_error(e)
                if not should_continue:
                    return
            else:
                raise
