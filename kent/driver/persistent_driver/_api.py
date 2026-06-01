"""APIMixin - Public inspection, control, and diagnostic APIs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from kent.driver.persistent_driver.sql_manager import (
    Page,
    RequestRecord,
    ResponseRecord,
    ResultRecord,
    SQLManager,
)

if TYPE_CHECKING:
    from kent.driver.persistent_driver.stats import DevDriverStats

logger = logging.getLogger(__name__)


@dataclass
class DiagnoseResult:
    """Result of running diagnose() on a response.

    Contains the yields produced by re-running a continuation,
    XPath observation data, and any errors that occurred.

    Attributes:
        response_id: The database ID of the response that was diagnosed.
        continuation: The continuation method name that was run.
        yields: List of yielded items with type and key attributes.
        simple_tree: Human-readable XPath observation tree.
        observer_json: JSON for UI highlighting.
        error: Error message if continuation raised an exception.
    """

    response_id: int
    continuation: str
    yields: list[dict[str, Any]]
    simple_tree: str
    observer_json: list[dict[str, Any]]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "response_id": self.response_id,
            "continuation": self.continuation,
            "yields": self.yields,
            "simple_tree": self.simple_tree,
            "observer_json": self.observer_json,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Serialize to JSON for API transport."""
        return json.dumps(self.to_dict())


class APIMixin:
    """Public inspection, control, listing, diagnostic, and cancellation APIs.

    Provides methods for step control (pause/resume),
    response content access, statistics, diagnosis, listing, and cancellation.
    """

    db: SQLManager
    scraper: BaseScraper

    if TYPE_CHECKING:

        async def _emit_progress(
            self, event_type: str, data: dict[str, Any]
        ) -> None: ...

    # --- Step Control ---

    async def pause_step(self, continuation: str) -> int:
        """Pause processing of requests for a specific continuation.

        Marks all pending requests for the given continuation as 'held'.
        Held requests are not picked up by workers but remain in the queue
        for later resume. Useful for temporarily stopping a problematic step
        while continuing to process other parts of the scraper.

        Args:
            continuation: The continuation method name to pause.

        Returns:
            Number of requests marked as held.
        """
        count = await self.db.pause_step(continuation)
        if count > 0:
            await self._emit_progress(
                "step_paused",
                {
                    "continuation": continuation,
                    "requests_held": count,
                },
            )
        return count

    async def resume_step(self, continuation: str) -> int:
        """Resume processing of held requests for a specific continuation.

        Marks all held requests for the given continuation as 'pending',
        making them available for workers to process again.

        Args:
            continuation: The continuation method name to resume.

        Returns:
            Number of requests restored to pending.
        """
        count = await self.db.resume_step(continuation)
        if count > 0:
            await self._emit_progress(
                "step_resumed",
                {
                    "continuation": continuation,
                    "requests_restored": count,
                },
            )
        return count

    async def get_held_count(self, continuation: str | None = None) -> int:
        """Get count of held requests, optionally filtered by continuation.

        Args:
            continuation: Optional continuation name to filter by.

        Returns:
            Count of held requests.
        """
        return await self.db.get_held_count(continuation)

    # --- Response Content Access ---

    async def get_response_content(self, request_id: int) -> bytes | None:
        """Get decompressed response content by request ID.

        Args:
            request_id: The database ID of the request.

        Returns:
            Decompressed content bytes, or None if response not found.
        """
        return await self.db.get_response_content(request_id)

    # --- Statistics ---

    async def get_stats(self) -> DevDriverStats:
        """Get comprehensive statistics about the driver state.

        Returns:
            DevDriverStats instance with queue, throughput, compression,
            result, and error statistics.
        """
        from kent.driver.persistent_driver.stats import (
            get_stats,
        )

        return await get_stats(self.db._session_factory)

    # --- Debugging / Diagnosis ---

    async def diagnose(
        self,
        request_id: int,
        speculation_cap: int = 3,  # Deprecated, kept for backwards compatibility
    ) -> DiagnoseResult:
        """Re-run a continuation against a stored response with XPath observation.

        This method retrieves a stored response, decompresses it, reconstructs
        the Response object, and re-runs the continuation method with a
        SelectorObserver active to capture all XPath/CSS queries.

        Useful for debugging "zero results" issues where the HTML structure
        may have changed or XPath queries are incorrect.

        Args:
            request_id: The database ID of the request whose response to diagnose.
            speculation_cap: Deprecated, no longer used.

        Returns:
            DiagnoseResult with yields, observation tree, and any errors.

        Raises:
            ValueError: If request_id not found or has no response.
        """
        from kent.common.selector_observer import (
            SelectorObserver,
        )

        # Get response and request data - all in one table now
        from kent.driver.persistent_driver.models import (
            Request as RequestModel,
        )

        async with self.db._session_factory() as session:
            from sqlmodel import select

            stmt = select(  # type: ignore[call-overload,misc]
                RequestModel.response_status_code,
                RequestModel.response_url,
                RequestModel.response_headers_json,
                RequestModel.continuation,
                RequestModel.method,
                RequestModel.url,
                RequestModel.accumulated_data_json,
                RequestModel.permanent_json,
            ).where(
                RequestModel.id == request_id,
                RequestModel.response_status_code.isnot(None),  # type: ignore[union-attr]
            )
            result = await session.execute(stmt)
            row = result.first()

        if row is None:
            raise ValueError(f"Response for request {request_id} not found")

        (
            status_code,
            url,
            headers_json,
            continuation_name,
            method,
            request_url,
            accumulated_data_json,
            permanent_json,
        ) = row

        # Decompress content
        content = await self.get_response_content(request_id)
        if content is None:
            content = b""

        # Reconstruct Response object
        headers = json.loads(headers_json) if headers_json else {}
        accumulated_data = (
            json.loads(accumulated_data_json) if accumulated_data_json else {}
        )
        permanent = json.loads(permanent_json) if permanent_json else {}

        http_params = HTTPRequestParams(
            method=HttpMethod(method),
            url=request_url,
        )
        # Create a Request to serve as the request context
        reconstructed_request = Request(
            request=http_params,
            continuation=continuation_name,
            current_location=request_url,
            accumulated_data=accumulated_data,
            permanent=permanent,
        )

        # Decode content to text for the Response
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")

        response = Response(
            status_code=status_code,
            url=url,
            content=content,
            text=text,
            headers=headers,
            request=reconstructed_request,
        )

        # Run continuation with observer
        yields: list[dict[str, Any]] = []
        error: str | None = None

        with SelectorObserver() as observer:
            try:
                continuation_method = self.scraper.get_continuation(
                    continuation_name
                )
                gen = continuation_method(response)

                for item in gen:
                    yield_info = self._describe_yield(item)
                    yields.append(yield_info)

            except Exception as e:
                import traceback

                error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        return DiagnoseResult(
            response_id=request_id,
            continuation=continuation_name,
            yields=yields,
            simple_tree=observer.simple_tree(),
            observer_json=observer.json(),
            error=error,
        )

    def _describe_yield(self, item: Any) -> dict[str, Any]:
        """Create a description of a yielded item for diagnose results."""
        from kent.data_types import ParsedData

        if isinstance(item, ParsedData):
            data = item.unwrap()
            data_str = str(data)
            return {
                "type": "ParsedData",
                "data_type": type(data).__name__,
                "preview": (
                    data_str[:200] + "..." if len(data_str) > 200 else data_str
                ),
            }
        elif isinstance(item, Request):
            if item.archive:
                return {
                    "type": "ArchiveRequest",
                    "url": item.request.url,
                    "expected_type": item.expected_type,
                }
            elif item.nonnavigating:
                return {
                    "type": "NonNavigatingRequest",
                    "url": item.request.url,
                }
            else:
                return {
                    "type": "NavigatingRequest",
                    "url": item.request.url,
                    "method": item.request.method.value,
                    "continuation": (
                        item.continuation
                        if isinstance(item.continuation, str)
                        else item.continuation.__name__
                    ),
                }
        elif item is None:
            return {"type": "None"}
        else:
            return {
                "type": type(item).__name__,
                "repr": repr(item)[:200],
            }

    # --- Web Interface Listing Methods ---
    # These delegate to SQLManager for the actual database operations

    async def list_requests(
        self,
        status: str | None = None,
        continuation: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Page[RequestRecord]:
        """List requests with optional filters and pagination."""
        return await self.db.list_requests(
            status=status,
            continuation=continuation,
            offset=offset,
            limit=limit,
        )

    async def list_responses(
        self,
        continuation: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Page[ResponseRecord]:
        """List responses with optional filters and pagination."""
        return await self.db.list_responses(
            continuation=continuation, offset=offset, limit=limit
        )

    async def list_results(
        self,
        result_type: str | None = None,
        is_valid: bool | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Page[ResultRecord]:
        """List results with optional filters and pagination."""
        return await self.db.list_results(
            result_type=result_type,
            is_valid=is_valid,
            offset=offset,
            limit=limit,
        )

    async def get_request(self, request_id: int) -> RequestRecord | None:
        """Get a single request by ID."""
        return await self.db.get_request(request_id)

    async def get_response(self, request_id: int) -> ResponseRecord | None:
        """Get a single response by request ID."""
        return await self.db.get_response(request_id)

    async def get_result(self, result_id: int) -> ResultRecord | None:
        """Get a single result by ID."""
        return await self.db.get_result(result_id)

    # --- Request Cancellation ---

    async def cancel_request(self, request_id: int) -> bool:
        """Cancel a pending request.

        Only pending or held requests can be cancelled. In-progress requests
        cannot be cancelled as they are already being processed.

        Args:
            request_id: The database ID of the request to cancel.

        Returns:
            True if the request was cancelled, False if not found or not cancellable.
        """
        cancelled = await self.db.cancel_request(request_id)
        if cancelled:
            await self._emit_progress(
                "request_cancelled",
                {
                    "request_id": request_id,
                },
            )
        return cancelled

    async def cancel_requests_by_continuation(self, continuation: str) -> int:
        """Cancel all pending/held requests for a continuation.

        Args:
            continuation: The continuation method name.

        Returns:
            Number of requests cancelled.
        """
        count = await self.db.cancel_requests_by_continuation(continuation)
        if count > 0:
            await self._emit_progress(
                "requests_batch_cancelled",
                {
                    "continuation": continuation,
                    "count": count,
                },
            )
        return count
