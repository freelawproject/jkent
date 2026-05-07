"""QueueMixin - DB-backed request queue operations."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse, urlunparse

from kent.data_types import (
    BaseRequest,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from kent.driver.persistent_driver.sql_manager import SQLManager

if TYPE_CHECKING:
    from kent.driver.persistent_driver._staging import StagedWrites


def _json_default(obj: Any) -> Any:
    """Handle date/datetime objects in json.dumps."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


class QueueMixin:
    """DB-backed queue: enqueue, dequeue, serialization/deserialization.

    Provides methods for persisting requests to SQLite with deduplication
    and reconstructing request objects from database rows.
    """

    db: SQLManager

    if TYPE_CHECKING:

        async def _emit_progress(
            self, event_type: str, data: dict[str, Any]
        ) -> None: ...

    async def enqueue_request(
        self,
        new_request: BaseRequest,
        context: Response | BaseRequest,
        parent_request_id: int | None = None,
    ) -> None:
        """Enqueue a new request to the database.

        Overrides AsyncDriver.enqueue_request to persist to SQLite.

        Args:
            new_request: The new request to enqueue.
            context: Response or originating request for URL resolution.
            parent_request_id: Optional parent request ID for tracking request relationships.
        """
        # Resolve the request from context
        resolved_request: Request = new_request.resolve_from(context)  # type: ignore[arg-type, assignment]

        # Check for duplicates before inserting
        dedup_key = resolved_request.deduplication_key
        if dedup_key is not None and not isinstance(dedup_key, str):
            # SkipDeduplicationCheck - allow the request
            dedup_key = None

        # Check if this dedup_key already exists
        if dedup_key and await self.db.check_dedup_key_exists(dedup_key):
            # Duplicate found - skip
            return

        # Serialize request data
        request_data = self._serialize_request(resolved_request)

        # Use provided parent_request_id, or look up from context if not provided
        parent_id: int | None = parent_request_id
        if (
            parent_id is None
            and isinstance(context, Response)
            and context.request
        ):
            parent_id = await self.db.find_parent_request_id(
                context.request.request.url
            )

        # Insert the request
        await self.db.insert_request(
            priority=resolved_request.priority,
            request_type=request_data["request_type"],
            method=request_data["method"],
            url=request_data["url"],
            headers_json=request_data["headers_json"],
            cookies_json=request_data["cookies_json"],
            body=request_data["body"],
            continuation=request_data["continuation"],
            current_location=request_data["current_location"],
            accumulated_data_json=request_data["accumulated_data_json"],
            permanent_json=request_data["permanent_json"],
            expected_type=request_data["expected_type"],
            dedup_key=dedup_key,
            parent_id=parent_id,
            is_speculative=request_data["is_speculative"],
            speculation_id=request_data["speculation_id"],
            verify=request_data["verify"],
            via_json=request_data["via_json"],
            bypass_rate_limit=request_data["bypass_rate_limit"],
        )

        # Emit progress event
        await self._emit_progress(
            "request_enqueued",
            {
                "url": request_data["url"],
                "continuation": request_data["continuation"],
                "priority": resolved_request.priority,
            },
        )

    async def _stage_enqueue_request(
        self,
        new_request: BaseRequest,
        context: Response | BaseRequest,
        parent_request_id: int | None,
        staged: StagedWrites,
    ) -> None:
        """Stage an enqueue for the parent step's flush.

        Mirrors ``enqueue_request`` but defers the DB insert and progress
        event until ``staged.flush()`` is called.
        """
        resolved_request: Request = new_request.resolve_from(context)  # type: ignore[arg-type, assignment]

        dedup_key = resolved_request.deduplication_key
        if dedup_key is not None and not isinstance(dedup_key, str):
            dedup_key = None

        request_data = self._serialize_request(resolved_request)
        request_data["priority"] = resolved_request.priority

        parent_id: int | None = parent_request_id
        if (
            parent_id is None
            and isinstance(context, Response)
            and context.request
        ):
            parent_id = await self.db.find_parent_request_id(
                context.request.request.url
            )

        progress_event = {
            "url": request_data["url"],
            "continuation": request_data["continuation"],
            "priority": resolved_request.priority,
        }

        staged.stage_request(
            request_data=request_data,
            dedup_key=dedup_key,
            parent_id=parent_id,
            progress_event=progress_event,
        )

    def _serialize_request(
        self,
        request: Request,
    ) -> dict[str, Any]:
        """Serialize a Request to dictionary for DB storage.

        Args:
            request: The request to serialize.

        Returns:
            Dictionary with serialized request data.
        """
        http_request = request.request

        # Get continuation name
        continuation = request.continuation
        if callable(continuation) and not isinstance(continuation, str):
            continuation = continuation.__name__

        # Determine request type and expected_type
        if request.archive:
            request_type = "archive"
            expected_type = request.expected_type
        elif request.nonnavigating:
            request_type = "non_navigating"
            expected_type = None
        else:
            request_type = "navigating"
            expected_type = None

        # Build permanent data
        permanent_data = dict(request.permanent) if request.permanent else {}

        # Serialize speculation_id as JSON tuple ["func_name", param_index, spec_id]
        speculation_id_json = None
        if request.speculation_id is not None:
            speculation_id_json = json.dumps(list(request.speculation_id))

        # Encode query params into the URL if present
        url = http_request.url
        if http_request.params:
            parsed = urlparse(url)
            if isinstance(http_request.params, bytes):
                query = http_request.params.decode()
            else:
                query = urlencode(http_request.params)
            if parsed.query:
                query = parsed.query + "&" + query
            url = urlunparse(parsed._replace(query=query))

        # Serialize via (ViaFormSubmit / ViaLink) as JSON
        via_json: str | None = None
        if request.via is not None:
            from kent.common.page_element import ViaFormSubmit, ViaLink

            if isinstance(request.via, ViaFormSubmit):
                via_json = json.dumps(
                    {
                        "type": "form_submit",
                        "form_selector": request.via.form_selector,
                        "submit_selector": request.via.submit_selector,
                        "field_data": request.via.field_data,
                        "description": request.via.description,
                    }
                )
            elif isinstance(request.via, ViaLink):
                via_json = json.dumps(
                    {
                        "type": "link",
                        "selector": request.via.selector,
                        "description": request.via.description,
                    }
                )

        return {
            "request_type": request_type,
            "method": http_request.method.value,
            "url": url,
            "headers_json": json.dumps(http_request.headers)
            if http_request.headers
            else None,
            "cookies_json": json.dumps(http_request.cookies)
            if http_request.cookies
            else None,
            "body": http_request.data
            if isinstance(http_request.data, bytes)
            else (
                json.dumps(http_request.data).encode()
                if http_request.data
                else None
            ),
            "continuation": continuation,
            "current_location": request.current_location,
            "accumulated_data_json": json.dumps(
                request.accumulated_data, default=_json_default
            )
            if request.accumulated_data
            else None,
            "permanent_json": json.dumps(permanent_data, default=_json_default)
            if permanent_data
            else None,
            "expected_type": expected_type,
            "is_speculative": request.is_speculative,
            "speculation_id": speculation_id_json,
            "verify": None
            if http_request.verify is True
            else (
                "false"
                if http_request.verify is False
                else str(http_request.verify)
            ),
            "via_json": via_json,
            "bypass_rate_limit": request.bypass_rate_limit,
        }

    async def _get_next_request(
        self,
    ) -> tuple[int, BaseRequest, int | None] | None:
        """Get the next pending request from the database.

        Returns:
            Tuple of (request_id, request, parent_request_id) or None
            if queue is empty.

        Notes:
            - Skips 'held' status requests
            - Skips requests in retry backoff (started_at > current time)
        """
        # Atomically dequeue the next pending request.
        # Uses UPDATE ... RETURNING to prevent race conditions where multiple
        # workers could select the same request.
        # Skip 'held' status requests
        # Skip requests in retry backoff (started_at is used to track retry-after time)
        row = await self.db.dequeue_next_request()

        if row is None:
            return None

        request_id = row[0]
        parent_request_id = row[19]  # Last column in RETURNING clause

        # Deserialize using the first 19 columns (excluding parent_request_id)
        request = self._deserialize_request(row[:19])
        return (request_id, request, parent_request_id)

    def _deserialize_request(self, row: tuple[Any, ...]) -> BaseRequest:
        """Deserialize a database row to a BaseRequest.

        Args:
            row: Database row tuple from requests table.

        Returns:
            Reconstructed Request with appropriate flags set based on
            request_type (navigating, non_navigating, or archive).
        """
        (
            _id,
            request_type,
            method,
            url,
            headers_json,
            cookies_json,
            body,
            continuation,
            current_location,
            accumulated_data_json,
            permanent_json,
            expected_type,
            priority,
            is_speculative,
            speculation_id_json,
            verify_raw,
            via_json_raw,
            bypass_rate_limit_raw,
            deduplication_key_raw,
        ) = row

        # Parse JSON fields
        headers = json.loads(headers_json) if headers_json else None
        cookies = json.loads(cookies_json) if cookies_json else None
        accumulated_data = (
            json.loads(accumulated_data_json) if accumulated_data_json else {}
        )
        permanent = json.loads(permanent_json) if permanent_json else {}

        # Parse speculation_id from JSON tuple ["func_name", param_index, spec_id]
        speculation_id: tuple[str, int, int] | None = None
        if speculation_id_json:
            parsed = json.loads(speculation_id_json)
            speculation_id = (parsed[0], parsed[1], parsed[2])

        # Decode body - if it's bytes that look like JSON, decode to dict
        # This handles form data that was serialized as JSON
        decoded_body: dict[str, Any] | bytes | None = None
        if body:
            if isinstance(body, bytes):
                try:
                    # Try to decode as JSON (form data case)
                    decoded_body = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Keep as bytes (raw body case)
                    decoded_body = body
            else:
                decoded_body = body

        # Convert verify from DB representation
        verify: bool | str = True
        if verify_raw is not None:
            verify = False if verify_raw == "false" else verify_raw

        # Create HTTP request params
        http_params = HTTPRequestParams(
            method=HttpMethod(method),
            url=url,
            headers=headers,
            cookies=cookies,
            data=decoded_body,
            verify=verify,
        )

        # Deserialize via (ViaFormSubmit / ViaLink)
        via: Any = None
        if via_json_raw:
            from kent.common.page_element import ViaFormSubmit, ViaLink

            via_data = json.loads(via_json_raw)
            if via_data["type"] == "form_submit":
                via = ViaFormSubmit(
                    form_selector=via_data["form_selector"],
                    submit_selector=via_data.get("submit_selector"),
                    field_data=via_data["field_data"],
                    description=via_data["description"],
                )
            elif via_data["type"] == "link":
                via = ViaLink(
                    selector=via_data["selector"],
                    description=via_data["description"],
                )

        bypass_rate_limit = bool(bypass_rate_limit_raw)

        # Create the appropriate request type
        if request_type == "archive":
            return Request(
                request=http_params,
                continuation=continuation,
                current_location=current_location,
                accumulated_data=accumulated_data,
                permanent=permanent,
                priority=priority,
                deduplication_key=deduplication_key_raw,
                archive=True,
                expected_type=expected_type,
                via=via,
                bypass_rate_limit=bypass_rate_limit,
            )
        elif request_type == "non_navigating":
            return Request(
                request=http_params,
                continuation=continuation,
                current_location=current_location,
                accumulated_data=accumulated_data,
                permanent=permanent,
                priority=priority,
                deduplication_key=deduplication_key_raw,
                nonnavigating=True,
                via=via,
                bypass_rate_limit=bypass_rate_limit,
            )
        else:  # navigating (default)
            return Request(
                request=http_params,
                continuation=continuation,
                current_location=current_location,
                accumulated_data=accumulated_data,
                permanent=permanent,
                priority=priority,
                deduplication_key=deduplication_key_raw,
                is_speculative=bool(is_speculative),
                speculation_id=speculation_id,
                via=via,
                bypass_rate_limit=bypass_rate_limit,
            )
