"""RequestQueueDB - DB-backed request queue operations for the unified driver."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse, urlunparse

from jkent.common.page_element import (
    ViaFormSubmit,
    ViaLink,
)
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    Selector,
)
from jkent.driver.database_engine.sql_manager import SQLManager

if TYPE_CHECKING:
    from jkent.driver.database_engine.staging import StagedWrites


def _selector_grammar(selector: str) -> str:
    """Best-effort selector grammar for legacy via rows lacking selector_type.

    Mirrors ``find_form``/``find_links``: unambiguous XPath prefixes are
    "xpath", everything else "css". Only used as a fallback — rows written
    after selector_type was added carry the real value.
    """
    return "xpath" if selector.startswith(("//", "./", "(")) else "css"


def _json_default(obj: Any) -> Any:
    """Handle date/datetime objects in json.dumps."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


class RequestQueueDB:
    """DB-backed queue: enqueue (staged), dequeue, (de)serialization.

    Provides methods for persisting requests to SQLite with deduplication
    and reconstructing request objects from database rows. Expects a
    ``db: SQLManager`` attribute supplied by the subclass.
    """

    db: SQLManager  # type: ignore

    async def _stage_enqueue_request(
        self,
        new_request: Request,
        context: Response | Request,
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
        request_data["priority"] = resolved_request.effective_priority

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
            "priority": resolved_request.effective_priority,
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
                # doseq so a repeated key (checkbox group / multi-select, whose
                # value is a list) encodes as repeated names — q=a&q=b — like a
                # browser, not as one urlencoded repr of the list.
                query = urlencode(http_request.params, doseq=True)
            if parsed.query:
                query = parsed.query + "&" + query
            url = urlunparse(parsed._replace(query=query))

        # Serialize via (ViaFormSubmit / ViaLink) as JSON
        via_json: str | None = None
        if request.via is not None:
            if isinstance(request.via, ViaFormSubmit):
                via_json = json.dumps(
                    {
                        "type": "form_submit",
                        "form_selector": request.via.form_selector.value,
                        "selector_type": request.via.form_selector.grammar,
                        "submit_selector": request.via.submit_selector,
                        "field_data": request.via.field_data,
                        "description": request.via.description,
                    }
                )
            elif isinstance(request.via, ViaLink):
                via_json = json.dumps(
                    {
                        "type": "link",
                        "selector": request.via.selector.value,
                        "selector_type": request.via.selector.grammar,
                        "description": request.via.description,
                    }
                )

        # Serialize the remaining HTTPRequestParams fields. tuple values
        # (timeout=(connect, read), auth=(user, pass), cert=(cert, key))
        # are stored as JSON lists; the deserializer re-tuples them.
        timeout_json: str | None = (
            json.dumps(http_request.timeout)
            if http_request.timeout is not None
            else None
        )
        json_data: str | None = (
            json.dumps(http_request.json)
            if http_request.json is not None
            else None
        )
        files_json: str | None = (
            json.dumps(http_request.files) if http_request.files else None
        )
        auth_json: str | None = (
            json.dumps(http_request.auth) if http_request.auth else None
        )
        proxies_json: str | None = (
            json.dumps(http_request.proxies) if http_request.proxies else None
        )
        cert_json: str | None = (
            json.dumps(http_request.cert) if http_request.cert else None
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
            "timeout_json": timeout_json,
            "json_data": json_data,
            "files_json": files_json,
            "auth_json": auth_json,
            "allow_redirects": http_request.allow_redirects,
            "proxies_json": proxies_json,
            "stream": http_request.stream,
            "cert_json": cert_json,
            "archive_hash_header": request.archive_hash_header,
            "reseedable": request.reseedable,
        }

    async def _get_next_request(
        self,
    ) -> tuple[int, Request, int | None] | None:
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
        parent_request_id = row[29]  # Last column in RETURNING clause

        # Deserialize using the first 29 columns (excluding parent_request_id)
        request = self._deserialize_request(row[:29])
        return (request_id, request, parent_request_id)

    def _deserialize_request(self, row: tuple[Any, ...]) -> Request:
        """Deserialize a database row to a Request.

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
            timeout_json_raw,
            json_data_raw,
            files_json_raw,
            auth_json_raw,
            allow_redirects_raw,
            proxies_json_raw,
            stream_raw,
            cert_json_raw,
            archive_hash_header_raw,
            reseedable_raw,
        ) = row

        # Parse JSON fields
        headers = json.loads(headers_json) if headers_json else None
        cookies = json.loads(cookies_json) if cookies_json else None
        accumulated_data: dict[str, Any] = (
            json.loads(accumulated_data_json) if accumulated_data_json else {}
        )
        permanent: dict[str, Any] = (
            json.loads(permanent_json) if permanent_json else {}
        )

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

        # Deserialize the remaining HTTPRequestParams fields. JSON has no
        # tuple type, so timeout / auth / cert come back as lists when
        # the scraper supplied a tuple; re-tuple them so equality with
        # the original HTTPRequestParams matches.
        timeout: float | tuple[float, float] | None
        if timeout_json_raw is None:
            timeout = None
        else:
            parsed_timeout = json.loads(timeout_json_raw)
            timeout = (
                tuple(parsed_timeout)  # type: ignore[assignment]
                if isinstance(parsed_timeout, list)
                else parsed_timeout
            )

        json_field: Any = (
            json.loads(json_data_raw) if json_data_raw is not None else None
        )
        files = json.loads(files_json_raw) if files_json_raw else None
        auth: tuple[str, str] | None
        if auth_json_raw:
            parsed_auth = json.loads(auth_json_raw)
            auth = (
                tuple(parsed_auth)  # type: ignore[assignment]
                if isinstance(parsed_auth, list)
                else parsed_auth
            )
        else:
            auth = None

        allow_redirects = (
            True if allow_redirects_raw is None else bool(allow_redirects_raw)
        )
        proxies = json.loads(proxies_json_raw) if proxies_json_raw else None
        stream = False if stream_raw is None else bool(stream_raw)
        cert: str | tuple[str, str] | None
        if cert_json_raw:
            parsed_cert = json.loads(cert_json_raw)
            cert = (
                tuple(parsed_cert)  # type: ignore[assignment]
                if isinstance(parsed_cert, list)
                else parsed_cert
            )
        else:
            cert = None

        # Create HTTP request params
        http_params = HTTPRequestParams(
            method=HttpMethod(method),
            url=url,
            headers=headers,
            cookies=cookies,
            data=decoded_body,
            json=json_field,
            files=files,
            auth=auth,
            timeout=timeout,
            allow_redirects=allow_redirects,
            proxies=proxies,
            verify=verify,
            stream=stream,
            cert=cert,
        )

        # Deserialize via (ViaFormSubmit / ViaLink)
        via: Any = None
        if via_json_raw:
            via_data = json.loads(via_json_raw)
            if via_data["type"] == "form_submit":
                via = ViaFormSubmit(
                    form_selector=Selector.of(
                        via_data["form_selector"],
                        via_data.get(
                            "selector_type",
                            _selector_grammar(via_data["form_selector"]),
                        ),
                    ),
                    submit_selector=via_data.get("submit_selector"),
                    field_data=via_data["field_data"],
                    description=via_data["description"],
                )
            elif via_data["type"] == "link":
                via = ViaLink(
                    selector=Selector.of(
                        via_data["selector"],
                        via_data.get(
                            "selector_type",
                            _selector_grammar(via_data["selector"]),
                        ),
                    ),
                    description=via_data["description"],
                )

        bypass_rate_limit = bool(bypass_rate_limit_raw)
        reseedable = None if reseedable_raw is None else bool(reseedable_raw)

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
                archive_hash_header=archive_hash_header_raw,
                via=via,
                bypass_rate_limit=bypass_rate_limit,
                reseedable=reseedable,
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
                reseedable=reseedable,
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
                reseedable=reseedable,
            )
