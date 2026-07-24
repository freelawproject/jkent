"""RequestQueueDB - DB-backed request queue operations for the unified driver."""

from __future__ import annotations

import base64
import json
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode, urlparse, urlunparse

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import (
    FilesType,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    VerifyType,
    via_from_json,
)
from jkent.driver.database_engine.sql_manager import SQLManager

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager._types import (
        PreresolvedResponse,
    )
    from jkent.driver.database_engine.staging import StagedWrites


def _json_default(obj: Any) -> Any:
    """Handle date/datetime objects in json.dumps."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def serialize_url_and_body(
    http_request: HTTPRequestParams,
) -> tuple[str, bytes | None]:
    """Encode an HTTPRequestParams into the ``(url, body)`` form the DB stores.

    Query params are folded into the URL with ``doseq=True`` so a repeated /
    list-valued key encodes as ``q=a&q=b`` like a browser, not as one
    urlencoded repr of the list. The body mirrors the queue's storage rule:
    bytes pass through verbatim (even ``b""``); any other truthy data is
    JSON-encoded; falsy non-bytes data (None, {}, []) stores as None.

    This is the single source of truth shared by
    ``RequestQueueDB.serialize_request`` (the write path) and replay's
    fallback-key derivation (``replay.source_index``), so the two can never
    drift — a drift here silently breaks replay key matching.
    """
    url = http_request.url
    if http_request.params:
        parsed = urlparse(url)
        if isinstance(http_request.params, bytes):
            query = http_request.params.decode()
        else:
            query = urlencode(http_request.params, doseq=True)
        if parsed.query:
            query = parsed.query + "&" + query
        # urlparse/urlunparse are AnyStr-typed; with a str url and str query
        # the result is always str, but the checker widens it to str | bytes.
        url = cast(str, urlunparse(parsed._replace(query=query)))

    body: bytes | None
    if isinstance(http_request.data, bytes):
        body = http_request.data
    elif http_request.data:
        body = json.dumps(http_request.data).encode()
    else:
        body = None

    return url, body


def _dump_if_present(value: Any) -> str | None:
    """JSON-encode ``value`` unless it is ``None``.

    Use for fields where a falsy-but-valid value (``0``, ``[]``, ``False``,
    ``0.0``) must be preserved — e.g. ``json`` and ``timeout``.
    """
    return json.dumps(value) if value is not None else None


def _dump_if_truthy(value: Any) -> str | None:
    """JSON-encode ``value`` unless it is falsy (None / empty dict / tuple)."""
    return json.dumps(value) if value else None


def _retuple(raw: str | None) -> Any:
    """Decode a JSON field, re-tupling lists (JSON has no tuple type).

    ``auth``, ``cert`` and ``timeout`` are stored as JSON lists when the
    scraper supplied a tuple; re-tuple them so equality with the original
    HTTPRequestParams matches.
    """
    if raw is None:
        return None
    parsed = json.loads(raw)
    return tuple(parsed) if isinstance(parsed, list) else parsed


def _serialize_verify(verify: VerifyType) -> str | None:
    """Encode ``verify`` (bool | CA-bundle path) for the single TEXT column.

    ``True`` -> NULL, ``False`` -> the sentinel ``"false"``, a path -> itself.
    A path literally equal to ``"false"`` would collide with the False
    sentinel on deserialize (silently disabling TLS verification), so reject
    it up front.
    """
    if verify is True:
        return None
    if verify is False:
        return "false"
    if verify == "false":
        raise ScraperConfigError(
            "verify='false' (a CA-bundle path) is ambiguous with "
            "verify=False in the request queue. Use verify=False to disable "
            "TLS verification, or point verify at a differently-named bundle."
        )
    return str(verify)


def _serialize_files(files: FilesType) -> str | None:
    """Serialize ``files`` to JSON, base64-encoding any binary content.

    Each value is tagged so the deserializer can rebuild it: a plain str is
    stored as text; bytes / file-like objects are read and base64-encoded; a
    file-tuple keeps its filename + trailing (content_type, headers) members
    around the (possibly binary) content.
    """
    if not files:
        return None
    return json.dumps(
        {name: _encode_file(value) for name, value in files.items()}
    )


def _encode_file(value: Any) -> dict[str, Any]:
    if isinstance(value, tuple):
        # FileTuple: (filename, fileobj[, content_type[, headers]]).
        return {
            "kind": "tuple",
            "filename": value[0],
            "content": _encode_file_content(value[1]),
            "extra": list(value[2:]),
        }
    return {"kind": "value", "content": _encode_file_content(value)}


def _encode_file_content(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"enc": "text", "data": content}
    if isinstance(content, bytes):
        return {
            "enc": "base64",
            "data": base64.b64encode(content).decode("ascii"),
        }
    if hasattr(content, "read"):
        raw = content.read()
        if isinstance(raw, str):
            return {"enc": "text", "data": raw}
        return {"enc": "base64", "data": base64.b64encode(raw).decode("ascii")}
    raise TypeError(
        f"file content of type {type(content).__name__} is not serializable"
    )


def _deserialize_files(raw: str | None) -> FilesType:
    """Inverse of :func:`_serialize_files`. Binary content rebuilds as bytes."""
    if not raw:
        return None
    decoded = json.loads(raw)
    return {name: _decode_file(spec) for name, spec in decoded.items()}


def _decode_file(spec: dict[str, Any]) -> Any:
    if spec.get("kind") == "tuple":
        content = _decode_file_content(spec["content"])
        return (spec["filename"], content, *spec["extra"])
    return _decode_file_content(spec["content"])


def _decode_file_content(content: dict[str, Any]) -> str | bytes:
    if content["enc"] == "text":
        return content["data"]
    return base64.b64decode(content["data"])


class RequestQueueDB:
    """DB-backed queue: enqueue (staged), dequeue, (de)serialization.

    Provides methods for persisting requests to SQLite with deduplication
    and reconstructing request objects from database rows. Expects a
    ``db: SQLManager`` attribute supplied by the subclass.

    Also tracks how many dequeued requests are still being handled
    (``in_flight_count``): a worker pool is pinned — retired workers are
    never replaced — so an idle worker must distinguish a drained queue from
    a momentary lull while a sibling's in-flight request may still enqueue
    children. The count is in-memory (all workers share one queue instance),
    so stale ``in_progress`` rows from a crashed prior run can never wedge
    idle workers the way a DB-status check could.
    """

    db: SQLManager  # type: ignore

    # Class-level default; instances mutate it into an instance attribute.
    # Subclass __init__s don't chain here, so a plain class default keeps
    # every subclass covered without touching their constructors.
    _in_flight: int = 0

    @property
    def in_flight_count(self) -> int:
        """Dequeued requests not yet marked done via :meth:`request_done`."""
        return self._in_flight

    def request_done(self) -> None:
        """Mark one dequeued request as fully handled (children enqueued)."""
        self._in_flight -= 1

    async def _prepare_enqueue(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None,
    ) -> tuple[dict[str, Any], str | None, int | None, dict[str, Any]]:
        """Resolve, serialize, and dedup/parent-resolve a request for enqueue.

        The shared core of both enqueue paths — the immediate
        ``RequestQueue.enqueue_request`` and the staged
        ``_stage_enqueue_request`` — so the two cannot drift on serialization,
        the dedup-key / priority / parent-id rules, or the progress payload.

        Returns ``(request_data, dedup_key, parent_id, progress_event)`` where
        ``request_data`` already carries the effective ``priority`` and is ready
        to splat into ``insert_request`` / ``stage_request``.
        """
        # Await any async field resolvers first: serialization needs concrete
        # values, and the regenerated dedup key must hash them.
        new_request = await new_request.resolve_deferred_fields()
        resolved_request: Request = new_request.resolve_from(context)  # type: ignore[arg-type, assignment]

        dedup_key = resolved_request.deduplication_key
        if dedup_key is not None and not isinstance(dedup_key, str):
            # SkipDeduplicationCheck sentinel -> always enqueue.
            dedup_key = None

        request_data = self.serialize_request(resolved_request)
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
        return request_data, dedup_key, parent_id, progress_event

    async def _stage_enqueue_request(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None,
        staged: StagedWrites,
        preresolved_response: PreresolvedResponse | None = None,
    ) -> None:
        """Stage an enqueue for the parent step's flush.

        Mirrors ``enqueue_request`` but defers the DB insert and progress
        event until ``staged.flush()`` is called. When ``preresolved_response``
        is set the row is inserted pre-resolved (response stored, worker skips
        the transport).
        """
        (
            request_data,
            dedup_key,
            parent_id,
            progress_event,
        ) = await self._prepare_enqueue(
            new_request, context, parent_request_id
        )
        staged.stage_request(
            request_data=request_data,
            dedup_key=dedup_key,
            parent_id=parent_id,
            progress_event=progress_event,
            preresolved_response=preresolved_response,
        )

    def serialize_request(
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

        # Fold query params into the URL and encode the body. Shared with
        # replay's key derivation so the two encodings can't drift.
        url, body = serialize_url_and_body(http_request)

        # Serialize via (ViaFormSubmit / ViaLink) as JSON
        via_json: str | None = None
        if request.via is not None:
            via_json = request.via.to_json()

        return {
            "request_type": request_type,
            "method": http_request.method.value,
            "url": url,
            "headers_json": _dump_if_truthy(http_request.headers),
            "cookies_json": _dump_if_truthy(http_request.cookies),
            "body": body,
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
            "verify": _serialize_verify(http_request.verify),
            "via_json": via_json,
            "bypass_rate_limit": request.bypass_rate_limit,
            # tuple values (timeout=(connect, read), auth=(user, pass),
            # cert=(cert, key)) store as JSON lists; the deserializer re-tuples.
            "timeout_json": _dump_if_present(http_request.timeout),
            "json_data": _dump_if_present(http_request.json),
            "files_json": _serialize_files(http_request.files),
            "auth_json": _dump_if_truthy(http_request.auth),
            "allow_redirects": http_request.allow_redirects,
            "proxies_json": _dump_if_truthy(http_request.proxies),
            "stream": http_request.stream,
            "cert_json": _dump_if_truthy(http_request.cert),
            "archive_hash_header": request.archive_hash_header,
            "reseedable": request.reseedable,
        }

    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
        """Get the next pending request from the database.

        Returns:
            Tuple of (request_id, request, parent_request_id, preresolved)
            or None if queue is empty. ``preresolved`` is True when the row
            already carries its response (promoted from a captured incidental)
            and the worker should skip the transport.

        Notes:
            - Skips 'held' status requests
            - Skips requests in retry backoff (started_at > current time)
        """
        # Atomically dequeue the next pending request.
        # Uses UPDATE ... RETURNING to prevent race conditions where multiple
        # workers could select the same request.
        # Skip 'held' status requests
        # Skip requests in retry backoff (started_at is used to track retry-after time)
        #
        # The in-flight count is raised BEFORE the dequeue so a claim is never
        # invisible: a sibling that reads an empty queue while this dequeue is
        # mid-commit still sees in_flight_count > 0 and keeps waiting instead
        # of retiring. The caller owns the matching request_done().
        self._in_flight += 1
        try:
            row = await self.db.dequeue_next_request()
        except BaseException:
            self._in_flight -= 1
            raise

        if row is None:
            self._in_flight -= 1
            return None

        request_id = row[0]
        parent_request_id = row[29]
        preresolved = bool(row[30])  # trailing RETURNING column

        # Deserialize using the first 29 columns (excluding parent_request_id
        # at [29] and preresolved at [30]).
        request = self._deserialize_request(row[:29])
        return (request_id, request, parent_request_id, preresolved)

    async def seconds_until_next_pending(self) -> float | None:
        """Delay until the soonest pending request is dequeuable.

        0.0 if one is ready now, the positive gap until the soonest retry
        still in backoff, or None if nothing is pending. See
        :meth:`SQLManager.seconds_until_next_pending`.
        """
        return await self.db.seconds_until_next_pending()

    async def restamp_request_start(self, request_id: int) -> None:
        """Re-stamp a request's start to now, after the rate-limit gate."""
        await self.db.restamp_request_start(request_id)

    def _deserialize_request(self, row: tuple[Any, ...]) -> Request:
        """Deserialize a database row to a Request.

        Args:
            row: Database row tuple from requests table.

        Returns:
            Reconstructed Request with appropriate flags set based on
            request_type (navigating, non_navigating, or archive).

        Note:
            ``HTTPRequestParams.params`` is *not* restored — it is folded
            into the stored URL on serialize (see ``serialize_url_and_body``),
            which is the single source of truth for the request target. The
            httpx transport sends the URL as-is and never re-sends ``params``,
            so the reconstructed request carries ``params=None`` with the
            query already in the URL. This is intentional: restoring ``params``
            would either drop the query (transport ignores ``params``) or
            double-encode it on re-serialization.
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

        # Decode body. Bytes that decode as JSON are reconstructed into the
        # original Python object (the common case: form data / JSON payloads
        # serialized via json.dumps). This is deliberately favored over raw
        # bytes — a raw bytes body that happens to be valid JSON (e.g.
        # b'{"a":1}', b'123') round-trips as the decoded object, not bytes.
        # Non-JSON / non-UTF-8 bytes are preserved verbatim. ``is not None``
        # (not truthiness) so an explicitly empty body ``b""`` survives as
        # b"" instead of collapsing to None.
        decoded_body: dict[str, Any] | bytes | None = None
        if body is not None:
            if isinstance(body, bytes):
                try:
                    decoded_body = json.loads(body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    decoded_body = body
            else:
                decoded_body = body

        # Convert verify from DB representation
        verify: bool | str = True
        if verify_raw is not None:
            verify = False if verify_raw == "false" else verify_raw

        # JSON has no tuple type, so timeout / auth / cert come back as lists
        # when the scraper supplied a tuple; _retuple restores them.
        timeout: float | tuple[float, float] | None = _retuple(
            timeout_json_raw
        )
        json_field: Any = (
            json.loads(json_data_raw) if json_data_raw is not None else None
        )
        files = _deserialize_files(files_json_raw)
        auth: tuple[str, str] | None = _retuple(auth_json_raw)

        allow_redirects = (
            True if allow_redirects_raw is None else bool(allow_redirects_raw)
        )
        proxies = json.loads(proxies_json_raw) if proxies_json_raw else None
        stream = False if stream_raw is None else bool(stream_raw)
        cert: str | tuple[str, str] | None = _retuple(cert_json_raw)

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
            via = via_from_json(via_json_raw)

        bypass_rate_limit = bool(bypass_rate_limit_raw)
        reseedable = None if reseedable_raw is None else bool(reseedable_raw)

        # Kwargs shared by every request type; the request_type only varies
        # the handful of flag/extra fields grafted on below.
        common: dict[str, Any] = {
            "request": http_params,
            "continuation": continuation,
            "current_location": current_location,
            "accumulated_data": accumulated_data,
            "permanent": permanent,
            "priority": priority,
            "deduplication_key": deduplication_key_raw,
            "via": via,
            "bypass_rate_limit": bypass_rate_limit,
            "reseedable": reseedable,
        }

        if request_type == "archive":
            return Request(
                **common,
                archive=True,
                expected_type=expected_type,
                archive_hash_header=archive_hash_header_raw,
            )
        elif request_type == "non_navigating":
            return Request(**common, nonnavigating=True)
        else:  # navigating (default)
            return Request(
                **common,
                is_speculative=bool(is_speculative),
                speculation_id=speculation_id,
            )
