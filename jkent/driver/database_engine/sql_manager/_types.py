"""Standalone types and utility functions for the sql_manager package.

Defines data transfer objects and the compute_cache_key helper used across
the driver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypedDict

if TYPE_CHECKING:
    from jkent.data_types import HttpMethod
    from jkent.driver.database_engine.enums import RequestType


@dataclass
class RequestInsert:
    """One ``requests``-row insert, as ``insert_request`` consumes it.

    The single definition of the insert payload: ``serialize_request``
    produces it, ``insert_request`` / ``insert_request_in_session`` consume
    it, so the field list exists once instead of being repeated across three
    signatures. ``dedup_key`` and ``parent_id`` keep the query layer's
    vocabulary and map to the ``deduplication_key`` / ``parent_request_id``
    columns via :meth:`column_values`.
    """

    request_type: RequestType
    method: HttpMethod
    url: str
    continuation: str
    current_location: str = ""
    priority: int = 9
    headers_json: str | None = None
    cookies_json: str | None = None
    body: bytes | None = None
    accumulated_data_json: str | None = None
    permanent_json: str | None = None
    expected_type: str | None = None
    dedup_key: str | None = None
    parent_id: int | None = None
    is_speculative: bool = False
    speculation_tracking_id: int | None = None
    speculative_index: int | None = None
    verify: str | None = None
    via_json: str | None = None
    bypass_rate_limit: bool = False
    timeout_json: str | None = None
    json_data: str | None = None
    files_json: str | None = None
    auth_json: str | None = None
    allow_redirects: bool = True
    proxies_json: str | None = None
    stream: bool = False
    cert_json: str | None = None
    archive_hash_header: str | None = None
    reseedable: bool | None = None

    _COLUMN_RENAMES: ClassVar[dict[str, str]] = {
        "dedup_key": "deduplication_key",
        "parent_id": "parent_request_id",
    }

    def column_values(self) -> dict[str, Any]:
        """Field values keyed by ``requests`` column name."""
        return {
            self._COLUMN_RENAMES.get(f.name, f.name): getattr(self, f.name)
            for f in fields(self)
        }


class IncidentalRequestParams(TypedDict):
    """Kwargs of one incidental-request insert, minus ``parent_request_id``.

    The record shape :meth:`IncidentalRequestStorageMixin.
    insert_incidental_requests` consumes — typed so a misspelled key is a
    type error at the call site instead of a ``TypeError`` mid-transaction.
    Every key is required: the capture listener initializes the full record
    up front and fills response fields in as they arrive.
    """

    resource_type: str
    method: str
    url: str
    headers_json: str | None
    body: bytes | None
    status_code: int | None
    response_headers_json: str | None
    content_compressed: bytes | None
    content_size_original: int | None
    content_size_compressed: int | None
    compression_dict_id: int | None
    started_at_ns: int | None
    completed_at_ns: int | None
    from_cache: bool | None
    failure_reason: str | None


@dataclass
class PreresolvedResponse:
    """Response columns to store on a pre-resolved request at insert time.

    Carries an already-compressed body (copied verbatim from the captured
    incidental's storage row, so no re-compression) plus the metadata the
    worker needs to reconstruct a :class:`~jkent.data_types.Response` when it
    runs the continuation without touching the transport.
    """

    status_code: int
    headers_json: str | None
    url: str
    content_compressed: bytes | None
    content_size_original: int | None
    content_size_compressed: int | None
    compression_dict_id: int | None


def compute_cache_key(
    method: str,
    url: str,
    body: bytes | None = None,
    headers_json: str | None = None,
) -> bytes:
    """Compute a cache key for response caching.

    Args:
        method: HTTP method (GET, POST, etc.).
        url: Request URL.
        body: Request body bytes (for POST/PUT requests).
        headers_json: JSON-encoded headers (optional).

    Returns:
        The raw 16-byte MD5 digest, as stored in ``requests.cache_key``.
    """
    hasher = hashlib.md5(usedforsecurity=False)
    hasher.update(method.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(url.encode("utf-8"))
    hasher.update(b"\x00")
    if body:
        hasher.update(body)
    hasher.update(b"\x00")
    if headers_json:
        hasher.update(headers_json.encode("utf-8"))
    return hasher.digest()


class IncidentalRequestDict(TypedDict):
    """JSON-ready shape of an :class:`IncidentalRequestRecord`.

    Mirrors the record's fields, with the derived properties
    (``compression_ratio``, ``duration_ms``) materialized and ``created_at``
    rendered as an ISO 8601 string.
    """

    id: int
    parent_request_id: int
    url: str
    headers_json: str | None
    started_at_ns: int | None
    completed_at_ns: int | None
    from_cache: bool | None
    created_at: str | None
    storage_id: int | None
    resource_type: str | None
    method: str | None
    status_code: int | None
    content_size_original: int | None
    content_size_compressed: int | None
    compression_ratio: float | None
    failure_reason: str | None
    duration_ms: float | None


@dataclass
class IncidentalRequestRecord:
    """Incidental request record joining metadata and storage tables.

    Represents a browser-initiated network request captured by Playwright,
    combining timing/metadata from incidental_requests with content/response
    data from incidental_request_storage.
    """

    id: int
    parent_request_id: int
    url: str
    headers_json: str | None
    started_at_ns: int | None
    completed_at_ns: int | None
    from_cache: bool | None
    created_at: datetime | None
    storage_id: int | None
    resource_type: str | None = None
    method: str | None = None
    status_code: int | None = None
    content_size_original: int | None = None
    content_size_compressed: int | None = None
    failure_reason: str | None = None

    @property
    def duration_ns(self) -> int | None:
        if self.started_at_ns is not None and self.completed_at_ns is not None:
            return self.completed_at_ns - self.started_at_ns
        return None

    @property
    def duration_ms(self) -> float | None:
        duration = self.duration_ns
        if duration is not None:
            return duration / 1_000_000
        return None

    @property
    def compression_ratio(self) -> float | None:
        """Compressed size over original size (lower is better).

        Same direction as the ``jkent.compression.ratio`` histogram, so a
        replay listing and a Grafana panel agree on what a "ratio" means.
        None if either size is unknown.
        """
        if self.content_size_original and self.content_size_compressed:
            return self.content_size_compressed / self.content_size_original
        return None

    def to_dict(self) -> IncidentalRequestDict:
        return {
            "id": self.id,
            "parent_request_id": self.parent_request_id,
            "url": self.url,
            "headers_json": self.headers_json,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "from_cache": self.from_cache,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "storage_id": self.storage_id,
            "resource_type": self.resource_type,
            "method": self.method,
            "status_code": self.status_code,
            "content_size_original": self.content_size_original,
            "content_size_compressed": self.content_size_compressed,
            "compression_ratio": self.compression_ratio,
            "failure_reason": self.failure_reason,
            "duration_ms": self.duration_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
