"""Standalone types and utility functions for the sql_manager package.

Defines data transfer objects and the compute_cache_key helper used across
the driver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


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
) -> str:
    """Compute a cache key for response caching.

    The cache key is a SHA256 hash of the request parameters that affect
    the response: method, URL, body, and headers.

    Args:
        method: HTTP method (GET, POST, etc.).
        url: Request URL.
        body: Request body bytes (for POST/PUT requests).
        headers_json: JSON-encoded headers (optional).

    Returns:
        Hex-encoded SHA256 hash string.
    """
    hasher = hashlib.sha256()
    hasher.update(method.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(url.encode("utf-8"))
    hasher.update(b"\x00")
    if body:
        hasher.update(body)
    hasher.update(b"\x00")
    if headers_json:
        hasher.update(headers_json.encode("utf-8"))
    return hasher.hexdigest()


def _compression_ratio(
    original: int | None, compressed: int | None
) -> float | None:
    """Ratio of original to compressed size, rounded to 2 dp.

    Returns None when either size is missing or zero (avoids div-by-zero
    and meaningless ratios).
    """
    if original and compressed:
        return round(original / compressed, 2)
    return None


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
    created_at: str | None
    storage_id: int | None
    # Fields from storage table (joined)
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
        return _compression_ratio(
            self.content_size_original, self.content_size_compressed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_request_id": self.parent_request_id,
            "url": self.url,
            "headers_json": self.headers_json,
            "started_at_ns": self.started_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "from_cache": self.from_cache,
            "created_at": self.created_at,
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
