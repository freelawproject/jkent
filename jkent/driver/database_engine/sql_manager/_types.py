"""Row-shaped models and utilities for the sql_manager package.

Every payload that crosses the SQLManager boundary is a pydantic model keyed
by *column name*, so one declaration serves as the insert payload, the
``RETURNING`` list, the row decoder (``model_validate(row)`` via
``from_attributes``), and the JSON-ready dump. Extra keys are rejected: a
misspelled column is a validation error at the call site, not a silent NULL.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from pydantic import BaseModel, computed_field

from jkent.common.serialization import JsonColumn
from jkent.data_types import HttpMethod, TimeoutType, VerifyType
from jkent.driver.database_engine.enums import RequestType
from jkent.driver.database_engine.models import RowModel

if TYPE_CHECKING:
    from sqlalchemy.orm import InstrumentedAttribute

    from jkent.driver.database_engine.models import Base


def model_columns(
    model: type[BaseModel],
    table: type[Base],
    *fallbacks: type[Base],
    partial: bool = False,
) -> dict[str, InstrumentedAttribute[Any]]:
    """*model*'s fields mapped to the ORM columns of the same name.

    Each field resolves against *table*, then each of *fallbacks* in order
    (for a model spanning a join). This is how a row model becomes a
    ``select``/``RETURNING`` list or an insert split, so the model is the
    only list of columns. A field with no column raises at import time
    unless *partial*, in which case it is left out.
    """
    columns: dict[str, InstrumentedAttribute[Any]] = {}
    for name in model.model_fields:
        for candidate in (table, *fallbacks):
            if name in candidate.__table__.columns:
                columns[name] = getattr(candidate, name)
                break
        else:
            if not partial:
                raise TypeError(
                    f"{model.__name__}.{name} is not a column of "
                    + ", ".join(t.__name__ for t in (table, *fallbacks))
                )
    return columns


class _RequestColumns(RowModel):
    """The ``requests`` columns an insert and a dequeue share, less the JSON
    columns, which an insert carries as text and a dequeue decodes."""

    request_type: RequestType
    method: HttpMethod
    url: str
    step: str
    current_location: str = ""
    priority: int = 9
    body: bytes | None = None
    body_is_form: bool = False
    expected_type: str | None = None
    deduplication_key: str | None = None
    parent_request_id: int | None = None
    is_speculative: bool = False
    speculation_tracking_id: int | None = None
    speculative_index: int | None = None
    via_json: str | None = None
    rate_limit: int = 0
    reseedable: bool | None = None


class RequestInsert(_RequestColumns):
    """One ``requests``-row insert, as ``insert_request`` consumes it.

    Fields are ``requests`` column names, so ``model_dump()`` is the INSERT
    payload.
    ``serialize_request`` produces it; the enqueue paths fill in
    ``priority``, ``deduplication_key`` and ``parent_request_id``.
    """

    headers_json: str | None = None
    cookies_json: str | None = None
    accumulated_data_json: str | None = None
    permanent_json: str | None = None
    timeout_json: str | None = None
    json_data: str | None = None
    verify_json: str = "true"


class DequeuedRow(_RequestColumns):
    """One dequeued request row: the insert columns plus queue state.

    The JSON columns come back decoded. The RETURNING list in
    ``dequeue_next_request`` is generated from ``model_fields``, so the row
    shape and this type cannot drift.
    """

    id: int
    preresolved: bool
    headers_json: Annotated[dict[str, str] | None, JsonColumn] = None
    cookies_json: Annotated[dict[str, str] | None, JsonColumn] = None
    accumulated_data_json: Annotated[dict[str, Any] | None, JsonColumn] = None
    permanent_json: Annotated[dict[str, Any] | None, JsonColumn] = None
    timeout_json: Annotated[TimeoutType, JsonColumn] = None
    json_data: Annotated[Any, JsonColumn] = None
    verify_json: Annotated[VerifyType, JsonColumn] = True


class InsertResult(NamedTuple):
    """What an insert resolved to: the row's id, and whether it is new.

    ``inserted`` is False when the dedup key already had a row — the
    constraint's ``ON CONFLICT IGNORE`` swallowed the INSERT and ``request_id``
    is the existing row's.
    """

    request_id: int
    inserted: bool


class CompressedPayload(RowModel):
    """A stored, compressed body.

    The four columns ``requests`` and ``incidental_request_storage`` both
    carry. ``content_compressed`` is None when there is no body.
    """

    content_compressed: bytes | None = None
    content_size_original: int | None = None
    content_size_compressed: int | None = None
    compression_dict_id: int | None = None


class StoredResponse(CompressedPayload):
    """A request's stored response columns.

    One shape for every direction: what ``store_response`` writes, what the
    pre-resolved insert path stores at enqueue time, and what
    ``get_stored_response`` reads back for ``Response`` reconstruction.
    Fields are ``requests`` column names, so ``model_dump()`` is the
    UPDATE/INSERT payload.

    ``content_compressed`` is None for a bodyless response (an empty body is
    stored as NULL, never ``b""``) and for archive responses, whose bytes
    live on disk.
    """

    response_status_code: int
    response_headers_json: str | None = None
    response_url: str


class ResultInsert(RowModel):
    """One ``results``-row insert: a serialized record and its validity."""

    result_type: str
    data_json: str
    is_valid: bool = True
    validation_errors_json: str | None = None


class IncidentalCapture(CompressedPayload):
    """One captured browser sub-request, as the Playwright listener builds it.

    The request-side fields are set when the browser issues the request; the
    response-side ones are filled in as the response arrives, so they
    default to None. ``replace_incidental_requests`` splits the fields between
    ``incidental_requests`` and ``incidental_request_storage`` by column
    name.

    ``content_md5`` is the digest of the body as delivered, before
    compression — the storage row's identity. Set it, the body and its sizes
    together with :meth:`set_body`.
    """

    resource_type: str
    method: str
    url: str
    headers_json: str | None = None
    body: bytes | None = None
    status_code: int | None = None
    response_headers_json: str | None = None
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    from_cache: bool | None = None
    failure_reason: str | None = None
    content_md5: bytes | None = None

    def set_body(self, content: bytes, compressed: bytes) -> None:
        """Record the delivered body ``content`` stored as ``compressed``."""
        self.content_compressed = compressed
        self.content_size_original = len(content)
        self.content_size_compressed = len(compressed)
        self.content_md5 = hashlib.md5(content, usedforsecurity=False).digest()


class IncidentalRequestRecord(RowModel):
    """An incidental request joined with its storage row.

    Everything but the payload sizes comes from ``incidental_requests``;
    ``content_size_original`` / ``content_size_compressed`` come from the
    content-addressed ``incidental_request_storage`` row and are None when
    the capture had no body to store.
    ``model_dump(mode="json")`` is the display shape: ``created_at`` renders
    as ISO 8601 and the derived durations/ratio are included.
    """

    id: int
    parent_request_id: int
    url: str
    resource_type: str
    method: str
    headers_json: str | None
    started_at_ns: int | None
    completed_at_ns: int | None
    from_cache: bool | None
    created_at: datetime | None
    storage_id: int | None
    status_code: int | None = None
    content_size_original: int | None = None
    content_size_compressed: int | None = None
    failure_reason: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ms(self) -> float | None:
        if self.started_at_ns is None or self.completed_at_ns is None:
            return None
        return (self.completed_at_ns - self.started_at_ns) / 1_000_000

    @computed_field  # type: ignore[prop-decorator]
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


def compute_cache_key(row: RequestInsert) -> bytes:
    """The ``requests.cache_key`` digest for *row*.

    Hashes the wire request as stored: method, URL, body and its JSON flag,
    headers, and the ``json`` payload, joined as a JSON array so no part can
    shift the boundary of another.

    Returns:
        The raw 16-byte MD5 digest.
    """
    identity = json.dumps(
        [
            row.method.value,
            row.url,
            None if row.body is None else row.body.hex(),
            row.body_is_form,
            row.headers_json,
            row.json_data,
        ]
    )
    return hashlib.md5(
        identity.encode("utf-8"), usedforsecurity=False
    ).digest()
