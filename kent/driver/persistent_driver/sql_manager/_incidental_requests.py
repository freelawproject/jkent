"""Incidental request storage operations with content deduplication."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from kent.driver.persistent_driver.models import (
    IncidentalRequest,
    IncidentalRequestStorage,
)
from kent.driver.persistent_driver.sql_manager._types import (
    IncidentalRequestRecord,
)

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.sql import Select

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


def incidental_record_select() -> Select[Any]:
    """Build the shared select() for IncidentalRequestRecord with storage join.

    Column order must match :func:`row_to_incidental_record`.
    """
    return select(  # type: ignore[call-overload,misc]
        IncidentalRequest.id,
        IncidentalRequest.parent_request_id,
        IncidentalRequest.url,
        IncidentalRequest.headers_json,
        IncidentalRequest.started_at_ns,
        IncidentalRequest.completed_at_ns,
        IncidentalRequest.from_cache,
        IncidentalRequest.created_at,
        IncidentalRequest.storage_id,
        IncidentalRequestStorage.resource_type,
        IncidentalRequestStorage.method,
        IncidentalRequestStorage.status_code,
        IncidentalRequestStorage.content_size_original,
        IncidentalRequestStorage.content_size_compressed,
        IncidentalRequestStorage.failure_reason,
    ).outerjoin(
        IncidentalRequestStorage,
        IncidentalRequest.storage_id == IncidentalRequestStorage.id,
    )


def row_to_incidental_record(row: Any) -> IncidentalRequestRecord:
    """Map a row from :func:`incidental_record_select` to a record."""
    return IncidentalRequestRecord(
        id=row[0],
        parent_request_id=row[1],
        url=row[2],
        headers_json=row[3],
        started_at_ns=row[4],
        completed_at_ns=row[5],
        from_cache=bool(row[6]) if row[6] is not None else None,
        created_at=row[7],
        storage_id=row[8],
        resource_type=row[9],
        method=row[10],
        status_code=row[11],
        content_size_original=row[12],
        content_size_compressed=row[13],
        failure_reason=row[14],
    )


class IncidentalRequestStorageMixin:
    """Insert and retrieve incidental browser requests with content dedup."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

    async def insert_incidental_request(
        self,
        parent_request_id: int,
        resource_type: str,
        method: str,
        url: str,
        headers_json: str | None = None,
        body: bytes | None = None,
        status_code: int | None = None,
        response_headers_json: str | None = None,
        content_compressed: bytes | None = None,
        content_size_original: int | None = None,
        content_size_compressed: int | None = None,
        compression_dict_id: int | None = None,
        started_at_ns: int | None = None,
        completed_at_ns: int | None = None,
        from_cache: bool = False,
        failure_reason: str | None = None,
    ) -> int:
        """Store an incidental request with content deduplication.

        Computes an MD5 of the compressed content. If a storage row with
        the same MD5 already exists, the new incidental_requests row
        reuses its storage_id instead of creating a duplicate.

        Returns:
            The database ID of the incidental_requests row.
        """
        content_md5 = None
        if content_compressed is not None:
            content_md5 = hashlib.md5(content_compressed).hexdigest()

        async with self._lock, self._session_factory() as session:
            storage_id: int | None = None

            # Dedup: look for an existing storage row with the same MD5
            if content_md5 is not None:
                result = await session.execute(
                    select(IncidentalRequestStorage.id)  # type: ignore[call-overload]
                    .where(IncidentalRequestStorage.content_md5 == content_md5)
                    .limit(1)
                )
                storage_id = result.scalar_one_or_none()

            # Create storage row if not deduped
            if storage_id is None:
                storage = IncidentalRequestStorage(
                    resource_type=resource_type,
                    url=url,
                    method=method,
                    body=body,
                    status_code=status_code,
                    response_headers_json=response_headers_json,
                    content_compressed=content_compressed,
                    content_size_original=content_size_original,
                    content_size_compressed=content_size_compressed,
                    compression_dict_id=compression_dict_id,
                    failure_reason=failure_reason,
                    content_md5=content_md5,
                )
                session.add(storage)
                await session.flush()
                storage_id = storage.id

            # Create the metadata row
            ir = IncidentalRequest(
                parent_request_id=parent_request_id,
                url=url,
                headers_json=headers_json,
                started_at_ns=started_at_ns,
                completed_at_ns=completed_at_ns,
                from_cache=from_cache,
                storage_id=storage_id,
            )
            session.add(ir)
            await session.commit()
            return ir.id  # type: ignore[return-value]

    async def get_incidental_requests(
        self, parent_request_id: int
    ) -> list[IncidentalRequestRecord]:
        """Get all incidental requests for a parent request.

        Joins with storage table to include content/response fields.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                incidental_record_select()
                .where(
                    IncidentalRequest.parent_request_id == parent_request_id  # type: ignore[arg-type]
                )
                .order_by(IncidentalRequest.started_at_ns.asc())  # type: ignore[union-attr]
            )
            return [row_to_incidental_record(row) for row in result.all()]

    async def get_incidental_request_by_id(
        self, incidental_id: int
    ) -> IncidentalRequestRecord | None:
        """Get a single incidental request by ID with storage data."""
        async with self._session_factory() as session:
            result = await session.execute(
                incidental_record_select().where(
                    IncidentalRequest.id == incidental_id  # type: ignore[arg-type]
                )
            )
            row = result.first()
            return row_to_incidental_record(row) if row is not None else None

    async def get_incidental_request_storage(
        self, storage_id: int
    ) -> dict[str, Any] | None:
        """Get raw storage row for decompression."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(IncidentalRequestStorage).where(
                    IncidentalRequestStorage.id == storage_id  # type: ignore[arg-type]
                )
            )
            s = result.scalar_one_or_none()
            if s is None:
                return None
            return {
                "id": s.id,
                "resource_type": s.resource_type,
                "url": s.url,
                "method": s.method,
                "body": s.body,
                "status_code": s.status_code,
                "response_headers_json": s.response_headers_json,
                "content_compressed": s.content_compressed,
                "content_size_original": s.content_size_original,
                "content_size_compressed": s.content_size_compressed,
                "compression_dict_id": s.compression_dict_id,
                "failure_reason": s.failure_reason,
                "content_md5": s.content_md5,
            }
