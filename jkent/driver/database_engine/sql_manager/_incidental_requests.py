"""Incidental request storage operations with content deduplication."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from jkent.driver.database_engine.models import (
    IncidentalRequest,
    IncidentalRequestStorage,
)
from jkent.driver.database_engine.sql_manager._types import (
    IncidentalRequestRecord,
)

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    from sqlalchemy.sql import Select


def incidental_record_select() -> Select[Any]:
    """Build the shared select() for IncidentalRequestRecord with storage join.

    Column order must match :func:`row_to_incidental_record`.
    """
    return select(
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

    _lock: asyncio.Lock  # type: ignore[misc]
    _session_factory: async_sessionmaker  # type: ignore[misc]

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

        Computes an MD5 of the compressed content and reuses an existing
        storage row only when the content MD5 **and** the per-request metadata
        carried on the storage row (resource_type, method, url, status_code,
        failure_reason, request ``body``) all match. Keying on more than the
        content hash means two requests with identical response bodies but
        different status/method/request-body/etc. get distinct storage rows
        instead of the second silently inheriting the first's metadata — which
        matters for incidental matching, where the stored request ``body`` is
        used to disambiguate several requests to the same URL.

        Returns:
            The database ID of the incidental_requests row.
        """
        async with self._lock, self._session_factory() as session:
            ir_id = await self._insert_incidental_in_session(
                session,
                parent_request_id=parent_request_id,
                resource_type=resource_type,
                method=method,
                url=url,
                headers_json=headers_json,
                body=body,
                status_code=status_code,
                response_headers_json=response_headers_json,
                content_compressed=content_compressed,
                content_size_original=content_size_original,
                content_size_compressed=content_size_compressed,
                compression_dict_id=compression_dict_id,
                started_at_ns=started_at_ns,
                completed_at_ns=completed_at_ns,
                from_cache=from_cache,
                failure_reason=failure_reason,
            )
            await session.commit()
            return ir_id

    async def insert_incidental_requests(
        self,
        parent_request_id: int,
        records: list[dict[str, Any]],
    ) -> list[int]:
        """Store a navigation's captured incidentals in one transaction.

        Each record is the kwargs of :meth:`insert_incidental_request` minus
        ``parent_request_id``. One lock acquisition / session / commit for the
        whole batch instead of one per subresource — a page with dozens of
        captured sub-requests was paying dozens of commits per navigation.
        Content dedup is preserved, including within the batch: the dedup
        SELECT sees the earlier records' flushed-but-uncommitted storage rows.

        Returns:
            The incidental_requests row IDs, in record order.
        """
        if not records:
            return []
        async with self._lock, self._session_factory() as session:
            ids = [
                await self._insert_incidental_in_session(
                    session, parent_request_id=parent_request_id, **record
                )
                for record in records
            ]
            await session.commit()
            return ids

    async def _insert_incidental_in_session(
        self,
        session: AsyncSession,
        *,
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
        """Dedup-and-insert one incidental inside an existing session (no commit)."""
        content_md5 = None
        if content_compressed is not None:
            content_md5 = hashlib.md5(content_compressed).digest()

        storage_id: int | None = None

        # Dedup: reuse a storage row only when the content hash AND the
        # metadata stored on that row all match (== None becomes IS NULL).
        if content_md5 is not None:
            result = await session.execute(
                select(IncidentalRequestStorage.id)
                .where(
                    IncidentalRequestStorage.content_md5 == content_md5,
                    IncidentalRequestStorage.resource_type == resource_type,
                    IncidentalRequestStorage.method == method,
                    IncidentalRequestStorage.url == url,
                    IncidentalRequestStorage.status_code == status_code,
                    IncidentalRequestStorage.failure_reason == failure_reason,
                    IncidentalRequestStorage.body == body,
                )
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
        await session.flush()
        return ir.id

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
                    IncidentalRequest.parent_request_id == parent_request_id
                )
                .order_by(IncidentalRequest.started_at_ns.asc())
            )
            return [row_to_incidental_record(row) for row in result.all()]

    async def get_incidental_request_by_id(
        self, incidental_id: int
    ) -> IncidentalRequestRecord | None:
        """Get a single incidental request by ID with storage data."""
        async with self._session_factory() as session:
            result = await session.execute(
                incidental_record_select().where(
                    IncidentalRequest.id == incidental_id
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
                    IncidentalRequestStorage.id == storage_id
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
                # Hex here, not in the column: this dict is a display/transport
                # surface that gets JSON-serialized downstream, where raw bytes
                # would not survive. The stored format stays the raw digest.
                "content_md5": (
                    s.content_md5.hex() if s.content_md5 is not None else None
                ),
            }
