"""Response storage operations for SQLManager."""

from __future__ import annotations

from typing import Final

from sqlalchemy import delete, select, update

from jkent.driver.database_engine.models import (
    ArchivedFile,
    Request,
    SpeculationOutcome,
)
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import (
    StoredResponse,
    model_columns,
)
from jkent.driver.database_engine.timestamps import now

_RESPONSE_COLUMNS: Final = model_columns(StoredResponse, Request).values()


class ResponseStorageMixin(SQLManagerBase):
    """Response and ArchivedFile operations."""

    async def store_response(
        self,
        request_id: int,
        response: StoredResponse,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> None:
        """Store an HTTP response by updating the request row.

        Args:
            request_id: The database ID of the request to update.
            response: The response columns to store.
            speculation_outcome: :class:`SpeculationOutcome` for a
                speculative request; None for an ordinary one.
        """
        async with self._write_session() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    **response.model_dump(),
                    speculation_outcome=speculation_outcome,
                    response_created_at=now(),
                )
            )
            await session.commit()

    async def store_archived_file(
        self,
        request_id: int,
        file_path: str,
        original_url: str,
        expected_type: str | None,
        file_size: int | None,
        content_hash: str | None,
    ) -> int:
        """Store archived file metadata, replacing the request's previous row.

        The response is stored before its step runs, so an attempt that
        fails afterwards — transiently, or by crashing before completion —
        downloads and stores again. Like the request row's stored response,
        the index describes the latest attempt only: one row per request.

        Args:
            request_id: The database ID of the associated request.
            file_path: Local file system path.
            original_url: URL the file was downloaded from.
            expected_type: Expected file type.
            file_size: File size in bytes, None if not measured.
            content_hash: SHA-256 hex digest of the file, None likewise.

        Returns:
            The database ID of the archived file record.
        """
        async with self._write_session() as session:
            await session.execute(
                delete(ArchivedFile).where(
                    ArchivedFile.request_id == request_id
                )
            )
            af = ArchivedFile(
                request_id=request_id,
                file_path=file_path,
                original_url=original_url,
                expected_type=expected_type,
                file_size=file_size,
                content_hash=content_hash,
            )
            session.add(af)
            await session.commit()
            return af.id

    async def get_stored_response(
        self, request_id: int
    ) -> StoredResponse | None:
        """Get a request's stored response.

        Returns None when the request does not exist or has no stored
        response — ``response_status_code IS NULL`` is the presence marker,
        so an unanswered request is not mistaken for one with an empty body.
        Serves both the worker's pre-resolved path (rebuilding a
        ``Response`` without touching the transport) and tab route
        interception (replaying a parent's page to a child tab).
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(*_RESPONSE_COLUMNS).where(
                    Request.id == request_id,
                    Request.response_status_code.isnot(None),
                )
            )
            row = result.first()
            return StoredResponse.model_validate(row) if row else None
