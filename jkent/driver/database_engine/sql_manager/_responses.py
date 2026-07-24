"""Response storage operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select, update
from sqlmodel import col

from jkent.driver.database_engine.models import (
    ArchivedFile,
    CompressionDict,
    Request,
)

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker


class ResponseStorageMixin:
    """Response, ArchivedFile, and CompressionDict operations."""

    _lock: asyncio.Lock  # type: ignore[misc]
    _session_factory: async_sessionmaker  # type: ignore[misc]

    # --- Response Storage ---

    async def store_response(
        self,
        request_id: int,
        status_code: int,
        headers_json: str | None,
        url: str,
        compressed_content: bytes | None,
        content_size_original: int,
        content_size_compressed: int,
        dict_id: int | None,
        continuation: str,
        speculation_outcome: str | None = None,
    ) -> int:
        """Store an HTTP response by updating the request row.

        Args:
            request_id: The database ID of the request to update.
            status_code: HTTP status code.
            headers_json: JSON-encoded response headers.
            url: Final URL after redirects.
            compressed_content: Compressed content bytes.
            content_size_original: Original content size.
            content_size_compressed: Compressed content size.
            dict_id: Compression dictionary ID if used.
            continuation: Continuation method name (unused, kept for API compat).
            speculation_outcome: For speculative requests: 'success', 'stopped', 'skipped'.

        Returns:
            The request_id (same as input).
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(Request)
                .where(col(Request.id) == request_id)
                .values(
                    response_status_code=status_code,
                    response_headers_json=headers_json,
                    response_url=url,
                    content_compressed=compressed_content,
                    content_size_original=content_size_original,
                    content_size_compressed=content_size_compressed,
                    compression_dict_id=dict_id,
                    speculation_outcome=speculation_outcome,
                    response_created_at=func.current_timestamp(),
                )
            )
            await session.commit()
            return request_id

    async def store_archived_file(
        self,
        request_id: int,
        file_path: str,
        original_url: str,
        expected_type: str | None,
        file_size: int,
        content_hash: str | None,
    ) -> int:
        """Store archived file metadata.

        Args:
            request_id: The database ID of the associated request.
            file_path: Local file system path.
            original_url: URL the file was downloaded from.
            expected_type: Expected file type.
            file_size: File size in bytes.
            content_hash: SHA256 hash of content.

        Returns:
            The database ID of the archived file record.
        """
        async with self._lock, self._session_factory() as session:
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
            return af.id  # type: ignore[return-value]

    async def get_response_compressed(
        self, request_id: int
    ) -> tuple[bytes | None, int | None] | None:
        """Get compressed response content and dict ID for a request.

        Args:
            request_id: The database ID of the request.

        Returns:
            Tuple of (compressed_content, dict_id), or None if the request
            does not exist or has no stored response. Mirrors the
            ``response_status_code IS NOT NULL`` guard used by the other
            response getters so an unanswered request is not mistaken for
            one with an empty body.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    col(Request.content_compressed),
                    col(Request.compression_dict_id),
                ).where(
                    col(Request.id) == request_id,
                    col(Request.response_status_code).isnot(None),
                )
            )
            row = result.first()
            return tuple(row) if row else None

    async def get_parent_response_for_tab(
        self, parent_request_id: int
    ) -> tuple[str, bytes, int | None, str | None, int] | None:
        """Get parent's stored response for tab route interception.

        Args:
            parent_request_id: The database ID of the parent request.

        Returns:
            Tuple of (response_url, content_compressed, compression_dict_id,
            response_headers_json, response_status_code) or None if no
            stored response exists.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    col(Request.response_url),
                    col(Request.content_compressed),
                    col(Request.compression_dict_id),
                    col(Request.response_headers_json),
                    col(Request.response_status_code),
                ).where(
                    col(Request.id) == parent_request_id,
                    col(Request.response_status_code).isnot(None),
                )
            )
            row = result.first()
            return tuple(row) if row else None

    async def get_stored_response(
        self, request_id: int
    ) -> tuple[int, str | None, str, bytes | None, int | None] | None:
        """Get a request's stored response for reconstruction.

        Returns ``(status_code, headers_json, url, content_compressed,
        compression_dict_id)`` or None when the request has no stored
        response. Used by the worker's pre-resolved path to rebuild a
        ``Response`` without touching the transport.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    col(Request.response_status_code),
                    col(Request.response_headers_json),
                    col(Request.response_url),
                    col(Request.content_compressed),
                    col(Request.compression_dict_id),
                ).where(
                    col(Request.id) == request_id,
                    col(Request.response_status_code).isnot(None),
                )
            )
            row = result.first()
            return tuple(row) if row else None

    async def get_compression_dict(self, dict_id: int) -> bytes | None:
        """Get compression dictionary data by ID.

        Args:
            dict_id: The database ID of the compression dictionary.

        Returns:
            Dictionary bytes if found, None otherwise.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(col(CompressionDict.dictionary_data)).where(
                    col(CompressionDict.id) == dict_id
                )
            )
            return result.scalar()

    async def has_compression_dict(self, continuation: str) -> bool:
        """Whether a trained compression dictionary exists for ``continuation``."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(col(CompressionDict.id))
                .where(col(CompressionDict.continuation) == continuation)
                .limit(1)
            )
            return result.first() is not None
