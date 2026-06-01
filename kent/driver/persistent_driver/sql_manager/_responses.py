"""Response storage operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update

from kent.driver.persistent_driver.models import (
    ArchivedFile,
    CompressionDict,
    Request,
)

if TYPE_CHECKING:
    import asyncio

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class ResponseStorageMixin:
    """Response, ArchivedFile, and CompressionDict operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

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
        from sqlalchemy import func

        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)  # type: ignore[arg-type]
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
            Tuple of (compressed_content, dict_id) or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Request.content_compressed,
                    Request.compression_dict_id,
                ).where(Request.id == request_id)
            )
            row = result.first()
            return tuple(row) if row else None  # type: ignore[return-value]

    async def get_cached_response(
        self, cache_key: str
    ) -> dict[str, Any] | None:
        """Look up a cached response by cache key.

        Returns the most recent successful (2xx) response for the given
        cache key, if one exists.

        Args:
            cache_key: The cache key (hash of method+url+body+headers).

        Returns:
            Dictionary with response data if found, None otherwise.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload,misc]
                    Request.id,
                    Request.response_status_code,
                    Request.response_headers_json,
                    Request.response_url,
                    Request.content_compressed,
                    Request.compression_dict_id,
                    Request.response_created_at,
                    Request.method,
                )
                .where(
                    Request.cache_key == cache_key,
                    Request.response_status_code >= 200,  # type: ignore[operator]
                    Request.response_status_code < 300,  # type: ignore[operator]
                )
                .order_by(Request.id.desc())  # type: ignore[union-attr]
                .limit(1)
            )
            row = result.first()
            if row is None:
                return None
            return {
                "id": row[0],
                "request_id": row[0],
                "status_code": row[1],
                "headers_json": row[2],
                "url": row[3],
                "content_compressed": row[4],
                "compression_dict_id": row[5],
                "created_at": row[6],
                "method": row[7],
            }

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
                select(  # type: ignore[call-overload,misc]
                    Request.response_url,
                    Request.content_compressed,
                    Request.compression_dict_id,
                    Request.response_headers_json,
                    Request.response_status_code,
                ).where(
                    Request.id == parent_request_id,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            row = result.first()
            return tuple(row) if row else None  # type: ignore[return-value]

    async def get_compression_dict(self, dict_id: int) -> bytes | None:
        """Get compression dictionary data by ID.

        Args:
            dict_id: The database ID of the compression dictionary.

        Returns:
            Dictionary bytes if found, None otherwise.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(CompressionDict.dictionary_data).where(  # type: ignore[call-overload]
                    CompressionDict.id == dict_id
                )
            )
            return result.scalar()
