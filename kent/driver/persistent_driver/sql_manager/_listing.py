"""Listing and read-only query operations for SQLManager."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import func, select

from kent.driver.persistent_driver.models import Request, Result
from kent.driver.persistent_driver.sql_manager._types import (
    Page,
    RequestRecord,
    ResponseRecord,
    ResultRecord,
)

if TYPE_CHECKING:
    import asyncio

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class ListingMixin:
    """Cross-model read-only listing and retrieval operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

    # --- Status ---

    async def get_run_status(
        self,
    ) -> Literal["unstarted", "in_progress", "done"]:
        """Check the current state of the scraper run.

        Returns:
            "unstarted": No requests in DB
            "in_progress": Pending or in_progress requests exist
            "done": No pending/in_progress but completed requests exist
        """
        active_count = await self.count_active_requests()  # type: ignore[attr-defined]
        if active_count > 0:
            return "in_progress"

        total_count = await self.count_all_requests()  # type: ignore[attr-defined]
        if total_count == 0:
            return "unstarted"

        return "done"

    # --- Listing Operations ---

    async def list_requests(
        self,
        status: str | None = None,
        continuation: str | None = None,
        offset: int = 0,
        limit: int = 50,
        sort: str = "queue",
    ) -> Page[RequestRecord]:
        """List requests with optional filters and pagination.

        Args:
            status: Filter by status.
            continuation: Filter by continuation method name.
            offset: Number of records to skip.
            limit: Maximum number of records to return.
            sort: Sort order - "queue" (default: priority, queue_counter),
                  "id_asc" (by id ascending), or "id_desc" (by id descending).

        Returns:
            Page of RequestRecord instances.
        """
        async with self._session_factory() as session:
            # Build WHERE conditions
            conditions = []
            if status:
                conditions.append(Request.status == status)
            if continuation:
                conditions.append(Request.continuation == continuation)

            # Count query
            count_stmt = select(func.count()).select_from(Request)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)  # type: ignore[arg-type]
            result = await session.execute(count_stmt)
            total = result.scalar() or 0

            # Data query
            data_stmt = select(*RequestRecord.select_columns(Request))
            for cond in conditions:
                data_stmt = data_stmt.where(cond)  # type: ignore[arg-type]

            if sort == "id_asc":
                data_stmt = data_stmt.order_by(Request.id.asc())  # type: ignore[union-attr]
            elif sort == "id_desc":
                data_stmt = data_stmt.order_by(Request.id.desc())  # type: ignore[union-attr]
            else:
                data_stmt = data_stmt.order_by(
                    Request.priority.asc(),  # type: ignore[attr-defined]
                    Request.queue_counter.asc(),  # type: ignore[attr-defined]
                )

            data_stmt = data_stmt.limit(limit).offset(offset)
            result = await session.execute(data_stmt)
            rows = result.all()

            items = [RequestRecord.from_row(row) for row in rows]

            return Page(
                items=items,
                total=total,
                offset=offset,
                limit=limit,
            )

    async def list_responses(
        self,
        continuation: str | None = None,
        request_id: int | None = None,
        speculation_outcome: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Page[ResponseRecord]:
        """List responses with optional filters and pagination.

        Queries requests that have a response (response_status_code IS NOT NULL).

        Args:
            continuation: Filter by continuation method name.
            request_id: Filter by request ID.
            speculation_outcome: Filter by speculation outcome.
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            Page of ResponseRecord instances.
        """
        async with self._session_factory() as session:
            conditions = [Request.response_status_code.isnot(None)]  # type: ignore[union-attr]
            if continuation:
                conditions.append(Request.continuation == continuation)
            if request_id:
                conditions.append(Request.id == request_id)
            if speculation_outcome:
                conditions.append(
                    Request.speculation_outcome == speculation_outcome
                )

            count_stmt = select(func.count()).select_from(Request)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            result = await session.execute(count_stmt)
            total = result.scalar() or 0

            data_stmt = select(  # type: ignore[call-overload,misc]
                Request.id,
                Request.response_status_code,
                Request.response_url,
                Request.content_size_original,
                Request.content_size_compressed,
                Request.continuation,
                Request.response_created_at,
                Request.compression_dict_id,
                Request.speculation_outcome,
            )
            for cond in conditions:
                data_stmt = data_stmt.where(cond)
            data_stmt = (
                data_stmt.order_by(Request.id.desc())  # type: ignore[union-attr]
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(data_stmt)
            rows = result.all()

            items = [
                ResponseRecord(
                    id=row[0],
                    status_code=row[1],
                    url=row[2],
                    content_size_original=row[3],
                    content_size_compressed=row[4],
                    continuation=row[5],
                    created_at=row[6],
                    compression_dict_id=row[7],
                    speculation_outcome=row[8],
                )
                for row in rows
            ]

            return Page(
                items=items,
                total=total,
                offset=offset,
                limit=limit,
            )

    async def list_results(
        self,
        result_type: str | None = None,
        is_valid: bool | None = None,
        request_id: int | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> Page[ResultRecord]:
        """List results with optional filters and pagination.

        Args:
            result_type: Filter by result type.
            is_valid: Filter by validation status.
            request_id: Filter by request ID.
            offset: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            Page of ResultRecord instances.
        """
        async with self._session_factory() as session:
            conditions = []
            if result_type:
                conditions.append(Result.result_type == result_type)
            if is_valid is not None:
                conditions.append(Result.is_valid == is_valid)
            if request_id:
                conditions.append(Result.request_id == request_id)

            count_stmt = select(func.count()).select_from(Result)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)  # type: ignore[arg-type]
            result = await session.execute(count_stmt)
            total = result.scalar() or 0

            data_stmt = select(  # type: ignore[call-overload]
                Result.id,
                Result.request_id,
                Result.result_type,
                Result.data_json,
                Result.is_valid,
                Result.validation_errors_json,
                Result.created_at,
            )
            for cond in conditions:
                data_stmt = data_stmt.where(cond)
            data_stmt = (
                data_stmt.order_by(Result.id.desc())  # type: ignore[union-attr]
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(data_stmt)
            rows = result.all()

            items = [
                ResultRecord(
                    id=row[0],
                    request_id=row[1],
                    result_type=row[2],
                    data_json=row[3],
                    is_valid=bool(row[4]),
                    validation_errors_json=row[5],
                    created_at=row[6],
                )
                for row in rows
            ]

            return Page(
                items=items,
                total=total,
                offset=offset,
                limit=limit,
            )

    async def get_request(self, request_id: int) -> RequestRecord | None:
        """Get a single request by ID.

        Args:
            request_id: The database ID of the request.

        Returns:
            RequestRecord or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(*RequestRecord.select_columns(Request)).where(
                    Request.id == request_id  # type: ignore[arg-type]
                )
            )
            row = result.first()
            if row is None:
                return None
            return RequestRecord.from_row(row)

    async def get_response(self, request_id: int) -> ResponseRecord | None:
        """Get response data for a request by its ID.

        Args:
            request_id: The database ID of the request.

        Returns:
            ResponseRecord or None if not found or no response stored.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload,misc]
                    Request.id,
                    Request.response_status_code,
                    Request.response_url,
                    Request.content_size_original,
                    Request.content_size_compressed,
                    Request.continuation,
                    Request.response_created_at,
                    Request.compression_dict_id,
                    Request.speculation_outcome,
                ).where(
                    Request.id == request_id,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            row = result.first()
            if row is None:
                return None
            return ResponseRecord(
                id=row[0],
                status_code=row[1],
                url=row[2],
                content_size_original=row[3],
                content_size_compressed=row[4],
                continuation=row[5],
                created_at=row[6],
                compression_dict_id=row[7],
                speculation_outcome=row[8],
            )

    async def get_result(self, result_id: int) -> ResultRecord | None:
        """Get a single result by ID.

        Args:
            result_id: The database ID of the result.

        Returns:
            ResultRecord or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Result.id,
                    Result.request_id,
                    Result.result_type,
                    Result.data_json,
                    Result.is_valid,
                    Result.validation_errors_json,
                    Result.created_at,
                ).where(Result.id == result_id)
            )
            row = result.first()
            if row is None:
                return None
            return ResultRecord(
                id=row[0],
                request_id=row[1],
                result_type=row[2],
                data_json=row[3],
                is_valid=bool(row[4]),
                validation_errors_json=row[5],
                created_at=row[6],
            )

    # --- Resume Request Operations ---

    async def get_permanent_json(self, request_id: int) -> str | None:
        """Get permanent_json field for a request.

        Args:
            request_id: The database ID of the request.

        Returns:
            The permanent_json string or None.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Request.permanent_json).where(Request.id == request_id)  # type: ignore[call-overload]
            )
            return result.scalar()

    async def get_predicate_result(self, request_id: int) -> bool:
        """Get predicate_result from a resume request's permanent_json.

        Args:
            request_id: The database ID of the resume request.

        Returns:
            The predicate_result boolean value.
        """
        permanent_json = await self.get_permanent_json(request_id)
        if permanent_json:
            data = json.loads(permanent_json)
            return data.get("predicate_result", False)
        return False

    # --- Statistics ---

    async def get_stats(self) -> Any:
        """Get comprehensive statistics about the driver state.

        Returns:
            DevDriverStats instance.
        """
        from kent.driver.persistent_driver.stats import get_stats

        return await get_stats(self._session_factory)

    # --- Response Content Access ---

    async def get_response_content(self, request_id: int) -> bytes | None:
        """Get decompressed response content by request ID.

        Args:
            request_id: The database ID of the request.

        Returns:
            Decompressed content bytes, or None if request not found
            or has no response.
        """
        from kent.driver.persistent_driver.compression import (
            decompress_response,
        )

        result = await self.get_response_compressed(request_id)  # type: ignore[attr-defined]
        if result is None:
            return None

        compressed, dict_id = result
        if not compressed:
            return b""

        return await decompress_response(
            self._session_factory, compressed, dict_id
        )

    async def get_response_content_with_headers(
        self, request_id: int
    ) -> tuple[bytes, str | None] | None:
        """Get decompressed response content and headers.

        Args:
            request_id: The database ID of the request.

        Returns:
            Tuple of (decompressed_content, headers_json) or None if not found.
        """
        from kent.driver.persistent_driver.compression import (
            decompress_response,
        )

        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Request.content_compressed,
                    Request.compression_dict_id,
                    Request.response_headers_json,
                ).where(
                    Request.id == request_id,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            row = result.first()

        if row is None:
            return None

        compressed_content, dict_id, headers_json = row

        if compressed_content is None:
            return (b"", headers_json)

        content = await decompress_response(
            self._session_factory, compressed_content, dict_id
        )
        return (content, headers_json)
