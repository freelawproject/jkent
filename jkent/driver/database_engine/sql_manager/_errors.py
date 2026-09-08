"""ErrorsMixin - storing and querying the ``errors`` table."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy import select

from jkent.driver.database_engine.errors import ErrorRecord, build_error
from jkent.driver.database_engine.models import Error, Request
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase

if TYPE_CHECKING:
    from sqlalchemy.sql import Select

    from jkent.driver.database_engine.enums import ErrorType


def _filter_errors(
    stmt: Select,
    error_type: ErrorType | str | None,
    continuation: str | None,
) -> Select:
    """Apply the shared error filters to a select over ``errors``.

    Used by both the listing and the count so a paginated total cannot
    drift from the rows the listing returns.

    Args:
        stmt: A select whose FROM is (or includes) ``errors``.
        error_type: Filter by error type, as an :class:`ErrorType` member or
            its label — ``CodedEnumType`` binds either to the stored code.
        continuation: Filter by continuation method name. Joins ``requests``,
            which drops errors raised outside a request context.

    Returns:
        The select with the requested filters applied.
    """
    if continuation is not None:
        stmt = stmt.join(Request, Error.request_id == Request.id).where(
            Request.continuation == continuation
        )
    if error_type is not None:
        stmt = stmt.where(Error.error_type == error_type)
    return stmt


class ErrorsMixin(SQLManagerBase):
    """Error storage and retrieval for SQLManager."""

    async def store_error(
        self,
        exc: Exception,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> int:
        """Store an error in the database.

        Extracts type-specific fields from the exception and stores them
        in the errors table.

        Args:
            exc: The exception to store.
            request_id: ID of the request that caused this error (if known).
            request_url: URL that triggered the error (fallback if not in
                exception).

        Returns:
            The database ID of the stored error.
        """
        error = build_error(exc, request_id, request_url)

        async with self.lock, self.session_factory() as session:
            session.add(error)
            await session.flush()
            error_id = error.id
            await session.commit()

        return error_id if error_id else 0

    async def get_error(self, error_id: int) -> ErrorRecord | None:
        """Get a single error by ID.

        Args:
            error_id: The error ID to retrieve.

        Returns:
            ErrorRecord if found, None otherwise.
        """
        async with self.session_factory() as session:
            error = await session.get(Error, error_id)
            if error is None:
                return None
            return ErrorRecord.from_model(error)

    async def list_errors(
        self,
        error_type: ErrorType | str | None = None,
        continuation: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        """List errors with optional filters, newest first.

        Args:
            error_type: Filter by error type, as an :class:`ErrorType` member
                or its label.
            continuation: Filter by continuation method name (requires join
                with requests).
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of ErrorRecord objects.
        """
        stmt = _filter_errors(select(Error), error_type, continuation)
        stmt = (
            stmt.order_by(Error.created_at.desc()).limit(limit).offset(offset)
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return [ErrorRecord.from_model(e) for e in result.scalars().all()]

    async def count_errors(
        self,
        error_type: ErrorType | str | None = None,
        continuation: str | None = None,
    ) -> int:
        """Count errors with optional filters.

        Accepts the same filters as :meth:`list_errors` so a paginated total
        matches the rows that listing would return.

        Args:
            error_type: Filter by error type (member or label, as
                :meth:`list_errors`).
            continuation: Filter by continuation method name (requires join
                with requests).

        Returns:
            Count of matching errors.
        """
        stmt = _filter_errors(
            select(sa.func.count()).select_from(Error),
            error_type,
            continuation,
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one()
