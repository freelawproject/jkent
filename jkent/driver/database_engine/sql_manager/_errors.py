"""ErrorsMixin - storing and querying the ``errors`` table."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import sqlalchemy as sa
from sqlalchemy import select

from jkent.driver.database_engine.errors import (
    ErrorRecord,
    build_error,
    error_message,
)
from jkent.driver.database_engine.models import Error, Request
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._requests import mark_failed

if TYPE_CHECKING:
    from sqlalchemy.sql import Select

    from jkent.driver.database_engine.enums import ErrorType

_TP = TypeVar("_TP", bound=tuple[Any, ...])


def _filter_errors(
    stmt: Select[_TP],
    error_type: ErrorType | None,
    step: str | None,
) -> Select[_TP]:
    """Apply the shared error filters to a select over ``errors``.

    Used by both the listing and the count so a paginated total cannot
    drift from the rows the listing returns.

    Args:
        stmt: A select whose FROM is (or includes) ``errors``.
        error_type: Filter by error type.
        step: Filter by step method name. Joins ``requests``,
            which drops errors raised outside a request context.

    Returns:
        The select with the requested filters applied.
    """
    if step is not None:
        stmt = stmt.join(Request, Error.request_id == Request.id).where(
            Request.step == step
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

        async with self._write_session() as session:
            session.add(error)
            await session.flush()
            error_id = error.id
            await session.commit()

        return error_id if error_id else 0

    async def fail_request(
        self,
        request_id: int,
        exc: Exception,
        request_url: str | None = None,
    ) -> int:
        """Mark ``request_id`` FAILED and store ``exc`` against it, atomically.

        One transaction for both, so a FAILED row always has its ``errors``
        row and neither lands without the other.

        Returns:
            The database ID of the stored error.
        """
        error = build_error(exc, request_id, request_url)
        async with self._write_session() as session:
            await session.execute(mark_failed(request_id, error_message(exc)))
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
            return ErrorRecord.model_validate(error)

    async def list_errors(
        self,
        error_type: ErrorType | None = None,
        step: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[ErrorRecord]:
        """List errors with optional filters, newest first.

        Args:
            error_type: Filter by error type.
            step: Filter by step method name (requires join
                with requests).
            offset: Number of records to skip.
            limit: Maximum records to return.

        Returns:
            List of ErrorRecord objects.
        """
        stmt = _filter_errors(select(Error), error_type, step)
        stmt = (
            stmt.order_by(Error.created_at.desc()).limit(limit).offset(offset)
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return [
                ErrorRecord.model_validate(e) for e in result.scalars().all()
            ]

    async def count_errors(
        self,
        error_type: ErrorType | None = None,
        step: str | None = None,
    ) -> int:
        """Count errors with optional filters.

        Accepts the same filters as :meth:`list_errors` so a paginated total
        matches the rows that listing would return.

        Args:
            error_type: Filter by error type.
            step: Filter by step method name (requires join
                with requests).

        Returns:
            Count of matching errors.
        """
        stmt = _filter_errors(
            select(sa.func.count()).select_from(Error),
            error_type,
            step,
        )

        async with self.session_factory() as session:
            result = await session.execute(stmt)
            return result.scalar_one()
