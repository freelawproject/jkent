"""Result storage operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kent.driver.persistent_driver.models import Result

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class ResultStorageMixin:
    """Result table database operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

    async def store_result(
        self,
        request_id: int,
        result_type: str,
        data_json: str,
        is_valid: bool = True,
        validation_errors_json: str | None = None,
    ) -> int:
        """Store a scraped result.

        Args:
            request_id: The database ID of the request that produced this.
            result_type: Pydantic model class name.
            data_json: JSON-encoded result data.
            is_valid: Whether the data passed validation.
            validation_errors_json: JSON-encoded validation errors if invalid.

        Returns:
            The database ID of the stored result.
        """
        async with self._lock, self._session_factory() as session:
            res_id = await self.store_result_in_session(
                session,
                request_id=request_id,
                result_type=result_type,
                data_json=data_json,
                is_valid=is_valid,
                validation_errors_json=validation_errors_json,
            )
            await session.commit()
            return res_id

    async def store_result_in_session(
        self,
        session: AsyncSession,
        *,
        request_id: int,
        result_type: str,
        data_json: str,
        is_valid: bool = True,
        validation_errors_json: str | None = None,
    ) -> int:
        """Stage a result row inside an existing session (no commit)."""
        res = Result(
            request_id=request_id,
            result_type=result_type,
            data_json=data_json,
            is_valid=is_valid,
            validation_errors_json=validation_errors_json,
        )
        session.add(res)
        await session.flush()
        return res.id  # type: ignore[return-value]
