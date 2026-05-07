"""Estimate storage operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kent.driver.persistent_driver.models import Estimate

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class EstimateStorageMixin:
    """Estimate table database operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

    async def store_estimate(
        self,
        request_id: int,
        expected_types_json: str,
        min_count: int,
        max_count: int | None = None,
    ) -> int:
        """Store an estimate from an EstimateData yield.

        Args:
            request_id: The database ID of the request that produced this.
            expected_types_json: JSON list of expected type name strings.
            min_count: Minimum expected result count.
            max_count: Maximum expected result count, or None for unbounded.

        Returns:
            The database ID of the stored estimate.
        """
        async with self._lock, self._session_factory() as session:
            est_id = await self.store_estimate_in_session(
                session,
                request_id=request_id,
                expected_types_json=expected_types_json,
                min_count=min_count,
                max_count=max_count,
            )
            await session.commit()
            return est_id

    async def store_estimate_in_session(
        self,
        session: AsyncSession,
        *,
        request_id: int,
        expected_types_json: str,
        min_count: int,
        max_count: int | None = None,
    ) -> int:
        """Stage an estimate row inside an existing session (no commit)."""
        estimate = Estimate(
            request_id=request_id,
            expected_types_json=expected_types_json,
            min_count=min_count,
            max_count=max_count,
        )
        session.add(estimate)
        await session.flush()
        return estimate.id  # type: ignore[return-value]
