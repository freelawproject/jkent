"""Result storage operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jkent.driver.database_engine.models import Result
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from jkent.driver.database_engine.sql_manager._types import ResultInsert


class ResultStorageMixin(SQLManagerBase):
    """Result table database operations."""

    async def store_result(self, request_id: int, result: ResultInsert) -> int:
        """Store a scraped result (own transaction).

        Args:
            request_id: The database ID of the request that produced this.
            result: The serialized record and its validation status.

        Returns:
            The database ID of the stored result.
        """
        async with self._write_session() as session:
            res_id = await self.store_result_in_session(
                session, request_id, result
            )
            await session.commit()
            return res_id

    async def store_result_in_session(
        self, session: AsyncSession, request_id: int, result: ResultInsert
    ) -> int:
        """Stage a result row inside an existing session (no commit)."""
        res = Result(request_id=request_id, **result.model_dump())
        session.add(res)
        await session.flush()
        return res.id
