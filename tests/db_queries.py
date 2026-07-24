"""Direct-model query helpers for asserting against a run database.

Small wrappers over ``select()`` on the SQLModel models, used where tests
need to inspect rows the driver wrote (the production code has no
inspection surface — it was deleted with ``ListingMixin``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlmodel import col

from jkent.driver.database_engine.models import Request, Result

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager import SQLManager


async def fetch_requests(
    sql_manager: SQLManager, status: str | None = None
) -> list[Request]:
    """All request rows, optionally filtered by status, in id order."""
    stmt = select(Request).order_by(col(Request.id))
    if status is not None:
        stmt = stmt.where(col(Request.status) == status)
    async with sql_manager._session_factory() as session:
        return list((await session.execute(stmt)).scalars().all())


async def fetch_results(
    sql_manager: SQLManager,
    request_id: int | None = None,
    is_valid: bool | None = None,
) -> list[Result]:
    """All result rows, optionally filtered, in id order."""
    stmt = select(Result).order_by(col(Result.id))
    if request_id is not None:
        stmt = stmt.where(col(Result.request_id) == request_id)
    if is_valid is not None:
        stmt = stmt.where(col(Result.is_valid) == is_valid)
    async with sql_manager._session_factory() as session:
        return list((await session.execute(stmt)).scalars().all())


async def get_request_row(
    sql_manager: SQLManager, request_id: int
) -> Request | None:
    """One request row by id, or None."""
    async with sql_manager._session_factory() as session:
        return (
            await session.execute(
                select(Request).where(col(Request.id) == request_id)
            )
        ).scalar_one_or_none()
