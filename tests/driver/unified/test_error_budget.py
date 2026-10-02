"""``ErrorBudget.request_failed`` writes both halves of a failure at once.

The request row's FAILED status and its ``errors`` row land in one
transaction: a crash between two writes would leave a FAILED row whose
error never landed — a failure nobody can diagnose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import ScraperAssumptionException
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import _errors
from jkent.driver.unified_driver.persistence import ErrorBudget

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager


async def _state(db: SQLManager, request_id: int) -> tuple[int, int]:
    async with db.session_factory() as session:
        status = (
            await session.execute(
                sa.text("SELECT status FROM requests WHERE id = :id"),
                {"id": request_id},
            )
        ).scalar_one()
        errors = (
            await session.execute(sa.text("SELECT COUNT(*) FROM errors"))
        ).scalar_one()
    return status, errors


async def test_request_failed_writes_the_row_and_its_error(
    sql_manager: SQLManager, insert_request: Callable[..., Awaitable[int]]
) -> None:
    request_id = await insert_request()
    budget = ErrorBudget(
        sql_manager, max_persistent_errors=None, stop=lambda: None
    )

    await budget.request_failed(
        request_id, ScraperAssumptionException("boom", "https://e/x")
    )

    assert await _state(sql_manager, request_id) == (
        RequestStatus.FAILED.code,
        1,
    )
    assert budget.persistent_error_count == 1


async def test_a_failure_to_file_the_error_leaves_the_row_unfailed(
    sql_manager: SQLManager,
    insert_request: Callable[..., Awaitable[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = await insert_request()

    def explode(*_: Any, **__: Any) -> Any:
        raise RuntimeError("errors row could not be built")

    monkeypatch.setattr(_errors, "build_error", explode)
    budget = ErrorBudget(
        sql_manager, max_persistent_errors=None, stop=lambda: None
    )

    with pytest.raises(RuntimeError):
        await budget.request_failed(
            request_id, ScraperAssumptionException("boom", "https://e/x")
        )

    assert await _state(sql_manager, request_id) == (
        RequestStatus.PENDING.code,
        0,
    )
