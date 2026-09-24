"""``ErrorBudget``: both halves of a failure at once, charged at most once.

The request row's FAILED status and its ``errors`` row land in one
transaction: a crash between two writes would leave a FAILED row whose
error never landed — a failure nobody can diagnose. A row enters FAILED
once, so a request is charged once; any report on a row the worker does not
hold raises and charges nothing.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import (
    ScraperAssumptionException,
    TransientException,
    TransientKind,
)
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Error
from jkent.driver.database_engine.sql_manager import (
    IllegalTransitionError,
    _errors,
)
from jkent.driver.unified_driver.persistence import ErrorBudget

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager

    InsertRequest = Callable[..., Awaitable[int]]
    Claim = Callable[[int], Awaitable[None]]


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


@pytest.fixture
async def claimed(insert_request: InsertRequest, claim: Claim) -> int:
    """A request row a worker holds, IN_PROGRESS."""
    request_id = await insert_request()
    await claim(request_id)
    return request_id


async def test_request_failed_writes_the_row_and_its_error(
    sql_manager: SQLManager, claimed: int
) -> None:
    budget = ErrorBudget(
        sql_manager, max_persistent_errors=None, stop=lambda: None
    )

    await budget.request_failed(
        claimed, ScraperAssumptionException("boom", "https://e/x")
    )

    assert await _state(sql_manager, claimed) == (
        RequestStatus.FAILED.code,
        1,
    )
    assert budget.persistent_error_count == 1


async def test_a_repeat_report_raises_and_charges_nothing(
    sql_manager: SQLManager, claimed: int
) -> None:
    """A second terminal report for one request is a caller bug."""
    budget, stops = _budget(sql_manager, 2)
    await budget.request_failed(
        claimed, ScraperAssumptionException("first", "https://e/x")
    )

    with pytest.raises(IllegalTransitionError):
        await budget.request_failed(
            claimed, ScraperAssumptionException("again", "https://e/x")
        )

    assert await _state(sql_manager, claimed) == (
        RequestStatus.FAILED.code,
        1,
    )
    assert budget.persistent_error_count == 1
    assert (budget.exhausted, stops.calls) == (False, 0)


@pytest.mark.parametrize(
    "status",
    [RequestStatus.PENDING, RequestStatus.COMPLETED, RequestStatus.STUBBED],
)
async def test_a_report_on_a_row_not_in_flight_raises(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    status: RequestStatus,
) -> None:
    """Only a claimed row can fail; the row, the errors and the count hold."""
    request_id = await insert_request()
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text("UPDATE requests SET status = :s WHERE id = :id"),
            {"s": status.code, "id": request_id},
        )
        await session.commit()
    budget, stops = _budget(sql_manager, 1)

    with pytest.raises(IllegalTransitionError):
        await budget.request_failed(request_id, RuntimeError("boom"))

    assert await _state(sql_manager, request_id) == (status.code, 0)
    assert budget.persistent_error_count == 0
    assert (budget.exhausted, stops.calls) == (False, 0)


async def test_a_failure_to_file_the_error_leaves_the_row_unfailed(
    sql_manager: SQLManager,
    claimed: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``errors`` insert fails after the row's UPDATE has run.

    The error row points at a request that does not exist, so the insert
    trips the foreign key at flush — inside the transaction that already
    marked the row FAILED. Rolling back must take the status with it.
    """
    real_build_error = _errors.build_error

    def orphaned(*args: Any, **kwargs: Any) -> Error:
        error = real_build_error(*args, **kwargs)
        error.request_id = claimed + 1_000
        return error

    monkeypatch.setattr(_errors, "build_error", orphaned)
    budget = ErrorBudget(
        sql_manager, max_persistent_errors=None, stop=lambda: None
    )

    with pytest.raises(sa.exc.IntegrityError):
        await budget.request_failed(
            claimed, ScraperAssumptionException("boom", "https://e/x")
        )

    assert await _state(sql_manager, claimed) == (
        RequestStatus.IN_PROGRESS.code,
        0,
    )
    assert budget.persistent_error_count == 0


class _Stops:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


def _budget(db: SQLManager, limit: int | None) -> tuple[ErrorBudget, _Stops]:
    stops = _Stops()
    return ErrorBudget(db, max_persistent_errors=limit, stop=stops), stops


async def _fail(
    budget: ErrorBudget,
    insert_request: InsertRequest,
    claim: Claim,
    exc: Exception,
) -> None:
    request_id = await insert_request()
    await claim(request_id)
    await budget.request_failed(request_id, exc)


async def test_transient_errors_are_filed_but_not_charged(
    sql_manager: SQLManager, insert_request: InsertRequest, claim: Claim
) -> None:
    budget, stops = _budget(sql_manager, 1)

    await _fail(
        budget,
        insert_request,
        claim,
        TransientException("503", kind=TransientKind.NETWORK),
    )

    assert budget.persistent_error_count == 0
    assert not budget.exhausted
    assert stops.calls == 0
    assert await sql_manager.count_errors() == 1


async def test_record_files_but_never_charges(
    sql_manager: SQLManager,
) -> None:
    """``record`` is replay's, and replay is outside the budget."""
    budget, stops = _budget(sql_manager, 1)

    await budget.record(ScraperAssumptionException("stubbed", "https://e/1"))

    assert budget.persistent_error_count == 0
    assert (budget.exhausted, stops.calls) == (False, 0)
    assert await sql_manager.count_errors() == 1


async def test_budget_stops_the_run_once_at_the_limit(
    sql_manager: SQLManager, insert_request: InsertRequest, claim: Claim
) -> None:
    budget, stops = _budget(sql_manager, 2)

    await _fail(
        budget,
        insert_request,
        claim,
        ScraperAssumptionException("one", "https://e/1"),
    )
    assert (budget.exhausted, stops.calls) == (False, 0)

    await _fail(
        budget,
        insert_request,
        claim,
        ScraperAssumptionException("two", "https://e/2"),
    )
    assert (budget.exhausted, stops.calls) == (True, 1)

    await _fail(
        budget,
        insert_request,
        claim,
        ScraperAssumptionException("three", "https://e/3"),
    )
    assert stops.calls == 1
    assert budget.persistent_error_count == 3
    assert "max_persistent_errors=2" in budget.exhaustion_message()


async def test_unbudgeted_run_counts_but_never_stops(
    sql_manager: SQLManager, insert_request: InsertRequest, claim: Claim
) -> None:
    budget, stops = _budget(sql_manager, None)

    for i in range(5):
        await _fail(
            budget, insert_request, claim, RuntimeError(f"unclassified {i}")
        )

    assert budget.persistent_error_count == 5
    assert (budget.exhausted, stops.calls) == (False, 0)


@pytest.mark.parametrize("limit", [0, -1])
def test_budget_below_one_is_rejected(
    sql_manager: SQLManager, limit: int
) -> None:
    """0 would stop on the first error; ``None`` is how to say "no budget"."""
    with pytest.raises(ValueError, match="max_persistent_errors"):
        _budget(sql_manager, limit)


def test_illegal_transition_error_survives_pickling() -> None:
    """It crosses process and Prefect task boundaries intact."""
    error = IllegalTransitionError(
        7, RequestStatus.IN_PROGRESS, RequestStatus.FAILED, None
    )

    error.add_note("context")

    restored = pickle.loads(pickle.dumps(error))

    assert str(restored) == str(error)
    assert restored.__notes__ == ["context"]
    assert (
        restored.request_id,
        restored.source,
        restored.target,
        restored.actual,
    ) == (7, RequestStatus.IN_PROGRESS, RequestStatus.FAILED, None)
