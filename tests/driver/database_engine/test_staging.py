"""``StagedWrites.flush``: its parent precondition, and callbacks that raise."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import icontract
import pytest

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.queue import EnqueuedEvent
from jkent.driver.database_engine.sql_manager import (
    RequestInsert,
    SQLManager,
)
from jkent.driver.database_engine.staging import StagedWrites
from tests.db_queries import get_request_row

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


async def test_raising_callback_does_not_lose_later_callbacks_or_events(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A callback raising after commit is logged; the rest still fire.

    The rows are already durable when the callbacks run, so the parent is
    completed either way. Propagating would drop the later callbacks and the
    progress events for children that *were* inserted.
    """
    parent_id = await insert_request()
    # ``flush`` is only ever reached for a request its worker holds, so the
    # parent has been dequeued and is ``in_progress`` by then; its contract
    # requires it. ``insert_request`` leaves the row ``pending``, so claim
    # it the way ``dequeue_next_request`` would.
    await sql_manager.dequeue_next_request()
    staged = StagedWrites(request_id=parent_id)
    staged.stage_request(
        request_data=RequestInsert(
            priority=5,
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url="https://example.com/child",
            step="parse",
            deduplication_key="child",
            parent_request_id=parent_id,
        ),
        progress_event=EnqueuedEvent(
            url="https://example.com/child", step="parse", priority=5
        ),
    )
    fired: list[str] = []

    async def boom() -> None:
        fired.append("boom")
        raise RuntimeError("on_data blew up")

    async def after() -> None:
        fired.append("after")

    staged.stage_callback(boom)
    staged.stage_callback(after)

    with caplog.at_level(logging.ERROR):
        events = await staged.flush(sql_manager)

    assert events == [
        EnqueuedEvent(
            url="https://example.com/child", step="parse", priority=5
        )
    ]
    assert fired == ["boom", "after"]
    assert "on_data blew up" in caplog.text
    parent = await get_request_row(sql_manager, parent_id)
    assert parent is not None
    assert parent.status == RequestStatus.COMPLETED


# --- The in_progress precondition -------------------------------------------
#
# ``flush`` marks its parent ``completed`` with no status guard in the
# UPDATE, which is safe only because a parent reaches it in exactly one
# state. The worker dispatches into ``_execute_one`` inside a ``try`` whose
# ``except`` arms hold every ``_fail_request``, so the flush path and the
# failure paths are mutually exclusive within one task. Nothing in the
# database says so, and a supervisor that failed an in-flight request from
# outside that sequence would silently turn the unconditional ``completed``
# into a status overwrite. The contract is what notices.


async def _staged_for(parent_id: int) -> StagedWrites:
    """A buffer with nothing in it: the precondition is all that is tested."""
    return StagedWrites(request_id=parent_id)


async def test_flushing_an_in_progress_parent_is_allowed(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """The one state the worker can actually be in when it flushes."""
    parent_id = await insert_request()
    await sql_manager.dequeue_next_request()

    staged = await _staged_for(parent_id)
    assert await staged.flush(sql_manager) == []

    row = await get_request_row(sql_manager, parent_id)
    assert row is not None
    assert row.status == RequestStatus.COMPLETED


async def test_flushing_a_failed_parent_is_refused(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """Alberto's case: a failed request must not be revived as completed."""
    parent_id = await insert_request()
    await sql_manager.dequeue_next_request()
    await sql_manager.mark_request_failed(parent_id, "timed out")

    staged = await _staged_for(parent_id)
    with pytest.raises(icontract.ViolationError, match="in_progress"):
        await staged.flush(sql_manager)

    row = await get_request_row(sql_manager, parent_id)
    assert row is not None
    assert row.status == RequestStatus.FAILED
    assert row.last_error == "timed out"


async def test_flushing_a_completed_parent_is_refused(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """A second flush of the same parent is a bug, not an idempotent no-op."""
    parent_id = await insert_request()
    await sql_manager.dequeue_next_request()
    await (await _staged_for(parent_id)).flush(sql_manager)

    with pytest.raises(icontract.ViolationError, match="in_progress"):
        await (await _staged_for(parent_id)).flush(sql_manager)


async def test_flushing_a_pending_parent_is_refused(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """A parent nobody claimed: the worker never reaches flush this way."""
    parent_id = await insert_request()

    staged = await _staged_for(parent_id)
    with pytest.raises(icontract.ViolationError, match="in_progress"):
        await staged.flush(sql_manager)


async def test_flushing_a_request_that_does_not_exist_is_refused(
    sql_manager: SQLManager,
) -> None:
    """The other silent pass: an UPDATE that matches no rows at all."""
    staged = await _staged_for(9999)
    with pytest.raises(icontract.ViolationError, match="in_progress"):
        await staged.flush(sql_manager)
