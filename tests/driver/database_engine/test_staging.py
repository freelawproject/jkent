"""``StagedWrites.flush``: once per buffer, and what survives a raising callback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.queue import EnqueuedEvent
from jkent.driver.database_engine.sql_manager import (
    RequestInsert,
    SQLManager,
)
from jkent.driver.database_engine.staging import StagedWrites
from tests.db_queries import fetch_requests, get_request_row

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


async def test_second_flush_is_refused(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
) -> None:
    """A buffer flushes once; the second call raises and writes nothing.

    A results row carries no dedup key and a deferred callback has no idea
    it already ran, so a re-flush would duplicate every result and re-fire
    every on_data. The buffer is also emptied on the way out, so even a
    caller reaching past the guard has nothing left to re-insert.
    """
    parent_id = await insert_request()
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

    async def once() -> None:
        fired.append("once")

    staged.stage_callback(once)

    assert len(await staged.flush(sql_manager)) == 1
    with pytest.raises(RuntimeError, match="already"):
        await staged.flush(sql_manager)

    assert fired == ["once"]
    rows = await fetch_requests(sql_manager)
    assert [row.url for row in rows] == [
        "https://example.com/test",
        "https://example.com/child",
    ]


async def test_reset_makes_a_flushed_buffer_usable_again(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
) -> None:
    """``reset`` clears the spent flag along with the buffers.

    The autowait retry path resets and re-stages; a reset buffer that still
    counted as flushed could never commit its retry.
    """
    parent_id = await insert_request()
    staged = StagedWrites(request_id=parent_id)
    assert await staged.flush(sql_manager) == []

    staged.reset()
    assert await staged.flush(sql_manager) == []
