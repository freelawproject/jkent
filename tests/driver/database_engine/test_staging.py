"""``StagedWrites.flush``: what survives a user callback that raises."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus, RequestType
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
        progress_event={"url": "https://example.com/child"},
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

    assert events == [{"url": "https://example.com/child"}]
    assert fired == ["boom", "after"]
    assert "on_data blew up" in caplog.text
    parent = await get_request_row(sql_manager, parent_id)
    assert parent is not None
    assert parent.status == RequestStatus.COMPLETED
