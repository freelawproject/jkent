"""Dequeue-ordering contract for the unified driver's RequestQueue.

``dequeue_next_request`` orders pending rows by ``priority`` ascending, then by
``queue_counter`` ascending — so lower priority values come out first, and ties
within a priority come out FIFO (insertion order). These tests enqueue via
``RequestQueue.enqueue_request`` and drain with ``_get_next_request``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import RequestQueue

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    """A fully-migrated SQLManager backed by a temp-file SQLite DB."""
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


def _context() -> Response:
    """A parent Response used as enqueue context for URL resolution."""
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/listing"
        ),
        continuation="parse_listing",
        current_location="https://example.com",
    )
    return Response(
        request=parent,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/listing",
    )


def _request(url: str, priority: int) -> Request:
    """A distinct-URL navigating request at ``priority``."""
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse_detail",
        current_location=url,
        priority=priority,
    )


async def _drain_urls(queue: RequestQueue) -> list[str]:
    """Dequeue everything and return the URLs in dequeue order."""
    urls: list[str] = []
    while (dequeued := await queue._get_next_request()) is not None:
        _request_id, restored, _parent_id = dequeued
        urls.append(restored.request.url)
    return urls


async def test_dequeues_in_ascending_priority_order(
    sql_manager: SQLManager,
) -> None:
    """Requests dequeue lowest-priority-value first, regardless of insert order."""
    queue = RequestQueue(sql_manager)
    context = _context()

    await queue.enqueue_request(_request("https://example.com/p9", 9), context)
    await queue.enqueue_request(_request("https://example.com/p1", 1), context)
    await queue.enqueue_request(_request("https://example.com/p5", 5), context)

    assert await _drain_urls(queue) == [
        "https://example.com/p1",
        "https://example.com/p5",
        "https://example.com/p9",
    ]


async def test_same_priority_dequeues_fifo(sql_manager: SQLManager) -> None:
    """Requests at the same priority dequeue FIFO (by queue_counter)."""
    queue = RequestQueue(sql_manager)
    context = _context()

    urls = [f"https://example.com/same/{i}" for i in range(5)]
    for url in urls:
        await queue.enqueue_request(_request(url, 5), context)

    assert await _drain_urls(queue) == urls


async def test_priority_then_fifo_within_priority(
    sql_manager: SQLManager,
) -> None:
    """Across mixed priorities, FIFO is preserved within each priority band."""
    queue = RequestQueue(sql_manager)
    context = _context()

    # Interleave priorities; insertion order recorded per priority band.
    await queue.enqueue_request(_request("https://example.com/a", 5), context)
    await queue.enqueue_request(_request("https://example.com/b", 1), context)
    await queue.enqueue_request(_request("https://example.com/c", 5), context)
    await queue.enqueue_request(_request("https://example.com/d", 1), context)
    await queue.enqueue_request(_request("https://example.com/e", 5), context)

    assert await _drain_urls(queue) == [
        "https://example.com/b",  # priority 1, enqueued first
        "https://example.com/d",  # priority 1, enqueued second
        "https://example.com/a",  # priority 5, enqueued first
        "https://example.com/c",  # priority 5, enqueued second
        "https://example.com/e",  # priority 5, enqueued third
    ]
