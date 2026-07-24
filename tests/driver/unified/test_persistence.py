"""Tests for the unified driver's RequestQueue and ResponseStorage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.database_engine.compression import decompress
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from tests.db_queries import fetch_results

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    """A fully-migrated SQLManager backed by a temp-file SQLite DB."""
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


def _parent_context() -> Response:
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/listing",
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


async def test_enqueue_dequeue_round_trip(sql_manager: SQLManager) -> None:
    """A Request survives enqueue -> dequeue (de)serialization."""
    queue = RequestQueue(sql_manager)
    context = _parent_context()

    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/detail/42",
            headers={"X-Test": "1"},
        ),
        continuation="parse_detail",
        current_location="https://example.com/detail/42",
        priority=7,
        accumulated_data={"foo": "bar"},
    )

    await queue.enqueue_request(original, context)

    dequeued = await queue.get_next_request()
    assert dequeued is not None
    _request_id, restored, _parent_id, _preresolved = dequeued

    assert isinstance(restored, Request)
    assert restored.request.url == "https://example.com/detail/42"
    assert restored.request.method == HttpMethod.POST
    assert restored.request.headers == {"X-Test": "1"}
    assert restored.continuation == "parse_detail"
    assert restored.priority == 7
    assert restored.accumulated_data == {"foo": "bar"}


async def test_enqueue_fires_progress_callback(
    sql_manager: SQLManager,
) -> None:
    """on_progress fires with a request_enqueued event on enqueue."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_progress(event_type: str, data: dict[str, Any]) -> None:
        events.append((event_type, data))

    queue = RequestQueue(sql_manager, on_progress=on_progress)
    context = _parent_context()
    req = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/x"
        ),
        continuation="parse_x",
        current_location="",
    )

    await queue.enqueue_request(req, context)

    assert len(events) == 1
    event_type, data = events[0]
    assert event_type == "request_enqueued"
    assert data["url"] == "https://example.com/x"
    assert data["continuation"] == "parse_x"


async def _seed_request(sql_manager: SQLManager) -> int:
    """Insert a request row and return its id (FK target for results/responses)."""
    queue = RequestQueue(sql_manager)
    await queue.enqueue_request(
        Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/seed"
            ),
            continuation="parse_seed",
            current_location="",
        ),
        _parent_context(),
    )
    dequeued = await queue.get_next_request()
    assert dequeued is not None
    return dequeued[0]


async def test_store_response_round_trip(sql_manager: SQLManager) -> None:
    """A stored Response can be read back with its content."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    response = Response(
        request=_parent_context().request,
        status_code=200,
        headers={"Content-Type": "text/html"},
        content=b"<html>body</html>",
        text="<html>body</html>",
        url="https://example.com/seed",
    )
    await storage.store_response(request_id, response, "parse_seed")

    stored = await sql_manager.get_stored_response(request_id)
    assert stored is not None
    status_code, _headers, _url, content_compressed, _dict_id = stored
    assert status_code == 200
    assert content_compressed is not None
    assert decompress(content_compressed) == b"<html>body</html>"


async def test_store_result_valid_and_invalid(sql_manager: SQLManager) -> None:
    """_store_result records valid and invalid results distinctly."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    await storage._store_result(request_id, {"a": 1}, is_valid=True)
    await storage._store_result(
        request_id,
        {"b": 2},
        is_valid=False,
        validation_errors=[{"loc": ("b",), "msg": "bad"}],
    )

    valid = await fetch_results(
        sql_manager, request_id=request_id, is_valid=True
    )
    invalid = await fetch_results(
        sql_manager, request_id=request_id, is_valid=False
    )
    assert len(valid) == 1
    assert len(invalid) == 1
    assert invalid[0].validation_errors_json is not None


async def test_handle_retry_backoff_and_ceiling(
    sql_manager: SQLManager,
) -> None:
    """handle_retry returns a sub-ceiling delay, then None once exhausted."""
    request_id = await _seed_request(sql_manager)
    # Small ceiling so the cumulative backoff trips quickly.
    storage = ResponseStorage(sql_manager, max_backoff_time=8.0)

    error = RuntimeError("transient")

    delay = await storage.handle_retry(request_id, error)
    assert delay is not None
    assert 0 < delay < 8.0

    # Drive cumulative backoff at/over the ceiling; eventually returns None.
    saw_none = False
    for _ in range(10):
        result = await storage.handle_retry(request_id, error)
        if result is None:
            saw_none = True
            break
        assert result < 8.0
    assert saw_none, "expected handle_retry to return None once over ceiling"
