"""Tests for the unified driver's RequestQueue and ResponseStorage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

from jkent.common.exceptions import HTTPResponseAssumptionException
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.database_engine.compression import decompress
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import (
    ErrorBudget,
    RequestQueue,
    ResponseStorage,
    RowOnlyErrorSink,
)
from tests.db_queries import fetch_results, get_request_row

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _parent_context() -> Response:
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/listing",
        ),
        step="parse_listing",
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
        step="parse_detail",
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
    assert restored.step == "parse_detail"
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
        step="parse_x",
        current_location="",
    )

    await queue.enqueue_request(req, context)

    assert len(events) == 1
    event_type, data = events[0]
    assert event_type == "request_enqueued"
    assert data["url"] == "https://example.com/x"
    assert data["step"] == "parse_x"


async def _seed_request(sql_manager: SQLManager) -> int:
    """Insert a request row and return its id (FK target for results/responses)."""
    queue = RequestQueue(sql_manager)
    await queue.enqueue_request(
        Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/seed"
            ),
            step="parse_seed",
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
    assert stored.response_status_code == 200
    assert stored.content_compressed is not None
    assert decompress(stored.content_compressed) == b"<html>body</html>"


async def test_store_result_valid_and_invalid(sql_manager: SQLManager) -> None:
    """_store_result records valid and invalid results distinctly."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    await storage._store_result(request_id, {"a": 1})
    await storage._store_result(
        request_id,
        {"b": 2},
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


async def test_handle_retry_floors_at_retry_after(
    sql_manager: SQLManager,
) -> None:
    """A server-sent Retry-After floors the computed backoff delay.

    The first retry's nominal backoff is far below 42s, so the returned
    delay being exactly the Retry-After proves the floor applied. The value
    arrives on the exception already parsed and clamped by the transport.
    """
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    error = HTTPResponseAssumptionException(
        429, [200], "https://example.com/x", retry_after=42.0
    )
    delay = await storage.handle_retry(request_id, error)
    assert delay == pytest.approx(42.0)


async def test_handle_retry_without_retry_after_is_unchanged(
    sql_manager: SQLManager,
) -> None:
    """No Retry-After on the error → the normal (small) backoff applies."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    error = HTTPResponseAssumptionException(
        429, [200], "https://example.com/x"
    )
    delay = await storage.handle_retry(request_id, error)
    assert delay is not None
    assert delay < 42.0


async def test_handle_retry_allows_a_retry_after_that_fills_the_budget(
    sql_manager: SQLManager,
) -> None:
    """A Retry-After equal to the whole budget still earns one retry.

    ``max_backoff_time`` is the most a request may wait in total; a wait
    that reaches it exactly is within it. With an exclusive bound a clamped
    ``Retry-After: 300`` under ``max_backoff_time=300`` failed the request
    with no retry at all. The second such wait does exceed it.
    """
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager, max_backoff_time=300.0)

    error = HTTPResponseAssumptionException(
        429, [200], "https://example.com/x", retry_after=300.0
    )
    assert await storage.handle_retry(request_id, error) == pytest.approx(
        300.0
    )
    assert await storage.handle_retry(request_id, error) is None


async def test_row_only_sink_logs_what_it_cannot_file(
    sql_manager: SQLManager,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``RowOnlyErrorSink.record`` has nowhere to file; it logs instead.

    The caller asked for the diagnosis to be kept. Dropping it without a
    trace left nothing — no ``errors`` row and no log line — to say the
    failure happened at all. ``request_failed`` already stores ``str(exc)``
    as ``last_error``, exactly like ``fail_request``.
    """
    request_id = await _seed_request(sql_manager)
    sink = RowOnlyErrorSink(ResponseStorage(sql_manager))

    with caplog.at_level(logging.WARNING):
        await sink.record(
            RuntimeError("replay miss"),
            request_id=request_id,
            request_url="https://example.com/seed",
        )

    assert "replay miss" in caplog.text
    assert "https://example.com/seed" in caplog.text

    await sink.request_failed(request_id, RuntimeError("gave up"))
    row = await get_request_row(sql_manager, request_id)
    assert row is not None
    assert row.last_error == "gave up"


async def _fail_via_budget(
    sql_manager: SQLManager, request_id: int, exc: Exception
) -> None:
    sink = ErrorBudget(
        sql_manager, max_persistent_errors=None, stop=lambda: None
    )
    await sink.request_failed(request_id, exc)


async def _fail_via_row_only_sink(
    sql_manager: SQLManager, request_id: int, exc: Exception
) -> None:
    await RowOnlyErrorSink(ResponseStorage(sql_manager)).request_failed(
        request_id, exc
    )


async def _schedule_a_retry(
    sql_manager: SQLManager, request_id: int, exc: Exception
) -> None:
    assert await ResponseStorage(sql_manager).handle_retry(request_id, exc)


@pytest.mark.parametrize(
    "write", [_fail_via_budget, _fail_via_row_only_sink, _schedule_a_retry]
)
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError(), "RuntimeError"),
        (TimeoutError(), "TimeoutError"),
        (RuntimeError("boom"), "boom"),
    ],
)
async def test_last_error_names_an_exception_that_has_no_message(
    sql_manager: SQLManager,
    write: Callable[[SQLManager, int, Exception], Awaitable[None]],
    exc: Exception,
    expected: str,
) -> None:
    """A bare exception reads as its class, not as "failed, for no reason"."""
    request_id = await _seed_request(sql_manager)
    await write(sql_manager, request_id, exc)
    row = await get_request_row(sql_manager, request_id)
    assert row is not None
    assert row.last_error == expected
