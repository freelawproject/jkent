"""``get_next_request`` releases its claim unless it hands one over.

The count is raised before the dequeue so a claim is never invisible to a
sibling mid-commit, which means every path out of the method that is *not*
a handover owes a decrement. The pool is pinned — a worker that raises is
never replaced — so a leaked claim is not a slow leak but a wedged run:
``in_flight_count`` never returns to zero, so every idle sibling waits
forever on a request nobody is holding, and the run cannot drain.

Deserialization is the interesting half. It raises for more reasons than it
looks: a rate-limit lane the table does not declare, a ``via_json`` that
matches no via shape, a malformed form body, and — since
``HTTPRequestParams`` began enforcing its own annotations — a pydantic
``ValidationError``. The tests drive each one rather than the structure, so
a new failure mode added later is covered without anyone remembering to add
it to a list.

Such a row is also marked ``failed`` on the way out. The dequeue has already
committed it ``in_progress``, and ``restore_queue`` resets ``in_progress``
to ``pending`` at startup, so a row that deterministically fails to
deserialize would otherwise be re-pended by every resume and fail the run
again, forever. The current run still dies — that is unchanged — but the
next one gets past it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from jkent.common.rate_limits import RateLimitTable
from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestType
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.sql_manager import DequeuedRow

_BASE: dict[str, Any] = {
    "id": 7,
    "preresolved": False,
    "request_type": RequestType.NAVIGATING,
    "method": HttpMethod.GET,
    "url": "https://example.com/x",
    "step": "parse",
}

#: Rows that reach ``_deserialize_request`` and fail there — past the
#: ``DequeuedRow`` validation that ``dequeue_next_request`` does, so the
#: claim is already taken when they blow up.
_UNDESERIALIZABLE: dict[str, dict[str, Any]] = {
    # A lane code with no lane: the table was built without the run's
    # stored lane names.
    "unknown rate-limit lane": {"rate_limit": 7},
    # Neither via shape.
    "via_json matching no shape": {"via_json": '{"type": "link"}'},
    # The decoder unpacks each entry into (name, value).
    "form body with a non-pair entry": {
        "body": b'[["a","b","c"]]',
        "body_is_form": True,
    },
    # A form body whose field names are not strings, which
    # HTTPRequestParams' `list[tuple[str, Any]]` now refuses. Unreachable
    # for rows written from here on, since enqueue rejects the same shape.
    "form body with non-string field names": {
        "body": b'[[1,"v"]]',
        "body_is_form": True,
    },
}


class _StubDB:
    """Hands back one prepared row, or raises instead of dequeuing."""

    def __init__(
        self,
        row: DequeuedRow | None = None,
        raises: BaseException | None = None,
        *,
        mark_failed_raises: Exception | None = None,
    ) -> None:
        self._row = row
        self._raises = raises
        self._mark_failed_raises = mark_failed_raises
        self.failed: list[tuple[int, str]] = []

    async def dequeue_next_request(self) -> DequeuedRow | None:
        if self._raises is not None:
            raise self._raises
        return self._row

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        if self._mark_failed_raises is not None:
            raise self._mark_failed_raises
        self.failed.append((request_id, error_message))


def _queue(db: _StubDB) -> RequestQueueDB:
    return RequestQueueDB(db, rate_limits=RateLimitTable.bare())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "extra", _UNDESERIALIZABLE.values(), ids=_UNDESERIALIZABLE
)
async def test_a_row_that_will_not_deserialize_releases_the_claim(
    extra: dict[str, Any],
) -> None:
    """The half Alberto's repro found: the raise is past the claim."""
    row = DequeuedRow.model_validate({**_BASE, **extra})
    db = _StubDB(row)
    queue = _queue(db)

    with pytest.raises(Exception):  # noqa: B017
        await queue.get_next_request()

    assert queue.in_flight_count == 0
    # Terminal, so the next resume does not re-pend it into the same crash.
    assert [request_id for request_id, _ in db.failed] == [_BASE["id"]]


@pytest.mark.parametrize(
    "extra", _UNDESERIALIZABLE.values(), ids=_UNDESERIALIZABLE
)
async def test_the_original_error_survives_a_failed_mark_failed(
    extra: dict[str, Any],
) -> None:
    """Recording the failure must not mask the reason for it."""
    row = DequeuedRow.model_validate({**_BASE, **extra})
    db = _StubDB(row, mark_failed_raises=RuntimeError("database is locked"))
    queue = _queue(db)

    with pytest.raises(Exception) as excinfo:
        await queue.get_next_request()

    assert not isinstance(excinfo.value, RuntimeError)
    assert queue.in_flight_count == 0


async def test_a_handed_over_row_is_not_marked_failed() -> None:
    db = _StubDB(DequeuedRow.model_validate(_BASE))
    queue = _queue(db)

    assert await queue.get_next_request() is not None
    assert db.failed == []


async def test_a_failed_dequeue_releases_the_claim() -> None:
    queue = _queue(_StubDB(raises=RuntimeError("database is locked")))

    with pytest.raises(RuntimeError):
        await queue.get_next_request()

    assert queue.in_flight_count == 0


async def test_a_cancelled_dequeue_releases_the_claim() -> None:
    """``BaseException``, not ``Exception``: a cancel must balance too."""
    queue = _queue(_StubDB(raises=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await queue.get_next_request()

    assert queue.in_flight_count == 0


async def test_an_empty_queue_releases_the_claim() -> None:
    queue = _queue(_StubDB(None))

    assert await queue.get_next_request() is None
    assert queue.in_flight_count == 0


async def test_a_handover_keeps_the_claim() -> None:
    """The one path that owes nothing: the caller now owns ``request_done``."""
    queue = _queue(_StubDB(DequeuedRow.model_validate(_BASE)))

    dequeued = await queue.get_next_request()

    assert dequeued is not None
    assert queue.in_flight_count == 1
    queue.request_done()
    assert queue.in_flight_count == 0
