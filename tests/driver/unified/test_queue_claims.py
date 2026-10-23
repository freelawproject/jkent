"""The in-flight claim discipline: one ``request_done`` per dequeued request.

``in_flight_count`` is the pinned worker pool's only termination signal — an
idle worker retires on a drained queue with nothing in flight, and is never
replaced. A count that drifts below zero can never return to zero, so every
idle worker would poll a finished run forever. The counter needs no database
to exercise; the dequeue side of the pairing is covered by
``test_queue_schedule_machine``.
"""

from __future__ import annotations

import pytest
from icontract import ViolationError

from jkent.driver.database_engine.queue import RequestQueueDB


def _queue() -> RequestQueueDB:
    """A queue with no database (the counter never touches one)."""
    return RequestQueueDB(None)  # type: ignore[arg-type]


def test_unpaired_request_done_is_refused() -> None:
    """``request_done`` with nothing in flight violates its contract.

    Contracts are enforced for the whole test suite (``tests/conftest.py``),
    so an unpaired call — a ``finally`` that fires on a path where the
    dequeue returned None, a caller releasing the same claim twice — fails
    the test that caused it instead of quietly arming a hang. In a release
    build the contract is a no-op and the method's ``max(0, ...)`` floor
    takes over: the count stays at zero either way, which is the property
    the pool depends on.
    """
    queue = _queue()
    with pytest.raises(ViolationError):
        queue.request_done()
    assert queue.in_flight_count == 0


def test_a_claim_is_released_once() -> None:
    """A claim raised by a dequeue is lowered by exactly one call."""
    queue = _queue()
    queue._in_flight = 1

    queue.request_done()

    assert queue.in_flight_count == 0
    with pytest.raises(ViolationError):
        queue.request_done()
