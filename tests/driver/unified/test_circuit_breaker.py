"""Tests for the run-scoped circuit breaker (:class:`CircuitBreaker`).

Contract under test:

- Closed by default; ``gate`` passes straight through.
- ``failure_threshold`` *consecutive* failures trip it; any success resets
  the count (and closes an open circuit — straggler evidence counts).
- Open: ``gate`` blocks; a success or the stop event wakes every waiter.
- Recovery is deadline-based: after ``recovery_timeout`` exactly one caller
  is admitted as the probe per window, so a probe whose verdict never
  arrives cannot wedge the gate.
- The policy is validated at construction and frozen after.

``time.monotonic`` is frozen by freezegun, so no test sleeps for real; the
blocked/unblocked distinction is asserted by yielding to the loop and
checking task completion, following the repo's no-wall-clock convention.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from freezegun import freeze_time
from freezegun.api import FrozenDateTimeFactory
from pydantic import ValidationError

from jkent import observability as obs
from jkent.driver.unified_driver.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clock() -> Iterator[FrozenDateTimeFactory]:
    """Freeze ``time.monotonic`` for every test; ``clock.tick(s)`` advances it.

    ``real_asyncio`` keeps the event loop on real time, so yields and
    ``wait_for`` timeouts behave normally while the breaker's clock stands
    still.
    """
    with freeze_time("2026-01-01", real_asyncio=True) as frozen:
        # freeze_time's return type spans its ticking variants; tick=False
        # (the default) always gives the frozen one.
        assert isinstance(frozen, FrozenDateTimeFactory)
        yield frozen


def make_breaker(
    *,
    failure_threshold: int = 3,
    recovery_timeout: float = 300.0,
    recovery_backoff: float = 1.0,
    max_recovery_timeout: float = 300.0,
    stop_event: asyncio.Event | None = None,
) -> CircuitBreaker:
    """Build a breaker with its own policy.

    ``recovery_backoff`` defaults to 1.0 — escalation off — so the tests of the
    fixed-window contract state their own spacing rather than inheriting the
    shipped curve. The escalation tests opt in explicitly.
    """
    return CircuitBreaker(
        CircuitBreakerPolicy(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            recovery_backoff=recovery_backoff,
            max_recovery_timeout=max_recovery_timeout,
        ),
        stop_event=stop_event,
    )


def trip(breaker: CircuitBreaker, count: int = 3) -> None:
    for _ in range(count):
        breaker.record_failure()


async def yield_loop(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def test_starts_closed_and_gate_passes() -> None:
    breaker = make_breaker()
    assert breaker.state == "closed"
    await asyncio.wait_for(breaker.gate(), timeout=1)


async def test_trips_after_threshold_consecutive_failures() -> None:
    breaker = make_breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"


async def test_success_resets_the_consecutive_count() -> None:
    breaker = make_breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"  # never 3 in a row


async def test_open_gate_blocks_and_success_wakes_all_waiters() -> None:
    breaker = make_breaker()
    trip(breaker)
    waiters = [asyncio.create_task(breaker.gate()) for _ in range(3)]
    await yield_loop()
    assert not any(t.done() for t in waiters)

    breaker.record_success()  # a straggler success is recovery evidence
    await yield_loop()
    assert breaker.state == "closed"
    assert all(t.done() for t in waiters)


async def test_deadline_admits_exactly_one_probe_per_window(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.tick(300.0)

    await asyncio.wait_for(breaker.gate(), timeout=1)  # the probe
    assert breaker.state == "half_open"

    # The next caller waits for the pushed-out deadline, not the old one.
    second = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not second.done()
    second.cancel()


async def test_probe_failure_reopens(clock: FrozenDateTimeFactory) -> None:
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.tick(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)

    breaker.record_failure()
    assert breaker.state == "open"


class _Counter:
    def __init__(self) -> None:
        self.total = 0

    def add(self, amount: int, attributes: object = None) -> None:
        self.total += amount


async def test_a_failed_probe_counts_as_a_reopen_not_a_trip(
    monkeypatch: pytest.MonkeyPatch,
    clock: FrozenDateTimeFactory,
) -> None:
    """``circuit.opens`` keeps meaning "trips from closed"."""
    counters = SimpleNamespace(
        circuit_opens=_Counter(), circuit_reopens=_Counter()
    )
    monkeypatch.setattr(obs, "instruments", lambda: counters)
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    for _ in range(2):
        clock.tick(300.0)
        await asyncio.wait_for(breaker.gate(), timeout=1)
        breaker.record_failure()
    assert (counters.circuit_opens.total, counters.circuit_reopens.total) == (
        1,
        2,
    )


async def test_probe_success_closes_and_wakes_waiters(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()

    clock.tick(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)  # probe admitted
    breaker.record_success()
    await yield_loop()
    assert breaker.state == "closed"
    assert waiter.done()


async def test_stuck_probe_does_not_wedge_the_gate(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.tick(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)  # probe, no verdict

    clock.tick(300.0)  # a full window with no record_* call
    await asyncio.wait_for(breaker.gate(), timeout=1)  # next probe admitted
    assert breaker.state == "half_open"


async def test_straggler_failure_during_open_pushes_the_deadline(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.tick(200.0)
    breaker.record_failure()  # in-flight straggler fails at t=200

    clock.tick(150.0)  # t=350: past the original deadline, not the new one
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()
    waiter.cancel()


async def test_stragglers_at_trip_time_do_not_escalate(
    clock: FrozenDateTimeFactory,
) -> None:
    """In-flight failures landing after the trip are not failed probes.

    With the shipped defaults (8 workers, backoff 2.0) escalating on them
    pushed the first probe of every outage out to the ceiling.
    """
    breaker = make_breaker(
        failure_threshold=8,
        recovery_timeout=30.0,
        recovery_backoff=2.0,
        max_recovery_timeout=300.0,
    )
    trip(breaker, 8)
    for _ in range(7):
        clock.tick(0.1)
        breaker.record_failure()
    assert breaker.recovery_window == 30.0

    clock.tick(30.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)
    assert breaker.state == "half_open"


async def test_stop_event_wakes_a_blocked_gate() -> None:
    stop_event = asyncio.Event()
    breaker = make_breaker(stop_event=stop_event)
    trip(breaker)
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()

    stop_event.set()
    await asyncio.wait_for(waiter, timeout=1)


async def test_wake_event_is_not_stale_across_open_periods() -> None:
    breaker = make_breaker()
    trip(breaker)
    breaker.record_success()  # first open period ends

    trip(breaker)  # second open period
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()  # a stale set event would let this through
    waiter.cancel()


async def test_first_trip_uses_the_base_window() -> None:
    breaker = make_breaker(recovery_timeout=30.0, recovery_backoff=2.0)
    trip(breaker)
    assert breaker.recovery_window == 30.0


async def test_failed_probe_escalates_the_window(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(
        recovery_timeout=30.0, recovery_backoff=2.0, max_recovery_timeout=300.0
    )
    trip(breaker)

    for expected in (60.0, 120.0, 240.0, 300.0, 300.0):
        clock.tick(breaker.recovery_window)
        await asyncio.wait_for(breaker.gate(), timeout=1)  # probe admitted
        breaker.record_failure()  # ...and it fails
        assert breaker.recovery_window == expected  # capped at the ceiling


async def test_close_resets_the_escalated_window(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=30.0, recovery_backoff=2.0)
    trip(breaker)
    clock.tick(30.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)
    breaker.record_failure()
    assert breaker.recovery_window == 60.0

    breaker.record_success()
    assert breaker.recovery_window == 30.0
    trip(breaker)  # an unrelated later hiccup pays the short window again
    assert breaker.recovery_window == 30.0


async def test_backoff_of_one_keeps_a_fixed_window(
    clock: FrozenDateTimeFactory,
) -> None:
    breaker = make_breaker(recovery_timeout=300.0, recovery_backoff=1.0)
    trip(breaker)
    clock.tick(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)
    breaker.record_failure()
    assert breaker.recovery_window == 300.0


async def test_stuck_probe_does_not_escalate(
    clock: FrozenDateTimeFactory,
) -> None:
    # Escalation is failure-driven: a probe that returns no verdict is not
    # evidence the server is worse, so the spacing must hold steady.
    breaker = make_breaker(recovery_timeout=30.0, recovery_backoff=2.0)
    trip(breaker)
    clock.tick(30.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)
    assert breaker.recovery_window == 30.0
    clock.tick(30.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)
    assert breaker.recovery_window == 30.0


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("failure_threshold", 0),
        ("recovery_timeout", 0.0),
        ("recovery_backoff", 0.5),
        ("max_recovery_timeout", -1.0),
    ],
)
def test_policy_rejects_a_bad_value(field: str, bad: float) -> None:
    with pytest.raises(ValidationError):
        CircuitBreakerPolicy.model_validate({field: bad})


def test_policy_rejects_a_base_window_above_the_ceiling() -> None:
    with pytest.raises(ValidationError):
        CircuitBreakerPolicy(
            recovery_timeout=600.0, max_recovery_timeout=300.0
        )


def test_policy_is_frozen() -> None:
    policy = CircuitBreakerPolicy()
    with pytest.raises(ValidationError):
        policy.recovery_timeout = 60.0  # type: ignore[misc]
