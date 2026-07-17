"""Tests for the run-scoped circuit breaker (:class:`CircuitBreaker`).

Contract under test:

- Closed by default; ``gate`` passes straight through.
- ``failure_threshold`` *consecutive* failures trip it; any success resets
  the count (and closes an open circuit — straggler evidence counts).
- Open: ``gate`` blocks; a success or the stop event wakes every waiter.
- Recovery is deadline-based: after ``recovery_timeout`` exactly one caller
  is admitted as the probe per window, so a probe whose verdict never
  arrives cannot wedge the gate.
- The policy is mutable; changes apply at the next transition.

Time is injected (a fake monotonic clock), so no test sleeps for real; the
blocked/unblocked distinction is asserted by yielding to the loop and
checking task completion, following the repo's no-wall-clock convention.
"""

from __future__ import annotations

import asyncio

from jkent.driver.unified_driver.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)


class Clock:
    """A manually-advanced stand-in for ``time.monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_breaker(
    *,
    failure_threshold: int = 3,
    recovery_timeout: float = 300.0,
    stop_event: asyncio.Event | None = None,
) -> tuple[CircuitBreaker, Clock]:
    clock = Clock()
    breaker = CircuitBreaker(
        CircuitBreakerPolicy(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        ),
        stop_event=stop_event,
        clock=clock,
    )
    return breaker, clock


def trip(breaker: CircuitBreaker, count: int = 3) -> None:
    for _ in range(count):
        breaker.record_failure()


async def yield_loop(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


async def test_default_policy() -> None:
    policy = CircuitBreakerPolicy()
    assert policy.failure_threshold == 3
    assert policy.recovery_timeout == 300.0


async def test_starts_closed_and_gate_passes() -> None:
    breaker, _ = make_breaker()
    assert breaker.state == "closed"
    await asyncio.wait_for(breaker.gate(), timeout=1)


async def test_trips_after_threshold_consecutive_failures() -> None:
    breaker, _ = make_breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"


async def test_success_resets_the_consecutive_count() -> None:
    breaker, _ = make_breaker(failure_threshold=3)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"  # never 3 in a row


async def test_open_gate_blocks_and_success_wakes_all_waiters() -> None:
    breaker, _ = make_breaker()
    trip(breaker)
    waiters = [asyncio.create_task(breaker.gate()) for _ in range(3)]
    await yield_loop()
    assert not any(t.done() for t in waiters)

    breaker.record_success()  # a straggler success is recovery evidence
    await yield_loop()
    assert breaker.state == "closed"
    assert all(t.done() for t in waiters)


async def test_deadline_admits_exactly_one_probe_per_window() -> None:
    breaker, clock = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.advance(300.0)

    await asyncio.wait_for(breaker.gate(), timeout=1)  # the probe
    assert breaker.state == "half_open"

    # The next caller waits for the pushed-out deadline, not the old one.
    second = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not second.done()
    second.cancel()


async def test_probe_failure_reopens() -> None:
    breaker, clock = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.advance(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)

    breaker.record_failure()
    assert breaker.state == "open"


async def test_probe_success_closes_and_wakes_waiters() -> None:
    breaker, clock = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()

    clock.advance(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)  # probe admitted
    breaker.record_success()
    await yield_loop()
    assert breaker.state == "closed"
    assert waiter.done()


async def test_stuck_probe_does_not_wedge_the_gate() -> None:
    breaker, clock = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.advance(300.0)
    await asyncio.wait_for(breaker.gate(), timeout=1)  # probe, no verdict

    clock.advance(300.0)  # a full window with no record_* call
    await asyncio.wait_for(breaker.gate(), timeout=1)  # next probe admitted
    assert breaker.state == "half_open"


async def test_straggler_failure_during_open_pushes_the_deadline() -> None:
    breaker, clock = make_breaker(recovery_timeout=300.0)
    trip(breaker)
    clock.advance(200.0)
    breaker.record_failure()  # in-flight straggler fails at t=200

    clock.advance(150.0)  # t=350: past the original deadline, not the new one
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()
    waiter.cancel()


async def test_stop_event_wakes_a_blocked_gate() -> None:
    stop_event = asyncio.Event()
    breaker, _ = make_breaker(stop_event=stop_event)
    trip(breaker)
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()

    stop_event.set()
    await asyncio.wait_for(waiter, timeout=1)


async def test_wake_event_is_not_stale_across_open_periods() -> None:
    breaker, _ = make_breaker()
    trip(breaker)
    breaker.record_success()  # first open period ends

    trip(breaker)  # second open period
    waiter = asyncio.create_task(breaker.gate())
    await yield_loop()
    assert not waiter.done()  # a stale set event would let this through
    waiter.cancel()


async def test_policy_mutation_applies_at_the_next_transition() -> None:
    breaker, _ = make_breaker(failure_threshold=5)
    breaker.record_failure()
    breaker.policy.failure_threshold = 2
    breaker.record_failure()
    assert breaker.state == "open"
