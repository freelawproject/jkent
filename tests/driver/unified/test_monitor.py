"""Tests for the concrete ``WorkerMonitor`` (jkent.driver.unified_driver).

Binds the real implementation to the reusable ``MonitorConformance`` suite,
then adds targeted ``run()``-loop tests with fast, pre-arranged conditions.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Iterable

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from jkent.driver.unified_driver import Monitor
from jkent.driver.unified_driver.orchestration import WorkerMonitor
from tests.driver.unified.test_monitor_conformance import MonitorConformance


class _FakeRun:
    """A run exposing a mutable ``active_worker_count`` and ``spawn_worker``."""

    def __init__(self, active_worker_count: int = 0) -> None:
        self.active_worker_count = active_worker_count
        self.spawn_calls = 0

    def spawn_worker(self) -> int:
        self.spawn_calls += 1
        self.active_worker_count += 1
        return self.active_worker_count


class _FakeRateLimiter:
    """A rate limiter exposing only ``max_rate_per_second``."""

    def __init__(self, max_rate_per_second: float | None) -> None:
        self._max_rate_per_second = max_rate_per_second

    async def gate(self, request: object) -> None:  # pragma: no cover
        return None

    @property
    def max_rate_per_second(self) -> float | None:
        return self._max_rate_per_second


def _pending(n: int) -> Callable[[], Awaitable[int]]:
    """A ``pending_requests`` callable that always reports ``n``."""

    async def pending() -> int:
        return n

    return pending


# --- conformance suite bound to the real implementation ------------------


class TestWorkerMonitor(MonitorConformance):
    """Run the conformance suite against the real ``WorkerMonitor``."""

    def make_monitor(
        self,
        *,
        max_workers: int,
        max_rate_per_second: float | None,
        active_worker_count: int = 0,
        durations: Iterable[float] = (),
    ) -> Monitor:
        monitor = WorkerMonitor(
            _FakeRun(active_worker_count=active_worker_count),
            _FakeRateLimiter(max_rate_per_second),
            max_workers=max_workers,
            pending_requests=_pending(0),
        )
        for d in durations:
            monitor.record_request_duration(d)
        return monitor

    @pytest.fixture
    def subject(self) -> Monitor:
        return WorkerMonitor(
            _FakeRun(),
            _FakeRateLimiter(2.0),
            max_workers=8,
            pending_requests=_pending(0),
        )

    # The inherited ``@given`` methods are re-declared here so this binding
    # owns its own function objects; sharing them with the reference suite's
    # subclass trips hypothesis's ``differing_executors`` health check.

    @pytest.mark.generative
    @given(
        durations=st.lists(st.floats(0.001, 100.0), min_size=1, max_size=50)
    )
    def test_avg_is_window_mean(self, durations: list[float]) -> None:
        monitor = self.make_monitor(max_workers=8, max_rate_per_second=2.0)
        for d in durations:
            monitor.record_request_duration(d)
        avg = monitor.recent_avg_request_duration_s()
        assert avg is not None
        assert avg == pytest.approx(sum(durations) / len(durations))

    @pytest.mark.generative
    @given(
        max_workers=st.integers(min_value=1, max_value=64),
        rate=st.floats(min_value=0.01, max_value=1_000.0),
        durations=st.lists(st.floats(0.001, 100.0), min_size=1, max_size=30),
    )
    def test_workers_needed_formula_and_clamp(
        self, max_workers: int, rate: float, durations: list[float]
    ) -> None:
        monitor = self.make_monitor(
            max_workers=max_workers,
            max_rate_per_second=rate,
            durations=durations,
        )
        avg = sum(durations) / len(durations)
        expected = max(1, min(math.ceil(rate * avg), max_workers))
        assert monitor.workers_needed() == expected
        assert 1 <= monitor.workers_needed() <= max_workers

    @pytest.mark.generative
    @given(
        max_workers=st.integers(min_value=1, max_value=64),
        durations=st.lists(st.floats(0.001, 100.0), min_size=1, max_size=30),
    )
    def test_unlimited_rate_always_max(
        self, max_workers: int, durations: list[float]
    ) -> None:
        monitor = self.make_monitor(
            max_workers=max_workers,
            max_rate_per_second=None,
            durations=durations,
        )
        assert monitor.workers_needed() == max_workers

    @pytest.mark.generative
    @given(
        max_workers=st.integers(min_value=1, max_value=64),
        active=st.integers(min_value=0, max_value=128),
    )
    def test_no_data_conservative_clamped(
        self, max_workers: int, active: int
    ) -> None:
        assume(active <= max_workers + 64)
        monitor = self.make_monitor(
            max_workers=max_workers,
            max_rate_per_second=2.0,
            active_worker_count=active,
        )
        assert monitor.recent_avg_request_duration_s() is None
        expected = max(1, min(active + 1, max_workers))
        assert monitor.workers_needed() == expected


# --- targeted run()-loop tests -------------------------------------------


async def test_run_exits_when_stopped() -> None:
    """The loop returns promptly when the stop event is pre-set."""
    run = _FakeRun(active_worker_count=2)
    monitor = WorkerMonitor(
        run,
        _FakeRateLimiter(2.0),
        max_workers=8,
        pending_requests=_pending(10),
        poll_interval=0.001,
    )
    monitor.stop_event.set()
    await asyncio.wait_for(monitor.run(), timeout=1.0)
    assert run.spawn_calls == 0


async def test_run_exits_when_idle_and_no_pending() -> None:
    """The loop exits once active == 0 and there is no pending work."""
    run = _FakeRun(active_worker_count=0)
    monitor = WorkerMonitor(
        run,
        _FakeRateLimiter(2.0),
        max_workers=8,
        pending_requests=_pending(0),
        poll_interval=0.001,
    )
    await asyncio.wait_for(monitor.run(), timeout=1.0)
    assert run.spawn_calls == 0


async def test_run_scales_up_under_load() -> None:
    """With pending work, low active count, and timing data, it spawns."""
    run = _FakeRun(active_worker_count=1)
    # 4 req/s * 1.0 s avg -> workers_needed = 4, capped at max_workers=3.
    monitor = WorkerMonitor(
        run,
        _FakeRateLimiter(4.0),
        max_workers=3,
        pending_requests=_pending(10),
        poll_interval=0.001,
    )
    monitor.record_request_duration(1.0)

    async def stop_soon() -> None:
        # Let the loop run a few cycles, then stop it.
        await asyncio.sleep(0.05)
        monitor.stop_event.set()

    await asyncio.gather(
        asyncio.wait_for(monitor.run(), timeout=2.0), stop_soon()
    )
    assert run.spawn_calls >= 1
    assert run.active_worker_count <= monitor.max_workers


async def test_run_does_not_spawn_without_pending_work() -> None:
    """No spawn when there is no pending work but workers are still active."""
    run = _FakeRun(active_worker_count=1)
    monitor = WorkerMonitor(
        run,
        _FakeRateLimiter(4.0),
        max_workers=8,
        pending_requests=_pending(0),
        poll_interval=0.001,
    )
    monitor.record_request_duration(1.0)

    async def stop_soon() -> None:
        await asyncio.sleep(0.05)
        monitor.stop_event.set()

    await asyncio.gather(
        asyncio.wait_for(monitor.run(), timeout=2.0), stop_soon()
    )
    assert run.spawn_calls == 0
