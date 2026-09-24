"""``LoopLagMonitor``: a background task sampling event-loop lag.

``instruments`` is replaced with a recorder, so no OTel SDK is needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jkent.observability import loop_monitor
from jkent.observability.loop_monitor import LoopLagMonitor


class _RecordingHistogram:
    def __init__(self) -> None:
        self.values: list[float] = []

    def record(self, value: float) -> None:
        self.values.append(value)


async def _stop(monitor: LoopLagMonitor) -> None:
    async with asyncio.timeout(1):
        await monitor.stop()


@pytest.fixture
def lag(monkeypatch: pytest.MonkeyPatch) -> _RecordingHistogram:
    recording = _RecordingHistogram()
    monkeypatch.delenv("JKENT_OTEL_LOOP_MONITOR", raising=False)
    monkeypatch.setattr(
        loop_monitor,
        "instruments",
        lambda: SimpleNamespace(loop_lag=recording),
    )
    return recording


async def test_records_non_negative_lag_samples(
    lag: _RecordingHistogram,
) -> None:
    monitor = LoopLagMonitor(interval=0.001)
    monitor.start()
    try:
        async with asyncio.timeout(2):
            while len(lag.values) < 3:
                await asyncio.sleep(0.001)
    finally:
        await _stop(monitor)
    assert all(v >= 0.0 for v in lag.values)


async def test_start_twice_runs_one_task(lag: _RecordingHistogram) -> None:
    monitor = LoopLagMonitor(interval=0.001)
    monitor.start()
    first = monitor._task
    monitor.start()
    try:
        assert monitor._task is first
    finally:
        await _stop(monitor)


async def test_stop_cancels_and_is_idempotent(
    lag: _RecordingHistogram,
) -> None:
    monitor = LoopLagMonitor(interval=0.001)
    await _stop(monitor)  # never started
    monitor.start()
    task = monitor._task
    assert task is not None
    loop = asyncio.get_running_loop()
    began = loop.time()
    await _stop(monitor)
    assert loop.time() - began < 0.5  # cancelled, not waited out
    assert task.cancelled()
    assert monitor._task is None
    await _stop(monitor)


@pytest.mark.parametrize("value", ["0", "false", "False"])
async def test_disabled_by_env(
    lag: _RecordingHistogram, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("JKENT_OTEL_LOOP_MONITOR", value)
    monitor = LoopLagMonitor(interval=0.001)
    monitor.start()
    assert monitor._task is None


async def test_a_cancelled_caller_of_stop_stays_cancelled(
    lag: _RecordingHistogram,
) -> None:
    """Cancelling the task awaiting ``stop()`` is not swallowed with the
    sampler's own cancellation."""
    monitor = LoopLagMonitor(interval=0.01)
    monitor.start()
    caller = asyncio.create_task(monitor.stop())
    await asyncio.sleep(0)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert monitor._task is None
