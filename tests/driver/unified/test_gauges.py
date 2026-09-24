"""Tests for :class:`RunGauges`' queue-backlog sampler.

The sampler is a background task that reads ``count_pending_requests`` on
an interval and publishes it as ``jkent.queue.pending``. A failed read is
logged and skipped — the next tick samples again — and cancellation stops
the task.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from jkent import observability as obs
from jkent.driver.unified_driver.gauges import RunGauges


class _RecordingGauge:
    def __init__(self) -> None:
        self.values: list[int] = []

    def set(self, value: int, _labels: dict[str, str]) -> None:
        self.values.append(value)


class _FlakyCounter:
    """``count_pending_requests`` raising for the first ``failures`` calls."""

    def __init__(self, failures: int, value: int) -> None:
        self.failures = failures
        self.value = value
        self.calls = 0

    async def count_pending_requests(self) -> int:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("database is locked")
        return self.value


@pytest.fixture
def gauge(monkeypatch: pytest.MonkeyPatch) -> _RecordingGauge:
    recording = _RecordingGauge()
    monkeypatch.setattr(obs, "sdk_active", lambda: True)
    monkeypatch.setattr(
        obs, "instruments", lambda: SimpleNamespace(queue_pending=recording)
    )
    monkeypatch.setattr(RunGauges, "SAMPLE_INTERVAL_S", 0.001)
    return recording


async def _until(predicate: Any, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.001)


@pytest.mark.parametrize("failures", [1, 3])
async def test_sampler_survives_a_failed_sample(
    gauge: _RecordingGauge,
    caplog: pytest.LogCaptureFixture,
    failures: int,
) -> None:
    counter = _FlakyCounter(failures=failures, value=7)
    gauges = RunGauges("scraper")
    with caplog.at_level(logging.WARNING):
        gauges.start_sampler(counter)  # type: ignore[arg-type]
        try:
            await _until(lambda: gauge.values)
        finally:
            await gauges.stop_sampler()
    assert gauge.values[0] == 7
    failed = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(failed) == failures
    assert all(r.exc_info is not None for r in failed)


async def test_stop_sampler_cancels_the_task(gauge: _RecordingGauge) -> None:
    gauges = RunGauges("scraper")
    gauges.start_sampler(_FlakyCounter(failures=0, value=1))  # type: ignore[arg-type]
    sampler = gauges._sampler
    assert sampler is not None
    await _until(lambda: gauge.values)
    await gauges.stop_sampler()
    assert sampler.cancelled()
    assert gauges._sampler is None
