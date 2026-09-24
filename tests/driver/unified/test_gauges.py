"""Tests for :class:`RunGauges`' queue-backlog sampler.

The sampler is a background task that reads ``count_pending_requests`` on
an interval and publishes it as ``jkent.queue.pending``. It is best-effort:
a failed read is logged and skipped, and the next tick samples again.
Stopping cancels the task and returns only once it has ended, so teardown
can close the database next.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from jkent import observability as obs
from jkent.driver.unified_driver.gauges import RunGauges

if TYPE_CHECKING:
    from collections.abc import Callable


class _RecordingGauge:
    def __init__(self) -> None:
        self.values: list[int] = []

    def set(self, value: int, _labels: dict[str, str]) -> None:
        self.values.append(value)


class _FirstSetRaises:
    """A gauge whose first ``set`` raises, as a rejecting exporter would."""

    def __init__(self) -> None:
        self.values: list[int] = []
        self.calls = 0

    def set(self, value: int, _labels: dict[str, str]) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("exporter rejected the point")
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


async def _until(
    predicate: Callable[[], object], timeout: float = 2.0
) -> None:
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


async def test_sampler_survives_a_failed_publish(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    flaky = _FirstSetRaises()
    monkeypatch.setattr(obs, "sdk_active", lambda: True)
    monkeypatch.setattr(
        obs, "instruments", lambda: SimpleNamespace(queue_pending=flaky)
    )
    monkeypatch.setattr(RunGauges, "SAMPLE_INTERVAL_S", 0.001)
    gauges = RunGauges("scraper")
    with caplog.at_level(logging.WARNING):
        gauges.start_sampler(_FlakyCounter(failures=0, value=4))  # type: ignore[arg-type]
        try:
            await _until(lambda: flaky.values)
        finally:
            await gauges.stop_sampler()
    assert flaky.values[0] == 4
    assert len(caplog.records) == 1
    assert gauges._sampler is None


async def test_stop_sampler_ends_the_task(gauge: _RecordingGauge) -> None:
    gauges = RunGauges("scraper")
    gauges.start_sampler(_FlakyCounter(failures=0, value=1))  # type: ignore[arg-type]
    sampler = gauges._sampler
    assert sampler is not None
    await _until(lambda: gauge.values)
    await gauges.stop_sampler()
    assert sampler.cancelled()
    assert gauges._sampler is None


async def test_every_sample_carries_the_start_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Samples land on the run's series even after the context changed."""
    seen: list[dict[str, str]] = []

    class _LabelGauge:
        def set(self, _value: int, labels: dict[str, str]) -> None:
            seen.append(labels)

    run_inst_id: str | None = "run-1"
    monkeypatch.setattr(obs, "sdk_active", lambda: True)
    monkeypatch.setattr(obs, "run_inst_id", lambda: run_inst_id)
    monkeypatch.setattr(
        obs,
        "instruments",
        lambda: SimpleNamespace(queue_pending=_LabelGauge()),
    )
    monkeypatch.setattr(RunGauges, "SAMPLE_INTERVAL_S", 0.001)
    gauges = RunGauges("scraper")
    gauges.start_sampler(_FlakyCounter(failures=0, value=1))  # type: ignore[arg-type]
    await _until(lambda: seen)
    run_inst_id = None
    await _until(lambda: len(seen) >= 2)
    await gauges.stop_sampler()
    assert all(
        labels == {"scraper": "scraper", "run_inst_id": "run-1"}
        for labels in seen
    )


class _SlowToUnwind(_FlakyCounter):
    """A second read that blocks, then takes ``release`` to unwind a cancel."""

    def __init__(self) -> None:
        super().__init__(failures=0, value=1)
        self.blocked = asyncio.Event()
        self.unwinding = asyncio.Event()
        self.release = asyncio.Event()

    @override
    async def count_pending_requests(self) -> int:
        if self.calls:
            self.blocked.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.unwinding.set()
                await self.release.wait()
                raise
        return await super().count_pending_requests()


async def test_stop_waits_for_the_sampler_to_end(
    gauge: _RecordingGauge,
) -> None:
    """Stop returns only once a read caught mid-flight has unwound."""
    counter = _SlowToUnwind()
    gauges = RunGauges("scraper")
    gauges.start_sampler(counter)  # type: ignore[arg-type]
    sampler = gauges._sampler
    assert sampler is not None
    await counter.blocked.wait()

    teardown = asyncio.create_task(gauges.stop_sampler())
    await counter.unwinding.wait()
    assert not teardown.done()

    counter.release.set()
    await teardown
    assert sampler.cancelled()


async def test_a_cancelled_stop_still_waits_for_the_sampler(
    gauge: _RecordingGauge,
) -> None:
    """Teardown cancelled mid-stop ends cancelled, but not before the sampler.

    Teardown closes the database next; returning while the sampler is still
    unwinding would let it touch a closed run db.
    """
    counter = _SlowToUnwind()
    gauges = RunGauges("scraper")
    gauges.start_sampler(counter)  # type: ignore[arg-type]
    sampler = gauges._sampler
    assert sampler is not None
    await counter.blocked.wait()

    teardown = asyncio.create_task(gauges.stop_sampler())
    await counter.unwinding.wait()
    teardown.cancel()
    await asyncio.sleep(0.01)
    assert not teardown.done()

    counter.release.set()
    with pytest.raises(asyncio.CancelledError):
        await teardown
    assert sampler.cancelled()
    assert gauges._sampler is None


async def test_failing_instruments_do_not_break_teardown(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken() -> SimpleNamespace:
        raise RuntimeError("meter provider misconfigured")

    monkeypatch.setattr(obs, "sdk_active", lambda: True)
    monkeypatch.setattr(obs, "instruments", broken)
    monkeypatch.setattr(RunGauges, "SAMPLE_INTERVAL_S", 0.001)
    gauges = RunGauges("scraper")
    with caplog.at_level(logging.WARNING):
        gauges.start_sampler(_FlakyCounter(failures=0, value=1))  # type: ignore[arg-type]
        await _until(lambda: caplog.records)
        await gauges.stop_sampler()
    assert gauges._sampler is None


def test_worker_count_carries_the_construction_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spawn/retire callback outside the run's context stays on its series."""
    seen: list[dict[str, str]] = []

    class _LabelGauge:
        def set(self, _value: int, labels: dict[str, str]) -> None:
            seen.append(labels)

    run_inst_id: str | None = "run-1"
    monkeypatch.setattr(obs, "run_inst_id", lambda: run_inst_id)
    monkeypatch.setattr(
        obs,
        "instruments",
        lambda: SimpleNamespace(worker_active=_LabelGauge()),
    )
    gauges = RunGauges("scraper")
    run_inst_id = None
    gauges.worker_active(3)
    assert seen == [{"scraper": "scraper", "run_inst_id": "run-1"}]
