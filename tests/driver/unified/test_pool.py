"""``WorkerPool`` — the pinned pool's spawn, ramp, drain, and cancel laws.

The hazard the ramp tests guard is the interaction with retirement: a worker
exits the moment nothing is pending and nothing is in flight, so a pool that
fills slowly must not let the run conclude before it has filled, and must not
keep adding workers to a run that has already drained or stopped.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from jkent.driver.unified_driver.pool import WorkerPool


class _BlockingWorker:
    """Stands in for a worker still processing: runs until ``release`` is set."""

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release

    async def run(self) -> None:
        await self._release.wait()


class _RetiringWorker:
    """Stands in for a worker that finds the queue drained and retires."""

    async def run(self) -> None:
        return None


class _Factory:
    """Records which worker ids were built."""

    def __init__(self, release: asyncio.Event | None = None) -> None:
        self.ids: list[int] = []
        self._release = release

    def __call__(self, worker_id: int):
        self.ids.append(worker_id)
        if self._release is None:
            return _RetiringWorker()
        return _BlockingWorker(self._release)


class _RampClock:
    """The pool's ramp sleep, advanced one interval per :meth:`tick`.

    Records every delay the ramp asked for; a sleep returns only when the
    test ticks (or the stop event is already set), so the ramp moves exactly
    as far as the test lets it.
    """

    def __init__(self) -> None:
        self.delays: list[float] = []
        self._ticks = asyncio.Semaphore(0)

    async def sleep(self, stop_event: asyncio.Event, delay: float) -> bool:
        self.delays.append(delay)
        if not stop_event.is_set():
            await self._ticks.acquire()
        return stop_event.is_set()

    async def tick(self) -> None:
        """Let one pending sleep return, and the ramp act on it."""
        self._ticks.release()
        for _ in range(3):
            await asyncio.sleep(0)


def _pool(
    factory: _Factory,
    *,
    size: int,
    ramp: float = 0.0,
    stop_event: asyncio.Event | None = None,
    on_change: Callable[[int], None] | None = None,
    clock: _RampClock | None = None,
) -> WorkerPool:
    return WorkerPool(
        size=size,
        ramp_interval=ramp,
        stop_event=stop_event or asyncio.Event(),
        make_worker=factory,
        on_change=on_change,
        **({} if clock is None else {"sleep": clock.sleep}),
    )


async def test_default_spawns_the_whole_pool_inline() -> None:
    """No interval configured spawns the whole pool up front."""
    release = asyncio.Event()
    factory = _Factory(release)
    pool = _pool(factory, size=4)
    try:
        pool.start()
        assert factory.ids == [0, 1, 2, 3]
        assert pool.active_count == 4
        assert pool.ramp_task is None
    finally:
        release.set()
        await pool.cancel()


async def test_spawn_ids_are_distinct_and_publish_count_changes() -> None:
    counts: list[int] = []
    pool = _pool(_Factory(), size=2, on_change=counts.append)
    a = pool.spawn()
    b = pool.spawn()
    assert a != b
    await pool.drain()
    await asyncio.sleep(0)  # let the done-callbacks fire
    assert pool.active_count == 0
    # Two spawns up, two retirements down.
    assert counts == [1, 2, 1, 0]


async def test_ramp_spawns_one_immediately_then_one_per_interval() -> None:
    """The pool still reaches ``size``, just not all at once."""
    interval = 7.5
    release = asyncio.Event()
    factory = _Factory(release)
    clock = _RampClock()
    pool = _pool(factory, size=4, ramp=interval, clock=clock)
    try:
        pool.start()
        # Worker 0 is synchronous so the run is never worker-less.
        assert factory.ids == [0]
        assert pool.ramp_task is not None
        for spawned in ([0, 1], [0, 1, 2], [0, 1, 2, 3]):
            await clock.tick()
            assert factory.ids == spawned
        await asyncio.wait_for(pool.ramp_task, timeout=5.0)
        assert clock.delays == [interval] * 3
    finally:
        release.set()
        await pool.cancel()


async def test_drain_does_not_return_while_the_ramp_is_pending() -> None:
    """An empty registry mid-ramp is not a drained queue.

    Without the ramp task in the drain's wait set this returns immediately
    and ``run()`` concludes with most of the pool never spawned. Worker 0
    retires at once, so the registry is empty while the ramp still sleeps.
    """
    clock = _RampClock()
    pool = _pool(_Factory(), size=3, ramp=0.05, clock=clock)
    pool.start()
    drain = asyncio.create_task(pool.drain())
    try:
        for _ in range(5):
            await asyncio.sleep(0)
        assert pool.active_count == 0
        assert pool.ramp_task is not None and not pool.ramp_task.done()
        assert not drain.done()
        # The ramp wakes to an empty registry (the queue drained) and ends,
        # which is what lets the drain return.
        await clock.tick()
        await asyncio.wait_for(drain, timeout=5.0)
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


async def test_stop_event_aborts_the_ramp() -> None:
    """A shutdown mid-ramp adds no further workers."""
    release = asyncio.Event()
    stop = asyncio.Event()
    factory = _Factory(release)
    pool = _pool(factory, size=5, ramp=0.05, stop_event=stop)
    try:
        pool.start()
        stop.set()
        assert pool.ramp_task is not None
        await asyncio.wait_for(pool.ramp_task, timeout=5.0)
        assert factory.ids == [0]
    finally:
        release.set()
        await pool.cancel()


async def test_ramp_gives_up_once_every_worker_has_retired() -> None:
    """A queue that drains mid-ramp ends the ramp instead of padding it.

    Retiring workers empty the registry before the first tick — the state a
    real pool reaches when its workers find nothing to do. Adding the
    remaining four would cost an interval each for workers that would retire
    on arrival.
    """
    factory = _Factory()
    pool = _pool(factory, size=5, ramp=0.01)
    pool.start()
    assert pool.ramp_task is not None
    await asyncio.wait_for(pool.ramp_task, timeout=5.0)
    assert factory.ids == [0]


async def test_cancel_cancels_a_pending_ramp() -> None:
    """Teardown stops the ramp; a survivor would spawn into a closing run."""
    release = asyncio.Event()
    pool = _pool(_Factory(release), size=5, ramp=30.0)
    pool.start()
    ramp = pool.ramp_task
    assert ramp is not None
    release.set()
    await pool.cancel()
    assert ramp.cancelled()
    assert pool.ramp_task is None
    assert pool.active_count == 0


async def test_drain_reraises_a_worker_failure() -> None:
    class _Exploding:
        async def run(self) -> None:
            raise RuntimeError("worker exploded")

    pool = WorkerPool(
        size=1,
        ramp_interval=0.0,
        stop_event=asyncio.Event(),
        make_worker=lambda _wid: _Exploding(),
    )
    pool.start()
    with pytest.raises(RuntimeError, match="worker exploded"):
        await pool.drain()


async def test_drain_outlives_a_worker_cancelled_from_outside() -> None:
    """A cancelled worker is not a failure: drain waits for the rest."""
    release = asyncio.Event()
    pool = _pool(_Factory(release), size=2)
    pool.start()
    drain = asyncio.create_task(pool.drain())
    try:
        await asyncio.sleep(0)
        pool._tasks[0].cancel()
        for _ in range(3):
            await asyncio.sleep(0)
        assert not drain.done()
        release.set()
        await asyncio.wait_for(drain, timeout=5.0)
    finally:
        release.set()
        await pool.cancel()


async def test_drain_reraises_a_ramp_spawned_worker_failure() -> None:
    """A worker spawned while drain waits, then dying, still fails the drain."""
    release = asyncio.Event()

    class _ExplodesAfterSpawn:
        async def run(self) -> None:
            raise RuntimeError("late worker exploded")

    def make(worker_id: int):
        if worker_id == 0:
            return _BlockingWorker(release)
        return _ExplodesAfterSpawn()

    pool = WorkerPool(
        size=2,
        ramp_interval=0.01,
        stop_event=asyncio.Event(),
        make_worker=make,
    )
    pool.start()
    drain = asyncio.create_task(pool.drain())
    # Let the ramp spawn worker 1 and let it die, then let worker 0 retire.
    while pool.ramp_task is not None and not pool.ramp_task.done():
        await asyncio.sleep(0.01)
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(RuntimeError, match="late worker exploded"):
        await drain


async def test_worker_death_is_logged_not_swallowed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The pool is pinned: a worker that raises is capacity the run never gets
    # back. The on-done callback retrieves the exception and says so, instead
    # of leaving it to surface as an asyncio "task exception was never
    # retrieved" line long after the pool quietly shrank.
    class _Exploding:
        async def run(self) -> None:
            raise RuntimeError("worker exploded")

    pool = WorkerPool(
        size=1,
        ramp_interval=0.0,
        stop_event=asyncio.Event(),
        make_worker=lambda _wid: _Exploding(),
    )
    with caplog.at_level(logging.ERROR):
        worker_id = pool.spawn()
        task = pool.task_for(worker_id)
        assert task is not None
        with pytest.raises(RuntimeError, match="worker exploded"):
            await task
    assert pool.task_for(worker_id) is None
    assert any("Worker 0 died" in record.message for record in caplog.records)


async def test_a_raising_gauge_does_not_hide_a_worker_death(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure is recorded and logged before the gauge hears of it.

    ``on_change`` is host code (a metrics gauge). When the retirement publish
    ran first and raised, the callback stopped before recording the death,
    so a drain begun after the worker died returned cleanly.
    """

    class _Exploding:
        async def run(self) -> None:
            raise RuntimeError("worker exploded")

    def gauge(count: int) -> None:
        if count == 0:
            raise ValueError("gauge exporter down")

    pool = WorkerPool(
        size=1,
        ramp_interval=0.0,
        stop_event=asyncio.Event(),
        make_worker=lambda _wid: _Exploding(),
        on_change=gauge,
    )
    with caplog.at_level(logging.ERROR):
        pool.start()
        for _ in range(3):
            await asyncio.sleep(0)
        assert pool.active_count == 0
        with pytest.raises(RuntimeError, match="worker exploded"):
            await pool.drain()
    assert any("Worker 0 died" in record.message for record in caplog.records)


@pytest.mark.parametrize("ramp", [0.0, 5.0])
async def test_a_second_start_is_refused(ramp: float) -> None:
    """start() brings the pool up once; again would double it (and, on a
    ramp, orphan the first ramp task where cancel() cannot reach it)."""
    release = asyncio.Event()
    factory = _Factory(release)
    pool = _pool(factory, size=2, ramp=ramp, clock=_RampClock())
    try:
        pool.start()
        with pytest.raises(RuntimeError, match="already started"):
            pool.start()
        assert factory.ids == ([0, 1] if ramp == 0 else [0])
    finally:
        release.set()
        await pool.cancel()
