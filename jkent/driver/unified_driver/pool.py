"""The pinned worker pool: spawn, ramp, drain, cancel.

:class:`WorkerPool` owns the worker task registry for a run. The pool is
*pinned*: it spawns ``size`` workers and never replaces one that exits — a
worker retires only when the queue is drained (nothing pending, nothing in flight),
so the pool cannot collapse while work can still appear. Spawning is
immediate by default; ``ramp_interval`` staggers it instead, which changes
only how fast the pool fills, not its size or its pinned-ness.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from jkent.driver.unified_driver.lifecycle import sleep_unless_stopped

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class Worker(Protocol):
    """What the pool runs: anything with an awaitable ``run``."""

    async def run(self) -> None: ...


class WorkerPool:
    """A pinned pool of worker tasks over one stop event.

    Args:
        size: Workers to spawn in total.
        ramp_interval: Seconds between spawns after the first; 0 spawns the
            whole pool inline.
        stop_event: The run's shutdown signal; a ramp in progress stops
            adding workers once it is set.
        make_worker: Builds worker ``n``; called once per spawn.
        on_change: Told the live worker count after every spawn and every
            retirement (the ``worker.active`` gauge).
        sleep: The ramp's stop-aware wait between spawns: sleeps the given
            seconds unless the event is set first, and returns whether it
            is. A seam for tests to drive the ramp without real time.
    """

    def __init__(
        self,
        *,
        size: int,
        ramp_interval: float,
        stop_event: asyncio.Event,
        make_worker: Callable[[int], Worker],
        on_change: Callable[[int], None] | None = None,
        sleep: Callable[
            [asyncio.Event, float], Awaitable[bool]
        ] = sleep_unless_stopped,
    ) -> None:
        self.size = size
        self.ramp_interval = ramp_interval
        self._stop_event = stop_event
        self._make_worker = make_worker
        self._on_change = on_change
        self._sleep = sleep
        self._tasks: dict[int, asyncio.Task[None]] = {}
        self._next_id = 0
        # Live only while a staggered startup is still adding workers.
        self._ramp_task: asyncio.Task[None] | None = None
        self._started = False
        # What each dead worker raised, in death order. Recorded by the
        # done-callback because a worker spawned while drain() is waiting
        # is not in its wait set, and has left the registry by the time
        # drain() looks again.
        self._failures: list[BaseException] = []

    @property
    def active_count(self) -> int:
        """Workers currently running."""
        return len(self._tasks)

    @property
    def ramp_task(self) -> asyncio.Task[None] | None:
        """The in-progress ramp, if a staggered startup is still adding."""
        return self._ramp_task

    def task_for(self, worker_id: int) -> asyncio.Task[None] | None:
        """The live task for ``worker_id``, or None once it has retired."""
        return self._tasks.get(worker_id)

    def spawn(self) -> int:
        """Create, register, and launch one worker; return its id."""
        worker_id = self._next_id
        self._next_id += 1
        worker = self._make_worker(worker_id)
        task = asyncio.create_task(worker.run())
        self._tasks[worker_id] = task

        def on_done(task: asyncio.Task[None], wid: int = worker_id) -> None:
            self._tasks.pop(wid, None)
            # The pool is pinned, so a worker that died of an exception is
            # capacity the run never gets back. Retrieve it and say so —
            # otherwise the loss is visible only as an asyncio "task exception
            # was never retrieved" line long after the fact.
            exc = None if task.cancelled() else task.exception()
            if exc is not None:
                self._failures.append(exc)
                logger.error(
                    "Worker %d died; the pool is down to %d worker(s)",
                    wid,
                    len(self._tasks),
                    exc_info=exc,
                )
            # Last: on_change is host code, and a raise from it must not
            # skip the bookkeeping drain() relies on.
            self._publish()

        # Before the publish, for the same reason: a registered task with
        # no callback would never leave the registry.
        task.add_done_callback(on_done)
        self._publish()
        return worker_id

    def start(self) -> None:
        """Bring the pool up, all at once or on a ramp.

        With ``ramp_interval`` at 0 the whole pool is spawned inline. With a
        ramp, worker 0 still starts synchronously — so the run is never
        momentarily worker-less — and the remainder are spawned by
        :meth:`_ramp` in the background, which :meth:`drain` watches
        alongside the workers themselves.

        Raises:
            RuntimeError: on a second call. It would double the pool and, on
                a ramp, replace the ramp task :meth:`cancel` knows about.
                (:meth:`spawn` stays unbounded: hosts add workers past
                ``size`` through ``Run.spawn_worker``.)
        """
        if self._started:
            raise RuntimeError("WorkerPool.start(): already started")
        self._started = True
        if self.ramp_interval <= 0:
            for _ in range(self.size):
                self.spawn()
            return
        self.spawn()
        if self.size > 1:
            self._ramp_task = asyncio.create_task(self._ramp())

    async def _ramp(self) -> None:
        """Add one worker per ``ramp_interval`` until the pool is full.

        Two early exits, both mandatory:

        * The stop event — a ramp must not keep adding workers to a run that
          is shutting down. The wait is stop-aware rather than a bare sleep so
          a Ctrl-C during a long ramp is prompt.
        * An empty worker registry. A worker retires only when nothing is
          pending and nothing is in flight pool-wide, so *every spawned worker
          having retired* means the queue drained during the ramp. Continuing
          would spawn each remaining worker only for it to retire immediately,
          padding the run by one interval apiece.

        Ramping cannot strand work: an early worker holding a request keeps
        ``in_flight_count`` above zero, and a worker that finds the queue
        momentarily empty sleeps and re-checks rather than retiring, so a late
        arrival is never the difference between drained and not.
        """
        for _ in range(self.size - 1):
            stopped = await self._sleep(self._stop_event, self.ramp_interval)
            if stopped or not self._tasks:
                return
            self.spawn()

    async def drain(self) -> None:
        """Await the pool until every worker has retired.

        A worker exits on stop or a drained queue (nothing pending, nothing
        in flight), so an empty pool means the scrape is drained — *unless* a
        staggered startup is still adding workers, in which case an empty
        registry only means the ramp has not caught up yet. The ramp task is
        therefore waited on alongside the workers: without it a slow ramp would
        let the drain return before the pool ever filled, ending the run early;
        with it, a ramp that raises also surfaces here rather than being
        swallowed as an unretrieved task exception. The first worker to have
        died re-raises here, whether or not it was in the wait set.
        """
        while True:
            if self._failures:
                raise self._failures[0]
            tasks = [t for t in self._tasks.values() if not t.done()]
            ramp = self._ramp_task
            if ramp is not None and not ramp.done():
                tasks.append(ramp)
            if not tasks:
                return
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                # A worker cancelled from outside is not a failure; the
                # rest of the pool still drains.
                exc = None if task.cancelled() else task.exception()
                if exc is not None:
                    raise exc

    async def cancel(self) -> None:
        """Cancel and await every live worker task, plus any pending ramp.

        Used on shutdown so no worker outlives ``run()`` and keeps issuing
        transport calls or DB writes against collaborators that ``aclose()`` is
        about to tear down. The ramp is cancelled first and in the same pass:
        left alive it would spawn *new* workers against those same
        mid-teardown collaborators. ``return_exceptions=True`` drains each
        task's result — the failure that triggered teardown and the
        CancelledErrors alike — so none is left unretrieved.
        """
        tasks: list[asyncio.Task[None]] = []
        if self._ramp_task is not None:
            tasks.append(self._ramp_task)
            self._ramp_task = None
        tasks.extend(self._tasks.values())
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _publish(self) -> None:
        if self._on_change is not None:
            self._on_change(len(self._tasks))
