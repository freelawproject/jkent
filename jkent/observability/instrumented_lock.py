"""An ``asyncio.Lock`` that measures contention.

The run's single :class:`~asyncio.Lock` serializes every SQLite access for a
run (dequeue, restamp, store, staged flush, dedup, counts). Under N concurrent
continuation workers it is the central serialization point — the "lock getting
fought over". :class:`InstrumentedLock` is a drop-in subclass that records two
things per use:

* ``jkent.db.lock.wait`` — time from requesting the lock to acquiring it
  (nonzero only under contention);
* ``jkent.db.lock.hold`` — time the holder kept it.

Being a subclass of :class:`asyncio.Lock`, it satisfies every existing
``asyncio.Lock`` annotation and ``async with`` site unchanged. Attributes come
from the ambient label context (``scraper`` / ``step``), so no call site needs
to pass anything.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from jkent.observability.metrics import current_labels, instruments


class InstrumentedLock(asyncio.Lock):
    """``asyncio.Lock`` that records acquire-wait and hold durations."""

    #: Set on acquire, cleared on release. Only ever read/written by the single
    #: coroutine that holds the lock, so a plain attribute is race-free.
    _held_since: float | None = None

    async def acquire(self) -> Literal[True]:
        start = time.monotonic()
        acquired = await super().acquire()
        inst = instruments()
        labels = current_labels()
        inst.lock_wait.record(time.monotonic() - start, labels)
        self._held_since = time.monotonic()
        return acquired

    def release(self) -> None:
        held_since = self._held_since
        self._held_since = None
        if held_since is not None:
            instruments().lock_hold.record(
                time.monotonic() - held_since, current_labels()
            )
        super().release()
