"""Event-loop lag monitor.

This is a utility class to monitor event loop lag.
This can be quite an important metric to monitor if we are running
many workers in a single thread.
"""

from __future__ import annotations

import asyncio
import os

from jkent.observability.metrics import instruments

DEFAULT_INTERVAL = 0.2


def _enabled() -> bool:
    return os.environ.get("JKENT_OTEL_LOOP_MONITOR", "1") not in (
        "0",
        "false",
        "False",
    )


class LoopLagMonitor:
    """Samples event-loop scheduling latency into ``jkent.event_loop.lag``.

    Usage (host process, once)::

        monitor = LoopLagMonitor()
        monitor.start()
        ...
        await monitor.stop()
    """

    def __init__(self, interval: float = DEFAULT_INTERVAL) -> None:
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Launch the sampling task (no-op if disabled or already running)."""
        if not _enabled() or self._task is not None:
            return
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Cancel and await the sampling task."""
        task = self._task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        lag = instruments().loop_lag
        while True:
            expected = loop.time() + self._interval
            await asyncio.sleep(self._interval)
            # Negative drift is impossible in principle; floor at 0 to guard
            # against clock granularity producing a tiny negative.
            lag.record(max(0.0, loop.time() - expected))
