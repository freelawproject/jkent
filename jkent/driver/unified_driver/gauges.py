"""The run-level gauges: live worker count and queue backlog.

Both are per-run state (attributes ``scraper`` and, when set,
``run_inst_id``) published as gauges, so the last value stands until updated.
The worker count is pushed by the pool on every spawn and retirement; the
backlog is sampled — and only when a real OTel SDK is installed, since the
sample is a DB read done solely to have something to record. A failed
sample is logged and skipped; the next tick samples again.

Keyed on ``run_inst_id``, each run is its own series: under cumulative
temporality a long-lived process keeps one frozen series per finished run
until it exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

from jkent import observability as obs

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)


class RunGauges:
    """Publishes ``jkent.worker.active`` and samples ``jkent.queue.pending``."""

    # How often the queue-backlog gauge samples count_pending_requests.
    SAMPLE_INTERVAL_S = 5.0

    def __init__(self, scraper_name: str) -> None:
        self._scraper_name = scraper_name
        self._sampler: asyncio.Task[None] | None = None

    def labels(self) -> dict[str, str]:
        """Per-run metric attributes: scraper name and (if set) run_inst_id."""
        labels = {"scraper": self._scraper_name}
        fid = obs.run_inst_id()
        if fid is not None:
            labels["run_inst_id"] = fid
        return labels

    def worker_active(self, count: int) -> None:
        """Publish the live worker count.

        With the pool pinned this flatlines at ``num_workers`` mid-run; its
        value is the two ends of the run — the startup ramp climbing to
        ``num_workers`` (when ``worker_ramp_interval`` is set), and the drain
        tail, how long the run limps along on the last one or two workers
        after the rest retire.
        """
        obs.instruments().worker_active.set(count, self.labels())

    def start_sampler(self, db: SQLManager) -> None:
        """Start sampling the pending-request backlog, if an SDK is active."""
        if self._sampler is None and obs.sdk_active():
            self._sampler = asyncio.create_task(self._sample(db))

    async def stop_sampler(self) -> None:
        """Cancel the backlog sampler, if one is running."""
        sampler = self._sampler
        if sampler is None:
            return
        sampler.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sampler
        self._sampler = None

    async def _sample(self, db: SQLManager) -> None:
        """Publish the backlog every ``SAMPLE_INTERVAL_S`` until cancelled.

        A sample that raises is logged and skipped, leaving the gauge at its
        last value until the next tick succeeds.
        """
        gauge = obs.instruments().queue_pending
        while True:
            try:
                pending = await db.count_pending_requests()
            except Exception:
                logger.warning(
                    "queue.pending sample failed; retrying next tick",
                    exc_info=True,
                )
            else:
                gauge.set(pending, self.labels())
            await asyncio.sleep(self.SAMPLE_INTERVAL_S)
