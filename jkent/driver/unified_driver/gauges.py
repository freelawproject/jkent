"""The run-level gauges: live worker count and queue backlog.

Both are per-run state (attributes ``scraper`` and, when set,
``run_inst_id``) published as gauges, so the last value stands until updated.
The worker count is pushed on every spawn and retirement by a
``WorkerPool`` whose ``on_change`` the host wires to
:meth:`RunGauges.worker_active`; the backlog is sampled — and only when a
real OTel SDK is installed, since the sample is a DB read done solely to have
something to record. Both are best-effort: a failed sample is logged and
skipped, and the next tick samples again.

Keyed on ``run_inst_id``, each run is its own series: under cumulative
temporality a long-lived process keeps one frozen series per finished run
until it exits.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, ClassVar

from jkent import observability as obs

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)


class RunGauges:
    """Publishes ``jkent.worker.active`` and samples ``jkent.queue.pending``."""

    # How often the queue-backlog gauge samples count_pending_requests.
    SAMPLE_INTERVAL_S: ClassVar[float] = 5.0

    def __init__(self, scraper_name: str) -> None:
        """Build the gauges for one run.

        Build it in the run's context: the labels are fixed here, so every
        worker count and backlog sample lands on the series of the context
        the run started in.

        Args:
            scraper_name: Name of the scraper, used as a metric label.
        """
        labels = {"scraper": scraper_name}
        fid = obs.run_inst_id()
        if fid is not None:
            labels["run_inst_id"] = fid
        self._labels = labels
        self._sampler: asyncio.Task[None] | None = None

    def labels(self) -> dict[str, str]:
        """Per-run metric attributes: scraper name and (if set) run_inst_id.

        Returns:
            A copy of the attribute dict attached to every metric this run
            publishes.
        """
        return dict(self._labels)

    def worker_active(self, count: int) -> None:
        """Publish the live worker count.

        With the pool pinned this flatlines at ``num_workers`` mid-run; its
        value is the two ends of the run — the startup ramp climbing to
        ``num_workers`` (when ``worker_ramp_interval`` is set), and the drain
        tail, how long the run limps along on the last one or two workers
        after the rest retire.
        """
        obs.instruments().worker_active.set(count, self._labels)

    def start_sampler(self, db: SQLManager) -> None:
        """Start sampling the pending-request backlog, if an SDK is active."""
        if self._sampler is None and obs.sdk_active():
            self._sampler = asyncio.create_task(self._sample(db))

    async def stop_sampler(self) -> None:
        """Cancel the backlog sampler, if one is running, and wait it out.

        Returns only once the sampler has ended, so the caller can close the
        run database next. A cancellation of the caller while waiting is
        held until then, and re-raised.

        Raises:
            CancelledError: The caller was cancelled while waiting.
        """  # noqa: DOC502 -- re-raised from a variable, which ruff can't see
        sampler, self._sampler = self._sampler, None
        if sampler is None:
            return
        sampler.cancel()
        cancelled: asyncio.CancelledError | None = None
        while not sampler.done():
            try:
                await asyncio.wait([sampler])
            except asyncio.CancelledError as exc:
                cancelled = exc
        if cancelled is not None:
            raise cancelled

    async def _sample(self, db: SQLManager) -> None:
        """Publish the backlog every ``SAMPLE_INTERVAL_S`` until cancelled."""
        while True:
            await _publish_backlog(db, self._labels)
            await asyncio.sleep(self.SAMPLE_INTERVAL_S)


async def _publish_backlog(db: SQLManager, labels: dict[str, str]) -> None:
    """Publish one backlog sample; log and skip one that raises.

    A skipped sample leaves the gauge at its last value until the next one
    succeeds.
    """
    try:
        obs.instruments().queue_pending.set(
            await db.count_pending_requests(), labels
        )
    except Exception:
        logger.warning(
            "queue.pending sample failed; retrying next tick",
            exc_info=True,
        )
