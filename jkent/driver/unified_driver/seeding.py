"""Getting work into a run: entry seeding and speculation templates.

:class:`Seeder` enqueues a fresh run's entry requests and sets up its
speculation templates. A run database is pinned to the seed set it was
created with; nothing adds entries to a populated one. It is a policy
object over a queue and a database; the run creates one at ``open`` and
delegates to it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jkent.driver._speculation_support import get_entry_requests
from jkent.driver.unified_driver.speculation import SpeculationManager

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper
    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.unified_driver.persistence import RequestQueue
    from jkent.driver.unified_driver.wiring import SeedParams

logger = logging.getLogger(__name__)


class Seeder:
    """Seeds entry requests and discovers speculation templates."""

    def __init__(
        self,
        scraper: BaseScraper[Any],
        queue: RequestQueue,
        db: SQLManager,
    ) -> None:
        self.scraper = scraper
        self._queue = queue
        self._db = db

    async def seed_entries(self, seed_params: SeedParams | None) -> None:
        """Enqueue the scraper's entry requests for a fresh queue.

        ``seed_params`` invocations go through ``initial_seed()``; without
        them every no-arg ``@entry`` is invoked (a scraper with none raises
        ``ScraperConfigError``). Speculative entries yield
        no requests here — they store templates on the scraper for
        :meth:`setup_speculation`.
        """
        for entry_request in get_entry_requests(self.scraper, seed_params):
            await self._queue.insert_root_request(entry_request)

    async def setup_speculation(
        self, seed_params: SeedParams | None
    ) -> SpeculationManager | None:
        """Build the speculation manager, load persisted state, seed probes.

        Discovery reads ``scraper._speculation_templates`` (populated by the
        ``initial_seed`` that :meth:`seed_entries` ran on a fresh queue). On
        resume the templates are empty, but :meth:`SpeculationManager.load`
        reconstructs them from persisted ``template_json``. Returns ``None``
        when the scraper has no speculation at all — the fact the worker
        needs, see :mod:`~jkent.driver.unified_driver.wiring`.
        """
        manager = SpeculationManager(
            self.scraper, self._queue, self._db, seed_params=seed_params
        )
        manager.discover()
        await manager.load()
        if not manager.has_state:
            return None
        await manager.seed()
        return manager
