"""LocalOnlyDriver: replay scraper runs from previous-run databases.

A subclass of :class:`PersistentDriver` that swaps the network for a
SQLite-backed lookup over one or more source DBs. The output is a regular
``PersistentDriver``-shaped run DB, resumable by ``kent run`` for any
requests that couldn't be served from the source data.

Exposed via the ``pdd replay`` CLI subcommand.
"""

from kent.driver.local_only_driver.errors import (
    LocalOnlyMiss,
    LocalOnlyScraperMismatchError,
)
from kent.driver.local_only_driver.local_only_driver import (
    LocalOnlyDriver,
    MatchMode,
    MissPolicy,
)
from kent.driver.local_only_driver.source_index import SourceIndex

__all__ = [
    "LocalOnlyDriver",
    "LocalOnlyMiss",
    "LocalOnlyScraperMismatchError",
    "MatchMode",
    "MissPolicy",
    "SourceIndex",
]
