"""Shared replay layer for the unified driver's replay transport.

- ``source_index`` -- the ``SourceIndex`` over previous-run source DBs and the
  replay-key helpers.
- ``error_pruning`` -- ``compute_pruning_plan`` (reseedable-anchor subtree pruning).
- ``errors`` -- the replay miss / scraper-mismatch exceptions.

The modules depend only on ``database_engine.compression`` and each other.
Consumers may import the submodules directly or the names re-exported here.
"""

from jkent.driver.replay.error_pruning import PruningPlan, compute_pruning_plan
from jkent.driver.replay.errors import ReplayScraperMismatchError
from jkent.driver.replay.source_index import (
    SourceIndex,
    fallback_replay_key_for_request,
)

__all__ = [
    "PruningPlan",
    "ReplayScraperMismatchError",
    "SourceIndex",
    "compute_pruning_plan",
    "fallback_replay_key_for_request",
]
