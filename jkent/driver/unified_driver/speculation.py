"""Speculation support for the unified driver.

:class:`SpeculationManager` holds the per-step :class:`SpeculationState` dict
and implements the two hooks of
:class:`~jkent.driver._speculation_support.AsyncSpeculationSupport` against a
:class:`~jkent.driver.unified_driver.persistence.RequestQueue` (enqueue) and a
:class:`~jkent.driver.database_engine.sql_manager.SQLManager` (persist). The
seed/extend/track engine lives in the shared base; this class only wires
dispatch and persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from jkent.common.serialization import dump_json_or_none
from jkent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
    discover_speculate_functions,
)
from jkent.driver.database_engine.sql_manager import SpeculationStateRecord

if TYPE_CHECKING:
    from pydantic import BaseModel

    from jkent.data_types import BaseScraper, Request
    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.unified_driver.persistence import RequestQueue

logger = logging.getLogger(__name__)


class SpeculationManager(AsyncSpeculationSupport):
    """Discovers, seeds, tracks, and persists speculation for a unified run."""

    def __init__(
        self,
        scraper: BaseScraper[Any],
        queue: RequestQueue,
        db: SQLManager,
        *,
        seed_params: list[dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        self.scraper = scraper
        self.seed_params = seed_params
        self._queue = queue
        self._db = db
        self._speculation_state: dict[str, SpeculationState] = {}
        self._speculation_lock = asyncio.Lock()

    # --- Hook implementations -------------------------------------------

    async def _enqueue_speculative(self, request: Request) -> None:
        """Insert a speculative probe into the queue, never deduplicated."""
        await self._queue.insert_root_request(request, deduplicate=False)

    async def _persist(self, spec_state: SpeculationState) -> None:
        """Upsert *spec_state* and cache the tracking row id on it.

        Also creates the tracking row before a state's first probe: the
        upsert writes the state as it stands (fresh counters on a first run,
        the loaded ones on a resume) and records the id probes point at.
        """
        # Speculation templates are pydantic models (Speculative subclasses
        # such as SpeculativeRange), so they always serialize. Persist the
        # JSON unconditionally so a resumed run can reconstruct a template
        # that the current discovery pass dropped — and so we never overwrite
        # a previously-stored template_json with NULL.
        template_json = cast(
            "BaseModel", spec_state.template
        ).model_dump_json()

        spec_state.tracking_id = await self._db.save_speculation_state(
            SpeculationStateRecord(
                func_name=spec_state.func_name,
                highest_successful_id=spec_state.highest_successful_id,
                consecutive_failures=spec_state.consecutive_failures,
                current_ceiling=spec_state.current_ceiling,
                stopped=spec_state.stopped,
                param_index=spec_state.param_index,
                template_json=template_json,
                seed_value_json=dump_json_or_none(spec_state.seed_value),
            )
        )

    # --- Lifecycle ------------------------------------------------------

    def discover(self) -> None:
        """Populate ``_speculation_state`` from discovered templates.

        Drops templates whose entry wasn't selected by ``seed_params``.
        """
        self._speculation_state = discover_speculate_functions(self.scraper)
        if self.seed_params is not None and self._speculation_state:
            selected = {name for inv in self.seed_params for name in inv}
            to_remove = [
                key
                for key, state in self._speculation_state.items()
                if state.base_func_name not in selected
            ]
            for key in to_remove:
                del self._speculation_state[key]

    @property
    def has_state(self) -> bool:
        """Whether any speculative templates were discovered."""
        return bool(self._speculation_state)

    async def load(self) -> None:
        """Load persisted speculation state from the DB for resumption.

        Updates ``_speculation_state`` with stored progress and reconstructs
        templates from ``template_json`` for states not in current discovery.
        A saved state that cannot be reconstructed (its entry is no longer
        speculative, or its template no longer validates) is skipped with a
        warning: the resumed run does not probe it.
        """
        saved_states = await self._db.load_all_speculation_states()

        for func_name, saved in saved_states.items():
            if func_name in self._speculation_state:
                spec_state = self._speculation_state[func_name]
                spec_state.tracking_id = saved.id
                spec_state.highest_successful_id = saved.highest_successful_id
                spec_state.consecutive_failures = saved.consecutive_failures
                spec_state.current_ceiling = saved.current_ceiling
                spec_state.stopped = saved.stopped
            elif saved.template_json:
                base_name = (
                    func_name.rsplit(":", 1)[0]
                    if ":" in func_name
                    else func_name
                )
                param_type = None
                for entry_info in self.scraper.list_speculative_entries():
                    if (
                        entry_info.func_name == base_name
                        and entry_info.speculative_param
                    ):
                        param_type = entry_info.param_types[
                            entry_info.speculative_param
                        ]
                        break

                if param_type is None:
                    logger.warning(
                        "No speculative entry %r for saved state %s, skipping",
                        base_name,
                        func_name,
                    )
                else:
                    try:
                        template = param_type.model_validate_json(
                            saved.template_json
                        )
                        self._speculation_state[func_name] = SpeculationState(
                            func_name=func_name,
                            tracking_id=saved.id,
                            template=template,
                            param_index=saved.param_index,
                            base_func_name=base_name,
                            highest_successful_id=saved.highest_successful_id,
                            consecutive_failures=saved.consecutive_failures,
                            current_ceiling=saved.current_ceiling,
                            stopped=saved.stopped,
                            seed_value=(
                                json.loads(saved.seed_value_json)
                                if saved.seed_value_json
                                else None
                            ),
                        )
                    except ValueError:
                        # pydantic's ValidationError and json's decode error
                        # are both ValueErrors.
                        logger.warning(
                            "Failed to deserialize template for %s, skipping",
                            func_name,
                            exc_info=True,
                        )

    async def persist_all(self) -> None:
        """Persist every speculation state (the final flush at close)."""
        for spec_state in self._speculation_state.values():
            await self._persist(spec_state)
