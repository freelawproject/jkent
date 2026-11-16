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

import logging
from typing import TYPE_CHECKING, Any, cast

from jkent.common.exceptions import ScraperConfigError
from jkent.common.speculative import PersistedSpeculative
from jkent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
    discover_speculate_functions,
)
from jkent.driver.database_engine.sql_manager import (
    SeedParams,
    SpeculationStateInsert,
)

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
        seed_params: SeedParams | None = None,
    ) -> None:
        """Bind the manager to one run's scraper, queue, and database.

        Args:
            scraper: The scraper whose speculative steps are managed.
            queue: The request queue speculative probes are inserted into.
            db: The run database holding persisted speculation state.
            seed_params: The run's seed parameters, or ``None``.
        """
        super().__init__(scraper)
        self.seed_params = seed_params
        self._queue = queue
        self._db = db

    # --- Hook implementations -------------------------------------------

    async def _enqueue_speculative(self, request: Request) -> None:
        """Insert a speculative probe into the queue, never deduplicated."""
        await self._queue.insert_root_request(request, deduplicate=False)

    async def _persist(self, spec_state: SpeculationState) -> int:
        """Upsert *spec_state* and return its tracking row id.

        Also creates the tracking row before a state's first probe: the
        upsert writes the state as it stands (fresh counters on a first run,
        the loaded ones on a resume).

        Returns:
            The tracking row id, which probes built from the state carry.
        """
        # Speculation templates are pydantic models (Speculative subclasses
        # such as SpeculativeRange), so they always serialize. Persist the
        # JSON unconditionally so a resumed run can reconstruct a template
        # that the current discovery pass dropped — and so we never overwrite
        # a previously-stored template_json with NULL.
        template_json = cast(
            "BaseModel", spec_state.template
        ).model_dump_json()

        return await self._db.save_speculation_state(
            SpeculationStateInsert(
                func_name=spec_state.func_name,
                highest_successful_id=spec_state.highest_successful_id,
                consecutive_failures=spec_state.consecutive_failures,
                current_ceiling=spec_state.current_ceiling,
                stopped=spec_state.stopped,
                param_index=spec_state.param_index,
                template_json=template_json,
            )
        )

    # --- Lifecycle ------------------------------------------------------

    def discover(self) -> None:
        """Populate ``_speculation_state`` from discovered templates.

        Drops templates whose entry wasn't selected by ``seed_params``.

        Raises:
            ScraperConfigError: Two templates were seeded from the same
                persisted cursor key, so there is no one cursor to hand
                back for it.
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
        seen: dict[str, str] = {}
        for state_key, spec_state in self._speculation_state.items():
            template = spec_state.template
            if not isinstance(template, PersistedSpeculative):
                continue
            persist_key = template.persist_key
            if persist_key is None:
                continue
            if persist_key in seen:
                raise ScraperConfigError(
                    f"speculative templates {seen[persist_key]!r} and "
                    f"{state_key!r} were both seeded from [{persist_key}]; "
                    "each persisted cursor may seed one template"
                )
            seen[persist_key] = state_key

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
                self._bind(spec_state, saved.id)
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
                        spec_state = SpeculationState(
                            func_name=func_name,
                            template=template,
                            param_index=saved.param_index,
                            base_func_name=base_name,
                            highest_successful_id=saved.highest_successful_id,
                            consecutive_failures=saved.consecutive_failures,
                            current_ceiling=saved.current_ceiling,
                            stopped=saved.stopped,
                        )
                    except ValueError:
                        # pydantic's ValidationError and json's decode error
                        # are both ValueErrors.
                        logger.warning(
                            "Failed to deserialize template for %s, skipping",
                            func_name,
                            exc_info=True,
                        )
                    else:
                        self._speculation_state[func_name] = spec_state
                        self._bind(spec_state, saved.id)

    def persisted_cursors(self) -> dict[str, PersistedSpeculative]:
        """Each persisted template's cursor for the next run, by key.

        Read-only: computed from the in-memory states, writing nothing.

        Returns:
            ``{persist_key: template.resume_after(highest_successful_id)}``
            for every template seeded from a ``[key]`` reference.
        """
        cursors: dict[str, PersistedSpeculative] = {}
        for spec_state in self._speculation_state.values():
            template = spec_state.template
            if (
                isinstance(template, PersistedSpeculative)
                and template.persist_key is not None
            ):
                cursors[template.persist_key] = template.resume_after(
                    spec_state.highest_successful_id
                )
        return cursors

    async def persist_all(self) -> None:
        """Persist every speculation state (the final flush at close)."""
        for spec_state in self._speculation_state.values():
            await self._save(spec_state)
