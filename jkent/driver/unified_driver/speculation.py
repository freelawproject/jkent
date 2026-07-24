"""Speculation support for the unified driver.

:class:`SpeculationManager` holds the per-step :class:`SpeculationState` dict
and implements the two abstract hooks of
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

from jkent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from jkent.data_types import BaseScraper, Request, Response
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
        """Serialize and insert a speculative probe into the queue."""
        request_data = self._queue.serialize_request(request)  # type: ignore[arg-type]
        # Spread the full serialized key set (as run.py's entry-request
        # enqueue does) so a speculative probe keeps every field — body, auth,
        # timeouts, redirects, bypass_rate_limit, cert, reseedable, etc. —
        # instead of silently dropping any not hand-listed here.
        await self._db.insert_request(  # type: ignore[misc]
            **request_data,
            priority=request.effective_priority,
            dedup_key=None,
            parent_id=None,
        )

    async def _after_outcome(self, spec_state: SpeculationState) -> None:
        """Persist updated speculation state inside the speculation lock."""
        # Speculation templates are pydantic models (Speculative subclasses
        # such as SpeculativeRange), so they always serialize. Persist the
        # JSON unconditionally so a resumed run can reconstruct a template
        # that the current discovery pass dropped — and so we never overwrite
        # a previously-stored template_json with NULL.
        template_json = cast(
            "BaseModel", spec_state.template
        ).model_dump_json()

        await self._db.save_speculation_state(
            func_name=spec_state.func_name,
            highest_successful_id=spec_state.highest_successful_id,
            consecutive_failures=spec_state.consecutive_failures,
            current_ceiling=spec_state.current_ceiling,
            stopped=spec_state.stopped,
            param_index=spec_state.param_index,
            template_json=template_json,
            seed_value_json=(
                json.dumps(spec_state.seed_value)
                if spec_state.seed_value is not None
                else None
            ),
        )

    # --- Lifecycle ------------------------------------------------------

    def discover(self) -> None:
        """Populate ``_speculation_state`` from discovered templates.

        Drops templates whose entry wasn't selected by ``seed_params``.
        """
        self._speculation_state = self._discover_speculate_functions()
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
        """
        saved_states = await self._db.load_all_speculation_states()

        for func_name, saved in saved_states.items():
            if func_name in self._speculation_state:
                spec_state = self._speculation_state[func_name]
                spec_state.highest_successful_id = saved[
                    "highest_successful_id"
                ]
                spec_state.consecutive_failures = saved["consecutive_failures"]
                spec_state.current_ceiling = saved["current_ceiling"]
                spec_state.stopped = saved["stopped"]
            elif "template_json" in saved and saved["template_json"]:
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

                if param_type is not None:
                    try:
                        template = param_type.model_validate_json(
                            saved["template_json"]
                        )
                        seed_value_json = saved.get("seed_value_json")
                        self._speculation_state[func_name] = SpeculationState(
                            func_name=func_name,
                            template=template,
                            param_index=saved["param_index"],
                            base_func_name=base_name,
                            highest_successful_id=saved[
                                "highest_successful_id"
                            ],
                            consecutive_failures=saved["consecutive_failures"],
                            current_ceiling=saved["current_ceiling"],
                            stopped=saved["stopped"],
                            seed_value=(
                                json.loads(seed_value_json)
                                if seed_value_json
                                else None
                            ),
                        )
                    except Exception:
                        logger.warning(
                            "Failed to deserialize template for %s, skipping",
                            func_name,
                        )

    async def seed(self) -> None:
        """Seed the queue with the initial speculative probe window."""
        await self._seed_speculative_queue()

    async def adopt_untracked(self, other: SpeculationManager) -> None:
        """Seed and absorb ``other``'s states that this manager isn't tracking.

        The run already seeded the states this manager holds, and a speculative
        probe insert carries no dedup key — so re-seeding a state already
        tracked here would double-probe it. Drop the states ``other`` shares
        with this manager, seed the genuinely new remainder onto the shared
        queue, and merge them in.
        """
        for key in list(other._speculation_state):
            if key in self._speculation_state:
                del other._speculation_state[key]
        if not other.has_state:
            return
        await other.seed()
        self._speculation_state.update(other._speculation_state)

    async def track_outcome(
        self, request: Request, response: Response
    ) -> None:
        """Record a speculative probe outcome and extend/stop as needed."""
        await self._track_speculation_outcome(request, response)

    async def persist_all(self) -> None:
        """Persist every speculation state (final flush, like old close)."""
        for spec_state in self._speculation_state.values():
            await self._after_outcome(spec_state)
