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
import logging
from typing import TYPE_CHECKING, Any

from jkent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
)

if TYPE_CHECKING:
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
        request_data = self._queue._serialize_request(request)  # type: ignore[arg-type]
        await self._db.insert_request(
            priority=request.effective_priority,
            request_type=request_data["request_type"],
            method=request_data["method"],
            url=request_data["url"],
            headers_json=request_data["headers_json"],
            cookies_json=request_data["cookies_json"],
            body=request_data["body"],
            continuation=request_data["continuation"],
            current_location=request_data["current_location"],
            accumulated_data_json=request_data["accumulated_data_json"],
            permanent_json=request_data["permanent_json"],
            expected_type=request_data["expected_type"],
            dedup_key=None,
            parent_id=None,
            is_speculative=request_data["is_speculative"],
            speculation_id=request_data["speculation_id"],
            verify=request_data.get("verify"),
        )

    async def _after_outcome(self, spec_state: SpeculationState) -> None:
        """Persist updated speculation state inside the speculation lock."""
        template_json = None
        if hasattr(spec_state.template, "model_dump_json"):
            template_json = spec_state.template.model_dump_json()  # type: ignore

        await self._db.save_speculation_state(
            func_name=spec_state.func_name,
            highest_successful_id=spec_state.highest_successful_id,
            consecutive_failures=spec_state.consecutive_failures,
            current_ceiling=spec_state.current_ceiling,
            stopped=spec_state.stopped,
            param_index=spec_state.param_index,
            template_json=template_json,
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

                if param_type is not None and hasattr(
                    param_type, "model_validate_json"
                ):
                    try:
                        template = param_type.model_validate_json(
                            saved["template_json"]
                        )
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
                        )
                    except Exception:
                        logger.warning(
                            "Failed to deserialize template for %s, skipping",
                            func_name,
                        )

    async def seed(self) -> None:
        """Seed the queue with the initial speculative probe window."""
        await self._seed_speculative_queue()

    async def track_outcome(
        self, request: Request, response: Response
    ) -> None:
        """Record a speculative probe outcome and extend/stop as needed."""
        await self._track_speculation_outcome(request, response)

    async def persist_all(self) -> None:
        """Persist every speculation state (final flush, like old close)."""
        for spec_state in self._speculation_state.values():
            await self._after_outcome(spec_state)
