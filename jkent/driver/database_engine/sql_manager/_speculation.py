"""Speculation tracking operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from jkent.driver.database_engine.models import (
    SpeculationTracking,
)

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker


class SpeculationStateDict(TypedDict):
    """Typed dict for speculation tracking state returned from DB."""

    func_name: str
    highest_successful_id: int
    consecutive_failures: int
    current_ceiling: int
    stopped: bool
    param_index: int
    template_json: str | None
    seed_value_json: str | None


def _row_to_speculation_state(row: Any) -> SpeculationStateDict:
    """Map a SpeculationTracking row to its state dict."""
    return SpeculationStateDict(
        func_name=row.func_name,
        highest_successful_id=row.highest_successful_id,
        consecutive_failures=row.consecutive_failures,
        current_ceiling=row.current_ceiling,
        stopped=bool(row.stopped),
        param_index=row.param_index,
        template_json=row.template_json,
        seed_value_json=row.seed_value_json,
    )


class SpeculationMixin:
    """SpeculationTracking operations for the Speculative protocol."""

    _lock: asyncio.Lock  # type: ignore[misc]
    _session_factory: async_sessionmaker  # type: ignore[misc]

    # --- Speculation Tracking ---

    async def save_speculation_state(
        self,
        func_name: str,
        highest_successful_id: int,
        consecutive_failures: int,
        current_ceiling: int,
        stopped: bool,
        param_index: int = 0,
        template_json: str | None = None,
        seed_value_json: str | None = None,
    ) -> None:
        """Save or update speculation tracking state.

        Args:
            func_name: State key (e.g. "fetch_case:0").
            highest_successful_id: Highest ID that returned 2xx.
            consecutive_failures: Count of failures beyond highest_successful_id.
            current_ceiling: Current upper bound of seeded IDs.
            stopped: Whether speculation has stopped for this entry.
            param_index: Index of this template in the params list.
            template_json: JSON serialization of the Speculative template.
            seed_value_json: JSON of the raw seed value the template came
                from (e.g. a host's "[cursor_key]" reference).
        """
        async with self._lock, self._session_factory() as session:
            stmt = sqlite_insert(SpeculationTracking).values(
                func_name=func_name,
                highest_successful_id=highest_successful_id,
                consecutive_failures=consecutive_failures,
                current_ceiling=current_ceiling,
                stopped=stopped,
                param_index=param_index,
                template_json=template_json,
                seed_value_json=seed_value_json,
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["func_name"],
                set_={
                    "highest_successful_id": stmt.excluded.highest_successful_id,
                    "consecutive_failures": stmt.excluded.consecutive_failures,
                    "current_ceiling": stmt.excluded.current_ceiling,
                    "stopped": stmt.excluded.stopped,
                    "param_index": stmt.excluded.param_index,
                    "template_json": stmt.excluded.template_json,
                    "seed_value_json": stmt.excluded.seed_value_json,
                    "updated_at": func.current_timestamp(),
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def load_all_speculation_states(
        self,
    ) -> dict[str, SpeculationStateDict]:
        """Load all speculation tracking states.

        Returns:
            Dict mapping func_name to their state dict.
        """
        async with self._session_factory() as session:
            result = await session.execute(select(SpeculationTracking))
            rows = result.scalars().all()
            return {
                row.func_name: _row_to_speculation_state(row) for row in rows
            }
