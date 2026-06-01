"""Speculation tracking operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from kent.driver.persistent_driver.models import (
    SpeculationTracking,
)

if TYPE_CHECKING:
    import asyncio

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class SpeculationStateDict(TypedDict):
    """Typed dict for speculation tracking state returned from DB."""

    func_name: str
    highest_successful_id: int
    consecutive_failures: int
    current_ceiling: int
    stopped: bool
    param_index: int
    template_json: str | None


class SpeculationMixin:
    """SpeculationTracking operations for the Speculative protocol."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

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
                    "updated_at": func.current_timestamp(),
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def load_speculation_state(
        self, func_name: str
    ) -> SpeculationStateDict | None:
        """Load speculation tracking state for a function.

        Args:
            func_name: State key.

        Returns:
            Dict with tracking fields, or None if no state exists.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(SpeculationTracking).where(
                    SpeculationTracking.func_name == func_name  # type: ignore[arg-type]
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "func_name": row.func_name,
                "highest_successful_id": row.highest_successful_id,
                "consecutive_failures": row.consecutive_failures,
                "current_ceiling": row.current_ceiling,
                "stopped": bool(row.stopped),
                "param_index": row.param_index,
                "template_json": row.template_json,
            }

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
                row.func_name: SpeculationStateDict(
                    func_name=row.func_name,
                    highest_successful_id=row.highest_successful_id,
                    consecutive_failures=row.consecutive_failures,
                    current_ceiling=row.current_ceiling,
                    stopped=bool(row.stopped),
                    param_index=row.param_index,
                    template_json=row.template_json,
                )
                for row in rows
            }

    async def get_all_speculation_progress(
        self,
    ) -> dict[str, int]:
        """Get highest_successful_id for all speculation tracking entries.

        Returns:
            Dict mapping func_name to their highest_successful_id.
        """
        states = await self.load_all_speculation_states()
        return {
            func_name: state["highest_successful_id"]
            for func_name, state in states.items()
        }

    async def clear_speculation_state(self, func_name: str) -> None:
        """Clear speculation tracking state for a function.

        Args:
            func_name: State key.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                delete(SpeculationTracking).where(
                    SpeculationTracking.func_name == func_name  # type: ignore[arg-type]
                )
            )
            await session.commit()

    async def clear_all_speculation_states(self) -> None:
        """Clear all speculation tracking states."""
        async with self._lock, self._session_factory() as session:
            await session.execute(delete(SpeculationTracking))
            await session.commit()
