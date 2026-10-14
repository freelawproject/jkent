"""Speculation tracking operations for SQLManager."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from jkent.driver.database_engine.models import SpeculationTracking
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import RowModel
from jkent.driver.database_engine.timestamps import now


class SpeculationStateRecord(RowModel):
    """One ``speculation_tracking`` row, in both directions.

    :meth:`SpeculationMixin.save_speculation_state` upserts one (``id`` is
    ignored on the way in — ``func_name`` is the conflict key) and
    :meth:`SpeculationMixin.load_all_speculation_states` reads them back.
    """

    id: int | None = None
    #: State key, e.g. ``"fetch_case:0"``.
    func_name: str
    #: Highest ID that returned 2xx; None before any has.
    highest_successful_id: int | None = None
    #: Failures counted past ``highest_successful_id``.
    consecutive_failures: int = 0
    #: Current upper bound of seeded IDs; None before seeding.
    current_ceiling: int | None = None
    #: Whether speculation has stopped for this entry.
    stopped: bool = False
    #: Index of this template in the params list.
    param_index: int = 0
    #: JSON serialization of the Speculative template.
    template_json: str | None = None
    #: JSON of the raw seed value the template came from (e.g. a host's
    #: "[cursor_key]" reference).
    seed_value_json: str | None = None


class SpeculationMixin(SQLManagerBase):
    """SpeculationTracking operations for the Speculative protocol."""

    async def save_speculation_state(
        self, state: SpeculationStateRecord
    ) -> int:
        """Save or update speculation tracking state.

        Returns the tracking row's id, which requests built from this
        template carry as ``requests.speculation_tracking_id`` — so callers
        can use this both to persist progress and to create the row up front,
        before its probes are enqueued.
        """
        values = state.model_dump(exclude={"id"})
        async with self._write_session() as session:
            stmt = sqlite_insert(SpeculationTracking).values(**values)
            upsert = stmt.on_conflict_do_update(
                index_elements=["func_name"],
                set_={
                    **{
                        column: getattr(stmt.excluded, column)
                        for column in values
                        if column != "func_name"
                    },
                    # Explicit: SQLAlchemy does not apply ``onupdate`` to an
                    # ON CONFLICT DO UPDATE SET clause, so the column default
                    # never fires on this path.
                    "updated_at": now(),
                },
            ).returning(SpeculationTracking.id)
            # DO UPDATE (never DO NOTHING) means the conflicting row is always
            # rewritten, so RETURNING yields a row on both the insert and the
            # update path — no second lookup to resolve the id.
            row_id = (await session.execute(upsert)).scalar_one()
            await session.commit()
            return row_id

    async def load_all_speculation_states(
        self,
    ) -> dict[str, SpeculationStateRecord]:
        """Load all speculation tracking states, keyed by ``func_name``."""
        async with self.session_factory() as session:
            result = await session.execute(select(SpeculationTracking))
            return {
                row.func_name: SpeculationStateRecord.model_validate(row)
                for row in result.scalars().all()
            }
