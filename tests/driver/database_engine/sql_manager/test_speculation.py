"""Tests for speculation tracking operations (_speculation.py)."""

from __future__ import annotations

import sqlalchemy as sa

from jkent.driver.database_engine.models import SpeculationTracking
from jkent.driver.database_engine.sql_manager import (
    SpeculationStateRecord,
    SQLManager,
)


async def test_upsert_keeps_the_row_id_and_rewrites_every_field(
    sql_manager: SQLManager,
) -> None:
    """Probe rows hold the tracking id as a foreign key, so saving progress
    must update the row it names rather than mint a new one."""
    first = await sql_manager.save_speculation_state(
        SpeculationStateRecord(func_name="fetch:0", template_json="{}")
    )
    progressed = SpeculationStateRecord(
        func_name="fetch:0",
        highest_successful_id=7,
        consecutive_failures=2,
        current_ceiling=12,
        stopped=True,
        param_index=0,
        template_json='{"t": 1}',
        seed_value_json='"[cursor]"',
    )
    second = await sql_manager.save_speculation_state(progressed)

    assert second == first
    saved = await sql_manager.load_all_speculation_states()
    assert list(saved) == ["fetch:0"]
    assert saved["fetch:0"] == progressed.model_copy(update={"id": first})


async def test_upsert_restamps_updated_at(sql_manager: SQLManager) -> None:
    await sql_manager.save_speculation_state(
        SpeculationStateRecord(func_name="fetch:0")
    )
    stale = "2000-01-01 00:00:00.000"
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.update(SpeculationTracking).values(
                updated_at=sa.literal_column(f"'{stale}'")
            )
        )
        await session.commit()

    await sql_manager.save_speculation_state(
        SpeculationStateRecord(func_name="fetch:0", current_ceiling=1)
    )

    async with sql_manager.session_factory() as session:
        stamped = (
            await session.execute(
                sa.select(sa.cast(SpeculationTracking.updated_at, sa.Text))
            )
        ).scalar_one()
    assert stamped > stale


async def test_load_keys_every_template_by_func_name(
    sql_manager: SQLManager,
) -> None:
    ids = {
        name: await sql_manager.save_speculation_state(
            SpeculationStateRecord(func_name=name, param_index=i)
        )
        for i, name in enumerate(["fetch:0", "fetch:1", "other:0"])
    }

    saved = await sql_manager.load_all_speculation_states()

    assert {name: s.id for name, s in saved.items()} == ids
    assert len(set(ids.values())) == 3
    assert saved["fetch:1"].param_index == 1
