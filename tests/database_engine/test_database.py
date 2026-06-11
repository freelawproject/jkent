"""Engine construction + schema baseline stamping (``database.py``).

The migration runner is gone (one schema version pre-0.1.0), so these pin
the two behaviors that replaced it — a fresh database is stamped at
``BASELINE_VERSION`` and an already-stamped database is left alone — plus
the ``**engine_kwargs`` pass-through hosts use to swap the pool class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlalchemy.pool import AsyncAdaptedQueuePool, NullPool

from jkent.driver.database_engine.database import (
    BASELINE_VERSION,
    init_database,
)

if TYPE_CHECKING:
    from pathlib import Path


async def test_fresh_database_is_stamped_at_baseline(tmp_path: Path) -> None:
    engine, factory = await init_database(tmp_path / "run.db")
    try:
        async with factory() as session:
            versions = (
                (
                    await session.execute(
                        sa.text("SELECT version FROM schema_info")
                    )
                )
                .scalars()
                .all()
            )
        assert versions == [BASELINE_VERSION]
    finally:
        await engine.dispose()


async def test_reopen_does_not_restamp(tmp_path: Path) -> None:
    db_path = tmp_path / "run.db"
    engine, _ = await init_database(db_path)
    await engine.dispose()

    engine, factory = await init_database(db_path)
    try:
        async with factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT count(*) FROM schema_info")
                )
            ).scalar()
        assert count == 1
    finally:
        await engine.dispose()


async def test_engine_kwargs_override_pool_class(tmp_path: Path) -> None:
    """Hosts can forward pool kwargs (jent's replay engine relies on this)."""
    default_engine, _ = await init_database(tmp_path / "a.db")
    pooled_engine, factory = await init_database(
        tmp_path / "b.db",
        poolclass=AsyncAdaptedQueuePool,
        pool_size=2,
        max_overflow=-1,
    )
    try:
        assert isinstance(default_engine.pool, NullPool)
        assert isinstance(pooled_engine.pool, AsyncAdaptedQueuePool)
        # PRAGMAs and schema init apply to custom-pool engines too.
        async with factory() as session:
            journal = (
                await session.execute(sa.text("PRAGMA journal_mode"))
            ).scalar()
            tables = (
                await session.execute(
                    sa.text(
                        "SELECT count(*) FROM sqlite_master WHERE type='table'"
                    )
                )
            ).scalar()
        assert journal == "wal"
        assert tables > 0
    finally:
        await default_engine.dispose()
        await pooled_engine.dispose()
