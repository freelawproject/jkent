"""Write transactions open ``BEGIN IMMEDIATE`` (``database.write_session``).

SQLite's deferred transaction takes its read snapshot at the first statement
and only then asks for the write lock. If another connection commits while it
waits, SQLite answers SQLITE_BUSY_SNAPSHOT *immediately* — the busy handler is
never called, so ``busy_timeout`` cannot absorb it and the caller sees
"database is locked". Writers therefore take the write lock up front.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import event

from jkent.driver.database_engine.database import init_database, write_session

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine


def _record_statements(engine: AsyncEngine, into: list[str]) -> None:
    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def _capture(
        _conn: Any,
        _cursor: Any,
        statement: str,
        *_args: Any,
    ) -> None:
        into.append(statement)


async def test_writers_begin_immediate_readers_begin_deferred(
    tmp_path: Path,
) -> None:
    engine, factory = await init_database(tmp_path / "run.db")
    statements: list[str] = []
    _record_statements(engine, statements)
    try:
        async with factory() as session:
            await session.execute(sa.text("SELECT count(*) FROM schema_info"))
        assert "BEGIN" in statements
        assert "BEGIN IMMEDIATE" not in statements

        statements.clear()
        async with write_session(factory, asyncio.Lock()) as session:
            await session.execute(
                sa.text("INSERT INTO schema_info (version) VALUES (2)")
            )
            await session.commit()
        # The lock is taken on entry, before any statement of our own.
        assert statements[0] == "BEGIN IMMEDIATE"
    finally:
        await engine.dispose()


async def test_write_session_holds_the_lock_it_is_given(
    tmp_path: Path,
) -> None:
    engine, factory = await init_database(tmp_path / "run.db")
    lock = asyncio.Lock()
    try:
        async with write_session(factory, lock):
            assert lock.locked()
        assert not lock.locked()
    finally:
        await engine.dispose()


async def test_read_then_write_survives_a_commit_from_another_handle(
    tmp_path: Path,
) -> None:
    """The regression: two handles on one file, one reading before writing.

    ``slow`` selects and then writes in the same transaction, with a window in
    the middle for ``other`` — a separate engine, i.e. a separate connection
    and a separate lock, as a browser transport's handle once was — to commit.
    Under a deferred BEGIN that window invalidates ``slow``'s snapshot and its
    write fails outright; under BEGIN IMMEDIATE ``other`` simply waits.
    """
    db_path = tmp_path / "run.db"
    slow_engine, slow_factory = await init_database(db_path)
    other_engine, other_factory = await init_database(db_path)
    selected = asyncio.Event()

    async def slow() -> None:
        async with write_session(slow_factory, asyncio.Lock()) as session:
            await session.execute(sa.text("SELECT count(*) FROM schema_info"))
            selected.set()
            # Long enough for `other` to commit if nothing is holding it off.
            await asyncio.sleep(0.2)
            await session.execute(
                sa.text("INSERT INTO schema_info (version) VALUES (2)")
            )
            await session.commit()

    async def other() -> None:
        await selected.wait()
        async with write_session(other_factory, asyncio.Lock()) as session:
            await session.execute(
                sa.text("INSERT INTO schema_info (version) VALUES (3)")
            )
            await session.commit()

    try:
        await asyncio.gather(slow(), other())

        async with slow_factory() as session:
            versions = (
                (
                    await session.execute(
                        sa.text("SELECT version FROM schema_info ORDER BY id")
                    )
                )
                .scalars()
                .all()
            )
        assert sorted(versions) == [1, 2, 3]
    finally:
        await slow_engine.dispose()
        await other_engine.dispose()
