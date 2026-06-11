"""Shared fixtures for unified-driver tests.

``memory_session_factory`` stands up a real, fully-migrated SQLite schema
entirely in memory (StaticPool so every session shares the one connection),
giving DB-backed contract tests a fast, isolated database with no temp files.

``schema_template`` is a once-built, empty, fully-migrated DB *file* that the
replay/archive rigs copy per hypothesis example (the replay ``SourceIndex``
opens source DBs read-only, so they must be real files).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from jkent.driver.database_engine.database import (
    get_session_factory,
    init_database,
)
from jkent.driver.database_engine.migrations import migrate_to
from jkent.driver.database_engine.scoped_session import ScopedSessionFactory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture(scope="session")
def schema_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A once-built, empty, fully-migrated DB file to copy per example."""
    path = tmp_path_factory.mktemp("schema_template") / "template.db"

    async def build() -> None:
        engine, _ = await init_database(path)
        await engine.dispose()

    asyncio.run(build())
    return path


@pytest.fixture
async def memory_session_factory() -> AsyncIterator[ScopedSessionFactory]:
    """An initialized in-memory SQLite DB, shared across sessions."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await migrate_to(engine)

    try:
        yield ScopedSessionFactory(get_session_factory(engine))
    finally:
        await engine.dispose()
