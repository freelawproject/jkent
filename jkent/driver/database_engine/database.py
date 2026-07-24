"""Database engine and session management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

from jkent.driver.database_engine.models import *  # noqa: F403

# The schema version that ``create_all`` (i.e. the current models) produces.
# There is only one version pre-0.1.0 — the day a second schema version
# exists, reintroduce a migration runner keyed on ``schema_info`` (the old
# one lived at ``database_engine/migrations`` until 2026-07).
BASELINE_VERSION = 1


async def create_engine_and_init(
    db_path: Path,
    echo: bool = False,
    **engine_kwargs: Any,
) -> AsyncEngine:
    """Create an async engine and initialize the database schema.

    Creates all tables if they don't exist. Configures WAL mode and
    foreign keys via connection event listeners.

    Args:
        db_path: Path to the SQLite database file.
        echo: Whether to echo SQL statements (for debugging).
        **engine_kwargs: Overrides merged over the defaults and passed to
            :func:`create_async_engine` — e.g. ``poolclass``/``pool_size``
            for hosts that want a connection pool instead of the default
            ``NullPool`` (with aiosqlite every pooled connection is a
            dedicated OS thread kept alive for the engine's lifetime).

    Returns:
        An initialized AsyncEngine.
    """
    url = f"sqlite+aiosqlite:///{db_path}"
    kwargs: dict[str, Any] = {
        "echo": echo,
        "connect_args": {"check_same_thread": False},
        "poolclass": NullPool,
        **engine_kwargs,
    }
    engine = create_async_engine(url, **kwargs)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

        # Stamp a fresh database at the baseline so any future migration
        # runner knows where to start; an already-stamped database is left
        # alone.
        current = (
            await conn.execute(sa.text("SELECT MAX(version) FROM schema_info"))
        ).scalar()
        if not current:
            await conn.execute(
                sa.text("INSERT INTO schema_info (version) VALUES (:v)"),
                {"v": BASELINE_VERSION},
            )

    return engine


def get_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker:
    """Create a session factory bound to the engine.

    Args:
        engine: The async engine to bind sessions to.

    Returns:
        An async session factory.
    """
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_database(
    db_path: Path,
    echo: bool = False,
    **engine_kwargs: Any,
) -> tuple[AsyncEngine, async_sessionmaker]:
    """Initialize database and return engine + session factory.

    This is the main entry point, replacing schema.init_database().

    Args:
        db_path: Path to the SQLite database file.
        echo: Whether to echo SQL statements.
        **engine_kwargs: Engine overrides forwarded to
            :func:`create_engine_and_init` (e.g. ``poolclass``).

    Returns:
        Tuple of (engine, session_factory).
    """
    engine = await create_engine_and_init(db_path, echo=echo, **engine_kwargs)
    return engine, get_session_factory(engine)
