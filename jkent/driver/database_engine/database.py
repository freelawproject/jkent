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
from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

from jkent.driver.database_engine.models import Base
from jkent.driver.database_engine.timestamps import require_subsec_support

# For future migrations if they should become necessary
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
            :func:`create_async_engine` — e.g. ``poolclass=NullPool`` for
            hosts that want connection-per-session semantics instead of the
            default persistent pool (the pool-sizing defaults are dropped
            automatically for pool classes that don't take them).

    Returns:
        An initialized AsyncEngine.
    """
    # Checked here rather than at import so the failure names the moment a
    # database is actually opened. Cached after the first call.
    require_subsec_support()

    url = f"sqlite+aiosqlite:///{db_path}"
    kwargs: dict[str, Any] = {
        "echo": echo,
        "connect_args": {"check_same_thread": False},
        # A persistent pool, not NullPool: with aiosqlite every connection is
        # a dedicated OS thread, so connection-per-session made each session
        # checkout a thread spawn + sqlite3.connect + PRAGMA setup — profiled
        # as dominating wall time on DB-chatty runs (jent replay). A few
        # long-lived threads are the cheaper trade.
        #
        # max_overflow=-1 (unbounded, overflow closed on release) preserves
        # NullPool's never-wait semantics: SQLManager methods can open a
        # session while their caller already holds one, so a bounded pool
        # could deadlock at exhaustion.
        "poolclass": AsyncAdaptedQueuePool,
        "pool_size": 5,
        "max_overflow": -1,
        **engine_kwargs,
    }
    # The sizing defaults only apply to queue pools; a host that overrides
    # poolclass (e.g. NullPool) must not inherit kwargs its pool rejects.
    if not issubclass(kwargs["poolclass"], QueuePool):
        for key in ("pool_size", "max_overflow"):
            if key not in engine_kwargs:
                kwargs.pop(key, None)
    engine = create_async_engine(url, **kwargs)

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        # WAL's default synchronous is FULL — an fsync on every commit, and
        # the request lifecycle commits several times per request. NORMAL in
        # WAL mode syncs only at checkpoint: an app crash loses nothing, a
        # power loss can lose the last few commits — fine for a resumable
        # run database.
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
