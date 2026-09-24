"""Database engine and session management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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

if TYPE_CHECKING:
    import asyncio
    from collections.abc import AsyncIterator, Mapping

    from sqlalchemy import Connection
    from sqlalchemy.engine import CursorResult
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
    from sqlalchemy.pool import ConnectionPoolEntry
    from sqlalchemy.sql import Executable

#: The schema this jkent reads and writes. A database stamped at any other
#: version is refused at open: there is no migration runner, so the only
#: supported answer is to recreate the run database.
BASELINE_VERSION = 3


class UnsupportedSchemaVersionError(RuntimeError):
    """A run database was stamped at a schema version jkent cannot read.

    Raised at open, before any table is touched, so the operator hears about
    it at startup rather than as ``no such column`` inside a worker hours in.
    """

    def __init__(self, db_path: Path, found: int) -> None:
        self.db_path = db_path
        self.found = found
        super().__init__(
            f"{db_path} is stamped at schema version {found}; this jkent "
            f"reads version {BASELINE_VERSION}. There is no migration "
            f"runner — recreate the run database."
        )


#: Execution option that opens a transaction as ``BEGIN IMMEDIATE`` instead of
#: SQLite's default deferred ``BEGIN``. Set by :func:`write_session`; read by
#: the engine's ``begin`` handler in :func:`create_engine_and_init`.
BEGIN_IMMEDIATE_OPTION = "jkent_begin_immediate"


async def _stamped_version(conn: AsyncConnection) -> int | None:
    """The version in ``schema_info``, or None for a database with no stamp.

    None covers both a brand-new file and one whose ``schema_info`` table
    does not exist yet — either way there is nothing to refuse and the
    caller stamps it.
    """
    has_table = (
        await conn.execute(
            sa.text(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'schema_info'"
            )
        )
    ).scalar()
    if not has_table:
        return None
    version: int | None = (
        await conn.execute(sa.text("SELECT MAX(version) FROM schema_info"))
    ).scalar()
    return version


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
        "connect_args": {"check_same_thread": False, "isolation_level": None},
        # A persistent pool, not NullPool: with aiosqlite every connection is
        # a dedicated OS thread, so connection-per-session made each session
        # checkout a thread spawn + sqlite3.connect + PRAGMA setup — profiled
        # as dominating wall time on DB-chatty runs (replay hosts). A few
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
    def _set_sqlite_pragma(
        dbapi_conn: DBAPIConnection, connection_record: ConnectionPoolEntry
    ) -> None:
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

    @event.listens_for(engine.sync_engine, "begin")
    def _begin(conn: Connection) -> None:
        if conn.get_execution_options().get(BEGIN_IMMEDIATE_OPTION, False):
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.exec_driver_sql("BEGIN")

    async with engine.connect() as conn:
        await conn.execution_options(**{BEGIN_IMMEDIATE_OPTION: True})
        async with conn.begin():
            # Check the stamp before create_all: create_all adds missing
            # tables but never missing columns, so a database from another
            # schema version opens cleanly here and fails much later, inside
            # a worker, as "no such column".
            existing = await _stamped_version(conn)
            if existing is not None and existing != BASELINE_VERSION:
                raise UnsupportedSchemaVersionError(db_path, existing)

            await conn.run_sync(Base.metadata.create_all)

            if existing is None:
                await conn.execute(
                    sa.text("INSERT INTO schema_info (version) VALUES (:v)"),
                    {"v": BASELINE_VERSION},
                )

    return engine


def get_session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
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
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
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


async def execute_rowcount(session: AsyncSession, stmt: Executable) -> int:
    """Execute a Core UPDATE/DELETE and return how many rows it touched.

    ``session.execute``'s typed overload promises a ``Result``, which has no
    ``rowcount``; what an UPDATE actually returns is a ``CursorResult``.
    """
    result = cast("CursorResult[Any]", await session.execute(stmt))
    return result.rowcount


@asynccontextmanager
async def write_session(
    session_factory: async_sessionmaker[AsyncSession],
    lock: asyncio.Lock,
) -> AsyncIterator[AsyncSession]:
    """Open a session for a write transaction, serialized by ``lock``.

    The single entry point for every mutation of a run database. It does two
    things a bare ``session_factory()`` does not:

    - holds ``lock``, serializing writers that share this manager, and
    - opens the transaction as ``BEGIN IMMEDIATE`` (see the ``begin`` handler
      in :func:`create_engine_and_init`), so a writer racing a *different*
      connection on the same file — another process, or a second handle in
      this one — waits out ``busy_timeout`` instead of failing outright with
      "database is locked".

    The caller still commits; the session rolls back on the way out if it
    doesn't.

    Example::

        async with write_session(self.session_factory, self.lock) as session:
            session.add(row)
            await session.commit()
    """
    async with lock, session_factory() as session:
        begin_immediate: Mapping[str, bool] = {BEGIN_IMMEDIATE_OPTION: True}
        await session.connection(execution_options=begin_immediate)
        yield session
