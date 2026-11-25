"""Database engine and session management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import quote

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
    from collections.abc import AsyncIterator, Iterable, Mapping

    from sqlalchemy import Connection, Delete, Update
    from sqlalchemy.engine import CursorResult
    from sqlalchemy.engine.interfaces import DBAPIConnection
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
    from sqlalchemy.pool import ConnectionPoolEntry

# For future migrations if they should become necessary
BASELINE_VERSION = 1


class UnsupportedSchemaVersionError(RuntimeError):
    """A run database was stamped at a schema version jkent cannot read.

    Raised at open, before any table is touched, so the operator hears about
    it at startup rather than as ``no such column`` inside a worker hours in.
    """

    def __init__(self, db_path: Path, found: int) -> None:
        """Initialize the error.

        Args:
            db_path: The run database that was opened.
            found: The schema version stamped in it.
        """
        self.db_path = db_path
        self.found = found
        super().__init__(
            f"{db_path} is stamped at schema version {found}; this jkent "
            f"reads version {BASELINE_VERSION}. There is no migration "
            f"runner — recreate the run database."
        )


class NotARunDatabaseError(RuntimeError):
    """A file opened read-only is missing the run-database schema.

    A read-only open cannot create what is missing, and reading on would
    only fail later as ``no such table`` — so it is refused at open.
    """

    def __init__(self, db_path: Path, missing: Iterable[str]) -> None:
        """Initialize the error.

        Args:
            db_path: The file that was opened.
            missing: The tables it lacks (``schema_info`` among them when
                the file carries no schema stamp).
        """
        self.db_path = db_path
        self.missing = sorted(missing)
        super().__init__(
            f"{db_path} is not a run database: missing tables "
            f"{', '.join(self.missing)}"
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

    Args:
        conn: An open connection to the run database.

    Returns:
        The stamped version, or ``None`` when the file carries no stamp.
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


async def _missing_tables(conn: AsyncConnection) -> set[str]:
    """The schema's tables that the open database does not have.

    Args:
        conn: An open connection to the database.

    Returns:
        Names of :data:`Base.metadata` tables absent from the file.
    """
    present = (
        await conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type = 'table'")
        )
    ).scalars()
    return set(Base.metadata.tables) - set(present)


async def create_engine_and_init(
    db_path: Path,
    echo: bool = False,
    *,
    read_only: bool = False,
    **engine_kwargs: Any,
) -> AsyncEngine:
    """Create an async engine and initialize the database schema.

    Creates all tables if they don't exist. Configures WAL mode and
    foreign keys via connection event listeners.

    With ``read_only``, the file is opened in SQLite's ``mode=ro`` instead:
    nothing is created, stamped or reconfigured, every write fails with
    "attempt to write a readonly database", and the file must already be a
    run database at :data:`BASELINE_VERSION`. (SQLite may still leave empty
    ``-wal``/``-shm`` sidecar files next to a WAL database.)

    Args:
        db_path: Path to the SQLite database file.
        echo: Whether to echo SQL statements (for debugging).
        read_only: Open an existing run database without writing to it.
        **engine_kwargs: Overrides merged over the defaults and passed to
            :func:`create_async_engine` — e.g. ``poolclass=NullPool`` for
            hosts that want connection-per-session semantics instead of the
            default persistent pool (the pool-sizing defaults are dropped
            automatically for pool classes that don't take them).

    Returns:
        An initialized AsyncEngine.

    Raises:
        UnsupportedSchemaVersionError: If the file is stamped at a schema
            version this jkent cannot read.
        FileNotFoundError: If ``read_only`` and ``db_path`` does not exist.
        NotARunDatabaseError: If ``read_only`` and the file lacks any of
            the schema's tables or carries no schema stamp.
    """
    # Checked here rather than at import so the failure names the moment a
    # database is actually opened. Cached after the first call.
    require_subsec_support()

    url: str | sa.URL
    if read_only:
        if not db_path.exists():
            raise FileNotFoundError(f"run database does not exist: {db_path}")
        # URI filename so SQLite itself refuses writes; built with
        # URL.create so the quoted path is not re-parsed as a URL.
        url = sa.URL.create(
            "sqlite+aiosqlite",
            database=f"file:{quote(str(db_path.resolve()))}?mode=ro",
            query={"uri": "true"},
        )
    else:
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
        # journal_mode is persisted in the file, so a reader leaves it be.
        if not read_only:
            cursor.execute("PRAGMA journal_mode=WAL")
            # WAL's default synchronous is FULL — an fsync on every commit,
            # and the request lifecycle commits several times per request.
            # NORMAL in WAL mode syncs only at checkpoint: an app crash loses
            # nothing, a power loss can lose the last few commits — fine for
            # a resumable run database.
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

    if read_only:
        try:
            async with engine.connect() as conn:
                if missing := await _missing_tables(conn):
                    raise NotARunDatabaseError(db_path, missing)
                existing = await _stamped_version(conn)
                if existing != BASELINE_VERSION:
                    # Tables all present, so schema_info is too; an empty
                    # one reads as no stamp.
                    if existing is None:
                        raise NotARunDatabaseError(db_path, ["schema_info"])
                    raise UnsupportedSchemaVersionError(db_path, existing)
        except BaseException:
            await engine.dispose()
            raise
        return engine

    async with engine.connect() as conn:
        await conn.execution_options(**{BEGIN_IMMEDIATE_OPTION: True})
        async with conn.begin():
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
    *,
    read_only: bool = False,
    **engine_kwargs: Any,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Initialize database and return engine + session factory.

    This is the main entry point, replacing schema.init_database().

    Args:
        db_path: Path to the SQLite database file.
        echo: Whether to echo SQL statements.
        read_only: Open an existing run database without writing to it
            (see :func:`create_engine_and_init`).
        **engine_kwargs: Engine overrides forwarded to
            :func:`create_engine_and_init` (e.g. ``poolclass``).

    Returns:
        Tuple of (engine, session_factory).
    """
    engine = await create_engine_and_init(
        db_path, echo=echo, read_only=read_only, **engine_kwargs
    )
    return engine, get_session_factory(engine)


async def execute_rowcount(
    session: AsyncSession, stmt: Update | Delete
) -> int:
    """Execute a Core UPDATE/DELETE statement.

    Args:
        session: The session to run it on.
        stmt: The UPDATE or DELETE to execute.

    Returns:
        How many rows it touched.
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

    Args:
        session_factory: Builds the session to open.
        lock: Serializes writers sharing this manager.

    Yields:
        The open session, inside a ``BEGIN IMMEDIATE`` transaction.
    """
    async with lock, session_factory() as session:
        begin_immediate: Mapping[str, bool] = {BEGIN_IMMEDIATE_OPTION: True}
        await session.connection(execution_options=begin_immediate)
        yield session
