"""SQLManagerBase - Core initialization and connection management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import zstandard as zstd
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from typing_extensions import Self

from jkent.driver.database_engine.database import init_database, write_session
from jkent.observability import InstrumentedLock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession


class DictEntry(NamedTuple):
    """A trained dictionary: its ``compression_dicts`` row id and the
    loaded zstd object."""

    dict_id: int
    dictionary: zstd.ZstdCompressionDict


class DictCache:
    """A database's in-memory cache of compression dictionaries.

    Without it, every stored response after compaction re-reads the ~110KB
    dictionary blob and rebuilds a ``ZstdCompressionDict`` from it.

    Dictionary rows are immutable once written — retraining mints a new
    row/version — so ``by_id`` entries never invalidate. ``latest`` maps a
    step to its newest :class:`DictEntry` and also caches the negative "no
    dictionary yet" result (the pre-compaction common case); both are
    dropped by ``train_compression_dict``, the only in-process writer. A run
    database has a single writing process, so no external training can slip
    past the negative cache.
    """

    __slots__ = ("by_id", "latest")

    def __init__(self) -> None:
        self.by_id: dict[int, zstd.ZstdCompressionDict] = {}
        self.latest: dict[str, DictEntry | None] = {}


class SQLManagerBase:
    """Core database connection and initialization for SQLManager.

    Provides the shared engine, session factory, and lock that all
    mixin classes depend on.

    Attributes:
        engine: The async SQLAlchemy engine the database is opened on.
        session_factory: Async session factory bound to :attr:`engine`.
        lock: The write lock every SQLManager mutation holds. One instance
            per manager, so concurrent writers actually serialize.
        dict_cache: The database's compression dictionaries, cached for
            the life of the manager (see
            :mod:`~jkent.driver.database_engine.compression`).

    Example::

        # Standalone usage for inspection
        async with SQLManager.open(db_path) as manager:
            params = await manager.get_seed_params()

        # With existing engine/session factory (for driver integration)
        manager = SQLManager(engine, session_factory)
        await manager.insert_request(params)
    """

    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Initialize with an engine and session factory.

        Args:
            engine: An async SQLAlchemy engine.
            session_factory: An async session factory bound to the engine.
        """
        self.engine: Final[AsyncEngine] = engine
        self.session_factory: Final[async_sessionmaker[AsyncSession]] = (
            session_factory
        )
        self.lock: Final[asyncio.Lock] = InstrumentedLock()
        self.dict_cache: Final[DictCache] = DictCache()

    @classmethod
    @asynccontextmanager
    async def open(cls, db_path: Path) -> AsyncIterator[Self]:
        """Open an existing database and create a SQLManager.

        A run creates its database with
        :func:`~jkent.driver.database_engine.database.init_database`; this
        opens one that already exists.

        Args:
            db_path: Path to the SQLite database file.

        Yields:
            SQLManager instance.

        Raises:
            FileNotFoundError: If ``db_path`` does not exist, or is a 0-byte
                file — SQLite opens that as an empty database, and a run
                database is never 0 bytes (WAL mode writes the header on
                creation). Nothing is created or written at the path.

        Example::

            async with SQLManager.open(db_path) as manager:
                params = await manager.get_seed_params()
        """
        if not db_path.exists():
            raise FileNotFoundError(f"run database does not exist: {db_path}")
        if db_path.stat().st_size == 0:
            raise FileNotFoundError(
                f"run database is empty (0 bytes): {db_path}"
            )
        engine, session_factory = await init_database(db_path)
        try:
            yield cls(engine, session_factory)
        finally:
            await engine.dispose()

    def _write_session(
        self,
    ) -> AbstractAsyncContextManager[AsyncSession, bool | None]:
        """Open a write session: this manager's lock + ``BEGIN IMMEDIATE``.

        Every mutating method goes through here rather than
        ``self.lock, self.session_factory()`` — see
        :func:`~jkent.driver.database_engine.database.write_session` for why
        a writer must not open a deferred transaction.
        """
        return write_session(self.session_factory, self.lock)
