"""SQLManagerBase - Core initialization and connection management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from typing_extensions import Self

from jkent.driver.database_engine.database import init_database, write_session
from jkent.driver.database_engine.models import Request
from jkent.observability import InstrumentedLock

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextlib import AbstractAsyncContextManager

    from sqlalchemy.ext.asyncio import AsyncSession


class SQLManagerBase:
    """Core database connection and initialization for SQLManager.

    Provides the shared engine, session factory, and lock that all
    mixin classes depend on.

    Attributes:
        engine: The async SQLAlchemy engine the database is opened on.
        session_factory: Async session factory bound to :attr:`engine`.
        lock: The write lock every SQLManager mutation holds. One instance
            per manager, so concurrent writers actually serialize.

    Example::

        # Standalone usage for inspection
        async with SQLManager.open(db_path) as manager:
            params = await manager.get_seed_params()

        # With existing engine/session factory (for driver integration)
        manager = SQLManager(engine, session_factory)
        await manager.store_response(request_id, response, continuation)
    """

    def __init__(
        self,
        engine: AsyncEngine,
        session_factory: async_sessionmaker,
    ) -> None:
        """Initialize with an engine and session factory.

        Args:
            engine: An async SQLAlchemy engine.
            session_factory: An async session factory bound to the engine.
        """
        self.engine: Final[AsyncEngine] = engine
        self.session_factory: Final[async_sessionmaker] = session_factory
        self.lock: Final[asyncio.Lock] = InstrumentedLock()
        # Not Final: reseeded from the DB on first use, then bumped per
        # enqueue.
        self._queue_counter: int | None = None

    @classmethod
    @asynccontextmanager
    async def open(cls, db_path: Path) -> AsyncIterator[Self]:
        """Open a database and create a SQLManager.

        Args:
            db_path: Path to the SQLite database file.

        Yields:
            SQLManager instance.

        Example::

            async with SQLManager.open(db_path) as manager:
                params = await manager.get_seed_params()
        """
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

    async def _ensure_queue_counter_seeded(
        self, session: AsyncSession
    ) -> None:
        """Seed the in-memory FIFO counter from the DB once.

        Runs a single ``max(queue_counter)`` scan the first time a counter
        is requested; subsequent calls are no-ops. Callers hold
        ``self.lock``, so this never races.
        """
        if self._queue_counter is None:
            result = await session.execute(
                select(func.max(Request.queue_counter))
            )
            self._queue_counter = result.scalar() or 0
