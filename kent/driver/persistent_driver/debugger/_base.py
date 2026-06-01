"""Base class for LocalDevDriverDebugger with core lifecycle and metadata."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlmodel import select

from kent.driver.persistent_driver.models import (
    Request,
    RunMetadata,
)
from kent.driver.persistent_driver.scoped_session import ScopedSessionFactory
from kent.driver.persistent_driver.sql_manager import (
    SQLManager,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class DebuggerBase:
    """Base class providing core lifecycle, metadata, and stats.

    Attributes:
        sql: The underlying SQLManager instance for database operations.
        read_only: Whether this instance is in read-only mode.
    """

    def __init__(
        self,
        sql: SQLManager,
        session_factory: ScopedSessionFactory,
        read_only: bool = True,
    ) -> None:
        """Initialize the debugger.

        Args:
            sql: SQLManager instance wrapping the database connection.
            session_factory: Async session factory for direct DB queries.
            read_only: If True, write operations will raise errors.
        """
        self.sql = sql
        self._session_factory = session_factory
        self.read_only = read_only

    @classmethod
    @asynccontextmanager
    async def open(
        cls, db_path: Path | str, read_only: bool = True
    ) -> AsyncIterator[Any]:
        """Open a database for debugging.

        Args:
            db_path: Path to the SQLite database file.
            read_only: If True, open in read-only mode (prevents writes).

        Yields:
            LocalDevDriverDebugger instance.

        Example:
            async with LocalDevDriverDebugger.open("run.db") as debugger:
                stats = await debugger.get_stats()
        """
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        from kent.driver.persistent_driver.database import (
            create_engine_and_init,
            get_session_factory,
        )

        if isinstance(db_path, str):
            db_path = Path(db_path)

        if read_only:
            # Read-only: don't create tables, use URI mode
            url = f"sqlite+aiosqlite:///file:{db_path}?mode=ro&uri=true"
            engine = create_async_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=NullPool,
            )
        else:
            engine = await create_engine_and_init(db_path)

        from kent.driver.persistent_driver.scoped_session import (
            ScopedSessionFactory,
        )

        session_factory = ScopedSessionFactory(get_session_factory(engine))
        sql = SQLManager(engine, session_factory)
        try:
            yield cls(sql, session_factory, read_only=read_only)
        finally:
            await session_factory.remove_all()
            await engine.dispose()

    def _require_write_mode(self) -> None:
        """Raise an error if in read-only mode.

        Raises:
            PermissionError: If the debugger is in read-only mode.
        """
        if self.read_only:
            raise PermissionError(
                "Operation requires write mode. Open with read_only=False."
            )

    # =========================================================================
    # Run Metadata and Stats
    # =========================================================================

    async def get_run_metadata(self) -> dict[str, Any] | None:
        """Get run metadata including scraper name, status, timestamps, and configuration.

        Returns:
            Dictionary with run metadata fields, or None if no metadata exists.

        Example:
            metadata = await debugger.get_run_metadata()
            print(f"Scraper: {metadata['scraper_name']}")
            print(f"Status: {metadata['status']}")
        """
        return await self.sql.get_run_metadata()

    async def get_run_status(self) -> dict[str, Any]:
        """Get run status with pending count or wrapped status indicator.

        Returns a status dictionary suitable for health reports and doctor commands.

        Returns:
            Dictionary with status information:
                - status: Current run status
                - is_running: Boolean indicating if run is in progress
                - pending_count: Number of pending requests (only if running)

        Example:
            status = await debugger.get_run_status()
            if status['is_running']:
                print(f"Run in progress: {status['pending_count']} pending requests")
        """
        # Get run status from metadata
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunMetadata.scraper_name, RunMetadata.status).where(
                    RunMetadata.id == 1
                )
            )
            row = result.first()
            if not row:
                return {
                    "status": "unknown",
                    "is_running": False,
                }

            status = row.status

        # Determine if run is in progress
        is_running = status in ("created", "running")

        result_dict: dict[str, Any] = {
            "status": status,
            "is_running": is_running,
        }

        # If running, include pending count
        if is_running:
            async with self._session_factory() as session:
                count_result = await session.execute(
                    select(sa.func.count())
                    .select_from(Request)
                    .where(Request.status == "pending")
                )
                row = count_result.scalar()  # type: ignore[assignment]
                pending_count = row if row else 0
                result_dict["pending_count"] = pending_count

        return result_dict

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics about the run.

        Returns:
            Dictionary with statistics including queue, throughput,
            compression, results, and errors.

        Example:
            stats = await debugger.get_stats()
            print(f"Total requests: {stats['queue']['total']}")
            print(f"Errors: {stats['errors']['total']}")
        """
        stats = await self.sql.get_stats()
        # Convert DevDriverStats to dict for consistent API
        return stats.to_dict()
