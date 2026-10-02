"""Tests for SQLManager context manager and initialization (_base.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from jkent.driver.database_engine.sql_manager import SQLManager


class TestSQLManagerContext:
    """Tests for SQLManager context manager and initialization."""

    async def test_open_context_manager(
        self, initialized_db: object, db_path: Path
    ) -> None:
        """SQLManager.open yields a usable manager inside the block."""
        async with SQLManager.open(db_path) as manager:
            count = await manager.count_pending_requests()
            assert count == 0

    async def test_open_disposes_engine_on_exit(
        self,
        initialized_db: object,
        db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Leaving the open() block disposes the underlying engine."""
        real_dispose = AsyncEngine.dispose
        disposed: list[object] = []

        async def spy_dispose(self: AsyncEngine, close: bool = True) -> None:
            disposed.append(self)
            await real_dispose(self, close)

        monkeypatch.setattr(AsyncEngine, "dispose", spy_dispose)

        async with SQLManager.open(db_path) as manager:
            engine = manager.engine
            assert not disposed  # not disposed while the block is open

        assert disposed == [engine]  # disposed exactly once, on exit

    async def test_open_missing_path_raises(self, tmp_path: Path) -> None:
        """Opening a path with no database raises and creates nothing."""
        missing = tmp_path / "missing.db"
        with pytest.raises(FileNotFoundError, match="missing.db"):
            async with SQLManager.open(missing):
                pass
        assert not missing.exists()

    async def test_open_empty_file_raises(self, tmp_path: Path) -> None:
        """A 0-byte file is not a run database and is left 0 bytes.

        SQLite treats an empty file as an empty database, so opening it
        would lay a fresh schema over it and read back an empty run.
        """
        empty = tmp_path / "empty.db"
        empty.touch()
        with pytest.raises(FileNotFoundError, match="empty.db"):
            async with SQLManager.open(empty):
                pass
        assert empty.stat().st_size == 0
