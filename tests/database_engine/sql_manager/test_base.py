"""Tests for SQLManager context manager and initialization (_base.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from jkent.driver.database_engine.sql_manager import SQLManager


class TestSQLManagerContext:
    """Tests for SQLManager context manager and initialization."""

    async def test_open_context_manager(self, db_path: Path) -> None:
        """SQLManager.open yields a usable manager inside the block."""
        async with SQLManager.open(db_path) as manager:
            assert manager._engine is not None
            # Can perform operations
            count = await manager.count_pending_requests()
            assert count == 0

    async def test_open_disposes_engine_on_exit(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Leaving the open() block disposes the underlying engine."""
        real_dispose = AsyncEngine.dispose
        disposed: list[object] = []

        async def spy_dispose(self: AsyncEngine, close: bool = True) -> None:
            disposed.append(self)
            await real_dispose(self, close)

        monkeypatch.setattr(AsyncEngine, "dispose", spy_dispose)

        async with SQLManager.open(db_path) as manager:
            engine = manager._engine
            assert not disposed  # not disposed while the block is open

        assert disposed == [engine]  # disposed exactly once, on exit

    async def test_engine_property(self, sql_manager: SQLManager) -> None:
        """Test engine is set on the manager."""
        assert sql_manager._engine is not None
