from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.sql_manager import SQLManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

# What the initialized_db fixture resolves to for its consumers.
_InitializedDB = tuple["AsyncEngine", "async_sessionmaker"]


@pytest.fixture
async def db_path(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def initialized_db(db_path: Path) -> AsyncIterator[_InitializedDB]:
    """Create an initialized database engine and session factory."""
    engine, session_factory = await init_database(db_path)
    yield engine, session_factory
    await engine.dispose()


@pytest.fixture
async def sql_manager(initialized_db: _InitializedDB) -> SQLManager:
    """Create a SQLManager instance."""
    engine, session_factory = initialized_db
    return SQLManager(engine, session_factory)


@pytest.fixture
def insert_request(
    sql_manager: SQLManager,
) -> Callable[..., Awaitable[int]]:
    """Factory that inserts a request with sensible defaults.

    ``insert_request`` has 14 required keyword arguments; the vast majority
    of tests only vary ``url``/``dedup_key``/``priority``/``continuation``.
    Use this to insert a request with defaults and override only what the
    test cares about::

        req_id = await insert_request(url="https://example.com/1", dedup_key="1")
    """

    async def _insert(**overrides: object) -> int:
        params: dict[str, object] = {
            "priority": 5,
            "request_type": "navigating",
            "method": "GET",
            "url": "https://example.com/test",
            "headers_json": None,
            "cookies_json": None,
            "body": None,
            "continuation": "parse",
            "current_location": "",
            "accumulated_data_json": None,
            "permanent_json": None,
            "expected_type": None,
            "dedup_key": None,
            "parent_id": None,
        }
        params.update(overrides)
        return await sql_manager.insert_request(**params)  # type: ignore[arg-type]

    return _insert
