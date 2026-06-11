"""Storage-method coverage for the unified driver's ResponseStorage.

Complements ``test_persistence.py`` (which covers ``_store_response``,
``_store_result``, and ``_handle_retry``) by exercising the remaining
``database_engine.storage.ResponseStorageDB`` methods against the unified
``ResponseStorage``: request-completion / -failure marking and archived-file
metadata storage.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import ResponseStorage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    """A fully-migrated SQLManager backed by a temp-file SQLite DB."""
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


async def _seed_request(sql_manager: SQLManager) -> int:
    """Insert one pending request row and return its id (FK target)."""
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests "
                "(status, priority, queue_counter, request_type, method, url, "
                " continuation, current_location) "
                "VALUES ('pending', 1, 1, 'navigating', 'GET', "
                "'https://example.com/x', 'parse', '')"
            )
        )
        await session.commit()
        return (
            await session.execute(sa.text("SELECT id FROM requests"))
        ).scalar_one()


async def test_mark_request_completed(sql_manager: SQLManager) -> None:
    """_mark_request_completed flips the row status to 'completed'."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    await storage._mark_request_completed(request_id)

    async with sql_manager._session_factory() as session:
        status = (
            await session.execute(
                sa.text("SELECT status FROM requests WHERE id = :id"),
                {"id": request_id},
            )
        ).scalar_one()
    assert status == "completed"


async def test_mark_request_failed(sql_manager: SQLManager) -> None:
    """_mark_request_failed sets status 'failed' and records the message."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)

    await storage._mark_request_failed(request_id, "boom")

    async with sql_manager._session_factory() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT status, last_error FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
        ).first()
    assert row is not None
    assert row[0] == "failed"
    assert row[1] == "boom"


async def test_store_archived_file(sql_manager: SQLManager) -> None:
    """_store_archived_file records path/url/type + computed size and hash."""
    request_id = await _seed_request(sql_manager)
    storage = ResponseStorage(sql_manager)
    content = b"%PDF-1.4 binary body \x00\x01\xff"

    archived_id = await storage._store_archived_file(
        request_id=request_id,
        file_path="/tmp/doc.pdf",
        original_url="https://example.com/doc.pdf",
        expected_type="pdf",
        content=content,
    )
    assert isinstance(archived_id, int)

    async with sql_manager._session_factory() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT file_path, original_url, expected_type, "
                    "file_size, content_hash "
                    "FROM archived_files WHERE request_id = :id"
                ),
                {"id": request_id},
            )
        ).first()
    assert row is not None
    assert row[0] == "/tmp/doc.pdf"
    assert row[1] == "https://example.com/doc.pdf"
    assert row[2] == "pdf"
    assert row[3] == len(content)
    assert row[4] == hashlib.sha256(content).hexdigest()
