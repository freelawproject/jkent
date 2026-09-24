"""Tests for response storage operations (_responses.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import sqlalchemy as sa

from jkent.driver.database_engine.compression import compress, decompress
from jkent.driver.database_engine.sql_manager import SQLManager, StoredResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


class TestResponseStorage:
    """Tests for response storage operations."""

    async def test_store_response(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test storing an HTTP response."""
        request_id = await insert_request()

        content = b"<html>Test content</html>"
        compressed = compress(content)

        await sql_manager.store_response(
            request_id,
            StoredResponse(
                response_status_code=200,
                response_headers_json=json.dumps(
                    {"Content-Type": "text/html"}
                ),
                response_url="https://example.com/test",
                content_compressed=compressed,
                content_size_original=len(content),
                content_size_compressed=len(compressed),
            ),
        )

        # Verify response was stored
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT response_status_code, content_size_original FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == 200
        assert row[1] == len(content)

    async def test_get_stored_response_round_trips(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """What was stored is what comes back, body included."""
        request_id = await insert_request()

        content = b"<html>Test content for retrieval</html>"
        compressed = compress(content)
        response = StoredResponse(
            response_status_code=200,
            response_url="https://example.com/test",
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
        )
        await sql_manager.store_response(request_id, response)

        stored = await sql_manager.get_stored_response(request_id)

        assert stored is not None
        assert stored == response
        assert stored.content_compressed is not None
        assert decompress(stored.content_compressed) == content

    async def test_get_stored_response_empty_body(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A headers-only response is stored (and found) with no body."""
        request_id = await insert_request(
            method="HEAD", url="https://example.com/resource"
        )

        await sql_manager.store_response(
            request_id,
            StoredResponse(
                response_status_code=200,
                response_headers_json=json.dumps(
                    {
                        "Content-Type": "application/pdf",
                        "Content-Length": "5000",
                    }
                ),
                response_url="https://example.com/resource",
                content_size_original=0,
                content_size_compressed=0,
            ),
        )

        stored = await sql_manager.get_stored_response(request_id)

        assert stored is not None
        assert stored.content_compressed is None

    async def test_get_stored_response_no_response(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A request with no stored response returns None (not an empty one)."""
        request_id = await insert_request(
            url="https://example.com/never-fetched"
        )

        # No store_response() call: the request exists but was never answered.
        assert await sql_manager.get_stored_response(request_id) is None


class TestArchivedFileStorage:
    async def test_re_storing_an_archive_replaces_its_row(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A request's archive index row describes its latest attempt.

        The response is stored before the step runs, so a step that fails
        transiently (or a crash before completion) re-downloads and
        re-stores. Appending left one ``archived_files`` row per attempt.
        """
        request_id = await insert_request()
        other_id = await insert_request(url="https://example.com/other.pdf")
        await sql_manager.store_archived_file(
            request_id=other_id,
            file_path="/archive/other.pdf",
            original_url="https://example.com/other.pdf",
            expected_type="pdf",
            file_size=1,
            content_hash="c",
        )

        for attempt, content_hash in enumerate(("a", "b")):
            await sql_manager.store_archived_file(
                request_id=request_id,
                file_path=f"/archive/{attempt}.pdf",
                original_url="https://example.com/doc.pdf",
                expected_type="pdf",
                file_size=10 + attempt,
                content_hash=content_hash,
            )

        async with sql_manager.session_factory() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT request_id, file_path, file_size, content_hash "
                        "FROM archived_files ORDER BY request_id"
                    )
                )
            ).all()
        assert [tuple(r) for r in rows] == [
            (request_id, "/archive/1.pdf", 11, "b"),
            (other_id, "/archive/other.pdf", 1, "c"),
        ]
