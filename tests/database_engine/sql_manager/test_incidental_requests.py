"""Tests for incidental request storage with content deduplication."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import sqlalchemy as sa

from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.sql_manager import (
    IncidentalRequestRecord,
    SQLManager,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


async def _create_parent_request(insert_request: InsertRequest) -> int:
    return await insert_request(url="https://example.com/page")


class TestIncidentalRequestStorage:
    async def test_insert_creates_both_rows(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"<script>alert(1)</script>"
        compressed = compress(content)

        ir_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/app.js",
            headers_json='{"Accept": "*/*"}',
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            started_at_ns=1000,
            completed_at_ns=2000,
        )

        assert ir_id is not None

        # Verify incidental_requests row
        async with sql_manager._session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT id, parent_request_id, url, storage_id "
                        "FROM incidental_requests WHERE id = :id"
                    ),
                    {"id": ir_id},
                )
            ).first()
        assert row is not None
        assert row[1] == parent_id
        assert row[2] == "https://cdn.example.com/app.js"
        assert row[3] is not None  # storage_id populated

        # Verify storage row
        storage_id = row[3]
        async with sql_manager._session_factory() as session:
            srow = (
                await session.execute(
                    sa.text(
                        "SELECT resource_type, method, status_code, "
                        "content_md5 FROM incidental_request_storage "
                        "WHERE id = :id"
                    ),
                    {"id": storage_id},
                )
            ).first()
        assert srow is not None
        assert srow[0] == "script"
        assert srow[1] == "GET"
        assert srow[3] == hashlib.md5(compressed).hexdigest()

    async def test_deduplication_same_content(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"body{margin:0}"
        compressed = compress(content)

        id1 = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="stylesheet",
            method="GET",
            url="https://cdn.example.com/style.css",
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            status_code=200,
        )
        id2 = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="stylesheet",
            method="GET",
            url="https://cdn.example.com/style.css",
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            status_code=200,
        )

        # Both rows should point to the same storage_id
        async with sql_manager._session_factory() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT storage_id FROM incidental_requests "
                        "WHERE id IN (:id1, :id2)"
                    ),
                    {"id1": id1, "id2": id2},
                )
            ).fetchall()
        assert rows[0][0] == rows[1][0]

        # Only one storage row
        async with sql_manager._session_factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert count == 1

    async def test_different_content_no_dedup(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)

        c1 = compress(b"content-a")
        c2 = compress(b"content-b")

        await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/a.js",
            content_compressed=c1,
            content_size_original=9,
            content_size_compressed=len(c1),
            status_code=200,
        )
        await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/b.js",
            content_compressed=c2,
            content_size_original=9,
            content_size_compressed=len(c2),
            status_code=200,
        )

        async with sql_manager._session_factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert count == 2

    async def test_same_content_different_metadata_no_dedup(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Identical bytes AND url but different status_code must NOT share a row.

        Regression: dedup keyed on the content MD5 alone made the second
        request silently inherit the first's status_code/method/etc. The two
        requests here differ *only* in status_code/failure_reason — same
        parent, resource_type, method, url and (empty) body — so this isolates
        the metadata component of the dedup key. If the key reverted to content
        MD5 alone, both would collapse onto one storage row and the count would
        be 1.
        """
        parent_id = await _create_parent_request(insert_request)
        content = b""  # e.g. an empty body returned by both a 200 and a 404
        compressed = compress(content)
        url = "https://api.example.com/resource"  # identical for both requests

        ok_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="fetch",
            method="GET",
            url=url,
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            status_code=200,
        )
        missing_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="fetch",
            method="GET",
            url=url,
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            status_code=404,
            failure_reason="not found",
        )

        # Two distinct storage rows despite identical content.
        async with sql_manager._session_factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert count == 2

        # Each request reports its own status, not the first writer's.
        ok = await sql_manager.get_incidental_request_by_id(ok_id)
        missing = await sql_manager.get_incidental_request_by_id(missing_id)
        assert ok is not None and missing is not None
        assert ok.status_code == 200
        assert missing.status_code == 404
        assert missing.failure_reason == "not found"

    async def test_no_content_creates_storage_row(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)

        ir_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="image",
            method="GET",
            url="https://cdn.example.com/img.png",
            failure_reason="net::ERR_BLOCKED_BY_CLIENT",
        )

        async with sql_manager._session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT storage_id FROM incidental_requests WHERE id = :id"
                    ),
                    {"id": ir_id},
                )
            ).first()
        assert row is not None
        assert row[0] is not None  # storage row still created

    async def test_dedup_across_parents(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        p1 = await _create_parent_request(insert_request)
        p2 = await _create_parent_request(insert_request)
        content = compress(b"shared-resource")

        id1 = await sql_manager.insert_incidental_request(
            parent_request_id=p1,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/shared.js",
            content_compressed=content,
            content_size_original=15,
            content_size_compressed=len(content),
            status_code=200,
        )
        id2 = await sql_manager.insert_incidental_request(
            parent_request_id=p2,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/shared.js",
            content_compressed=content,
            content_size_original=15,
            content_size_compressed=len(content),
            status_code=200,
        )

        async with sql_manager._session_factory() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT storage_id FROM incidental_requests "
                        "WHERE id IN (:id1, :id2)"
                    ),
                    {"id1": id1, "id2": id2},
                )
            ).fetchall()
        assert rows[0][0] == rows[1][0]

    async def test_get_incidental_requests(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = compress(b"hello")

        await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/1.js",
            content_compressed=content,
            content_size_original=5,
            content_size_compressed=len(content),
            status_code=200,
            started_at_ns=1000,
            completed_at_ns=2000,
        )
        await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="stylesheet",
            method="GET",
            url="https://cdn.example.com/2.css",
            content_compressed=content,
            content_size_original=5,
            content_size_compressed=len(content),
            status_code=200,
            started_at_ns=3000,
            completed_at_ns=4000,
        )

        records = await sql_manager.get_incidental_requests(parent_id)
        assert len(records) == 2
        assert all(isinstance(r, IncidentalRequestRecord) for r in records)
        # Ordered by started_at_ns asc
        assert records[0].url == "https://cdn.example.com/1.js"
        assert records[1].url == "https://cdn.example.com/2.css"
        assert records[0].resource_type == "script"
        assert records[0].status_code == 200

    async def test_get_incidental_request_by_id(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = compress(b"test")

        ir_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="script",
            method="POST",
            url="https://api.example.com/track",
            headers_json='{"Content-Type": "application/json"}',
            content_compressed=content,
            content_size_original=4,
            content_size_compressed=len(content),
            status_code=204,
            started_at_ns=5000,
            completed_at_ns=6000,
            from_cache=True,
        )

        record = await sql_manager.get_incidental_request_by_id(ir_id)
        assert record is not None
        assert isinstance(record, IncidentalRequestRecord)
        assert record.parent_request_id == parent_id
        assert record.method == "POST"
        assert record.status_code == 204
        assert record.from_cache is True
        assert record.storage_id is not None

    async def test_get_incidental_request_not_found(
        self, sql_manager: SQLManager
    ) -> None:
        record = await sql_manager.get_incidental_request_by_id(99999)
        assert record is None

    async def test_get_incidental_request_storage(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"storage-test"
        compressed = compress(content)

        ir_id = await sql_manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="document",
            method="GET",
            url="https://example.com/doc",
            content_compressed=compressed,
            content_size_original=len(content),
            content_size_compressed=len(compressed),
            status_code=200,
            response_headers_json='{"Content-Type": "text/html"}',
        )

        record = await sql_manager.get_incidental_request_by_id(ir_id)
        assert record is not None
        assert record.storage_id is not None
        storage = await sql_manager.get_incidental_request_storage(
            record.storage_id
        )
        assert storage is not None
        assert storage["content_compressed"] == compressed
        assert (
            storage["response_headers_json"] == '{"Content-Type": "text/html"}'
        )
        assert storage["content_md5"] == hashlib.md5(compressed).hexdigest()


class TestIncidentalRequestRecord:
    def test_duration_ms(self) -> None:
        r = IncidentalRequestRecord(
            id=1,
            parent_request_id=1,
            url="https://example.com",
            headers_json=None,
            started_at_ns=1_000_000,
            completed_at_ns=3_500_000,
            from_cache=False,
            created_at=None,
            storage_id=1,
        )
        assert r.duration_ns == 2_500_000
        assert r.duration_ms == 2.5

    def test_duration_ms_no_timing(self) -> None:
        r = IncidentalRequestRecord(
            id=1,
            parent_request_id=1,
            url="https://example.com",
            headers_json=None,
            started_at_ns=None,
            completed_at_ns=None,
            from_cache=None,
            created_at=None,
            storage_id=None,
        )
        assert r.duration_ns is None
        assert r.duration_ms is None

    def test_compression_ratio(self) -> None:
        r = IncidentalRequestRecord(
            id=1,
            parent_request_id=1,
            url="https://example.com",
            headers_json=None,
            started_at_ns=None,
            completed_at_ns=None,
            from_cache=None,
            created_at=None,
            storage_id=1,
            content_size_original=1000,
            content_size_compressed=250,
        )
        assert r.compression_ratio == 4.0

    def test_to_dict(self) -> None:
        r = IncidentalRequestRecord(
            id=42,
            parent_request_id=10,
            url="https://example.com/resource",
            headers_json="{}",
            started_at_ns=1_000_000,
            completed_at_ns=2_000_000,
            from_cache=False,
            created_at="2024-01-01",
            storage_id=5,
            resource_type="script",
            method="GET",
            status_code=200,
            content_size_original=100,
            content_size_compressed=50,
        )
        d = r.to_dict()
        assert d["id"] == 42
        assert d["duration_ms"] == 1.0
        assert d["compression_ratio"] == 2.0
        assert d["resource_type"] == "script"

    def test_to_json(self) -> None:
        r = IncidentalRequestRecord(
            id=1,
            parent_request_id=1,
            url="https://example.com",
            headers_json=None,
            started_at_ns=None,
            completed_at_ns=None,
            from_cache=None,
            created_at=None,
            storage_id=None,
        )
        parsed = json.loads(r.to_json())
        assert parsed["id"] == 1
