"""Tests for incidental request storage with content deduplication."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from pydantic import ValidationError

from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.sql_manager import (
    IncidentalCapture,
    IncidentalRequestRecord,
    SQLManager,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


async def _create_parent_request(insert_request: InsertRequest) -> int:
    return await insert_request(url="https://example.com/page")


async def _insert_incidental(
    manager: SQLManager,
    parent_request_id: int,
    content: bytes | None = None,
    **fields: Any,
) -> int:
    """Insert one captured incidental and return its ``incidental_requests`` id.

    ``content`` is the response body as the browser delivered it.
    """
    capture = IncidentalCapture(**fields)
    if content is not None:
        capture.set_body(content, compress(content))
    ids = await manager.replace_incidental_requests(
        parent_request_id, [capture]
    )
    return ids[0]


def _capture(
    content: bytes | None = None,
    *,
    resource_type: str = "script",
    method: str = "GET",
    **fields: Any,
) -> IncidentalCapture:
    capture = IncidentalCapture(
        resource_type=resource_type, method=method, **fields
    )
    if content is not None:
        capture.set_body(content, compress(content))
    return capture


class TestReplaceIncidentalRequests:
    async def test_a_retry_replaces_the_previous_attempts_captures(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Each navigation attempt's batch replaces the parent's last one.

        A retried navigation captures its sub-requests again; appending
        would leave both attempts under the parent, and a ``Singular``
        incidental would then find two matches.
        """
        parent_id = await _create_parent_request(insert_request)
        other_id = await _create_parent_request(insert_request)
        await sql_manager.replace_incidental_requests(
            other_id, [_capture(url="https://cdn.example.com/other.js")]
        )

        await sql_manager.replace_incidental_requests(
            parent_id,
            [
                _capture(b"a", url="https://api.example.com/detail"),
                _capture(url="https://cdn.example.com/app.js"),
            ],
        )
        await sql_manager.replace_incidental_requests(
            parent_id, [_capture(b"b", url="https://api.example.com/detail")]
        )

        records = await sql_manager.get_incidental_requests(parent_id)
        assert [r.url for r in records] == ["https://api.example.com/detail"]
        assert records[0].content_size_original == 1
        [other] = await sql_manager.get_incidental_requests(other_id)
        assert other.url == "https://cdn.example.com/other.js"

    async def test_an_attempt_that_captured_nothing_clears_the_parent(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        await sql_manager.replace_incidental_requests(
            parent_id, [_capture(url="https://cdn.example.com/app.js")]
        )

        assert (
            await sql_manager.replace_incidental_requests(parent_id, []) == []
        )
        assert await sql_manager.get_incidental_requests(parent_id) == []


class TestIncidentalRequestStorage:
    async def test_insert_creates_both_rows(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"<script>alert(1)</script>"

        ir_id = await _insert_incidental(
            sql_manager,
            parent_request_id=parent_id,
            resource_type="script",
            method="GET",
            url="https://cdn.example.com/app.js",
            headers_json='{"Accept": "*/*"}',
            content=content,
            started_at_ns=1000,
            completed_at_ns=2000,
        )

        # Verify incidental_requests row
        async with sql_manager.session_factory() as session:
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

        # The fetch's own metadata is on the capture row, not the shared one.
        async with sql_manager.session_factory() as session:
            crow = (
                await session.execute(
                    sa.text(
                        "SELECT resource_type, method, status_code "
                        "FROM incidental_requests WHERE id = :id"
                    ),
                    {"id": ir_id},
                )
            ).first()
        assert crow is not None
        assert crow[0] == "script"
        assert crow[1] == "GET"

        # The storage row is the payload and its digest, nothing else.
        storage_id = row[3]
        async with sql_manager.session_factory() as session:
            srow = (
                await session.execute(
                    sa.text(
                        "SELECT content_md5 FROM incidental_request_storage "
                        "WHERE id = :id"
                    ),
                    {"id": storage_id},
                )
            ).first()
        assert srow is not None
        assert srow[0] == hashlib.md5(content).digest()

    @pytest.mark.parametrize(
        "same_parent", [True, False], ids=["same-parent", "across-parents"]
    )
    async def test_deduplication_same_content(
        self,
        sql_manager: SQLManager,
        insert_request: InsertRequest,
        same_parent: bool,
    ) -> None:
        """Two captures of the same body share one storage row, whether one
        page load fetched it twice or two page loads fetched it once each."""
        first_parent = await _create_parent_request(insert_request)
        second_parent = (
            first_parent
            if same_parent
            else await _create_parent_request(insert_request)
        )
        content = b"body{margin:0}"

        def capture() -> IncidentalCapture:
            return _capture(
                content,
                resource_type="stylesheet",
                url="https://cdn.example.com/style.css",
                status_code=200,
            )

        if same_parent:
            ids = await sql_manager.replace_incidental_requests(
                first_parent, [capture(), capture()]
            )
        else:
            ids = [
                *await sql_manager.replace_incidental_requests(
                    first_parent, [capture()]
                ),
                *await sql_manager.replace_incidental_requests(
                    second_parent, [capture()]
                ),
            ]

        async with sql_manager.session_factory() as session:
            storage_ids = (
                (
                    await session.execute(
                        sa.text(
                            "SELECT storage_id FROM incidental_requests "
                            "WHERE id IN (:id1, :id2)"
                        ),
                        {"id1": ids[0], "id2": ids[1]},
                    )
                )
                .scalars()
                .all()
            )
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert len(storage_ids) == 2 and len(set(storage_ids)) == 1
        assert storage_ids[0] is not None
        assert count == 1

    async def test_one_body_is_one_row_however_it_was_compressed(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Identity is the delivered bytes, not their encoding on disk."""
        parent_id = await _create_parent_request(insert_request)
        content = b"$(function(){ /* jquery */ });"
        captures = []
        for encoded in (compress(content), b"another encoding of that body"):
            capture = IncidentalCapture(
                resource_type="script",
                method="GET",
                url="https://cdn.example.com/jquery.js",
            )
            capture.set_body(content, encoded)
            captures.append(capture)
        await sql_manager.replace_incidental_requests(parent_id, captures)

        async with sql_manager.session_factory() as session:
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

        c1 = b"content-a"
        c2 = b"content-b"

        await sql_manager.replace_incidental_requests(
            parent_id,
            [
                _capture(
                    c1, url="https://cdn.example.com/a.js", status_code=200
                ),
                _capture(
                    c2, url="https://cdn.example.com/b.js", status_code=200
                ),
            ],
        )

        async with sql_manager.session_factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert count == 2

    async def test_same_content_different_metadata_shares_payload(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Identical bytes share one payload row; the metadata stays per-fetch.

        The two captures differ *only* in status_code/failure_reason — same
        parent, resource_type, method, url and (empty) body — so this isolates
        what the storage key does and does not cover. Storage is
        content-addressed, so one row holds the bytes for both; what each
        request reported is a column of ``incidental_requests`` and is read
        back per occurrence. Folding that metadata back into the storage key
        is what made the table quadratic against URLs carrying a render-time
        cache-buster, so both halves of this matter: one payload row, two
        honest status codes.
        """
        parent_id = await _create_parent_request(insert_request)
        content = b""  # e.g. an empty body returned by both a 200 and a 404
        url = "https://api.example.com/resource"  # identical for both requests

        ok_id, missing_id = await sql_manager.replace_incidental_requests(
            parent_id,
            [
                _capture(
                    content, resource_type="fetch", url=url, status_code=200
                ),
                _capture(
                    content,
                    resource_type="fetch",
                    url=url,
                    status_code=404,
                    failure_reason="not found",
                ),
            ],
        )

        # One storage row: identical bytes, and the bytes are the whole key.
        async with sql_manager.session_factory() as session:
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert count == 1

        # Each request still reports its own status, not the first writer's.
        ok = await sql_manager.get_incidental_request_by_id(ok_id)
        missing = await sql_manager.get_incidental_request_by_id(missing_id)
        assert ok is not None and missing is not None
        assert ok.status_code == 200
        assert missing.status_code == 404
        assert missing.failure_reason == "not found"

    async def test_shared_payload_keeps_the_first_captures_headers(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Headers ride on the shared row, first writer wins.

        Two regressions go red here: headers back in the storage key (two
        rows), and a later capture overwriting the stored headers.
        """
        parent_id = await _create_parent_request(insert_request)
        ids = await sql_manager.replace_incidental_requests(
            parent_id,
            [
                _capture(
                    b"console.log(1)",
                    url="https://cdn.example.com/app.js",
                    status_code=200,
                    response_headers_json=headers,
                )
                for headers in ('{"ETag": "a"}', '{"ETag": "b"}')
            ],
        )

        records = [
            await sql_manager.get_incidental_request_by_id(rid) for rid in ids
        ]
        storage_ids = {r.storage_id for r in records if r is not None}
        [storage_id] = storage_ids  # one shared payload row
        assert storage_id is not None
        storage = await sql_manager.get_incidental_request_storage(storage_id)
        assert storage is not None
        assert storage.response_headers_json == '{"ETag": "a"}'

    async def test_no_content_creates_no_storage_row(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A capture with no body stores nothing and keeps its own failure text.

        There is no payload to address, so there is no row to address it
        with — a bodyless capture used to mint a storage row whose only
        content column was a NULL digest.
        """
        parent_id = await _create_parent_request(insert_request)

        ir_id = await _insert_incidental(
            sql_manager,
            parent_request_id=parent_id,
            resource_type="image",
            method="GET",
            url="https://cdn.example.com/img.png",
            failure_reason="net::ERR_BLOCKED_BY_CLIENT",
        )

        async with sql_manager.session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT storage_id FROM incidental_requests WHERE id = :id"
                    ),
                    {"id": ir_id},
                )
            ).first()
            count = (
                await session.execute(
                    sa.text("SELECT COUNT(*) FROM incidental_request_storage")
                )
            ).scalar()
        assert row is not None
        assert row[0] is None
        assert count == 0

        record = await sql_manager.get_incidental_request_by_id(ir_id)
        assert record is not None
        assert record.failure_reason == "net::ERR_BLOCKED_BY_CLIENT"
        assert record.resource_type == "image"

    async def test_get_incidental_requests(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"hello"

        # Stored out of order: the read sorts by start time.
        await sql_manager.replace_incidental_requests(
            parent_id,
            [
                _capture(
                    content,
                    resource_type="stylesheet",
                    url="https://cdn.example.com/2.css",
                    status_code=200,
                    started_at_ns=3000,
                    completed_at_ns=4000,
                ),
                _capture(
                    content,
                    url="https://cdn.example.com/1.js",
                    status_code=200,
                    started_at_ns=1000,
                    completed_at_ns=2000,
                ),
            ],
        )

        records = await sql_manager.get_incidental_requests(parent_id)
        assert len(records) == 2
        # Ordered by started_at_ns asc
        assert records[0].url == "https://cdn.example.com/1.js"
        assert records[1].url == "https://cdn.example.com/2.css"
        assert records[0].resource_type == "script"
        assert records[0].status_code == 200

    async def test_get_incidental_request_by_id(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        parent_id = await _create_parent_request(insert_request)
        content = b"test"

        ir_id = await _insert_incidental(
            sql_manager,
            parent_request_id=parent_id,
            resource_type="script",
            method="POST",
            url="https://api.example.com/track",
            headers_json='{"Content-Type": "application/json"}',
            content=content,
            status_code=204,
            started_at_ns=5000,
            completed_at_ns=6000,
            from_cache=True,
        )

        record = await sql_manager.get_incidental_request_by_id(ir_id)
        assert record is not None
        assert record.parent_request_id == parent_id
        assert record.method == "POST"
        assert record.status_code == 204
        assert record.from_cache is True
        assert record.storage_id is not None

    async def test_created_at_reads_back_as_utc(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """The stored stamp comes back aware, in UTC, and dumps with ``Z``."""
        parent_id = await _create_parent_request(insert_request)
        ir_id = await _insert_incidental(
            sql_manager,
            parent_request_id=parent_id,
            resource_type="document",
            method="GET",
            url="https://example.com/doc",
        )
        record = await sql_manager.get_incidental_request_by_id(ir_id)
        assert record is not None and record.created_at is not None
        assert record.created_at.tzinfo is timezone.utc
        assert record.model_dump(mode="json")["created_at"].endswith("Z")

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

        ir_id = await _insert_incidental(
            sql_manager,
            parent_request_id=parent_id,
            resource_type="document",
            method="GET",
            url="https://example.com/doc",
            content=content,
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
        assert storage.content_compressed == compressed
        assert storage.response_headers_json == '{"Content-Type": "text/html"}'
        assert storage.content_md5 == hashlib.md5(content).digest()


def _record_fields(**overrides: Any) -> dict[str, Any]:
    """Keyword arguments for a minimal :class:`IncidentalRequestRecord`."""
    fields: dict[str, Any] = {
        "id": 1,
        "parent_request_id": 1,
        "url": "https://example.com",
        "resource_type": "document",
        "method": "GET",
        "headers_json": None,
        "started_at_ns": None,
        "completed_at_ns": None,
        "from_cache": None,
        "created_at": None,
        "storage_id": None,
    }
    fields.update(overrides)
    return fields


class TestIncidentalRequestRecord:
    @pytest.mark.parametrize("column", ["resource_type", "method"])
    def test_not_null_column_is_required(self, column: str) -> None:
        """A record without a NOT NULL ``incidental_requests`` column is invalid."""
        fields = _record_fields()
        del fields[column]
        with pytest.raises(ValidationError, match=column):
            IncidentalRequestRecord(**fields)

    @pytest.mark.parametrize("column", ["resource_type", "method"])
    def test_not_null_column_rejects_none(self, column: str) -> None:
        with pytest.raises(ValidationError, match=column):
            IncidentalRequestRecord(**_record_fields(**{column: None}))

    def test_duration_ms(self) -> None:
        r = IncidentalRequestRecord(
            **_record_fields(
                started_at_ns=1_000_000,
                completed_at_ns=3_500_000,
                from_cache=False,
                storage_id=1,
            )
        )
        assert r.duration_ms == 2.5

    def test_aware_created_at_dumps_with_a_z_suffix(self) -> None:
        stamp = datetime(2026, 9, 22, 13, 45, 1, 250_000, tzinfo=timezone.utc)
        r = IncidentalRequestRecord(**_record_fields(created_at=stamp))
        assert r.model_dump(mode="json")["created_at"] == (
            "2026-09-22T13:45:01.250000Z"
        )

    def test_duration_ms_no_timing(self) -> None:
        r = IncidentalRequestRecord(**_record_fields())
        assert r.duration_ms is None

    def test_compression_ratio(self) -> None:
        r = IncidentalRequestRecord(
            **_record_fields(
                storage_id=1,
                content_size_original=1000,
                content_size_compressed=250,
            )
        )
        assert r.compression_ratio == 0.25
