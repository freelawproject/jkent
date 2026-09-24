"""Storage-method coverage for the unified driver's ResponseStorage.

Complements ``test_persistence.py`` (which covers ``store_response``,
``_store_result``, and ``handle_retry``) by exercising the remaining
``database_engine.storage.ResponseStorageDB`` methods against the unified
``ResponseStorage``: request-completion / -failure marking and archived-file
metadata storage.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import (
    ArchiveResponse,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.storage import serialize_result
from jkent.driver.unified_driver.persistence import ResponseStorage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


async def test_archive_response_records_its_file(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """An ``ArchiveResponse``'s path, type, size and digest reach the row."""
    request_id = await insert_request()  # the archived_files FK target
    content = b"%PDF-1.4 binary body \x00\x01\xff"
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/doc.pdf"
        ),
        step="parse",
        archive=True,
        expected_type="pdf",
    )
    response = ArchiveResponse(
        status_code=200,
        headers={},
        content=b"",
        url="https://example.com/doc.pdf",
        request=request,
        file_url="/tmp/doc.pdf",
        file_size=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
    )

    await ResponseStorage(sql_manager).store_response(
        request_id, response, "parse"
    )

    async with sql_manager.session_factory() as session:
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
    assert tuple(row) == (
        "/tmp/doc.pdf",
        "https://example.com/doc.pdf",
        "pdf",
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def test_invalid_result_stores_bytes_as_their_repr() -> None:
    """An invalid result is diagnostic: its bytes are shown, not refused."""
    row = serialize_result(
        {"pdf": b"%PDF"},
        [{"loc": ["pdf"], "msg": "not text", "input": b"%PDF"}],
    )
    assert json.loads(row.data_json) == {"pdf": "b'%PDF'"}
    assert (
        json.loads(row.validation_errors_json or "")[0]["input"] == "b'%PDF'"
    )


def test_valid_result_refuses_bytes() -> None:
    with pytest.raises(TypeError, match="bytes"):
        serialize_result({"pdf": b"%PDF"})


async def test_archive_response_without_a_file_raises(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """An ``ArchiveResponse`` naming no file is refused, not stored bodiless.

    An archive handler whose ``save_stream`` (or skip decision) hands back
    ``""`` used to complete the request with neither a body nor an
    ``archived_files`` row — the download vanished without an error.
    """
    request_id = await insert_request()
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/doc.pdf"
        ),
        step="parse",
        archive=True,
        expected_type="pdf",
    )
    response = ArchiveResponse(
        status_code=200,
        headers={},
        content=b"",
        url="https://example.com/doc.pdf",
        request=request,
        file_url="",
    )

    with pytest.raises(ScraperConfigError, match="file_url"):
        await ResponseStorage(sql_manager).store_response(
            request_id, response, "parse"
        )

    assert await sql_manager.get_stored_response(request_id) is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"retry_jitter": 1.0}, "retry_jitter"),
        ({"retry_jitter": -0.1}, "retry_jitter"),
        ({"retry_base_delay": 0.0}, "retry_base_delay"),
        ({"retry_base_delay": -1.0}, "retry_base_delay"),
        ({"max_backoff_time": 0.0}, "max_backoff_time"),
    ],
)
def test_storage_rejects_degenerate_retry_parameters(
    sql_manager: SQLManager, kwargs: dict[str, float], match: str
) -> None:
    """Retry knobs that defeat the backoff are refused at construction.

    A jitter of 1 or more can draw a zero or negative wait, so the floor
    catches a share of every generation and the pool retries in lockstep
    again; a non-positive base delay never grows past the floor at all.
    """
    with pytest.raises(ValueError, match=match):
        ResponseStorage(sql_manager, **kwargs)
