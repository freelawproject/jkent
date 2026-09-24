"""Tests for Playwright driver database persistence.

Tests schema extensions and SQLManager methods for incidental requests.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.sql_manager import (
    IncidentalCapture,
    SQLManager,
)

InsertRequest = Callable[..., Awaitable[int]]


async def _insert_incidental(
    manager: SQLManager,
    parent_request_id: int,
    content: bytes | None = None,
    **fields: Any,
) -> int:
    """Insert one captured incidental and return its ``incidental_requests`` id."""
    capture = IncidentalCapture(**fields)
    if content is not None:
        capture.set_body(content, compress(content))
    ids = await manager.replace_incidental_requests(
        parent_request_id, [capture]
    )
    return ids[0]


async def test_insert_incidental_request(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """Test inserting an incidental request."""
    # Create a parent request first
    await sql_manager.init_run_metadata(
        scraper_name="test_scraper",
        scraper_version="1.0",
        num_workers=1,
        max_backoff_time=60.0,
    )

    # Insert a test request to be the parent
    parent_id = await insert_request()

    # Insert an incidental request
    content = b"body { color: red }"
    incidental_id = await _insert_incidental(
        sql_manager,
        parent_request_id=parent_id,
        resource_type="stylesheet",
        method="GET",
        url="https://example.com/style.css",
        headers_json='{"Accept": "text/css"}',
        body=None,
        status_code=200,
        response_headers_json='{"Content-Type": "text/css"}',
        content=content,
        started_at_ns=1000000000,
        completed_at_ns=1000001000,
        from_cache=False,
        failure_reason=None,
    )

    assert incidental_id > 0

    # Retrieve the incidental request
    incidental = await sql_manager.get_incidental_request_by_id(incidental_id)
    assert incidental is not None
    assert incidental.parent_request_id == parent_id
    assert incidental.resource_type == "stylesheet"
    assert incidental.method == "GET"
    assert incidental.url == "https://example.com/style.css"
    assert incidental.status_code == 200
    assert incidental.from_cache is False
    # The payload sizes come from the storage row, through the join.
    compressed_size = len(compress(content))
    assert (
        incidental.content_size_original,
        incidental.content_size_compressed,
    ) == (len(content), compressed_size)
    assert incidental.compression_ratio == compressed_size / len(content)


async def test_get_incidental_requests_by_parent(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """Test retrieving all incidental requests for a parent request."""
    # Setup parent request
    await sql_manager.init_run_metadata(
        scraper_name="test_scraper",
        scraper_version="1.0",
        num_workers=1,
        max_backoff_time=60.0,
    )

    parent_id = await insert_request()

    # Insert multiple incidental requests
    resource_types = ["stylesheet", "script", "image", "xhr"]
    await sql_manager.replace_incidental_requests(
        parent_id,
        [
            IncidentalCapture(
                resource_type=resource_type,
                method="GET",
                url=f"https://example.com/resource{i}.{resource_type}",
                status_code=200,
                started_at_ns=1000000000 + i * 1000,
                completed_at_ns=1000001000 + i * 1000,
            )
            for i, resource_type in enumerate(resource_types)
        ],
    )

    # Retrieve all incidental requests
    incidentals = await sql_manager.get_incidental_requests(parent_id)
    assert len(incidentals) == 4
    assert [r.resource_type for r in incidentals] == resource_types
    # No body, so no storage row to join: the payload columns stay empty.
    for record in incidentals:
        assert record.storage_id is None
        assert (
            record.content_size_original,
            record.content_size_compressed,
            record.compression_ratio,
        ) == (None, None, None)


## Migration tests removed -- old schema.py migration logic was deleted
## as part of the SQLAlchemy refactor. Migrations will be handled by
## Alembic going forward.
