"""Tests for Playwright driver database persistence.

Tests schema extensions and SQLManager methods for incidental requests
and browser configuration.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import sqlalchemy as sa

from jkent.driver.database_engine.database import (
    BASELINE_VERSION,
    init_database,
)
from jkent.driver.database_engine.sql_manager import SQLManager


async def test_schema_includes_incidental_requests_table():
    """Verify the database schema includes incidental_requests table."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, session_factory = await init_database(db_path)

        # Check that incidental_requests table exists
        async with session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='incidental_requests'"
                )
            )
            row = result.first()
        assert row is not None
        assert row[0] == "incidental_requests"

        # Check schema has expected columns
        async with session_factory() as session:
            result = await session.execute(
                sa.text("PRAGMA table_info(incidental_requests)")
            )
            columns = result.all()
        column_names = {col[1] for col in columns}

        expected_columns = {
            "id",
            "parent_request_id",
            "url",
            "headers_json",
            "started_at_ns",
            "completed_at_ns",
            "from_cache",
            "created_at",
            "storage_id",
        }
        assert expected_columns.issubset(column_names)

        await engine.dispose()


async def test_schema_includes_browser_config_json_field():
    """Verify run_metadata table has browser_config_json field."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, session_factory = await init_database(db_path)

        # Check that run_metadata has browser_config_json column
        async with session_factory() as session:
            result = await session.execute(
                sa.text("PRAGMA table_info(run_metadata)")
            )
            columns = result.all()
        column_names = {col[1] for col in columns}

        assert "browser_config_json" in column_names

        await engine.dispose()


async def test_schema_version_is_baseline():
    """The current schema is the baseline version (1)."""
    assert BASELINE_VERSION == 1


async def test_insert_incidental_request():
    """Test inserting an incidental request."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, session_factory = await init_database(db_path)
        manager = SQLManager(engine, session_factory)

        # Create a parent request first
        await manager.init_run_metadata(
            scraper_name="test_scraper",
            scraper_version="1.0",
            num_workers=1,
            max_backoff_time=60.0,
        )

        # Insert a test request to be the parent
        parent_id = await manager.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="start",
            current_location="https://example.com",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key="test-key",
            parent_id=None,
        )

        # Insert an incidental request
        incidental_id = await manager.insert_incidental_request(
            parent_request_id=parent_id,
            resource_type="stylesheet",
            method="GET",
            url="https://example.com/style.css",
            headers_json='{"Accept": "text/css"}',
            body=None,
            status_code=200,
            response_headers_json='{"Content-Type": "text/css"}',
            content_compressed=b"compressed css content",
            content_size_original=1024,
            content_size_compressed=512,
            compression_dict_id=None,
            started_at_ns=1000000000,
            completed_at_ns=1000001000,
            from_cache=False,
            failure_reason=None,
        )

        assert incidental_id > 0

        # Retrieve the incidental request
        incidental = await manager.get_incidental_request_by_id(incidental_id)
        assert incidental is not None
        assert incidental.parent_request_id == parent_id
        assert incidental.resource_type == "stylesheet"
        assert incidental.method == "GET"
        assert incidental.url == "https://example.com/style.css"
        assert incidental.status_code == 200
        assert incidental.from_cache is False

        await engine.dispose()


async def test_get_incidental_requests_by_parent():
    """Test retrieving all incidental requests for a parent request."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, session_factory = await init_database(db_path)
        manager = SQLManager(engine, session_factory)

        # Setup parent request
        await manager.init_run_metadata(
            scraper_name="test_scraper",
            scraper_version="1.0",
            num_workers=1,
            max_backoff_time=60.0,
        )

        parent_id = await manager.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="start",
            current_location="https://example.com",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key="test-key",
            parent_id=None,
        )

        # Insert multiple incidental requests
        resource_types = ["stylesheet", "script", "image", "xhr"]
        for i, resource_type in enumerate(resource_types):
            await manager.insert_incidental_request(
                parent_request_id=parent_id,
                resource_type=resource_type,
                method="GET",
                url=f"https://example.com/resource{i}.{resource_type}",
                status_code=200,
                started_at_ns=1000000000 + i * 1000,
                completed_at_ns=1000001000 + i * 1000,
            )

        # Retrieve all incidental requests
        incidentals = await manager.get_incidental_requests(parent_id)
        assert len(incidentals) == 4
        assert [r.resource_type for r in incidentals] == resource_types

        await engine.dispose()


async def test_browser_config_persistence():
    """Test storing and retrieving browser configuration."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        engine, session_factory = await init_database(db_path)
        manager = SQLManager(engine, session_factory)

        browser_config = {
            "browser_type": "chromium",
            "headless": True,
            "viewport": {"width": 1280, "height": 720},
            "user_agent": "Mozilla/5.0 Test",
            "locale": "en-US",
            "timezone_id": "America/New_York",
        }

        # Initialize with browser config
        await manager.init_run_metadata(
            scraper_name="test_scraper",
            scraper_version="1.0",
            num_workers=1,
            max_backoff_time=60.0,
            browser_config=browser_config,
        )

        # Retrieve metadata and check browser config
        metadata = await manager.get_run_metadata()
        assert metadata is not None
        assert metadata["browser_config"] == browser_config

        await engine.dispose()


## Migration tests removed -- old schema.py migration logic was deleted
## as part of the SQLAlchemy refactor. Migrations will be handled by
## Alembic going forward.
