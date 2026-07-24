"""Tests for run metadata operations (_run_metadata.py)."""

from __future__ import annotations

import sqlalchemy as sa

from jkent.driver.database_engine.sql_manager import SQLManager


class TestRunMetadata:
    """Tests for run metadata operations."""

    async def test_init_run_metadata_new(
        self, sql_manager: SQLManager
    ) -> None:
        """Test initializing new run metadata."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        # Verify metadata was created
        async with sql_manager._session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT scraper_name, num_workers FROM run_metadata WHERE id = 1"
                )
            )
            row = result.first()
        assert row is not None
        assert row[0] == "TestScraper"
        assert row[1] == 2

    async def test_init_run_metadata_idempotent(
        self, sql_manager: SQLManager
    ) -> None:
        """Test init_run_metadata doesn't create duplicates."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        # Call again - should not create duplicate
        await sql_manager.init_run_metadata(
            scraper_name="DifferentScraper",
            scraper_version="2.0.0",
            num_workers=4,
            max_backoff_time=120.0,
        )

        async with sql_manager._session_factory() as session:
            result = await session.execute(
                sa.text("SELECT COUNT(*) FROM run_metadata")
            )
            row = result.first()
        assert row[0] == 1  # type: ignore[index]

    async def test_update_run_status(self, sql_manager: SQLManager) -> None:
        """Test updating run status."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        await sql_manager.update_run_status("running")

        async with sql_manager._session_factory() as session:
            result = await session.execute(
                sa.text("SELECT status FROM run_metadata WHERE id = 1")
            )
            row = result.first()
        assert row[0] == "running"  # type: ignore[index]
