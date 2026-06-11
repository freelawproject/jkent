"""Tests for run metadata operations (_run_metadata.py)."""

from __future__ import annotations

import sqlalchemy as sa

from jkent.driver.database_engine.enums import RunStatus
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
        async with sql_manager.session_factory() as session:
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

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text("SELECT COUNT(*) FROM run_metadata")
            )
            row = result.first()
        assert row[0] == 1  # type: ignore[index]

    async def test_init_run_metadata_restamps_session_config(
        self, sql_manager: SQLManager
    ) -> None:
        """A resumed run records its own config, not the first run's.

        ``run.py`` calls ``init_run_metadata`` on every open. Without the
        re-stamp, a run resumed with more workers -- or after a scraper
        fix -- kept reporting the values the *first* invocation used, so
        ``scraper_version`` described only the first half of the corpus.
        """
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=4,
            max_backoff_time=60.0,
            jitter=0.1,
            seed_params=[{"entry": {"a": 1}}],
        )

        # Resume: same database, new session configuration, and no
        # seed_params (the resume path does not re-pass them).
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.1.0",
            num_workers=8,
            max_backoff_time=120.0,
            jitter=0.25,
        )

        metadata = await sql_manager.get_run_metadata()
        assert metadata is not None
        assert metadata["scraper_version"] == "1.1.0"
        assert metadata["num_workers"] == 8
        assert metadata["max_backoff_time"] == 120.0
        assert metadata["jitter"] == 0.25
        # The merge-managed column is left alone: blanking it here would
        # discard params add_seed_params had folded in.
        assert metadata["seed_params"] == [{"entry": {"a": 1}}]

    async def test_update_run_status(self, sql_manager: SQLManager) -> None:
        """Test updating run status."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        await sql_manager.update_run_status(RunStatus.RUNNING)

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text("SELECT status FROM run_metadata WHERE id = 1")
            )
            row = result.first()
        assert row[0] == RunStatus.RUNNING.code  # type: ignore[index]

    async def test_close_run_interrupts_a_running_run(
        self, sql_manager: SQLManager
    ) -> None:
        """close_run stamps INTERRUPTED + ended_at on a run still RUNNING."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )
        await sql_manager.update_run_status(RunStatus.RUNNING)

        await sql_manager.close_run()

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, ended_at FROM run_metadata WHERE id = 1"
                )
            )
            row = result.first()
        assert row is not None
        assert row[0] == RunStatus.INTERRUPTED.code
        assert row[1] is not None

    async def test_close_run_leaves_a_finalized_run_alone(
        self, sql_manager: SQLManager
    ) -> None:
        """close_run must not restamp ended_at on an already-finalized run.

        ended_at means "when the run ended", not "when the database was last
        closed" — post-run processing and later inspection sessions close the
        file again, and each close would otherwise grow the recorded duration.
        """
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )
        await sql_manager.finalize_run(RunStatus.COMPLETED, None)

        async with sql_manager.session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT status, ended_at FROM run_metadata WHERE id = 1"
                    )
                )
            ).first()
        assert row is not None
        finalized_ended_at = row[1]

        await sql_manager.close_run()

        async with sql_manager.session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT status, ended_at FROM run_metadata WHERE id = 1"
                    )
                )
            ).first()
        assert row is not None
        assert row[0] == RunStatus.COMPLETED.code
        assert row[1] == finalized_ended_at
