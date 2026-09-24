"""Tests for run metadata operations (_run_metadata.py)."""

from __future__ import annotations

import pytest
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
            scraper_name="TestScraper",
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
        assert metadata.scraper_version == "1.1.0"
        assert metadata.num_workers == 8
        assert metadata.max_backoff_time == 120.0
        assert metadata.jitter == 0.25
        # seed_params_json is write-once.
        assert metadata.seed_params == [{"entry": {"a": 1}}]

    async def test_resume_with_a_different_scraper_is_refused(
        self, sql_manager: SQLManager
    ) -> None:
        """A run database belongs to the scraper that created it.

        Opening it with another scraper would re-stamp the session config
        under the first scraper's name and dispatch its queued requests to
        steps of a class that never enqueued them.
        """
        await sql_manager.init_run_metadata(
            scraper_name="pkg.a:ScraperA",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        with pytest.raises(ValueError, match="pkg.a:ScraperA"):
            await sql_manager.init_run_metadata(
                scraper_name="pkg.b:ScraperB",
                scraper_version="9.9.9",
                num_workers=8,
                max_backoff_time=120.0,
            )

        metadata = await sql_manager.get_run_metadata()
        assert metadata is not None
        assert metadata.scraper_name == "pkg.a:ScraperA"
        assert metadata.scraper_version == "1.0.0"
        assert metadata.num_workers == 2

    async def test_reopen_with_different_seed_params_is_refused(
        self, sql_manager: SQLManager
    ) -> None:
        """Seeds other than the recorded ones would go unrecorded.

        ``seed_params_json`` is write-once, so accepting new params here
        leaves the database describing a seed set it was not seeded with.
        """
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
            seed_params=[{"entry": {"a": 1}}],
        )

        with pytest.raises(ValueError, match="seed_params"):
            await sql_manager.init_run_metadata(
                scraper_name="TestScraper",
                scraper_version="1.1.0",
                num_workers=2,
                max_backoff_time=60.0,
                seed_params=[{"entry": {"a": 2}}],
            )

        metadata = await sql_manager.get_run_metadata()
        assert metadata is not None
        assert metadata.seed_params == [{"entry": {"a": 1}}]
        assert metadata.scraper_version == "1.0.0"

    async def test_reopen_with_the_same_seed_params_is_accepted(
        self, sql_manager: SQLManager
    ) -> None:
        """A fresh run that failed before seeding retries with its params."""
        seed_params = [{"entry": {"a": 1}}]
        for version in ("1.0.0", "1.1.0"):
            await sql_manager.init_run_metadata(
                scraper_name="TestScraper",
                scraper_version=version,
                num_workers=2,
                max_backoff_time=60.0,
                seed_params=seed_params,
            )

        metadata = await sql_manager.get_run_metadata()
        assert metadata is not None
        assert metadata.scraper_version == "1.1.0"
        assert metadata.seed_params == seed_params

    async def test_update_run_status(self, sql_manager: SQLManager) -> None:
        """Test updating run status."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )

        await sql_manager.update_run_status_running()

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text("SELECT status FROM run_metadata WHERE id = 1")
            )
            row = result.first()
        assert row[0] == RunStatus.RUNNING.code  # type: ignore[index]

    async def test_resume_keeps_the_first_start_and_clears_the_end(
        self, sql_manager: SQLManager
    ) -> None:
        """A resumed run's duration spans every session, not just the last."""
        await sql_manager.init_run_metadata(
            scraper_name="TestScraper",
            scraper_version="1.0.0",
            num_workers=2,
            max_backoff_time=60.0,
        )
        await sql_manager.update_run_status_running()
        first = "2026-01-01 00:00:00.000"
        async with sql_manager.session_factory() as session:
            await session.execute(
                sa.text(
                    "UPDATE run_metadata SET started_at = :t WHERE id = 1"
                ),
                {"t": first},
            )
            await session.commit()
        await sql_manager.close_run()

        await sql_manager.update_run_status_running()

        async with sql_manager.session_factory() as session:
            row = (
                await session.execute(
                    sa.text(
                        "SELECT started_at, ended_at FROM run_metadata "
                        "WHERE id = 1"
                    )
                )
            ).first()
        assert row is not None
        assert (row[0], row[1]) == (first, None)

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
        await sql_manager.update_run_status_running()

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
