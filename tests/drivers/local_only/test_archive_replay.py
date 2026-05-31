"""Archive request replay.

When the scraper yields an ``archive=True`` Request whose source DB has
a matching ``archived_files`` row, replay must reference that row's
``file_path`` verbatim — no copy, no re-download.

Uses :class:`BugCourtScraperWithArchive` which yields PDF/MP3 archive
requests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.persistent_driver.persistent_driver import PersistentDriver
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.scraper.example.bug_court import BugCourtScraperWithArchive


def _make_scraper(server_url: str) -> BugCourtScraperWithArchive:
    scraper = BugCourtScraperWithArchive()
    scraper.BASE_URL = server_url  # type: ignore[misc]
    scraper.rate_limits = []  # type: ignore[misc]
    return scraper


async def _archive_paths(db_path: Path) -> list[str]:
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text("SELECT file_path FROM archived_files ORDER BY id ASC")
            )
        ).all()
    return [r[0] for r in rows]


@pytest.mark.asyncio
async def test_archive_replay_references_source_paths_verbatim(
    bug_court_server, tmp_path: Path
) -> None:
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    storage_dir = tmp_path / "source_storage"
    storage_dir.mkdir()

    async with PersistentDriver.open(
        _make_scraper(bug_court_server.url),
        source_db,
        storage_dir=storage_dir,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    source_files = await _archive_paths(source_db)
    assert source_files, "Persistent run produced no archived files"

    async with LocalOnlyDriver.open(
        scraper=_make_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    replay_files = await _archive_paths(out_db)

    # Every replayed archive points at the same source-side path. The
    # output DB references the source's storage_dir verbatim.
    assert sorted(replay_files) == sorted(source_files)
    for p in replay_files:
        assert Path(p).exists(), f"replayed file path {p} doesn't exist"
        # The replayed path is inside the original storage_dir.
        assert str(storage_dir) in p
