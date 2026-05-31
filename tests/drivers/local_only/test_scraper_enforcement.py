"""Single-scraper enforcement at LocalOnlyDriver startup.

Every source DB must record the same scraper *class* (versions may
differ). Mismatch raises :class:`LocalOnlyScraperMismatchError` with
the offending DB and scraper name in the message.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.local_only_driver.errors import (
    LocalOnlyScraperMismatchError,
)
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    run_with_persistent_driver,
)


def _stamp_scraper(db_path: Path, name: str) -> None:
    """Overwrite ``run_metadata.scraper_name`` in an already-built DB."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "UPDATE run_metadata SET scraper_name = ? WHERE id = 1", (name,)
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_class_match_succeeds_when_version_differs(
    bug_court_server, tmp_path: Path
) -> None:
    """Same class, no version recorded — should open without error."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    # Should not raise.
    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ):
        pass


@pytest.mark.asyncio
async def test_class_mismatch_raises_with_path_and_name(
    bug_court_server, tmp_path: Path
) -> None:
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    _stamp_scraper(source_db, "some.other.module:OtherScraper")

    with pytest.raises(LocalOnlyScraperMismatchError) as excinfo:
        async with LocalOnlyDriver.open(
            scraper=make_bug_court_scraper(bug_court_server.url),
            db_path=out_db,
            source_db_paths=[source_db],
            miss_policy="raise",
            mode="curr-error-free",
            num_workers=2,
            enable_monitor=False,
        ):
            pass

    msg = str(excinfo.value)
    assert "OtherScraper" in msg
    assert str(source_db) in msg
