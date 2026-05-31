"""Concurrent ``fetch_response`` calls against a shared source DB.

sqlite3 connections with ``check_same_thread=False`` can be used from
multiple threads but not *simultaneously*. The worker pool will fan out
``asyncio.to_thread(self.source_index.fetch_response, entry)`` across N
threads — without per-connection serialization, that interleaves
statement handles on the same connection and raises
``sqlite3.InterfaceError: bad parameter or other API misuse`` (or, more
insidiously, returns the wrong row's ``compression_dict_id``).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from kent.driver.local_only_driver.source_index import SourceIndex
from kent.driver.persistent_driver.persistent_driver import PersistentDriver
from tests.conftest import AioHttpTestServer
from tests.drivers.local_only.conftest import make_bug_court_scraper


async def _build_source_with_many_rows(server_url: str, db_path: Path) -> int:
    """Run BugCourtScraper end-to-end and return the row count."""
    async with PersistentDriver.open(
        make_bug_court_scraper(server_url),
        db_path,
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)
    conn = sqlite3.connect(str(db_path))
    n = conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0]
    conn.close()
    return n


@pytest.mark.asyncio
async def test_concurrent_fetch_response_does_not_corrupt(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Many simultaneous fetch_response calls all succeed and return valid HTML."""
    source_db = tmp_path / "source.db"
    await _build_source_with_many_rows(bug_court_server.url, source_db)

    idx = SourceIndex(source_db_paths=[source_db])
    try:
        # Collect every index entry.
        rows = idx._index_conn.execute(
            "SELECT dedup_key FROM source_index"
        ).fetchall()
        assert len(rows) > 1, "fixture should have multiple rows"
        entries = [idx.lookup(r[0]) for r in rows]

        # Fire all fetches concurrently. Without the per-connection lock,
        # this is the path that raised "bad parameter or other API misuse"
        # in production with --workers 8.
        async def fetch(entry):  # type: ignore[no-untyped-def]
            return await asyncio.to_thread(idx.fetch_response, entry)

        results = await asyncio.gather(
            *(fetch(e) for e in entries * 10)  # 10x concurrency stress
        )
        for fetched in results:
            assert fetched.status_code >= 200
            assert isinstance(fetched.content, bytes)
            # Decompressed HTML should at least look plausibly like the
            # mock-server output, not garbled bytes from a dict mismatch.
            assert (
                b"<html" in fetched.content
                or b"<table" in fetched.content
                or fetched.content.startswith(b"{")
            )
    finally:
        idx.close()


@pytest.mark.asyncio
async def test_replay_with_two_source_dbs_and_workers(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """End-to-end multi-DB replay with workers>1 doesn't crash on the conn lock."""
    from kent.driver.local_only_driver import LocalOnlyDriver

    src_a = tmp_path / "a.db"
    src_b = tmp_path / "b.db"
    out = tmp_path / "out.db"
    await _build_source_with_many_rows(bug_court_server.url, src_a)
    await _build_source_with_many_rows(bug_court_server.url, src_b)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out,
        source_db_paths=[src_a, src_b],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=4,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    # No assertion needed beyond "ran without raising"; if the
    # concurrency bug were back, the workers would raise InterfaceError.
