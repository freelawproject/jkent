"""Continuation errors during ``pdd replay {strict,lenient}`` become stubs.

When the source serves a response and the scraper's continuation
raises (e.g. an HTMLStructuralAssumptionException because the stored
HTML doesn't match the current scraper code's assumptions), strict and
lenient modes route the exception through the miss policy. With the
default ``--miss stub`` that means the row ends as ``pending`` in the
output — ready for a downstream ``kent run`` to re-fetch it.

error-stubs mode keeps the default worker behavior (marks the row
failed) so the operator sees which descendants of HATEOAS anchors are
still broken.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa
import zstandard as zstd

from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.conftest import AioHttpTestServer
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    run_with_persistent_driver,
)


async def _corrupt_one_detail_response(db_path: Path) -> str:
    """Replace one ``/cases/<docket>`` row's stored HTML with garbage.

    BugCourtScraper.parse_detail uses :class:`CheckedHtmlElement` to
    require exactly one ``//div[@class='case-details']`` element. A
    body of ``<html><body>empty</body></html>`` will fail that check
    with ``HTMLStructuralAssumptionException``, which is what we want
    to exercise.

    Returns the URL of the corrupted row so tests can target it.
    """
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT id, url, compression_dict_id FROM requests "
        "WHERE url LIKE '%/cases/%' AND response_status_code IS NOT NULL "
        "ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if row is None:
        conn.close()
        raise AssertionError("source DB has no /cases/<docket> row to corrupt")
    rid, url, _dict_id = row
    # Compress against no dictionary so the source-side decompress just
    # works — even if the source originally trained a dict, NULL-dict
    # decompression on a NULL-dict compressed blob is the simplest path.
    garbage = b"<html><body>broken</body></html>"
    compressed = zstd.ZstdCompressor().compress(garbage)
    conn.execute(
        "UPDATE requests SET content_compressed = ?, "
        "content_size_original = ?, content_size_compressed = ?, "
        "compression_dict_id = NULL WHERE id = ?",
        (compressed, len(garbage), len(compressed), rid),
    )
    conn.commit()
    conn.close()
    return url


async def _row_status(db_path: Path, url_substring: str) -> str | None:
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text("SELECT status FROM requests WHERE url LIKE :pat"),
                {"pat": f"%{url_substring}%"},
            )
        ).all()
    if not rows:
        return None
    return rows[0][0]


async def _result_count(db_path: Path) -> int:
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        count = (
            await session.execute(sa.text("SELECT COUNT(*) FROM results"))
        ).scalar() or 0
    return count


@pytest.mark.asyncio
async def test_strict_continuation_error_becomes_pending(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """In strict mode with --miss stub, a parse error ends as pending."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    corrupted_url = await _corrupt_one_detail_response(source_db)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="prev-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    docket_segment = corrupted_url.rsplit("/", 1)[-1]
    assert (await _row_status(out_db, docket_segment)) == "pending", (
        f"corrupted detail page {corrupted_url} should be pending"
    )

    # All the non-corrupted detail pages still produced results.
    assert await _result_count(out_db) > 0


@pytest.mark.asyncio
async def test_lenient_continuation_error_becomes_pending(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Same behavior in lenient mode (mode 2 retry-failure path)."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    corrupted_url = await _corrupt_one_detail_response(source_db)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    docket_segment = corrupted_url.rsplit("/", 1)[-1]
    assert (await _row_status(out_db, docket_segment)) == "pending", (
        f"corrupted detail page {corrupted_url} should be pending"
    )


@pytest.mark.asyncio
async def test_lenient_continuation_error_raises_with_miss_raise(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """With --miss raise, a continuation error aborts the replay run."""
    from kent.common.exceptions import RequestFailedHalt

    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    await _corrupt_one_detail_response(source_db)

    with pytest.raises(RequestFailedHalt):
        async with LocalOnlyDriver.open(
            scraper=make_bug_court_scraper(bug_court_server.url),
            db_path=out_db,
            source_db_paths=[source_db],
            miss_policy="raise",
            mode="curr-error-free",
            num_workers=1,
            enable_monitor=False,
        ) as driver:
            await driver.run(setup_signal_handlers=False)
