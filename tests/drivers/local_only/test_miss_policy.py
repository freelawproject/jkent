"""Miss-policy behavior: raise / skip / stub.

These tests inject a known-incomplete source DB (one branch is missing
its detail-page rows) and verify each policy.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa

from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    read_pending_dedup_keys,
    run_with_persistent_driver,
)


async def _delete_detail_responses(db_path: Path) -> None:
    """Strip stored response content from every ``/cases/<id>`` row.

    The list page (``/cases``) stays intact; only the detail-page
    fulfillments are removed. With the list page still present, the
    scraper will yield detail requests but they'll all be misses.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        UPDATE requests
        SET response_status_code = NULL,
            content_compressed = NULL,
            response_headers_json = NULL,
            response_url = NULL
        WHERE url LIKE '%/cases/%'
        """
    )
    conn.commit()
    conn.close()


async def _status_counts(db_path: Path) -> dict[str, int]:
    """Map ``status`` → count for the requests table."""
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text(
                    "SELECT status, COUNT(*) FROM requests GROUP BY status"
                )
            )
        ).all()
    return {r[0]: r[1] for r in rows}


@pytest.mark.asyncio
async def test_miss_stub_writes_pending_rows(
    bug_court_server, tmp_path: Path
) -> None:
    """`--miss stub` leaves unfulfillable requests as status='pending'."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    await _delete_detail_responses(source_db)

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

    counts = await _status_counts(out_db)
    # The list page itself was served (it was kept in source). Every
    # detail-page request that was yielded became a stub → pending.
    assert counts.get("pending", 0) > 0
    pending_keys = await read_pending_dedup_keys(out_db)
    assert any("cases" in str(k) for k in pending_keys) or pending_keys


@pytest.mark.asyncio
async def test_miss_skip_deletes_the_row(
    bug_court_server, tmp_path: Path
) -> None:
    """`--miss skip` removes the row entirely; no failed/pending residue.

    A replayed run must not carry ``failed`` rows out the back, so
    ``skip`` honors its "log + drop, no output row" semantic literally
    by deleting the row from the output DB.
    """
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    await _delete_detail_responses(source_db)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="skip",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    counts = await _status_counts(out_db)
    assert counts.get("failed", 0) == 0
    assert counts.get("pending", 0) == 0
    # The /cases entry stayed completed; every detail row was deleted.
    assert counts.get("completed", 0) == 1


@pytest.mark.asyncio
async def test_miss_raise_propagates(bug_court_server, tmp_path: Path) -> None:
    """`--miss raise` aborts the run when an unfulfillable request appears."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    await _delete_detail_responses(source_db)

    from kent.common.exceptions import RequestFailedHalt

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
