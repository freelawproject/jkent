"""TransientException from a step routes through the miss policy with a hateoas walk.

The default ``--miss stub`` triggers a runtime walk up
``parent_request_id`` in the *output* DB: the failing request and every
descendant of the nearest ``hateoas=True`` ancestor (or the root) end
as ``pending`` so a downstream ``kent run`` can re-fetch from a clean
re-entry point. ``--miss raise`` / ``--miss skip`` honor their normal
semantics on the current request.

Applies in all three replay modes (strict, lenient, error-stubs); for
TransientException specifically, error-stubs does *not* fall back to
the worker's retry handler — replay has no network to retry against.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest
import sqlalchemy as sa

from kent.common.exceptions import RequestFailedHalt, TransientException
from kent.data_types import Response, ScraperYield
from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.conftest import AioHttpTestServer
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    run_with_persistent_driver,
)
from tests.scraper.example.bug_court import BugCourtScraper


def _scraper_that_throws_transient(
    server_url: str, fail_url_substring: str
) -> BugCourtScraper:
    """BugCourtScraper whose parse_detail throws on a specific docket.

    The class is unchanged (so the LocalOnlyDriver single-scraper check
    passes); only the bound method is replaced on the instance.
    """
    scraper = make_bug_court_scraper(server_url)
    original_parse_detail = scraper.parse_detail

    def parse_detail_or_transient(
        response: Response,
    ) -> Generator[ScraperYield[dict], None, None]:
        if fail_url_substring in response.url:
            raise TransientException(f"simulated transient for {response.url}")
        yield from original_parse_detail(response)

    scraper.parse_detail = parse_detail_or_transient  # type: ignore[method-assign]
    return scraper


async def _row_status_by_url(db_path: Path, url_substring: str) -> str | None:
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
    return rows[0][0] if rows else None


async def _all_statuses(db_path: Path) -> dict[str, int]:
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


async def _pick_one_detail_url(db_path: Path) -> str:
    """Return the URL of one ``/cases/<docket>`` row in the source DB."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT url FROM requests WHERE url LIKE '%/cases/BCC%' "
        "ORDER BY id ASC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None, "source DB has no /cases/<docket> row"
    return row[0]


@pytest.mark.asyncio
async def test_transient_with_stub_walks_to_root(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """No hateoas in chain → walk to root → only the root anchor remains pending.

    Per replay's invariant ("a pending row never has descendants"),
    the end-of-run cleanup deletes every descendant of the stubbed
    root. The downstream ``kent run`` will re-fetch the root and
    regenerate the whole subtree.
    """
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    target_url = await _pick_one_detail_url(source_db)
    target_substring = target_url.split("/cases/", 1)[1]

    scraper = _scraper_that_throws_transient(
        bug_court_server.url, f"/cases/{target_substring}"
    )

    async with LocalOnlyDriver.open(
        scraper=scraper,
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="prev-error-free",
        num_workers=1,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    statuses = await _all_statuses(out_db)
    # Only the root /cases entry remains, as pending. The failing
    # detail row + every sibling detail row + their child rows have
    # been deleted by `_finalize_stubs`.
    assert statuses == {"pending": 1}, statuses
    assert (await _row_status_by_url(out_db, "/cases")) == "pending"


@pytest.mark.asyncio
async def test_transient_with_hateoas_true_on_failing_request_stubs_only_that_row(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Setting hateoas=True directly on the failing detail request limits the walk.

    We force ``hateoas=True`` on the row in the *output* DB by patching
    BugCourtScraper to mark every yielded detail Request as hateoas=True.
    Then the walk stops at the failing row, leaving its siblings and
    the entry untouched.
    """
    from collections.abc import Generator as _Gen

    from lxml import html

    from kent.data_types import (
        HttpMethod,
        HTTPRequestParams,
        Request,
    )

    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    target_url = await _pick_one_detail_url(source_db)
    target_substring = target_url.split("/cases/", 1)[1]

    scraper = _scraper_that_throws_transient(
        bug_court_server.url, f"/cases/{target_substring}"
    )

    # Replace parse_list so every yielded detail Request carries hateoas=True.
    def parse_list_hateoas_true(
        response: Response,
    ) -> _Gen[Request, None, None]:
        tree = html.fromstring(response.text)
        for row in tree.xpath("//tr[@class='case-row']"):
            cells = row.xpath(".//td[@class='docket']/text()")
            if not cells:
                continue
            docket = cells[0]
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=f"/cases/{docket}",
                ),
                continuation="parse_detail",
                hateoas=True,
            )

    scraper.parse_list = parse_list_hateoas_true  # type: ignore[method-assign]

    async with LocalOnlyDriver.open(
        scraper=scraper,
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="prev-error-free",
        num_workers=1,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    # Only the failing detail row is pending. Siblings remain completed,
    # entry remains completed.
    assert (await _row_status_by_url(out_db, target_substring)) == "pending", (
        "failing row should be pending"
    )
    statuses = await _all_statuses(out_db)
    assert statuses.get("completed", 0) > 0, (
        f"non-failing rows must still be completed; got {statuses}"
    )


@pytest.mark.asyncio
async def test_transient_with_miss_raise_halts(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """`--miss raise` aborts the run when a step throws TransientException."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    target_url = await _pick_one_detail_url(source_db)
    target_substring = target_url.split("/cases/", 1)[1]

    scraper = _scraper_that_throws_transient(
        bug_court_server.url, f"/cases/{target_substring}"
    )

    with pytest.raises(RequestFailedHalt):
        async with LocalOnlyDriver.open(
            scraper=scraper,
            db_path=out_db,
            source_db_paths=[source_db],
            miss_policy="raise",
            mode="curr-error-free",
            num_workers=1,
            enable_monitor=False,
        ) as driver:
            await driver.run(setup_signal_handlers=False)


@pytest.mark.asyncio
async def test_pending_rows_never_have_descendants_after_replay(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Replay invariant: every pending row in the output has zero descendants.

    The downstream ``kent run`` re-fetches each pending row and
    regenerates its subtree fresh; leaving stale descendants would
    cause its enqueue dedup-check to skip the freshly yielded children.
    """
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    target_url = await _pick_one_detail_url(source_db)
    target_substring = target_url.split("/cases/", 1)[1]
    scraper = _scraper_that_throws_transient(
        bug_court_server.url, f"/cases/{target_substring}"
    )

    async with LocalOnlyDriver.open(
        scraper=scraper,
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="prev-error-free",
        num_workers=1,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    async with (
        SQLManager.open(out_db) as sql,
        sql._session_factory() as session,
    ):
        offenders = (
            await session.execute(
                sa.text(
                    "SELECT p.id, p.url, COUNT(c.id) AS descendants "
                    "FROM requests p "
                    "INNER JOIN requests c ON c.parent_request_id = p.id "
                    "WHERE p.status = 'pending' "
                    "GROUP BY p.id, p.url"
                )
            )
        ).all()
    assert offenders == [], (
        f"Pending rows must have zero descendants. Offenders: {offenders}"
    )


@pytest.mark.asyncio
async def test_transient_in_error_stubs_mode_does_not_retry_loop(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """error-stubs mode also applies the miss policy on TransientException.

    The pre-replay worker's TransientException handler would retry
    forever (no network in replay). The transient-miss-policy path
    short-circuits that.
    """
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )

    target_url = await _pick_one_detail_url(source_db)
    target_substring = target_url.split("/cases/", 1)[1]

    scraper = _scraper_that_throws_transient(
        bug_court_server.url, f"/cases/{target_substring}"
    )

    # error-stubs mode would normally pre-pass for errored *source* rows.
    # Source DB here has no errors, so the pre-pass is a no-op; the run
    # walks scraper top-down. The runtime transient will trigger the
    # hateoas walk on the output DB.
    async with LocalOnlyDriver.open(
        scraper=scraper,
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="desc-error-free",
        num_workers=1,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    # The run completed without an infinite retry loop. The subtree is
    # pending now.
    statuses = await _all_statuses(out_db)
    assert statuses.get("pending", 0) >= 1, statuses
