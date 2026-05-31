"""Idempotency of ``pdd replay``.

Two replays of the same source DB must produce the same parsed-data and
completed-request set. The same property must hold for a replay-of-replay:
running LocalOnlyDriver against its own output reproduces the same data.

Only fields we know are intentionally fresh on each run are allowed to
differ (``request_id`` / ``created_at`` / ``completed_at`` timestamps).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kent.driver.local_only_driver import LocalOnlyDriver
from tests.conftest import AioHttpTestServer
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    read_completed_dedup_keys,
    read_result_data_json,
    run_with_persistent_driver,
)


def _sorted_by_docket(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: r.get("docket", ""))


@pytest.mark.asyncio
async def test_replay_matches_persistent_run(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Replaying a PersistentDriver run yields the same parsed data."""
    source_db = tmp_path / "source.db"
    replay_db = tmp_path / "replay1.db"

    scraper = make_bug_court_scraper(bug_court_server.url)
    await run_with_persistent_driver(scraper, source_db)
    source_results = _sorted_by_docket(await read_result_data_json(source_db))
    source_completed = await read_completed_dedup_keys(source_db)

    assert source_results, "Persistent run produced no results — bad fixture"

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=replay_db,
        source_db_paths=[source_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    replay_results = _sorted_by_docket(await read_result_data_json(replay_db))
    replay_completed = await read_completed_dedup_keys(replay_db)

    assert len(replay_results) == len(source_results)
    assert replay_results == source_results
    assert replay_completed == source_completed


@pytest.mark.asyncio
async def test_replay_of_replay_is_idempotent(
    bug_court_server: AioHttpTestServer, tmp_path: Path
) -> None:
    """Replaying a replay yields the same data again."""
    source_db = tmp_path / "source.db"
    replay1_db = tmp_path / "replay1.db"
    replay2_db = tmp_path / "replay2.db"

    scraper = make_bug_court_scraper(bug_court_server.url)
    await run_with_persistent_driver(scraper, source_db)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=replay1_db,
        source_db_paths=[source_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=replay2_db,
        source_db_paths=[replay1_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    r1 = _sorted_by_docket(await read_result_data_json(replay1_db))
    r2 = _sorted_by_docket(await read_result_data_json(replay2_db))
    assert r1 == r2
    assert await read_completed_dedup_keys(
        replay1_db
    ) == await read_completed_dedup_keys(replay2_db)
