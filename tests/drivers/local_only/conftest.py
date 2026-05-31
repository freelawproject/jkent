"""Shared fixtures for LocalOnlyDriver tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sqlalchemy as sa

from kent.driver.persistent_driver.persistent_driver import PersistentDriver
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.scraper.example.bug_court import BugCourtScraper


def make_bug_court_scraper(server_url: str) -> BugCourtScraper:
    """Return a BugCourtScraper pinned to ``server_url``."""
    scraper = BugCourtScraper()
    scraper.BASE_URL = server_url  # type: ignore[misc]
    scraper.rate_limits = []  # type: ignore[misc]
    return scraper


async def run_with_persistent_driver(
    scraper: BugCourtScraper, db_path: Path
) -> None:
    """Run a scraper end-to-end with PersistentDriver, no monitor."""
    async with PersistentDriver.open(
        scraper,
        db_path,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)


async def read_result_data_json(db_path: Path) -> list[dict[str, Any]]:
    """Read every ``results.data_json`` from a finished DB.

    Order is not preserved; sort the result list by caller's choice of key.
    """
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text("SELECT data_json FROM results ORDER BY id ASC")
            )
        ).all()
    return [json.loads(r[0]) for r in rows]


async def read_completed_dedup_keys(db_path: Path) -> set[str]:
    """Read every completed Request's dedup_key from a finished DB."""
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text(
                    "SELECT deduplication_key FROM requests "
                    "WHERE status = 'completed' "
                    "AND deduplication_key IS NOT NULL"
                )
            )
        ).all()
    return {r[0] for r in rows}


async def read_pending_dedup_keys(db_path: Path) -> set[str]:
    """Read every pending Request's dedup_key from a DB."""
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text(
                    "SELECT deduplication_key FROM requests "
                    "WHERE status = 'pending' "
                    "AND deduplication_key IS NOT NULL"
                )
            )
        ).all()
    return {r[0] for r in rows}
