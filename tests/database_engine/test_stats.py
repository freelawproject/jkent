"""Tests for the stats queries (database_engine/stats.py).

Pins the math the dashboards read: queue counts by status and
continuation, the throughput-duration branch (previously uncovered),
result/error tallies, and the top-level ``get_stats`` aggregation.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import (
    get_error_stats,
    get_queue_stats,
    get_result_stats,
    get_stats,
    get_throughput_stats,
)


async def _insert_request(
    sql_manager: SQLManager,
    *,
    status: str,
    continuation: str = "parse",
    queue_counter: int,
    started_at: str | None = None,
    completed_at: str | None = None,
) -> None:
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, queue_counter, "
                "method, url, continuation, current_location, started_at, "
                "completed_at) "
                "VALUES (:status, 9, :qc, 'GET', 'https://s', :cont, '', "
                ":started, :completed)"
            ),
            {
                "status": status,
                "qc": queue_counter,
                "cont": continuation,
                "started": started_at,
                "completed": completed_at,
            },
        )
        await session.commit()


async def test_queue_stats_counts_by_status_and_continuation(
    sql_manager: SQLManager,
) -> None:
    statuses = [
        ("pending", "list"),
        ("pending", "detail"),
        ("in_progress", "list"),
        ("completed", "detail"),
        ("failed", "detail"),
        ("held", "list"),
    ]
    for i, (status, continuation) in enumerate(statuses, start=1):
        await _insert_request(
            sql_manager,
            status=status,
            continuation=continuation,
            queue_counter=i,
        )

    stats = await get_queue_stats(sql_manager._session_factory)
    assert stats.pending == 2
    assert stats.in_progress == 1
    assert stats.completed == 1
    assert stats.failed == 1
    assert stats.held == 1
    assert stats.total == 6
    assert stats.by_continuation["list"] == {
        "pending": 1,
        "in_progress": 1,
        "held": 1,
    }
    assert stats.by_continuation["detail"] == {
        "pending": 1,
        "completed": 1,
        "failed": 1,
    }
    assert stats.to_dict()["total"] == 6


async def test_throughput_stats_duration_math(sql_manager: SQLManager) -> None:
    """Three requests of 10s each over a one-minute window."""
    windows = [
        ("2026-06-12 12:00:00", "2026-06-12 12:00:10"),
        ("2026-06-12 12:00:20", "2026-06-12 12:00:30"),
        ("2026-06-12 12:00:50", "2026-06-12 12:01:00"),
    ]
    for i, (started_at, completed_at) in enumerate(windows, start=1):
        await _insert_request(
            sql_manager,
            status="completed",
            queue_counter=i,
            started_at=started_at,
            completed_at=completed_at,
        )

    stats = await get_throughput_stats(sql_manager._session_factory)
    assert stats.total_completed == 3
    # First start 12:00:00 -> last completion 12:01:00 = 60 seconds.
    # (julianday arithmetic is float-based, so approx.)
    assert stats.total_duration_seconds == pytest.approx(60.0, abs=1e-3)
    assert stats.requests_per_minute == pytest.approx(3.0, abs=1e-3)
    assert stats.average_response_time_seconds == pytest.approx(10.0, abs=1e-3)
    assert stats.to_dict()["requests_per_minute"] == pytest.approx(
        3.0, abs=1e-3
    )


async def test_throughput_stats_empty_db(sql_manager: SQLManager) -> None:
    stats = await get_throughput_stats(sql_manager._session_factory)
    assert stats.total_completed == 0
    assert stats.total_duration_seconds == 0.0
    assert stats.requests_per_minute == 0.0


async def test_get_stats_aggregates_with_run_metadata(
    sql_manager: SQLManager,
) -> None:
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO run_metadata (id, scraper_name, status, "
                "base_delay, jitter, num_workers, max_backoff_time) "
                "VALUES (1, 'StatsScraper', 'running', 0, 0, 1, 0)"
            )
        )
        await session.commit()
    await _insert_request(sql_manager, status="pending", queue_counter=1)

    stats = await get_stats(sql_manager._session_factory)
    assert stats.scraper_name == "StatsScraper"
    assert stats.run_status == "running"
    assert stats.queue.pending == 1
    payload = stats.to_dict()
    assert payload["queue"]["pending"] == 1
    assert payload["scraper_name"] == "StatsScraper"
    assert isinstance(stats.to_json(), str)


async def test_result_and_error_stats_empty(sql_manager: SQLManager) -> None:
    results = await get_result_stats(sql_manager._session_factory)
    assert (results.total, results.valid, results.invalid) == (0, 0, 0)
    errors = await get_error_stats(sql_manager._session_factory)
    assert (errors.total, errors.unresolved, errors.resolved) == (0, 0, 0)
