"""Tests for the stats queries (database_engine/stats.py).

Pins the math for progress logging reads, all through the one public
entry point ``get_stats``: queue counts by status and continuation, the
throughput-duration branch, result/error tallies, and run-metadata fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import sqlalchemy as sa

from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import (
    get_run_summary,
    get_stats,
    read_run_summary,
)
from jkent.driver.unified_driver.run import ScrapeRun

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


async def _insert_request(
    sql_manager: SQLManager,
    *,
    status: str,
    continuation: str = "parse",
    queue_counter: int,
    started_at: str | None = None,
    completed_at: str | None = None,
    request_type: str = "navigating",
) -> int:
    async with sql_manager._session_factory() as session:
        result = await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, queue_counter, "
                "method, url, continuation, current_location, started_at, "
                "completed_at, request_type) "
                "VALUES (:status, 9, :qc, 'GET', 'https://s', :cont, '', "
                ":started, :completed, :rtype) RETURNING id"
            ),
            {
                "status": status,
                "qc": queue_counter,
                "cont": continuation,
                "started": started_at,
                "completed": completed_at,
                "rtype": request_type,
            },
        )
        request_id = result.scalar_one()
        await session.commit()
    return request_id


async def _insert_result(
    sql_manager: SQLManager, *, request_id: int, result_type: str, valid: bool
) -> None:
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO results (request_id, result_type, data_json, "
                "is_valid) VALUES (:rid, :rtype, '{}', :valid)"
            ),
            {"rid": request_id, "rtype": result_type, "valid": valid},
        )
        await session.commit()


async def _insert_error(
    sql_manager: SQLManager,
    *,
    request_id: int | None,
    error_type: str,
    error_class: str = "ValueError",
    message: str = "boom",
) -> None:
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO errors (request_id, error_type, error_class, "
                "message, request_url, is_resolved) "
                "VALUES (:rid, :etype, :eclass, :msg, 'https://s', 0)"
            ),
            {
                "rid": request_id,
                "etype": error_type,
                "eclass": error_class,
                "msg": message,
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

    stats = (await get_stats(sql_manager._session_factory)).queue
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

    stats = (await get_stats(sql_manager._session_factory)).throughput
    assert stats.total_completed == 3
    # First start 12:00:00 -> last completion 12:01:00 = 60 seconds.
    # (julianday arithmetic is float-based, so approx.)
    assert stats.total_duration_seconds == pytest.approx(60.0, abs=1e-3)
    assert stats.requests_per_minute == pytest.approx(3.0, abs=1e-3)
    assert stats.average_response_time_seconds == pytest.approx(10.0, abs=1e-3)


async def test_throughput_stats_empty_db(sql_manager: SQLManager) -> None:
    stats = (await get_stats(sql_manager._session_factory)).throughput
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


async def test_result_and_error_stats_empty(sql_manager: SQLManager) -> None:
    stats = await get_stats(sql_manager._session_factory)
    assert (
        stats.results.total,
        stats.results.valid,
        stats.results.invalid,
    ) == (
        0,
        0,
        0,
    )
    assert (
        stats.errors.total,
        stats.errors.unresolved,
        stats.errors.resolved,
    ) == (0, 0, 0)


async def _seed_summary_fixture(sql_manager: SQLManager) -> None:
    """Two page continuations, one archive request, three errors.

    Yields a summary where one error is archive-only, one rides a page
    request, and one is orphaned (no request row) — covering every
    aggregation branch including the failure-classification inputs.
    """
    list_id = await _insert_request(
        sql_manager, status="completed", continuation="list", queue_counter=1
    )
    detail_ok = await _insert_request(
        sql_manager,
        status="completed",
        continuation="detail",
        queue_counter=2,
    )
    detail_bad = await _insert_request(
        sql_manager, status="failed", continuation="detail", queue_counter=3
    )
    archive_bad = await _insert_request(
        sql_manager,
        status="failed",
        continuation="save_pdf",
        queue_counter=4,
        request_type="archive",
    )

    await _insert_result(
        sql_manager, request_id=list_id, result_type="CaseData", valid=True
    )
    await _insert_result(
        sql_manager, request_id=detail_ok, result_type="CaseData", valid=False
    )
    await _insert_result(
        sql_manager, request_id=detail_ok, result_type="Docket", valid=True
    )

    await _insert_error(
        sql_manager,
        request_id=detail_bad,
        error_type="transient",
        message="first",
    )
    await _insert_error(sql_manager, request_id=archive_bad, error_type="http")
    await _insert_error(sql_manager, request_id=None, error_type="orphan")


async def test_run_summary_aggregates(sql_manager: SQLManager) -> None:
    await _seed_summary_fixture(sql_manager)

    summary = await get_run_summary(sql_manager._session_factory)

    assert summary.requests_by_status == [
        ("detail", "completed", 1),
        ("detail", "failed", 1),
        ("list", "completed", 1),
        ("save_pdf", "failed", 1),
    ]
    assert summary.requests_total == 4
    assert summary.results_by_type == [
        ("CaseData", 1, 1),
        ("Docket", 1, 0),
    ]
    assert summary.errors_total == 3
    assert summary.errors_by_type == {"transient": 1, "http": 1, "orphan": 1}
    assert len(summary.error_rows) == 3
    assert summary.error_rows[0] == (
        "transient",
        "ValueError",
        "first",
        "https://s",
    )
    assert (None, "orphan", 1) in summary.errors_by_continuation
    assert ("detail", "transient", 1) in summary.errors_by_continuation
    assert ("save_pdf", "http", 1) in summary.errors_by_continuation
    assert summary.errored_requests == 2
    assert summary.archive_error_total == 1


async def test_run_summary_error_rows_limit(sql_manager: SQLManager) -> None:
    await _seed_summary_fixture(sql_manager)

    summary = await get_run_summary(
        sql_manager._session_factory, error_rows_limit=1
    )

    assert len(summary.error_rows) == 1
    assert summary.errors_total == 3  # limit trims detail rows, not counts


async def test_run_summary_empty_db(sql_manager: SQLManager) -> None:
    summary = await get_run_summary(sql_manager._session_factory)

    assert summary.requests_by_status == []
    assert summary.results_by_type == []
    assert summary.errors_total == 0
    assert summary.errors_by_type == {}
    assert summary.error_rows == []
    assert summary.errors_by_continuation == []
    assert summary.requests_total == 0
    assert summary.errored_requests == 0
    assert summary.archive_error_total == 0


async def test_read_run_summary_from_path(tmp_path: Path) -> None:
    """The path-based reader works on a closed database (post-run path)."""
    db_path = tmp_path / "run.db"
    async with SQLManager.open(db_path) as manager:
        await _seed_summary_fixture(manager)

    summary = await read_run_summary(db_path)

    assert summary.requests_total == 4
    assert summary.errors_total == 3
    assert summary.archive_error_total == 1


async def test_run_stats_method(
    sql_manager: SQLManager, tmp_path: Path
) -> None:
    """ScrapeRun.stats() is the public wrapper over get_stats."""
    run = ScrapeRun(cast("Any", object()), tmp_path / "unused.db")

    with pytest.raises(RuntimeError, match="not open"):
        await run.stats()

    await _insert_request(sql_manager, status="pending", queue_counter=1)
    run._db = sql_manager
    stats = await run.stats()
    assert stats.queue.pending == 1
