"""Tests for the stats queries (database_engine/stats.py).

Pins the math for progress logging reads, all through the one public
entry point ``get_stats``: queue counts by status and step, the
throughput-duration branch, result/error tallies, and run-metadata fields.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import (
    NotARunDatabaseError,
    init_database,
)
from jkent.driver.database_engine.enums import (
    ErrorType,
    RequestStatus,
    RequestType,
    RunStatus,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import (
    get_run_summary,
    get_stats,
    read_run_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


async def _insert_request(
    sql_manager: SQLManager,
    *,
    status: str,
    step: str = "parse",
    started_at: str | None = None,
    completed_at: str | None = None,
    request_type: str = "navigating",
) -> int:
    # Raw SQL: the insert path can set neither ``status`` nor explicit
    # ``started_at``/``completed_at`` text, and these rows need all three.
    async with sql_manager.session_factory() as session:
        result = await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, "
                "method, url, step, current_location, started_at, "
                "completed_at, request_type) "
                "VALUES (:status, 9, :method, 'https://s', :cont, '', "
                ":started, :completed, :rtype) RETURNING id"
            ),
            {
                "status": RequestStatus(status).code,
                "method": HttpMethod.GET.code,
                "cont": step,
                "started": started_at,
                "completed": completed_at,
                "rtype": RequestType(request_type).code,
            },
        )
        request_id = result.scalar_one()
        await session.commit()
    return request_id


async def _start_run(sql_manager: SQLManager, started_at: str) -> None:
    """Write the run_metadata row the throughput window starts from."""
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO run_metadata (id, scraper_name, status, "
                "started_at, jitter, num_workers, max_backoff_time) "
                "VALUES (1, 'StatsScraper', :status, :started, 0, 1, 0)"
            ),
            {"status": RunStatus.RUNNING.code, "started": started_at},
        )
        await session.commit()


async def _insert_result(
    sql_manager: SQLManager, *, request_id: int, result_type: str, valid: bool
) -> None:
    async with sql_manager.session_factory() as session:
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
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO errors (request_id, error_type, error_class, "
                "message, request_url) "
                "VALUES (:rid, :etype, :eclass, :msg, 'https://s')"
            ),
            {
                "rid": request_id,
                "etype": ErrorType(error_type).code,
                "eclass": error_class,
                "msg": message,
            },
        )
        await session.commit()


async def test_queue_stats_counts_by_status_and_step(
    sql_manager: SQLManager,
) -> None:
    statuses = [
        ("pending", "list"),
        ("pending", "detail"),
        ("in_progress", "list"),
        ("completed", "detail"),
        ("failed", "detail"),
        # Replay-only status: must still land in a counter and in total.
        ("stubbed", "list"),
    ]
    for status, step in statuses:
        await _insert_request(
            sql_manager,
            status=status,
            step=step,
        )

    stats = (await get_stats(sql_manager.session_factory)).queue
    assert stats.pending == 2
    assert stats.in_progress == 1
    assert stats.completed == 1
    assert stats.failed == 1
    assert stats.stubbed == 1
    assert stats.total == 6
    assert stats.by_step["list"] == {
        "pending": 1,
        "in_progress": 1,
        "stubbed": 1,
    }
    assert stats.by_step["detail"] == {
        "pending": 1,
        "completed": 1,
        "failed": 1,
    }


async def test_throughput_stats_duration_math(sql_manager: SQLManager) -> None:
    """Three requests of 10s each over a one-minute window."""
    await _start_run(sql_manager, "2026-06-12 12:00:00")
    windows = [
        ("2026-06-12 12:00:00", "2026-06-12 12:00:10"),
        ("2026-06-12 12:00:20", "2026-06-12 12:00:30"),
        ("2026-06-12 12:00:50", "2026-06-12 12:01:00"),
    ]
    for started_at, completed_at in windows:
        await _insert_request(
            sql_manager,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
        )

    stats = (await get_stats(sql_manager.session_factory)).throughput
    assert stats.total_completed == 3
    # Run start 12:00:00 -> last completion 12:01:00 = 60 seconds.
    # (julianday arithmetic is float-based, so approx.)
    assert stats.total_duration_seconds == pytest.approx(60.0, abs=1e-3)
    assert stats.requests_per_minute == pytest.approx(3.0, abs=1e-3)
    assert stats.average_response_time_seconds == pytest.approx(10.0, abs=1e-3)


async def test_throughput_window_spans_failed_requests(
    sql_manager: SQLManager,
) -> None:
    """A failed tail is still time the run spent; it widens the window."""
    await _start_run(sql_manager, "2026-06-12 12:00:00")
    await _insert_request(
        sql_manager,
        status="completed",
        started_at="2026-06-12 12:00:00",
        completed_at="2026-06-12 12:00:10",
    )
    await _insert_request(
        sql_manager,
        status="failed",
        started_at="2026-06-12 12:00:50",
        completed_at="2026-06-12 12:01:00",
    )

    stats = (await get_stats(sql_manager.session_factory)).throughput
    assert stats.total_completed == 1
    assert stats.total_duration_seconds == pytest.approx(60.0, abs=1e-3)
    assert stats.requests_per_minute == pytest.approx(1.0, abs=1e-3)
    assert stats.average_response_time_seconds == pytest.approx(10.0, abs=1e-3)


async def test_throughput_window_of_an_all_failed_run(
    sql_manager: SQLManager,
) -> None:
    """A run that completed nothing still reports the time it spent."""
    await _start_run(sql_manager, "2026-06-12 12:00:00")
    await _insert_request(
        sql_manager,
        status="failed",
        started_at="2026-06-12 12:00:00",
        completed_at="2026-06-12 12:01:00",
    )

    stats = (await get_stats(sql_manager.session_factory)).throughput
    assert stats.total_completed == 0
    assert stats.total_duration_seconds == pytest.approx(60.0, abs=1e-3)
    assert stats.requests_per_minute == 0.0
    assert stats.average_response_time_seconds is None


async def test_throughput_window_starts_at_the_run_not_a_restamped_row(
    sql_manager: SQLManager,
) -> None:
    """A retry or restamp moving the first row's started_at does not move it.

    The run began at 12:00:00, but its only request's started_at was
    rewritten to 12:00:30 by a retry; the window still starts at the run.
    """
    await _start_run(sql_manager, "2026-06-12 12:00:00")
    await _insert_request(
        sql_manager,
        status="completed",
        started_at="2026-06-12 12:00:30",
        completed_at="2026-06-12 12:00:40",
    )

    stats = (await get_stats(sql_manager.session_factory)).throughput
    assert stats.total_duration_seconds == pytest.approx(40.0, abs=1e-3)
    assert stats.average_response_time_seconds == pytest.approx(10.0, abs=1e-3)


async def test_throughput_window_is_empty_before_the_run_starts(
    sql_manager: SQLManager,
) -> None:
    """No run start recorded means no window, whatever the rows say."""
    await _insert_request(
        sql_manager,
        status="completed",
        started_at="2026-06-12 12:00:00",
        completed_at="2026-06-12 12:00:10",
    )

    stats = (await get_stats(sql_manager.session_factory)).throughput
    assert stats.total_completed == 1
    assert stats.total_duration_seconds == 0.0
    assert stats.requests_per_minute == 0.0


async def test_stats_empty_db(sql_manager: SQLManager) -> None:
    """Every sub-report of ``get_stats`` reads zero on an untouched database."""
    stats = await get_stats(sql_manager.session_factory)
    assert stats.throughput.total_completed == 0
    assert stats.throughput.total_duration_seconds == 0.0
    assert stats.throughput.requests_per_minute == 0.0
    assert (
        stats.results.total,
        stats.results.valid,
        stats.results.invalid,
    ) == (
        0,
        0,
        0,
    )
    assert stats.errors.total == 0


async def test_get_stats_aggregates_with_run_metadata(
    sql_manager: SQLManager,
) -> None:
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO run_metadata (id, scraper_name, status, "
                "jitter, num_workers, max_backoff_time) "
                "VALUES (1, 'StatsScraper', :status, 0, 1, 0)"
            ),
            {"status": RunStatus.RUNNING.code},
        )
        await session.commit()
    await _insert_request(sql_manager, status="pending")

    stats = await get_stats(sql_manager.session_factory)
    assert stats.scraper_name == "StatsScraper"
    assert stats.run_status == "running"
    assert stats.queue.pending == 1


async def _seed_summary_fixture(sql_manager: SQLManager) -> None:
    """Two page steps, one archive request, three errors.

    Yields a summary where one error is archive-only, one rides a page
    request, and one is orphaned (no request row) — covering every
    aggregation branch including the failure-classification inputs.
    """
    list_id = await _insert_request(
        sql_manager, status="completed", step="list"
    )
    detail_ok = await _insert_request(
        sql_manager,
        status="completed",
        step="detail",
    )
    detail_bad = await _insert_request(
        sql_manager, status="failed", step="detail"
    )
    archive_bad = await _insert_request(
        sql_manager,
        status="failed",
        step="save_pdf",
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
    await _insert_error(
        sql_manager, request_id=archive_bad, error_type="persistent"
    )
    await _insert_error(sql_manager, request_id=None, error_type="unknown")


async def test_run_summary_aggregates(sql_manager: SQLManager) -> None:
    await _seed_summary_fixture(sql_manager)

    summary = await get_run_summary(sql_manager.session_factory)

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
    # Keys are ``ErrorType`` members; lookups by label work because members
    # are ``str`` subclasses with the label's hash and equality.
    assert summary.errors_by_type == {
        "transient": 1,
        "persistent": 1,
        "unknown": 1,
    }
    assert len(summary.error_rows) == 3
    assert summary.error_rows[0] == (
        "transient",
        "ValueError",
        "first",
        "https://s",
    )
    # Ordered by step (SQLite sorts the orphaned error's NULL step first),
    # and complete: the breakdown sums to the error total.
    assert summary.errors_by_step == [
        (None, "unknown", 1),
        ("detail", "transient", 1),
        ("save_pdf", "persistent", 1),
    ]
    assert summary.errored_requests == 2
    assert summary.archive_error_total == 1
    assert not summary.only_archive_failures


async def test_run_summary_error_rows_limit(sql_manager: SQLManager) -> None:
    await _seed_summary_fixture(sql_manager)

    summary = await get_run_summary(
        sql_manager.session_factory, error_rows_limit=1
    )

    assert len(summary.error_rows) == 1
    assert summary.errors_total == 3  # limit trims detail rows, not counts


async def test_run_summary_empty_db(sql_manager: SQLManager) -> None:
    summary = await get_run_summary(sql_manager.session_factory)

    assert summary.requests_by_status == []
    assert summary.results_by_type == []
    assert summary.errors_total == 0
    assert summary.errors_by_type == {}
    assert summary.error_rows == []
    assert summary.errors_by_step == []
    assert summary.requests_total == 0
    assert summary.errored_requests == 0
    assert summary.archive_error_total == 0
    assert not summary.only_archive_failures


async def test_read_run_summary_from_path(db_path: Path) -> None:
    """The path-based reader works on a closed database (post-run path)."""
    # Not ``sql_manager``: its engine would still be open when the reader
    # runs, and the point is reading after the writer has gone.
    engine, session_factory = await init_database(db_path)
    try:
        await _seed_summary_fixture(SQLManager(engine, session_factory))
    finally:
        await engine.dispose()

    summary = await read_run_summary(db_path)

    assert summary.requests_total == 4
    assert summary.errors_total == 3
    assert summary.archive_error_total == 1


async def test_read_run_summary_missing_path_raises(tmp_path: Path) -> None:
    """A missing database raises instead of materializing an empty one.

    The reporting path runs exactly when run databases get moved/renamed; a
    stale path must not yield a plausible all-zeroes summary plus a fresh
    empty schema at the wrong location.
    """
    missing = tmp_path / "nope.db"
    with pytest.raises(FileNotFoundError):
        await read_run_summary(missing)
    assert not missing.exists()


async def test_read_run_summary_empty_file_raises(tmp_path: Path) -> None:
    """A 0-byte file raises instead of summarizing as an all-zeroes run.

    A truncated copy or a ``touch``-ed placeholder is as wrong a path as a
    missing one, and opening it would write a whole schema into it.
    """
    empty = tmp_path / "empty.db"
    empty.touch()
    with pytest.raises(FileNotFoundError):
        await read_run_summary(empty)
    assert empty.stat().st_size == 0


async def test_read_run_summary_foreign_database_raises(
    tmp_path: Path,
) -> None:
    """A non-run SQLite file raises and is not given a schema.

    The reader opens read-only, so it never creates tables in whatever
    file the path happens to name.
    """
    other = tmp_path / "other.db"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
    conn.close()

    with pytest.raises(NotARunDatabaseError):
        await read_run_summary(other)
    with sqlite3.connect(other) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    conn.close()
    assert tables == [("unrelated",)]


async def test_run_summary_only_archive_failures(
    sql_manager: SQLManager,
) -> None:
    """Every error on an archive request: pages fine, archiving failed."""
    await _insert_request(sql_manager, status="completed", step="list")
    archive_bad = await _insert_request(
        sql_manager, status="failed", step="save_pdf", request_type="archive"
    )
    await _insert_error(
        sql_manager, request_id=archive_bad, error_type="persistent"
    )

    summary = await get_run_summary(sql_manager.session_factory)

    assert summary.only_archive_failures


async def test_run_summary_page_failed_without_error_row(
    sql_manager: SQLManager,
) -> None:
    """A page failed with no error row is still a failed page.

    An undeserializable row is marked FAILED without filing an error, so
    the error rows alone (all on the archive request here) read as
    archive-only and the summary reported every page fine.
    """
    await _insert_request(sql_manager, status="failed", step="list")
    archive_bad = await _insert_request(
        sql_manager, status="failed", step="save_pdf", request_type="archive"
    )
    await _insert_error(
        sql_manager, request_id=archive_bad, error_type="persistent"
    )

    summary = await get_run_summary(sql_manager.session_factory)

    assert summary.errors_total == summary.archive_error_total == 1
    assert summary.failed_page_requests == 1
    assert not summary.only_archive_failures


async def test_throughput_counts_completed_rows_without_timestamps(
    sql_manager: SQLManager,
) -> None:
    """A COMPLETED row with no timestamps still counts as completed.

    It contributes nothing to the window or the average, but dropping it
    would make ``total_completed`` disagree with ``QueueStats.completed``.
    """
    await _start_run(sql_manager, "2026-06-12 12:00:00")
    await _insert_request(
        sql_manager,
        status="completed",
        started_at="2026-06-12 12:00:00",
        completed_at="2026-06-12 12:00:30",
    )
    await _insert_request(sql_manager, status="completed")

    stats = await get_stats(sql_manager.session_factory)

    assert stats.throughput.total_completed == stats.queue.completed == 2
    assert stats.throughput.total_duration_seconds == pytest.approx(
        30.0, abs=1e-3
    )
    assert stats.throughput.average_response_time_seconds == pytest.approx(
        30.0, abs=1e-3
    )
