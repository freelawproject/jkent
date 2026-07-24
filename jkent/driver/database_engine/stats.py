"""Statistics dataclasses and queries for a run database.

``get_stats`` aggregates queue/throughput/result/error statistics for a run;
We can poll this for periodic aggregates for logging purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import sqlalchemy as sa
from sqlmodel import col, select

from jkent.driver.database_engine.models import (
    Error,
    Request,
    Result,
    RunMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass
class QueueStats:
    """Statistics about the request queue.

    Attributes:
        pending: Number of pending requests.
        in_progress: Number of requests currently being processed.
        completed: Number of successfully completed requests.
        failed: Number of failed requests.
        held: Number of held (paused) requests.
        total: Total number of requests.
        by_continuation: Counts by continuation method name.
    """

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    held: int = 0
    total: int = 0
    by_continuation: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class ThroughputStats:
    """Statistics about request throughput.

    Attributes:
        total_completed: Total requests completed.
        total_duration_seconds: Total duration from first to last request.
        requests_per_minute: Average requests completed per minute.
        average_response_time_seconds: Average time between start and completion.
    """

    total_completed: int = 0
    total_duration_seconds: float = 0.0
    requests_per_minute: float = 0.0
    average_response_time_seconds: float = 0.0


@dataclass
class ResultStats:
    """Statistics about scraped results.

    Attributes:
        total: Total number of results.
        valid: Number of valid results.
        invalid: Number of invalid results.
        by_type: Counts by result type (Pydantic model name).
    """

    total: int = 0
    valid: int = 0
    invalid: int = 0
    by_type: dict[str, int] = field(default_factory=dict)


@dataclass
class ErrorStats:
    """Statistics about errors.

    Attributes:
        total: Total number of errors.
        unresolved: Number of unresolved errors.
        resolved: Number of resolved errors.
        by_type: Counts by error type (structural, validation, transient).
        by_continuation: Counts by continuation method name.
    """

    total: int = 0
    unresolved: int = 0
    resolved: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_continuation: dict[str, int] = field(default_factory=dict)


@dataclass
class DevDriverStats:
    """Combined statistics for a run.

    Attributes:
        queue: Queue statistics.
        throughput: Throughput statistics.
        results: Result statistics.
        errors: Error statistics.
        run_status: Current run status.
        scraper_name: Name of the scraper.
    """

    queue: QueueStats
    throughput: ThroughputStats
    results: ResultStats
    errors: ErrorStats
    run_status: str = "unknown"
    scraper_name: str = ""


async def _queue_stats(session: AsyncSession) -> QueueStats:
    """Compute queue stats on an existing session."""
    # Get counts by status
    result = await session.execute(
        select(col(Request.status), sa.func.count()).group_by(
            col(Request.status)
        )
    )
    rows = result.all()

    stats = QueueStats()
    for status, count in rows:
        if status == "pending":
            stats.pending = count
        elif status == "in_progress":
            stats.in_progress = count
        elif status == "completed":
            stats.completed = count
        elif status == "failed":
            stats.failed = count
        elif status == "held":
            stats.held = count

    stats.total = (
        stats.pending
        + stats.in_progress
        + stats.completed
        + stats.failed
        + stats.held
    )

    # Get counts by continuation
    result = await session.execute(  # type: ignore[assignment]
        select(
            col(Request.continuation),
            col(Request.status),
            sa.func.count(),
        ).group_by(col(Request.continuation), col(Request.status))
    )
    rows = result.all()

    for continuation, status, count in rows:
        if continuation not in stats.by_continuation:
            stats.by_continuation[continuation] = {}
        stats.by_continuation[continuation][status] = count

    return stats


async def _throughput_stats(session: AsyncSession) -> ThroughputStats:
    """Compute throughput stats on an existing session."""
    result = await session.execute(
        select(
            sa.func.count(),
            sa.func.min(col(Request.started_at)),
            sa.func.max(col(Request.completed_at)),
            sa.func.avg(
                (
                    sa.func.julianday(col(Request.completed_at))
                    - sa.func.julianday(col(Request.started_at))
                )
                * 86400
            ),
        ).where(
            col(Request.status) == "completed",
            col(Request.started_at).isnot(None),
            col(Request.completed_at).isnot(None),
        )
    )
    row = result.first()

    stats = ThroughputStats()
    if row and row[0] > 0:
        stats.total_completed = row[0]

        # Calculate duration from first to last request
        if row[1] and row[2]:
            duration_result = await session.execute(
                select(
                    (
                        sa.func.julianday(sa.literal(row[2]))
                        - sa.func.julianday(sa.literal(row[1]))
                    )
                    * 86400
                )
            )
            duration_row = duration_result.first()
            if duration_row and duration_row[0] is not None:
                stats.total_duration_seconds = duration_row[0]
                if stats.total_duration_seconds > 0:
                    stats.requests_per_minute = (
                        stats.total_completed / stats.total_duration_seconds
                    ) * 60

        if row[3]:
            stats.average_response_time_seconds = row[3]

    return stats


async def _result_stats(session: AsyncSession) -> ResultStats:
    """Compute result stats on an existing session."""
    result = await session.execute(
        select(
            sa.func.count(),
            sa.func.sum(
                sa.case((col(Result.is_valid) == sa.true(), 1), else_=0)
            ),
            sa.func.sum(
                sa.case((col(Result.is_valid) == sa.false(), 1), else_=0)
            ),
        )
    )
    row = result.first()

    stats = ResultStats()
    if row:
        stats.total = row[0]
        stats.valid = row[1] or 0
        stats.invalid = row[2] or 0

    # Get counts by type
    result = await session.execute(  # type: ignore[assignment]
        select(col(Result.result_type), sa.func.count()).group_by(
            col(Result.result_type)
        )
    )
    rows = result.all()
    for result_type, count in rows:
        stats.by_type[result_type] = count

    return stats


async def _error_stats(session: AsyncSession) -> ErrorStats:
    """Compute error stats on an existing session."""
    result = await session.execute(
        select(
            sa.func.count(),
            sa.func.sum(
                sa.case((col(Error.is_resolved) == sa.false(), 1), else_=0)
            ),
            sa.func.sum(
                sa.case((col(Error.is_resolved) == sa.true(), 1), else_=0)
            ),
        )
    )
    row = result.first()

    stats = ErrorStats()
    if row:
        stats.total = row[0]
        stats.unresolved = row[1] or 0
        stats.resolved = row[2] or 0

    # Get counts by type
    result = await session.execute(  # type: ignore[assignment]
        select(col(Error.error_type), sa.func.count()).group_by(
            col(Error.error_type)
        )
    )
    rows = result.all()
    for error_type, count in rows:
        stats.by_type[error_type] = count

    # Get counts by continuation (via joined requests)
    result = await session.execute(  # type: ignore[assignment]
        select(col(Request.continuation), sa.func.count(col(Error.id)))
        .join(Request, col(Error.request_id) == col(Request.id))
        .group_by(col(Request.continuation))
    )
    rows = result.all()
    for continuation, count in rows:
        stats.by_continuation[continuation] = count

    return stats


async def get_stats(
    session_factory: async_sessionmaker,
) -> DevDriverStats:
    """Get all statistics for a run database.

    Runs every sub-query on a single shared session/connection instead of
    opening one per sub-stat.

    Args:
        session_factory: Async session factory.

    Returns:
        DevDriverStats instance with all statistics.
    """
    async with session_factory() as session:
        # Get run metadata
        result = await session.execute(
            select(
                col(RunMetadata.scraper_name), col(RunMetadata.status)
            ).where(col(RunMetadata.id) == 1)
        )
        row = result.first()
        scraper_name = row[0] if row else ""
        run_status = row[1] if row else "unknown"

        return DevDriverStats(
            queue=await _queue_stats(session),
            throughput=await _throughput_stats(session),
            results=await _result_stats(session),
            errors=await _error_stats(session),
            run_status=run_status,
            scraper_name=scraper_name,
        )


@dataclass
class RunSummary:
    """Post-run aggregates for a run database.

    The driver records per-request failures (HTTP errors, structural /
    validation assumption failures) as ``errors`` rows rather than raising,
    so a run can complete while still holding failed work. This gathers what
    the run actually did — the queue, the harvested results, and the recorded
    errors — plus the counts a host needs to classify the failure mode.

    Attributes:
        requests_by_status: ``(continuation, status, count)`` triples,
            ordered by continuation then status.
        results_by_type: ``(result_type, valid, invalid)`` triples, ordered
            by result type.
        errors_total: Total error-row count.
        errors_by_type: ``{error_type: count}``.
        error_rows: The first ``error_rows_limit`` error detail rows in id
            order, as ``(error_type, error_class, message, request_url)``.
        errors_by_continuation: ``(continuation, error_type, count)``
            triples ordered by continuation then error type; ``None``
            continuation means the error's request row is gone.
        requests_total: Total request count.
        errored_requests: Distinct requests with at least one error row.
        archive_error_total: Error rows on archive (file-download) requests
            (``request_type == 'archive'``). ``archive_error_total ==
            errors_total`` means the scraped pages all succeeded and only
            file archiving failed.
    """

    requests_by_status: list[tuple[str, str, int]] = field(
        default_factory=list
    )
    results_by_type: list[tuple[str, int, int]] = field(default_factory=list)
    errors_total: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    error_rows: list[tuple[str, str, str, str]] = field(default_factory=list)
    errors_by_continuation: list[tuple[str | None, str, int]] = field(
        default_factory=list
    )
    requests_total: int = 0
    errored_requests: int = 0
    archive_error_total: int = 0


async def get_run_summary(
    session_factory: async_sessionmaker,
    *,
    error_rows_limit: int = 50,
) -> RunSummary:
    """Aggregate a :class:`RunSummary` on a single session.

    Args:
        session_factory: Async session factory for the run database.
        error_rows_limit: How many error detail rows to include.

    Returns:
        RunSummary with queue/result/error aggregates.
    """
    summary = RunSummary()
    async with session_factory() as session:
        rows = (
            await session.execute(
                select(
                    col(Request.continuation),
                    col(Request.status),
                    sa.func.count(),
                )
                .group_by(col(Request.continuation), col(Request.status))
                .order_by(col(Request.continuation), col(Request.status))
            )
        ).all()
        summary.requests_by_status = [tuple(r) for r in rows]
        summary.requests_total = sum(n for _, _, n in rows)

        rows = (
            await session.execute(
                select(
                    col(Result.result_type),
                    sa.func.sum(
                        sa.case(
                            (col(Result.is_valid) == sa.true(), 1), else_=0
                        )
                    ),
                    sa.func.sum(
                        sa.case(
                            (col(Result.is_valid) == sa.false(), 1), else_=0
                        )
                    ),
                )
                .group_by(col(Result.result_type))
                .order_by(col(Result.result_type))
            )
        ).all()
        summary.results_by_type = [tuple(r) for r in rows]

        summary.errors_total = (
            await session.execute(select(sa.func.count()).select_from(Error))
        ).scalar_one()

        rows = (
            await session.execute(
                select(col(Error.error_type), sa.func.count()).group_by(
                    col(Error.error_type)
                )
            )
        ).all()
        summary.errors_by_type = dict(rows)

        rows = (
            await session.execute(
                select(
                    col(Error.error_type),
                    col(Error.error_class),
                    col(Error.message),
                    col(Error.request_url),
                )
                .order_by(col(Error.id))
                .limit(error_rows_limit)
            )
        ).all()
        summary.error_rows = [tuple(r) for r in rows]

        # continuation lives on the request, not the error, so join through;
        # an outer join keeps errors whose request row is gone (None key).
        rows = (
            await session.execute(
                select(
                    col(Request.continuation),
                    col(Error.error_type),
                    sa.func.count(),
                )
                .select_from(Error)
                .join(
                    Request,
                    col(Error.request_id) == col(Request.id),
                    isouter=True,
                )
                .group_by(col(Request.continuation), col(Error.error_type))
                .order_by(col(Request.continuation), col(Error.error_type))
            )
        ).all()
        summary.errors_by_continuation = [tuple(r) for r in rows]

        summary.errored_requests = (
            await session.execute(
                select(
                    sa.func.count(sa.distinct(col(Error.request_id)))
                ).where(col(Error.request_id).isnot(None))
            )
        ).scalar_one()

        summary.archive_error_total = (
            await session.execute(
                select(sa.func.count())
                .select_from(Error)
                .join(Request, col(Error.request_id) == col(Request.id))
                .where(col(Request.request_type) == "archive")
            )
        ).scalar_one()

    return summary


async def read_run_summary(
    db_path: Path,
    *,
    error_rows_limit: int = 50,
) -> RunSummary:
    """Read a :class:`RunSummary` from a run database file.

    Opens the database itself, so it works after the run object is gone —
    the post-run reporting path. A database the driver never initialized
    yields empty aggregates (opening creates the empty schema).

    Args:
        db_path: Path to the run's SQLite database.
        error_rows_limit: How many error detail rows to include.
    """
    # Deferred import: sql_manager mixins import sibling database_engine
    # modules, so a top-level import here would be cycle-prone.
    from jkent.driver.database_engine.sql_manager import (  # noqa: PLC0415
        SQLManager,
    )

    async with SQLManager.open(db_path) as manager:
        return await get_run_summary(
            manager._session_factory, error_rows_limit=error_rows_limit
        )
