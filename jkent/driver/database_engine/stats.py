"""Statistics dataclasses and queries for a run database.

``get_stats`` aggregates queue/throughput/result/error statistics for a run;
We can poll this for periodic aggregates for logging purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, NamedTuple

import sqlalchemy as sa
from sqlalchemy import select

from jkent.driver.database_engine.models import (
    Error,
    ErrorType,
    Request,
    RequestStatus,
    RequestType,
    Result,
    RunMetadata,
    RunStatus,
)
from jkent.driver.database_engine.sql_manager import (
    RUN_METADATA_ID,
    SQLManager,
)
from jkent.driver.database_engine.timestamps import epoch_seconds

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
        stubbed: Number of stubbed requests (replay databases only).
        total: Total number of requests, counted from the grouped rows —
            always equals ``count(*)`` even for a status this dataclass has
            no field for.
        by_continuation: Counts by continuation method name.
    """

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    held: int = 0
    stubbed: int = 0
    total: int = 0
    by_continuation: dict[str, dict[RequestStatus, int]] = field(
        default_factory=dict
    )


@dataclass
class ThroughputStats:
    """Statistics about request throughput.

    Attributes:
        total_completed: Total requests completed.
        total_duration_seconds: Wall clock from the earliest ``started_at``
            to the latest ``completed_at`` across *every* session this
            database has seen, not the current one. A run interrupted on
            Monday and resumed on Wednesday spans both days plus the idle
            gap between them.
        requests_per_minute: ``total_completed`` over
            ``total_duration_seconds``. Inherits the window above, so on a
            resumed database this is a lifetime average including time no
            run was open — not the rate the run is currently achieving.
        average_response_time_seconds: Mean ``completed_at - started_at``
            over completed requests, excluding pre-resolved ones for the
            same reason as
            :meth:`~jkent.driver.database_engine.sql_manager.SQLManager.avg_completed_request_duration_s`:
            they never touch the transport, so they finish inside a
            millisecond and would drag the mean toward zero without any
            fetch having happened.
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
        by_type: Counts by error type (structural, validation, transient).
        by_continuation: Counts by continuation method name; the ``None``
            key holds errors whose request row is gone (or that were stored
            without one), so the breakdown always sums to ``total``.
    """

    total: int = 0
    by_type: dict[ErrorType, int] = field(default_factory=dict)
    by_continuation: dict[str | None, int] = field(default_factory=dict)


@dataclass
class RunStats:
    """Combined statistics for a run.

    Attributes:
        queue: Queue statistics.
        throughput: Throughput statistics.
        results: Result statistics.
        errors: Error statistics.
        run_status: Current run status; None when no metadata row exists.
        scraper_name: Name of the scraper.
    """

    queue: QueueStats
    throughput: ThroughputStats
    results: ResultStats
    errors: ErrorStats
    run_status: RunStatus | None = None
    scraper_name: str = ""


#: QueueStats field for each RequestStatus. Total is counted from the rows
#: themselves, so a status missing here (a future addition) still counts
#: toward ``total`` instead of silently vanishing from the aggregate.
_STATUS_FIELDS: Final[dict[RequestStatus, str]] = {
    RequestStatus.PENDING: "pending",
    RequestStatus.IN_PROGRESS: "in_progress",
    RequestStatus.COMPLETED: "completed",
    RequestStatus.FAILED: "failed",
    RequestStatus.HELD: "held",
    RequestStatus.STUBBED: "stubbed",
}


async def _queue_stats(session: AsyncSession) -> QueueStats:
    """Compute queue stats on an existing session."""
    # Get counts by status
    result = await session.execute(
        select(Request.status, sa.func.count()).group_by(Request.status)
    )
    rows = result.all()

    stats = QueueStats()
    for status, count in rows:
        field_name = _STATUS_FIELDS.get(status)
        if field_name is not None:
            setattr(stats, field_name, count)
    stats.total = sum(count for _, count in rows)

    # Get counts by continuation
    result = await session.execute(  # type: ignore[assignment]
        select(
            Request.continuation,
            Request.status,
            sa.func.count(),
        ).group_by(Request.continuation, Request.status)
    )
    rows = result.all()

    for continuation, status, count in rows:
        if continuation not in stats.by_continuation:
            stats.by_continuation[continuation] = {}
        stats.by_continuation[continuation][status] = count

    return stats


async def _throughput_stats(session: AsyncSession) -> ThroughputStats:
    """Compute throughput stats on an existing session."""
    # First-to-last duration is aggregated in the same SELECT (max epoch -
    # min epoch) rather than re-sending the min/max timestamps to SQLite in
    # a second round trip just to subtract them.
    duration_s = epoch_seconds(Request.completed_at) - epoch_seconds(
        Request.started_at
    )
    result = await session.execute(
        select(
            sa.func.count(),
            # Pre-resolved requests are excluded from the *average* only,
            # via CASE rather than a WHERE: they are genuinely completed
            # requests, so dropping them from the row set would make
            # total_completed disagree with QueueStats.completed and
            # shorten the duration window. They contribute NULL here,
            # which avg() ignores. Same exclusion, and the same reason, as
            # SQLManager.avg_completed_request_duration_s: a request whose
            # response was attached at enqueue time never asked the site
            # anything, so it has nothing to say about how long the site
            # takes to answer.
            sa.func.avg(
                sa.case(
                    (Request.preresolved == sa.false(), duration_s),
                    else_=None,
                )
            ),
            sa.func.max(epoch_seconds(Request.completed_at))
            - sa.func.min(epoch_seconds(Request.started_at)),
        ).where(
            Request.status == RequestStatus.COMPLETED,
            Request.started_at.isnot(None),
            Request.completed_at.isnot(None),
        )
    )
    row = result.first()

    stats = ThroughputStats()
    if row and row[0] > 0:
        stats.total_completed = row[0]

        if row[2] is not None:
            stats.total_duration_seconds = row[2]
            if stats.total_duration_seconds > 0:
                stats.requests_per_minute = (
                    stats.total_completed / stats.total_duration_seconds
                ) * 60

        if row[1]:
            stats.average_response_time_seconds = row[1]

    return stats


async def _result_stats(session: AsyncSession) -> ResultStats:
    """Compute result stats on an existing session."""
    result = await session.execute(
        select(
            sa.func.count(),
            sa.func.sum(sa.case((Result.is_valid == sa.true(), 1), else_=0)),
            sa.func.sum(sa.case((Result.is_valid == sa.false(), 1), else_=0)),
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
        select(Result.result_type, sa.func.count()).group_by(
            Result.result_type
        )
    )
    rows = result.all()
    for result_type, count in rows:
        stats.by_type[result_type] = count

    return stats


async def _error_stats(session: AsyncSession) -> ErrorStats:
    """Compute error stats on an existing session."""
    result = await session.execute(select(sa.func.count()).select_from(Error))
    row = result.first()

    stats = ErrorStats()
    if row:
        stats.total = row[0]

    # Get counts by type
    result = await session.execute(  # type: ignore[assignment]
        select(Error.error_type, sa.func.count()).group_by(Error.error_type)
    )
    rows = result.all()
    for error_type, count in rows:
        stats.by_type[error_type] = count

    # Get counts by continuation (via joined requests). Outer join, matching
    # get_run_summary's errors_by_continuation: an error stored without a
    # request row (or whose row is gone) lands under the None key instead of
    # dropping out and desyncing this breakdown from ``total``.
    result = await session.execute(  # type: ignore[assignment]
        select(Request.continuation, sa.func.count(Error.id))
        .select_from(Error)
        .join(Request, Error.request_id == Request.id, isouter=True)
        .group_by(Request.continuation)
    )
    rows = result.all()
    for continuation, count in rows:
        stats.by_continuation[continuation] = count

    return stats


async def get_stats(
    session_factory: async_sessionmaker,
) -> RunStats:
    """Get all statistics for a run database.

    Runs every sub-query on a single shared session/connection instead of
    opening one per sub-stat.

    Args:
        session_factory: Async session factory.

    Returns:
        RunStats instance with all statistics.
    """
    async with session_factory() as session:
        # Get run metadata
        result = await session.execute(
            select(RunMetadata.scraper_name, RunMetadata.status).where(
                RunMetadata.id == RUN_METADATA_ID
            )
        )
        row = result.first()
        scraper_name = row[0] if row else ""
        run_status = row[1] if row else None

        return RunStats(
            queue=await _queue_stats(session),
            throughput=await _throughput_stats(session),
            results=await _result_stats(session),
            errors=await _error_stats(session),
            run_status=run_status,
            scraper_name=scraper_name,
        )


class StatusCount(NamedTuple):
    """One ``group by continuation, status`` row of the requests table."""

    continuation: str
    status: RequestStatus
    request_count: int


class ResultTypeCount(NamedTuple):
    """One result type's valid/invalid split."""

    result_type: str
    valid: int
    invalid: int


class ErrorRow(NamedTuple):
    """One error detail row, as the post-run report prints it."""

    error_type: ErrorType
    error_class: str
    message: str
    request_url: str


class ErrorContinuationCount(NamedTuple):
    """Error rows grouped by the continuation their request belongs to."""

    #: None when the error's request row is gone, or it was stored
    #: without one.
    continuation: str | None
    error_type: ErrorType
    error_count: int


@dataclass
class RunSummary:
    """Post-run aggregates for a run database.

    The driver records per-request failures (HTTP errors, structural /
    validation assumption failures) as ``errors`` rows rather than raising,
    so a run can complete while still holding failed work. This gathers what
    the run actually did — the queue, the harvested results, and the recorded
    errors — plus the counts a host needs to classify the failure mode.

    Every ``list`` field holds :class:`~typing.NamedTuple` rows, so a
    caller can unpack them positionally or read them by name.

    Attributes:
        requests_by_status: :class:`StatusCount` rows, ordered by
            continuation then status.
        results_by_type: :class:`ResultTypeCount` rows, ordered by result
            type.
        errors_total: Total error-row count.
        errors_by_type: ``{error_type: count}``.
        error_rows: The first ``error_rows_limit`` :class:`ErrorRow` rows,
            in id order.
        errors_by_continuation: :class:`ErrorContinuationCount` rows
            ordered by continuation then error type; a ``None``
            continuation means the error's request row is gone.
        requests_total: Total request count.
        errored_requests: Distinct requests with at least one error row.
        archive_error_total: Error rows on archive (file-download) requests
            (``request_type == RequestType.ARCHIVE``). ``archive_error_total ==
            errors_total`` means the scraped pages all succeeded and only
            file archiving failed.
    """

    requests_by_status: list[StatusCount] = field(default_factory=list)
    results_by_type: list[ResultTypeCount] = field(default_factory=list)
    errors_total: int = 0
    errors_by_type: dict[ErrorType, int] = field(default_factory=dict)
    error_rows: list[ErrorRow] = field(default_factory=list)
    errors_by_continuation: list[ErrorContinuationCount] = field(
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
                    Request.continuation,
                    Request.status,
                    sa.func.count(),
                )
                .group_by(Request.continuation, Request.status)
                .order_by(Request.continuation, Request.status)
            )
        ).all()
        summary.requests_by_status = [StatusCount._make(r) for r in rows]
        summary.requests_total = sum(n for _, _, n in rows)

        rows = (
            await session.execute(
                select(
                    Result.result_type,
                    sa.func.sum(
                        sa.case((Result.is_valid == sa.true(), 1), else_=0)
                    ),
                    sa.func.sum(
                        sa.case((Result.is_valid == sa.false(), 1), else_=0)
                    ),
                )
                .group_by(Result.result_type)
                .order_by(Result.result_type)
            )
        ).all()
        summary.results_by_type = [ResultTypeCount._make(r) for r in rows]

        summary.errors_total = (
            await session.execute(select(sa.func.count()).select_from(Error))
        ).scalar_one()

        rows = (
            await session.execute(
                select(Error.error_type, sa.func.count()).group_by(
                    Error.error_type
                )
            )
        ).all()
        summary.errors_by_type = dict(rows)

        rows = (
            await session.execute(
                select(
                    Error.error_type,
                    Error.error_class,
                    Error.message,
                    Error.request_url,
                )
                .order_by(Error.id)
                .limit(error_rows_limit)
            )
        ).all()
        summary.error_rows = [ErrorRow._make(r) for r in rows]

        # continuation lives on the request, not the error, so join through;
        # an outer join keeps errors whose request row is gone (None key).
        rows = (
            await session.execute(
                select(
                    Request.continuation,
                    Error.error_type,
                    sa.func.count(),
                )
                .select_from(Error)
                .join(
                    Request,
                    Error.request_id == Request.id,
                    isouter=True,
                )
                .group_by(Request.continuation, Error.error_type)
                .order_by(Request.continuation, Error.error_type)
            )
        ).all()
        summary.errors_by_continuation = [
            ErrorContinuationCount._make(r) for r in rows
        ]

        summary.errored_requests = (
            await session.execute(
                select(sa.func.count(sa.distinct(Error.request_id))).where(
                    Error.request_id.isnot(None)
                )
            )
        ).scalar_one()

        summary.archive_error_total = (
            await session.execute(
                select(sa.func.count())
                .select_from(Error)
                .join(Request, Error.request_id == Request.id)
                .where(Request.request_type == RequestType.ARCHIVE)
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
    the post-run reporting path.

    Args:
        db_path: Path to the run's SQLite database.
        error_rows_limit: How many error detail rows to include.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist. This runs on the
            post-run reporting path — exactly when run databases get moved
            and renamed — and opening a missing path would both mask the
            mistake (a plausible all-zeroes summary) and materialize it (a
            freshly created empty schema at the wrong location).
    """
    if not db_path.exists():
        raise FileNotFoundError(f"run database does not exist: {db_path}")

    async with SQLManager.open(db_path) as manager:
        return await get_run_summary(
            manager.session_factory, error_rows_limit=error_rows_limit
        )
