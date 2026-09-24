"""Statistics dataclasses and queries for a run database.

Two readers share one set of grouped queries:

- :func:`get_stats` — the *live* progress surface (``ScrapeRun.stats``):
  queue counts by status, throughput, result and error totals. Read-only,
  so a host can poll it alongside the scrape's writers.
- :func:`get_run_summary` / :func:`read_run_summary` — the *post-run*
  report: the same grouped rows in list form, plus the error detail rows
  and the counts a host needs to classify the failure mode.

Every aggregate is derived from a ``GROUP BY`` row set rather than queried
on its own: ``QueueStats.total`` is the sum of ``by_status``, and
``ResultStats.total`` / ``ErrorStats.total`` are summed from their rows when
the ``from_rows`` factory builds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

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
    from collections.abc import Iterable, Mapping
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


# --- Grouped rows ---------------------------------------------------------


class StatusCount(NamedTuple):
    """One ``group by step, status`` row of the requests table."""

    step: str
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


class ErrorStepCount(NamedTuple):
    """Error rows grouped by the step their request belongs to."""

    #: None when the error's request row is gone, or it was stored
    #: without one.
    step: str | None
    error_type: ErrorType
    error_count: int


# --- Live stats -----------------------------------------------------------


@dataclass
class QueueStats:
    """Statistics about the request queue.

    ``by_status`` is the data; the named counters are views over it, so a
    status this module has never heard of still lands in ``by_status`` and
    ``total`` instead of vanishing from the aggregate.

    Attributes:
        by_status: Row count per :class:`RequestStatus` (absent = 0).
        by_step: Counts by step method name, then status.
    """

    by_status: dict[RequestStatus, int] = field(default_factory=dict)
    by_step: dict[str, dict[RequestStatus, int]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[StatusCount]) -> QueueStats:
        """Fold ``group by step, status`` rows into both breakdowns."""
        stats = cls()
        for step, status, count in rows:
            stats.by_status[status] = stats.by_status.get(status, 0) + count
            stats.by_step.setdefault(step, {})[status] = count
        return stats

    def count(self, statuses: Iterable[RequestStatus]) -> int:
        """Rows in any of ``statuses`` (e.g. ``RequestStatus.active()``)."""
        return sum(self.by_status.get(s, 0) for s in statuses)

    @property
    def total(self) -> int:
        """Every row, whatever its status."""
        return sum(self.by_status.values())

    @property
    def active(self) -> int:
        """Rows the run still owns (:meth:`RequestStatus.active`)."""
        return self.count(RequestStatus.active())

    @property
    def pending(self) -> int:
        return self.by_status.get(RequestStatus.PENDING, 0)

    @property
    def in_progress(self) -> int:
        return self.by_status.get(RequestStatus.IN_PROGRESS, 0)

    @property
    def completed(self) -> int:
        return self.by_status.get(RequestStatus.COMPLETED, 0)

    @property
    def failed(self) -> int:
        return self.by_status.get(RequestStatus.FAILED, 0)

    @property
    def stubbed(self) -> int:
        """Replay databases only."""
        return self.by_status.get(RequestStatus.STUBBED, 0)


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
            over completed requests, excluding pre-resolved ones: they never
            touch the transport, so they finish inside a millisecond and
            would drag the mean toward zero without any fetch having
            happened.
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

    @classmethod
    def from_rows(cls, rows: Iterable[ResultTypeCount]) -> ResultStats:
        stats = cls()
        for result_type, valid, invalid in rows:
            stats.valid += valid
            stats.invalid += invalid
            stats.by_type[result_type] = valid + invalid
        stats.total = stats.valid + stats.invalid
        return stats


@dataclass
class ErrorStats:
    """Statistics about errors.

    Attributes:
        total: Total number of errors.
        by_type: Counts by error type (structural, validation, transient).
        by_step: Counts by step method name; the ``None``
            key holds errors whose request row is gone (or that were stored
            without one), so the breakdown always sums to ``total``.
    """

    total: int = 0
    by_type: dict[ErrorType, int] = field(default_factory=dict)
    by_step: dict[str | None, int] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls,
        by_type: Mapping[ErrorType, int],
        by_step: Iterable[ErrorStepCount],
    ) -> ErrorStats:
        stats = cls(by_type=dict(by_type), total=sum(by_type.values()))
        for step, _, count in by_step:
            stats.by_step[step] = stats.by_step.get(step, 0) + count
        return stats


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


# --- The shared grouped queries ------------------------------------------


async def _request_status_counts(session: AsyncSession) -> list[StatusCount]:
    rows = await session.execute(
        select(Request.step, Request.status, sa.func.count())
        .group_by(Request.step, Request.status)
        .order_by(Request.step, Request.status)
    )
    return [StatusCount._make(r) for r in rows.all()]


async def _result_type_counts(session: AsyncSession) -> list[ResultTypeCount]:
    rows = await session.execute(
        select(
            Result.result_type,
            sa.func.coalesce(
                sa.func.sum(
                    sa.case((Result.is_valid == sa.true(), 1), else_=0)
                ),
                0,
            ),
            sa.func.coalesce(
                sa.func.sum(
                    sa.case((Result.is_valid == sa.false(), 1), else_=0)
                ),
                0,
            ),
        )
        .group_by(Result.result_type)
        .order_by(Result.result_type)
    )
    return [ResultTypeCount._make(r) for r in rows.all()]


async def _error_type_counts(session: AsyncSession) -> dict[ErrorType, int]:
    rows = await session.execute(
        select(Error.error_type, sa.func.count()).group_by(Error.error_type)
    )
    return dict(rows.tuples().all())


async def _error_step_counts(
    session: AsyncSession,
) -> list[ErrorStepCount]:
    # step lives on the request, not the error, so join through; an
    # outer join keeps errors whose request row is gone (None key) so this
    # breakdown always sums to the error total.
    rows = await session.execute(
        select(Request.step, Error.error_type, sa.func.count())
        .select_from(Error)
        .join(Request, Error.request_id == Request.id, isouter=True)
        .group_by(Request.step, Error.error_type)
        .order_by(Request.step, Error.error_type)
    )
    return [ErrorStepCount._make(r) for r in rows.all()]


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
            # which avg() ignores. A request whose response was attached at
            # enqueue time never asked the site anything, so it has nothing
            # to say about how long the site takes to answer.
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


async def get_stats(
    session_factory: async_sessionmaker[AsyncSession],
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
        result = await session.execute(
            select(RunMetadata.scraper_name, RunMetadata.status).where(
                RunMetadata.id == RUN_METADATA_ID
            )
        )
        row = result.first()
        scraper_name = row[0] if row else ""
        run_status = row[1] if row else None

        return RunStats(
            queue=QueueStats.from_rows(await _request_status_counts(session)),
            throughput=await _throughput_stats(session),
            results=ResultStats.from_rows(await _result_type_counts(session)),
            errors=ErrorStats.from_rows(
                await _error_type_counts(session),
                await _error_step_counts(session),
            ),
            run_status=run_status,
            scraper_name=scraper_name,
        )


# --- Post-run summary -----------------------------------------------------


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
            step then status.
        results_by_type: :class:`ResultTypeCount` rows, ordered by result
            type.
        errors_total: Total error-row count.
        errors_by_type: ``{error_type: count}``.
        error_rows: The first ``error_rows_limit`` :class:`ErrorRow` rows,
            in id order.
        errors_by_step: :class:`ErrorStepCount` rows
            ordered by step then error type; a ``None``
            step means the error's request row is gone.
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
    errors_by_step: list[ErrorStepCount] = field(default_factory=list)
    requests_total: int = 0
    errored_requests: int = 0
    archive_error_total: int = 0


async def get_run_summary(
    session_factory: async_sessionmaker[AsyncSession],
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
        summary.requests_by_status = await _request_status_counts(session)
        summary.requests_total = sum(
            n for _, _, n in summary.requests_by_status
        )
        summary.results_by_type = await _result_type_counts(session)
        summary.errors_by_type = await _error_type_counts(session)
        summary.errors_total = sum(summary.errors_by_type.values())
        summary.errors_by_step = await _error_step_counts(session)

        rows = await session.execute(
            select(
                Error.error_type,
                Error.error_class,
                Error.message,
                Error.request_url,
            )
            .order_by(Error.id)
            .limit(error_rows_limit)
        )
        summary.error_rows = [ErrorRow._make(r) for r in rows.all()]

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
        FileNotFoundError: If ``db_path`` does not exist or is a 0-byte
            file (see :meth:`SQLManager.open`). This runs on the
            post-run reporting path — exactly when run databases get moved
            and renamed — and opening a missing path would both mask the
            mistake (a plausible all-zeroes summary) and materialize it (a
            freshly created empty schema at the wrong location).
    """
    async with SQLManager.open(db_path) as manager:
        return await get_run_summary(
            manager.session_factory, error_rows_limit=error_rows_limit
        )
