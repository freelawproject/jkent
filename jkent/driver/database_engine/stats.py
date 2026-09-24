"""Statistics dataclasses and queries for a run database.

Two readers share one set of grouped queries:

- :func:`get_stats` — the *live* progress surface (``ScrapeRun.stats``):
  queue counts by status, throughput, result and error totals. Read-only,
  so a host can poll it alongside the scrape's writers.
- :func:`get_run_summary` / :func:`read_run_summary` — the *post-run*
  report: the same grouped rows in list form, plus the error detail rows
  and the counts a host needs to classify the failure mode.

Each number is computed in one place. Every aggregate is derived from a
``GROUP BY`` row set rather than queried on its own, and every total is a
property over the folded counts, so it cannot drift from them.
:class:`ThroughputStats` takes its completed count from :class:`QueueStats`,
and :class:`RunSummary` keeps the raw rows and reads its totals through
those same folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
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
    from collections.abc import Iterable
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

    #: None when the error was stored without a request (raised outside
    #: one); deleting a request cascades to its errors.
    step: str | None
    error_type: ErrorType
    error_count: int


# --- Live stats -----------------------------------------------------------


@dataclass
class QueueStats:
    """Statistics about the request queue.

    ``by_status`` is the data; the named counters are views over it, so
    ``total`` counts every status, including ones without a named counter.

    Attributes:
        by_status: Row count per :class:`RequestStatus` (absent = 0).
        by_step: Counts by step method name, then status.
    """

    by_status: dict[RequestStatus, int] = field(default_factory=dict)
    by_step: dict[str, dict[RequestStatus, int]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[StatusCount]) -> QueueStats:
        """Fold ``group by step, status`` rows into both breakdowns.

        Args:
            rows: ``(step, status, count)`` rows from the queue query.

        Returns:
            The counts, broken down by status and by step.
        """
        stats = cls()
        for step, status, count in rows:
            stats.by_status[status] = stats.by_status.get(status, 0) + count
            stats.by_step.setdefault(step, {})[status] = count
        return stats

    def count(self, statuses: Iterable[RequestStatus]) -> int:
        """Count rows in any of ``statuses``.

        Args:
            statuses: The statuses to total, e.g. ``RequestStatus.active()``.

        Returns:
            How many rows carry one of them.
        """
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
        """Rows waiting to be dequeued."""
        return self.by_status.get(RequestStatus.PENDING, 0)

    @property
    def in_progress(self) -> int:
        """Rows leased by a worker."""
        return self.by_status.get(RequestStatus.IN_PROGRESS, 0)

    @property
    def completed(self) -> int:
        """Rows that finished successfully."""
        return self.by_status.get(RequestStatus.COMPLETED, 0)

    @property
    def failed(self) -> int:
        """Rows that ended in a terminal failure."""
        return self.by_status.get(RequestStatus.FAILED, 0)

    @property
    def stubbed(self) -> int:
        """Replay databases only."""
        return self.by_status.get(RequestStatus.STUBBED, 0)


@dataclass
class ThroughputStats:
    """Statistics about request throughput.

    Attributes:
        total_completed: :attr:`QueueStats.completed`, carried here for
            :attr:`requests_per_minute`.
        total_duration_seconds: Wall clock from the run's first start
            (``run_metadata.started_at``) to the latest ``completed_at``
            over every request, whatever its status. It spans *every*
            session this database has seen, not the current one: a run
            interrupted on Monday and resumed on Wednesday spans both days
            plus the idle gap between them.
        average_response_time_seconds: Mean ``completed_at - started_at``
            over completed requests, excluding pre-resolved ones: they never
            touch the transport, so they finish inside a millisecond and
            would drag the mean toward zero without any fetch having
            happened. None when no completed request touched the transport.
    """

    total_completed: int = 0
    total_duration_seconds: float = 0.0
    average_response_time_seconds: float | None = None

    @property
    def requests_per_minute(self) -> float:
        """``total_completed`` per minute of ``total_duration_seconds``.

        Inherits that window, so on a resumed database this is a lifetime
        average including time no run was open — not the rate the run is
        currently achieving.
        """
        if self.total_duration_seconds <= 0:
            return 0.0
        return self.total_completed / self.total_duration_seconds * 60


@dataclass
class ResultStats:
    """Statistics about scraped results.

    Attributes:
        valid: Number of valid results.
        invalid: Number of invalid results.
        by_type: Counts by result type (Pydantic model name).
    """

    valid: int = 0
    invalid: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Valid plus invalid results."""
        return self.valid + self.invalid

    @classmethod
    def from_rows(cls, rows: Iterable[ResultTypeCount]) -> ResultStats:
        """Fold ``group by result_type`` rows into the result breakdown.

        Args:
            rows: ``(result_type, valid, invalid)`` rows.

        Returns:
            The result counts, by validity and by type.
        """
        stats = cls()
        for result_type, valid, invalid in rows:
            stats.valid += valid
            stats.invalid += invalid
            stats.by_type[result_type] = valid + invalid
        return stats


@dataclass
class ErrorStats:
    """Statistics about errors.

    Attributes:
        by_type: Counts by error type (structural, validation, transient).
        by_step: Counts by step method name; the ``None``
            key holds errors stored without a request, so the breakdown
            always sums to ``total``.
    """

    by_type: dict[ErrorType, int] = field(default_factory=dict)
    by_step: dict[str | None, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """All error rows, with or without a request."""
        return sum(self.by_step.values())

    @classmethod
    def from_rows(cls, rows: Iterable[ErrorStepCount]) -> ErrorStats:
        """Fold ``group by step, error_type`` rows into both breakdowns.

        Args:
            rows: ``(step, error_type, count)`` rows; the step is ``None``
                for errors stored without a request.

        Returns:
            The error counts, by type and by step.
        """
        stats = cls()
        for step, error_type, count in rows:
            stats.by_type[error_type] = (
                stats.by_type.get(error_type, 0) + count
            )
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
            sa.func.sum(sa.case((Result.is_valid == sa.true(), 1), else_=0)),
            sa.func.sum(sa.case((Result.is_valid == sa.false(), 1), else_=0)),
        )
        .group_by(Result.result_type)
        .order_by(Result.result_type)
    )
    return [ResultTypeCount._make(r) for r in rows.all()]


async def _error_step_counts(
    session: AsyncSession,
) -> list[ErrorStepCount]:
    # step lives on the request, not the error, so join through; an
    # outer join keeps errors stored without a request (None key) so this
    # breakdown always sums to the error total.
    rows = await session.execute(
        select(Request.step, Error.error_type, sa.func.count())
        .select_from(Error)
        .join(Request, Error.request_id == Request.id, isouter=True)
        .group_by(Request.step, Error.error_type)
        .order_by(Request.step, Error.error_type)
    )
    return [ErrorStepCount._make(r) for r in rows.all()]


async def _throughput_stats(
    session: AsyncSession, completed: int
) -> ThroughputStats:
    """Compute throughput stats on an existing session.

    Args:
        session: An open session on the run database.
        completed: :attr:`QueueStats.completed`, counted there.

    Returns:
        The run's throughput statistics.
    """
    # The run's start comes from run_metadata, not min(requests.started_at):
    # a retry or the post-gate restamp overwrites a row's started_at, so the
    # earliest one left need not be when the run began. The window is
    # aggregated in the same SELECT rather than re-sending the timestamps to
    # SQLite in a second round trip just to subtract them.
    run_started = (
        select(epoch_seconds(RunMetadata.started_at))
        .where(RunMetadata.id == RUN_METADATA_ID)
        .scalar_subquery()
    )
    duration_s = epoch_seconds(Request.completed_at) - epoch_seconds(
        Request.started_at
    )
    result = await session.execute(
        select(
            # Pre-resolved requests never asked the site anything, so they
            # have nothing to say about how long it takes to answer; they
            # contribute NULL here, which avg() ignores.
            sa.func.avg(
                sa.case(
                    (
                        sa.and_(
                            Request.status == RequestStatus.COMPLETED,
                            Request.preresolved == sa.false(),
                        ),
                        duration_s,
                    ),
                    else_=None,
                )
            ),
            sa.func.max(epoch_seconds(Request.completed_at)) - run_started,
        )
    )
    average, window = result.one()
    return ThroughputStats(
        total_completed=completed,
        total_duration_seconds=window or 0.0,
        average_response_time_seconds=average,
    )


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

        queue = QueueStats.from_rows(await _request_status_counts(session))
        return RunStats(
            queue=queue,
            throughput=await _throughput_stats(session, queue.completed),
            results=ResultStats.from_rows(await _result_type_counts(session)),
            errors=ErrorStats.from_rows(await _error_step_counts(session)),
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
        error_rows: The first ``error_rows_limit`` :class:`ErrorRow` rows,
            in id order.
        errors_by_step: :class:`ErrorStepCount` rows
            ordered by step then error type; a ``None``
            step means the error was stored without a request.
        errored_requests: Distinct requests with at least one error row.
        archive_error_total: Error rows on archive (file-download) requests
            (``request_type == RequestType.ARCHIVE``). See
            :attr:`only_archive_failures`.
        failed_page_requests: Non-archive requests that ended FAILED,
            whether or not an error row was filed for them (an
            undeserializable row is failed without one).
    """

    requests_by_status: list[StatusCount] = field(default_factory=list)
    results_by_type: list[ResultTypeCount] = field(default_factory=list)
    error_rows: list[ErrorRow] = field(default_factory=list)
    errors_by_step: list[ErrorStepCount] = field(default_factory=list)
    errored_requests: int = 0
    archive_error_total: int = 0
    failed_page_requests: int = 0

    # Folded once: get_run_summary fills the rows before anything reads the
    # properties over them.
    @cached_property
    def _queue_stats(self) -> QueueStats:
        return QueueStats.from_rows(self.requests_by_status)

    @cached_property
    def _error_stats(self) -> ErrorStats:
        return ErrorStats.from_rows(self.errors_by_step)

    @property
    def requests_total(self) -> int:
        """Total request count."""
        return self._queue_stats.total

    @property
    def errors_total(self) -> int:
        """Total error-row count."""
        return self._error_stats.total

    @property
    def errors_by_type(self) -> dict[ErrorType, int]:
        """``{error_type: count}``."""
        return self._error_stats.by_type

    @property
    def only_archive_failures(self) -> bool:
        """Errors were recorded, all on archive requests, and no page failed.

        The scraped pages all succeeded; only file archiving failed. False
        for a run with no errors at all.
        """
        return (
            0 < self.errors_total == self.archive_error_total
            and self.failed_page_requests == 0
        )


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
        summary.results_by_type = await _result_type_counts(session)
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

        # count(DISTINCT) skips the NULL request_id of request-less errors,
        # and the outer join keeps them out of the archive sum.
        errored, archive = (
            await session.execute(
                select(
                    sa.func.count(sa.distinct(Error.request_id)),
                    sa.func.coalesce(
                        sa.func.sum(
                            sa.case(
                                (
                                    Request.request_type
                                    == RequestType.ARCHIVE,
                                    1,
                                ),
                                else_=0,
                            )
                        ),
                        0,
                    ),
                )
                .select_from(Error)
                .outerjoin(Request, Error.request_id == Request.id)
            )
        ).one()
        summary.errored_requests = errored
        summary.archive_error_total = archive
        summary.failed_page_requests = (
            await session.execute(
                select(sa.func.count()).where(
                    Request.status == RequestStatus.FAILED,
                    Request.request_type != RequestType.ARCHIVE,
                )
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

    Returns:
        The run's summary, read from the database on disk.

    Raises:
        FileNotFoundError: If ``db_path`` does not exist or is a 0-byte
            file (see :meth:`SQLManager.open`). This runs on the
            post-run reporting path — exactly when run databases get moved
            and renamed — and opening a missing path would both mask the
            mistake (a plausible all-zeroes summary) and materialize it (a
            freshly created empty schema at the wrong location).
        NotARunDatabaseError: If the file is not a run database. It is
            opened read-only, so a reader never adds tables to it.
    """  # noqa: DOC502 -- raised by SQLManager.open, not inline
    async with SQLManager.open(db_path, read_only=True) as manager:
        return await get_run_summary(
            manager.session_factory, error_rows_limit=error_rows_limit
        )
