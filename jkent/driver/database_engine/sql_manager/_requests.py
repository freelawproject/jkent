"""Request queue operations for SQLManager."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Final, NamedTuple

import sqlalchemy as sa
from sqlalchemy import case, func, insert, or_, select, update

from jkent.driver.database_engine.models import Request, RequestStatus
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import (
    DequeuedRow,
    InsertResult,
    RequestInsert,
    StoredResponse,
    compute_cache_key,
    model_columns,
)
from jkent.driver.database_engine.stored_body import training_sample_clauses
from jkent.driver.database_engine.timestamps import (
    TIMESTAMP_FORMAT,
    epoch_seconds,
    now,
)

if TYPE_CHECKING:
    from sqlalchemy import Select
    from sqlalchemy.ext.asyncio import AsyncSession

    from jkent.driver.database_engine.enums import SpeculationOutcome

logger = logging.getLogger(__name__)

_DEQUEUED_COLUMNS: Final = model_columns(DequeuedRow, Request).values()


def mark_failed(request_id: int, error_message: str) -> sa.Update:
    """The UPDATE that marks ``request_id`` FAILED with ``error_message``."""
    return (
        update(Request)
        .where(Request.id == request_id)
        .values(
            status=RequestStatus.FAILED,
            completed_at=now(),
            last_error=error_message,
        )
    )


def next_pending_id() -> Select[tuple[int]]:
    """The id of the row the next dequeue claims.

    Pending and past any retry backoff, by priority, then FIFO by id —
    the order ``idx_requests_status_priority`` stores, so no sort.
    """
    return (
        select(Request.id)
        .where(
            Request.status == RequestStatus.PENDING,
            or_(
                Request.started_at.is_(None),
                Request.started_at <= func.datetime("now", "subsec"),
            ),
        )
        .order_by(Request.priority.asc(), Request.id.asc())
        .limit(1)
    )


class RetryState(NamedTuple):
    """A request's retry bookkeeping, as :meth:`get_retry_state` reads it."""

    retry_count: int
    cumulative_backoff: float


class RequestQueueMixin(SQLManagerBase):
    """Request table database operations."""

    async def _find_by_dedup_key_in_session(
        self, session: AsyncSession, dedup_key: str
    ) -> int | None:
        """Find a request ID by deduplication key inside an existing session."""
        result = await session.execute(
            select(Request.id).where(Request.deduplication_key == dedup_key)
        )
        return result.scalar()

    async def insert_request(
        self,
        params: RequestInsert,
        *,
        preresolved_response: StoredResponse | None = None,
    ) -> InsertResult:
        """Insert a new request into the queue (own transaction).

        See :meth:`insert_request_in_session` — this is the same insert
        wrapped in its own write session and commit.
        """
        async with self._write_session() as session:
            result = await self.insert_request_in_session(
                session, params, preresolved_response=preresolved_response
            )
            await session.commit()
            return result

    async def insert_request_in_session(
        self,
        session: AsyncSession,
        params: RequestInsert,
        *,
        preresolved_response: StoredResponse | None = None,
    ) -> InsertResult:
        """Insert a request inside an existing session (no commit).

        Deduplication is the ``uq_requests_dedup_key`` constraint's job: its
        ``ON CONFLICT IGNORE`` turns a duplicate — against a committed row, an
        earlier row in this same transaction, or a racing writer — into a
        no-op INSERT, which is then resolved to the existing row's id. No
        lookup precedes the INSERT, so the common (new-key) case is one
        statement. The insert runs as a Core INSERT so the ignored case is a
        missing RETURNING row rather than an ORM ``FlushError`` that poisons
        the whole session.

        When ``preresolved_response`` is given, the request is inserted with
        its response columns already populated and ``preresolved=True`` — one
        atomic INSERT, so a promoted incidental (see the driver's
        ``Request.incidental`` handling) lands with its body in the same
        transaction that enqueues it. The worker then skips the transport and
        runs the step against this stored response. A pre-resolved
        insert that deduplicates drops the captured response in favour of the
        existing row, and logs a warning.

        Returns:
            :class:`InsertResult`: the row's id and whether it is new.
        """
        values = params.model_dump()
        values.update(
            status=RequestStatus.PENDING,
            cache_key=compute_cache_key(params),
        )
        if preresolved_response is not None:
            values.update(
                preresolved_response.model_dump(),
                preresolved=True,
                response_created_at=now(),
            )

        result = await session.execute(
            insert(Request).values(**values).returning(Request.id)
        )
        inserted_id = result.scalar()
        if inserted_id is not None:
            return InsertResult(inserted_id, True)

        # The DDL-level ON CONFLICT IGNORE swallowed the insert: another
        # writer (or an earlier row in this same transaction) owns this
        # dedup key. Resolve to that row's id.
        existing = (
            await self._find_by_dedup_key_in_session(
                session, params.deduplication_key
            )
            if params.deduplication_key is not None
            else None
        )
        if existing is None:
            raise RuntimeError(
                f"INSERT for {params.url!r} was ignored but no row holds "
                f"dedup key {params.deduplication_key!r}"
            )
        if preresolved_response is not None:
            # Deduplicating an ordinary enqueue is routine; deduplicating a
            # pre-resolved one discards a captured response body.
            logger.warning(
                "Pre-resolved insert for %s deduplicated against request "
                "%d (dedup key %r); the captured response was dropped.",
                params.url,
                existing,
                params.deduplication_key,
            )
        return InsertResult(existing, False)

    async def dequeue_next_request(
        self,
    ) -> DequeuedRow | None:
        """Atomically dequeue the next pending request.

        This method atomically selects and marks a request as 'in_progress'
        in a single database operation using UPDATE ... RETURNING. This
        prevents race conditions where multiple workers could select the
        same request.

        The retry-backoff gate compares ``started_at`` against
        ``datetime('now', 'subsec')``: ``schedule_retry`` writes the stamp
        with its fractional part intact, and a whole-second ``'now'`` would
        quantize eligibility to second boundaries — re-synchronizing the
        very retries the sub-second jitter was written to spread out.

        Returns:
            :class:`DequeuedRow` or None if the queue is empty.
        """
        async with self._write_session() as session:
            stmt = (
                update(Request)
                .where(Request.id == next_pending_id().scalar_subquery())
                .values(
                    status=RequestStatus.IN_PROGRESS,
                    started_at=now(),
                )
                .returning(*_DEQUEUED_COLUMNS)
            )
            result = await session.execute(stmt)
            row = result.first()
            await session.commit()
            return DequeuedRow.model_validate(row) if row else None

    async def mark_request_completed(
        self,
        request_id: int,
        *,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> None:
        """Mark a request as completed.

        Args:
            request_id: The database ID of the request.
            speculation_outcome: Written to the row when given — the outcome
                of a probe completed without storing a response. ``None``
                leaves the column as it is.
        """
        async with self._write_session() as session:
            await self.mark_request_completed_in_session(session, request_id)
            if speculation_outcome is not None:
                await session.execute(
                    update(Request)
                    .where(Request.id == request_id)
                    .values(speculation_outcome=speculation_outcome)
                )
            await session.commit()

    async def mark_request_completed_in_session(
        self, session: AsyncSession, request_id: int
    ) -> None:
        """Mark a request as completed inside an existing session (no commit)."""
        await session.execute(
            update(Request)
            .where(Request.id == request_id)
            .values(
                status=RequestStatus.COMPLETED,
                completed_at=now(),
            )
        )

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        """Mark a request as failed.

        Args:
            request_id: The database ID of the request.
            error_message: Error message describing the failure.
        """
        async with self._write_session() as session:
            await session.execute(mark_failed(request_id, error_message))
            await session.commit()

    async def get_retry_state(self, request_id: int) -> RetryState | None:
        """Get retry state for a request.

        Args:
            request_id: The database ID of the request.

        Returns:
            :class:`RetryState`, or None if the request does not exist.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(Request.retry_count, Request.cumulative_backoff).where(
                    Request.id == request_id
                )
            )
            row = result.first()
            return RetryState._make(row) if row else None

    async def schedule_retry(
        self,
        request_id: int,
        new_cumulative_backoff: float,
        delay_seconds: float,
        error: str,
    ) -> None:
        """Schedule a request for retry with backoff.

        The delay is not stored as a number — it is expressed by pushing
        ``started_at`` that far into the future, which is the gate
        ``dequeue_next_request`` actually honours.

        Args:
            request_id: The database ID of the request.
            new_cumulative_backoff: Updated cumulative backoff time.
            delay_seconds: How long the request must wait before it is
                eligible again. Must be finite and non-negative.
            error: Error message from the current attempt.

        Raises:
            ValueError: If *delay_seconds* is NaN, infinite or negative.
                SQLite answers a modifier it cannot parse with NULL rather
                than an error, and a NULL ``started_at`` is not "no
                schedule" — it is the ready-now sentinel both
                ``dequeue_next_request`` and ``seconds_until_next_pending``
                honour. Such a delay would therefore *delete* the backoff
                and re-dispatch the request in a tight loop against a host
                that just signalled distress, silently. Also raised, with
                nothing written, for a finite delay SQLite still renders as
                NULL — one landing past 9999-12-31, or large enough to
                format in exponent notation.
        """
        if not math.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError(
                f"retry delay must be finite and non-negative, got "
                f"{delay_seconds!r} for request {request_id}"
            )
        # The whole of the retry schedule: ``dequeue_next_request`` only
        # claims a pending row once ``started_at`` is in the past, so pushing
        # it forward *is* the wait. Written in the millisecond format every
        # other timestamp uses, with the fractional delay intact — the jitter
        # that separates two workers' retries is usually sub-second, and
        # truncating here would throw exactly that away.
        retry_at = sa.func.strftime(
            TIMESTAMP_FORMAT, "now", f"+{delay_seconds} seconds"
        )
        async with self._write_session() as session:
            # Rendered first, so a NULL is refused rather than stored.
            if (await session.execute(select(retry_at))).scalar() is None:
                raise ValueError(
                    f"retry delay {delay_seconds!r} for request {request_id} "
                    "cannot be scheduled: SQLite renders its retry time as "
                    "NULL"
                )
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status=RequestStatus.PENDING,
                    retry_count=Request.retry_count + 1,
                    cumulative_backoff=new_cumulative_backoff,
                    last_error=error,
                    started_at=retry_at,
                )
            )
            await session.commit()

    async def count_pending_requests(self) -> int:
        """Count pending requests in the queue."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status == RequestStatus.PENDING)
            )
            return result.scalar() or 0

    async def seconds_until_next_pending(self) -> float | None:
        """Seconds until the soonest pending request becomes dequeuable.

        Returns 0.0 when a pending request is ready now (``started_at`` is
        NULL or already past — e.g. a retry whose backoff has elapsed), the
        positive gap until the soonest future-scheduled request when every
        pending request is still in retry backoff, or None when there are no
        pending requests at all. Lets a worker that just dequeued None sleep
        exactly until the next retry is ready instead of retiring and leaving
        it to the slow monitor poll.

        Uses ``datetime('now', 'subsec')`` on both sides for the same reason
        as ``dequeue_next_request``: retry stamps carry milliseconds, and a
        whole-second "now" would quantize both the gate and the reported gap.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(
                    func.min(
                        case(
                            (
                                or_(
                                    Request.started_at.is_(None),
                                    Request.started_at
                                    <= func.datetime("now", "subsec"),
                                ),
                                0.0,
                            ),
                            else_=(
                                epoch_seconds(Request.started_at)
                                - epoch_seconds(func.datetime("now", "subsec"))
                            ),
                        )
                    )
                ).where(Request.status == RequestStatus.PENDING)
            )
            value = result.scalar()
            if value is None:
                return None
            return max(0.0, float(value))

    async def restamp_request_start(self, request_id: int) -> None:
        """Reset a request's start timestamps to now, after the rate gate.

        ``started_at`` is stamped at dequeue, before the rate-limiter gate, so
        a DB-derived duration would otherwise include time spent waiting for a
        token. Re-stamping just before execution makes the persisted start
        mark the execute region rather than the queue wait.
        """
        async with self._write_session() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    started_at=now(),
                )
            )
            await session.commit()

    async def count_active_requests(self) -> int:
        """Count the rows in :meth:`RequestStatus.active` (pending + claimed)."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status.in_(RequestStatus.active()))
            )
            return result.scalar() or 0

    async def count_all_requests(self) -> int:
        """Count all requests in the database."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(Request)
            )
            return result.scalar() or 0

    async def has_any_requests(self) -> bool:
        """Whether the database holds any request rows at all."""
        return await self.count_all_requests() > 0

    async def resolved_response_count(self, step: str) -> int:
        """Count a step's dictionary-training samples.

        The rows ``train_compression_dict`` samples from; see
        :func:`~jkent.driver.database_engine.stored_body.training_sample_clauses`.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(*training_sample_clauses(step))
            )
            return result.scalar() or 0
