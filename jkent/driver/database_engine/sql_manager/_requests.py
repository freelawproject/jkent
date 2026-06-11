"""Request queue operations for SQLManager."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Final, NamedTuple

import sqlalchemy as sa
from sqlalchemy import case, func, insert, or_, select, update

from jkent.driver.database_engine.models import (
    Request,
    RequestStatus,
    RequestType,
)
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import (
    PreresolvedResponse,
    RequestInsert,
    compute_cache_key,
)
from jkent.driver.database_engine.timestamps import (
    TIMESTAMP_FORMAT,
    epoch_seconds,
    now,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from jkent.data_types import HttpMethod

logger = logging.getLogger(__name__)

#: How many dedup keys go in one ``IN (...)`` predicate. SQLite's
#: ``SQLITE_MAX_VARIABLE_NUMBER`` is 999 on builds before 3.32 and 32766
#: after; 500 stays under the older ceiling with room for the rest of the
#: statement's binds, so the chunking holds on any build we run on.
DEDUP_KEY_CHUNK_SIZE: Final = 500


class DequeuedRow(NamedTuple):
    """One dequeued request row.

    Field names are ``requests`` column names, and the RETURNING list in
    ``dequeue_next_request`` is generated from ``DequeuedRow._fields`` — the
    row shape and this type cannot drift.
    """

    id: int
    request_type: RequestType
    method: HttpMethod
    url: str
    headers_json: str | None
    cookies_json: str | None
    body: bytes | None
    continuation: str
    current_location: str
    accumulated_data_json: str | None
    permanent_json: str | None
    expected_type: str | None
    priority: int
    is_speculative: bool
    speculation_tracking_id: int | None
    speculative_index: int | None
    verify: str | None
    via_json: str | None
    bypass_rate_limit: bool
    deduplication_key: str | None
    timeout_json: str | None
    json_data: str | None
    files_json: str | None
    auth_json: str | None
    allow_redirects: bool
    proxies_json: str | None
    stream: bool
    cert_json: str | None
    archive_hash_header: str | None
    reseedable: bool | None
    parent_request_id: int | None
    preresolved: bool


class RetryState(NamedTuple):
    """A request's retry bookkeeping, as :meth:`get_retry_state` reads it."""

    retry_count: int
    cumulative_backoff: float


class RequestQueueMixin(SQLManagerBase):
    """Request table database operations."""

    async def check_dedup_key_exists(self, dedup_key: str) -> bool:
        """Check if a deduplication key already exists.

        Args:
            dedup_key: The deduplication key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        async with self.session_factory() as session:
            return (
                await self._find_by_dedup_key_in_session(session, dedup_key)
                is not None
            )

    async def _find_by_dedup_key_in_session(
        self, session: AsyncSession, dedup_key: str
    ) -> int | None:
        """Find a request ID by deduplication key inside an existing session."""
        result = await session.execute(
            select(Request.id).where(Request.deduplication_key == dedup_key)
        )
        return result.scalar()

    async def _find_existing_dedup_keys_in_session(
        self, session: AsyncSession, dedup_keys: list[str]
    ) -> set[str]:
        """Return which of *dedup_keys* already exist as requests.

        One query per chunk instead of a point lookup per key, so a step that
        yields many deduplicated children does not issue an N+1 of selects.
        Chunked to stay under SQLite's bound-parameter limit.
        """
        existing: set[str] = set()
        for start in range(0, len(dedup_keys), DEDUP_KEY_CHUNK_SIZE):
            chunk = dedup_keys[start : start + DEDUP_KEY_CHUNK_SIZE]
            result = await session.execute(
                select(Request.deduplication_key).where(
                    Request.deduplication_key.in_(chunk)
                )
            )
            existing.update(row[0] for row in result.all())
        return existing

    async def _get_next_queue_counter_in_session(
        self, session: AsyncSession
    ) -> int:
        """Next queue counter, computed in memory after a one-time seed.

        Reuses the caller's session for the one-time
        ``max(queue_counter)`` seed so no extra connection is opened.
        """
        await self._ensure_queue_counter_seeded(session)
        assert self._queue_counter is not None
        self._queue_counter += 1
        return self._queue_counter

    async def find_parent_request_id(self, url: str) -> int | None:
        """Find the request ID for a given URL.

        Used to link child requests to their parent.

        Args:
            url: The URL of the parent request.

        Returns:
            Request ID if found, None otherwise.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(Request.id)
                .where(
                    Request.url == url,
                    Request.status.in_(
                        [
                            RequestStatus.COMPLETED,
                            RequestStatus.IN_PROGRESS,
                        ]
                    ),
                )
                .order_by(Request.id.desc())
                .limit(1)
            )
            return result.scalar()

    async def insert_request(
        self,
        params: RequestInsert,
        *,
        skip_dedup_check: bool = False,
        preresolved_response: PreresolvedResponse | None = None,
    ) -> int:
        """Insert a new request into the queue (own transaction).

        See :meth:`insert_request_in_session` — this is the same insert
        wrapped in its own write session and commit.

        Returns:
            The ID of the newly inserted request, or the existing ID if
            deduplicated.
        """
        async with self._write_session() as session:
            req_id = await self.insert_request_in_session(
                session,
                params,
                skip_dedup_check=skip_dedup_check,
                preresolved_response=preresolved_response,
            )
            await session.commit()
            return req_id

    async def insert_request_in_session(
        self,
        session: AsyncSession,
        params: RequestInsert,
        *,
        skip_dedup_check: bool = False,
        preresolved_response: PreresolvedResponse | None = None,
    ) -> int:
        """Insert a request inside an existing session (no commit).

        Performs the dedup check and INSERT in the same session so callers
        can compose multiple writes into a single transaction. Pass
        ``skip_dedup_check=True`` when the caller has already checked the
        dedup key against committed rows (e.g. a batched staging flush) to
        avoid a redundant per-row lookup. The insert runs as a Core INSERT
        so that when the ``uq_requests_dedup_key`` ON CONFLICT IGNORE
        constraint fires (a race, or a duplicate key inside the same staged
        batch) the duplicate resolves to the existing row's id instead of
        surfacing as an ORM ``FlushError`` that poisons the whole session.

        When ``preresolved_response`` is given, the request is inserted with
        its response columns already populated and ``preresolved=True`` — one
        atomic INSERT, so a promoted incidental (see the driver's
        ``Request.incidental`` handling) lands with its body in the same
        transaction that enqueues it. The worker then skips the transport and
        runs the continuation against this stored response. A pre-resolved
        insert that deduplicates drops the captured response (the existing
        row already answered this key) — logged as a warning because the
        promotion's continuation will not run.
        """
        if not skip_dedup_check and params.dedup_key is not None:
            existing = await self._find_by_dedup_key_in_session(
                session, params.dedup_key
            )
            if existing is not None:
                self._warn_if_preresolved_dropped(
                    existing, params, preresolved_response
                )
                return existing

        queue_counter = await self._get_next_queue_counter_in_session(session)

        values = params.column_values()
        values.update(
            status=RequestStatus.PENDING,
            queue_counter=queue_counter,
            cache_key=compute_cache_key(
                params.method, params.url, params.body, params.headers_json
            ),
        )
        if preresolved_response is not None:
            # A pre-resolved request lands with its response columns already
            # set and preresolved=True, in this same INSERT — the worker will
            # run the continuation against it without hitting the transport.
            values.update(
                preresolved=True,
                response_status_code=preresolved_response.status_code,
                response_headers_json=preresolved_response.headers_json,
                response_url=preresolved_response.url,
                content_compressed=preresolved_response.content_compressed,
                content_size_original=(
                    preresolved_response.content_size_original
                ),
                content_size_compressed=(
                    preresolved_response.content_size_compressed
                ),
                compression_dict_id=preresolved_response.compression_dict_id,
                response_created_at=now(),
            )

        result = await session.execute(
            insert(Request).values(**values).returning(Request.id)
        )
        inserted_id = result.scalar()
        if inserted_id is not None:
            return inserted_id

        # The DDL-level ON CONFLICT IGNORE swallowed the insert: another
        # writer (or an earlier row in this same transaction) owns this
        # dedup key. Resolve to that row's id.
        existing = (
            await self._find_by_dedup_key_in_session(session, params.dedup_key)
            if params.dedup_key is not None
            else None
        )
        if existing is None:
            raise RuntimeError(
                f"INSERT for {params.url!r} was ignored but no row holds "
                f"dedup key {params.dedup_key!r}"
            )
        self._warn_if_preresolved_dropped(
            existing, params, preresolved_response
        )
        return existing

    @staticmethod
    def _warn_if_preresolved_dropped(
        existing_id: int,
        params: RequestInsert,
        preresolved_response: PreresolvedResponse | None,
    ) -> None:
        """Make a deduplicated pre-resolved insert loud.

        Deduplicating an ordinary enqueue is routine; deduplicating a
        pre-resolved one silently discards a captured response body and the
        continuation that promotion was meant to run, so it must at least be
        observable.
        """
        if preresolved_response is not None:
            logger.warning(
                "Pre-resolved insert for %s deduplicated against request "
                "%d (dedup key %r); the captured response was dropped and "
                "its continuation will not run.",
                params.url,
                existing_id,
                params.dedup_key,
            )

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
            subq = (
                select(Request.id)
                .where(
                    Request.status == RequestStatus.PENDING,
                    or_(
                        Request.started_at.is_(None),
                        Request.started_at <= func.datetime("now", "subsec"),
                    ),
                )
                .order_by(
                    Request.priority.asc(),
                    Request.queue_counter.asc(),
                )
                .limit(1)
                .scalar_subquery()
            )

            stmt = (
                update(Request)
                .where(Request.id == subq)
                .values(
                    status=RequestStatus.IN_PROGRESS,
                    started_at=now(),
                )
                .returning(
                    *(getattr(Request, name) for name in DequeuedRow._fields)
                )
            )
            result = await session.execute(stmt)
            row = result.first()
            await session.commit()
            return DequeuedRow._make(row) if row else None

    async def mark_request_completed(self, request_id: int) -> None:
        """Mark a request as completed.

        Args:
            request_id: The database ID of the request.
        """
        async with self._write_session() as session:
            await self.mark_request_completed_in_session(session, request_id)
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
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status=RequestStatus.FAILED,
                    completed_at=now(),
                    last_error=error_message,
                )
            )
            await session.commit()

    async def get_retry_state(self, request_id: int) -> RetryState | None:
        """Get retry state for a request.

        Args:
            request_id: The database ID of the request.

        Returns:
            :class:`RetryState`, or None if the request does not exist.
            ``cumulative_backoff`` reads NULL as 0.0 — a request that has
            never been retried has accumulated no backoff.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(Request.retry_count, Request.cumulative_backoff).where(
                    Request.id == request_id
                )
            )
            row = result.first()
            if row is None:
                return None
            return RetryState(row[0], row[1] or 0.0)

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
                that just signalled distress, silently. The floor in
                ``ResponseStorage.handle_retry`` keeps real delays far from
                this, so the check is about the failure mode, not a
                reachable value today.
        """
        if not math.isfinite(delay_seconds) or delay_seconds < 0:
            raise ValueError(
                f"retry delay must be finite and non-negative, got "
                f"{delay_seconds!r} for request {request_id}"
            )
        async with self._write_session() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status=RequestStatus.PENDING,
                    retry_count=Request.retry_count + 1,
                    cumulative_backoff=new_cumulative_backoff,
                    last_error=error,
                    # The whole of the retry schedule: ``dequeue_next_request``
                    # only claims a pending row once ``started_at`` is in the
                    # past, so pushing it forward *is* the wait. Written in the
                    # millisecond format every other timestamp uses, with the
                    # fractional delay intact — the jitter that separates two
                    # workers' retries is usually sub-second, and truncating
                    # here would throw exactly that away.
                    started_at=sa.func.strftime(
                        TIMESTAMP_FORMAT,
                        "now",
                        f"+{delay_seconds} seconds",
                    ),
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
        """Count pending and in_progress requests."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(
                    Request.status.in_(
                        [RequestStatus.PENDING, RequestStatus.IN_PROGRESS]
                    )
                )
            )
            return result.scalar() or 0

    async def count_all_requests(self) -> int:
        """Count all requests in the database."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(Request)
            )
            return result.scalar() or 0

    async def avg_completed_request_duration_s(
        self, sample_size: int = 20
    ) -> float | None:
        """Average duration of recently fetched requests, in seconds.

        Differences ``completed_at`` and ``started_at`` over the last
        *sample_size* completed requests, in Unix seconds with the fractional
        part (see ``timestamps.epoch_seconds``).

        Pre-resolved requests are excluded. They never touch the transport —
        their response was attached at enqueue time — so they complete inside
        a millisecond and would round to a zero duration, dragging the average
        toward zero without any fetch having happened. This average exists to
        describe how long the site takes to answer, and a request that never
        asked it anything has nothing to contribute.

        Returns:
            Average duration in seconds, or None if no completed
            requests with timing data exist.
        """
        duration_s = epoch_seconds(Request.completed_at) - epoch_seconds(
            Request.started_at
        )
        subq = (
            select(duration_s.label("duration_s"))
            .where(
                Request.status == RequestStatus.COMPLETED,
                Request.preresolved == sa.false(),
                Request.started_at.isnot(None),
                Request.completed_at.isnot(None),
            )
            .order_by(Request.id.desc())
            .limit(sample_size)
            .subquery()
        )
        async with self.session_factory() as session:
            result = await session.execute(select(func.avg(subq.c.duration_s)))
            avg_s = result.scalar()
            if avg_s is None or avg_s <= 0:
                return None
            return float(avg_s)

    async def continuations_needing_compression_dict(
        self, threshold: int = 1000
    ) -> list[str]:
        """Find continuations with enough responses to train a dictionary.

        Returns continuations that have at least *threshold* responses
        whose ``compression_dict_id`` is NULL (i.e. not yet compressed
        with a trained dictionary).

        Counts only COMPLETED rows, matching
        :meth:`resolved_response_count` and ``train_compression_dict`` —
        see the note there on why a body on a non-completed row is not
        training material.

        Args:
            threshold: Minimum number of undict-compressed responses
                required before a continuation is returned.

        Returns:
            List of continuation names meeting the threshold.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(Request.continuation)
                .where(
                    Request.status == RequestStatus.COMPLETED,
                    Request.response_status_code.isnot(None),
                    Request.content_compressed.isnot(None),
                    Request.compression_dict_id.is_(None),
                )
                .group_by(Request.continuation)
                .having(func.count() >= threshold)
            )
            return [row[0] for row in result.all()]

    async def resolved_response_count(self, continuation: str) -> int:
        """Count a continuation's requests that carry a compressible body.

        Mirrors ``train_compression_dict``'s own filter (COMPLETED, a
        stored ``response_status_code`` *and* a non-NULL
        ``content_compressed``): archive requests set a status code but
        store no body, so counting them would seed a compactor that then
        trains over zero responses. Counting only rows with a body keeps
        the seed and the training set consistent.

        The COMPLETED filter matters for the same reason. The worker
        deliberately writes bodies onto rows that did not succeed —
        ``_handle_transient`` stores a debug snapshot before scheduling a
        retry, and the persistent-HTTP arm stores the observed error
        response and then marks the request FAILED, where it stays. On a
        site that 404s or 403s a meaningful share of a continuation's URLs
        those rows would otherwise both push the continuation over the
        compactor threshold and land in the random training sample, so a
        continuation's dictionary would be trained partly on error pages
        nobody reads back.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(
                    Request.continuation == continuation,
                    Request.status == RequestStatus.COMPLETED,
                    Request.response_status_code.isnot(None),
                    Request.content_compressed.isnot(None),
                )
            )
            return result.scalar() or 0

    # --- Step Control ---

    async def pause_step(self, continuation: str) -> int:
        """Pause processing of requests for a continuation.

        Marks all pending requests as 'held'.

        Args:
            continuation: The continuation method name.

        Returns:
            Number of requests marked as held.
        """
        async with self._write_session() as session:
            result = await session.execute(
                update(Request)
                .where(
                    Request.status == RequestStatus.PENDING,
                    Request.continuation == continuation,
                )
                .values(status=RequestStatus.HELD)
            )
            await session.commit()
            return result.rowcount  # type: ignore[attr-defined]

    async def resume_step(self, continuation: str) -> int:
        """Resume processing of held requests.

        Args:
            continuation: The continuation method name.

        Returns:
            Number of requests restored to pending.
        """
        async with self._write_session() as session:
            result = await session.execute(
                update(Request)
                .where(
                    Request.status == RequestStatus.HELD,
                    Request.continuation == continuation,
                )
                .values(status=RequestStatus.PENDING)
            )
            await session.commit()
            return result.rowcount  # type: ignore[attr-defined]

    async def get_held_count(self, continuation: str | None = None) -> int:
        """Get count of held requests.

        Args:
            continuation: Optional continuation name filter.

        Returns:
            Count of held requests.
        """
        async with self.session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(Request)
                .where(Request.status == RequestStatus.HELD)
            )
            if continuation:
                stmt = stmt.where(Request.continuation == continuation)
            result = await session.execute(stmt)
            return result.scalar() or 0

    # --- Request Cancellation ---

    async def cancel_request(self, request_id: int) -> bool:
        """Cancel a pending request.

        Args:
            request_id: The database ID of the request.

        Returns:
            True if cancelled, False if not found or not cancellable.
        """
        async with self._write_session() as session:
            result = await session.execute(
                update(Request)
                .where(
                    Request.id == request_id,
                    Request.status.in_(
                        [RequestStatus.PENDING, RequestStatus.HELD]
                    ),
                )
                .values(
                    status=RequestStatus.FAILED,
                    completed_at=now(),
                    last_error="Cancelled by user",
                )
            )
            await session.commit()
            return result.rowcount > 0  # type: ignore[attr-defined]

    async def cancel_requests_by_continuation(self, continuation: str) -> int:
        """Cancel all pending/held requests for a continuation.

        Args:
            continuation: The continuation method name.

        Returns:
            Number of requests cancelled.
        """
        async with self._write_session() as session:
            result = await session.execute(
                update(Request)
                .where(
                    Request.continuation == continuation,
                    Request.status.in_(
                        [RequestStatus.PENDING, RequestStatus.HELD]
                    ),
                )
                .values(
                    status=RequestStatus.FAILED,
                    completed_at=now(),
                    last_error="Cancelled by user (batch)",
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[attr-defined]
