"""Request queue operations for SQLManager."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import case, func, or_, select, update

from jkent.driver.database_engine.models import (
    Request,
    RequestStatus,
    RequestType,
)
from jkent.driver.database_engine.sql_manager._types import (
    PreresolvedResponse,
    compute_cache_key,
)
from jkent.driver.database_engine.timestamps import (
    TIMESTAMP_FORMAT,
    epoch_seconds,
    now,
)

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from jkent.data_types import HttpMethod


class RequestQueueMixin:
    """Request table database operations."""

    _lock: asyncio.Lock  # type: ignore[misc]
    _session_factory: async_sessionmaker  # type: ignore[misc]

    async def check_dedup_key_exists(self, dedup_key: str) -> bool:
        """Check if a deduplication key already exists.

        Args:
            dedup_key: The deduplication key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        async with self._session_factory() as session:
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
        for start in range(0, len(dedup_keys), 500):
            chunk = dedup_keys[start : start + 500]
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
        await self._ensure_queue_counter_seeded(session)  # type: ignore[attr-defined]
        assert self._queue_counter is not None  # type: ignore[attr-defined, has-type]
        self._queue_counter += 1  # type: ignore[attr-defined]
        return self._queue_counter  # type: ignore[attr-defined]

    async def find_parent_request_id(self, url: str) -> int | None:
        """Find the request ID for a given URL.

        Used to link child requests to their parent.

        Args:
            url: The URL of the parent request.

        Returns:
            Request ID if found, None otherwise.
        """
        async with self._session_factory() as session:
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
        priority: int,
        request_type: RequestType,
        method: HttpMethod,
        url: str,
        headers_json: str | None,
        cookies_json: str | None,
        body: bytes | None,
        continuation: str,
        current_location: str,
        accumulated_data_json: str | None,
        permanent_json: str | None,
        expected_type: str | None,
        dedup_key: str | None,
        parent_id: int | None,
        is_speculative: bool = False,
        speculation_tracking_id: int | None = None,
        speculative_index: int | None = None,
        verify: str | None = None,
        via_json: str | None = None,
        bypass_rate_limit: bool = False,
        timeout_json: str | None = None,
        json_data: str | None = None,
        files_json: str | None = None,
        auth_json: str | None = None,
        allow_redirects: bool = True,
        proxies_json: str | None = None,
        stream: bool = False,
        cert_json: str | None = None,
        archive_hash_header: str | None = None,
        reseedable: bool | None = None,
        skip_dedup_check: bool = False,
    ) -> int:
        """Insert a new request into the queue.

        Args:
            priority: Request priority (lower = higher priority).
            request_type: Type of request (navigating, non_navigating, etc.).
            method: HTTP method.
            url: Request URL.
            headers_json: JSON-encoded headers.
            cookies_json: JSON-encoded cookies.
            body: Request body bytes.
            continuation: Continuation method name.
            current_location: Current navigation location.
            accumulated_data_json: JSON-encoded accumulated data.
            permanent_json: JSON-encoded permanent data.
            expected_type: Expected type for archive requests.
            dedup_key: Deduplication key.
            parent_id: Parent request ID.
            is_speculative: Whether this is a speculative request.
            speculation_tracking_id: ``speculation_tracking`` row this probe
                came from. The row is created before the probe is enqueued.
            speculative_index: Position of this probe in its template's
                sequence (the ``Speculative.from_int()`` argument).
            bypass_rate_limit: If True, skip rate limiting for this request.
            skip_dedup_check: Pass ``True`` when the caller has already checked
                the dedup key against committed rows, to avoid re-running the
                same lookup inside this insert (the ``uq_requests_dedup_key``
                ON CONFLICT IGNORE constraint still backstops races).

        Returns:
            The ID of the newly inserted request, or the existing ID if
            deduplicated.
        """
        async with self._lock, self._session_factory() as session:
            req_id = await self.insert_request_in_session(
                session,
                priority=priority,
                request_type=request_type,
                method=method,
                url=url,
                headers_json=headers_json,
                cookies_json=cookies_json,
                body=body,
                continuation=continuation,
                current_location=current_location,
                accumulated_data_json=accumulated_data_json,
                permanent_json=permanent_json,
                expected_type=expected_type,
                dedup_key=dedup_key,
                parent_id=parent_id,
                is_speculative=is_speculative,
                speculation_tracking_id=speculation_tracking_id,
                speculative_index=speculative_index,
                verify=verify,
                via_json=via_json,
                bypass_rate_limit=bypass_rate_limit,
                timeout_json=timeout_json,
                json_data=json_data,
                files_json=files_json,
                auth_json=auth_json,
                allow_redirects=allow_redirects,
                proxies_json=proxies_json,
                stream=stream,
                cert_json=cert_json,
                archive_hash_header=archive_hash_header,
                reseedable=reseedable,
                skip_dedup_check=skip_dedup_check,
            )
            await session.commit()
            return req_id

    async def insert_request_in_session(
        self,
        session: AsyncSession,
        *,
        priority: int,
        request_type: RequestType,
        method: HttpMethod,
        url: str,
        headers_json: str | None,
        cookies_json: str | None,
        body: bytes | None,
        continuation: str,
        current_location: str,
        accumulated_data_json: str | None,
        permanent_json: str | None,
        expected_type: str | None,
        dedup_key: str | None,
        parent_id: int | None,
        is_speculative: bool = False,
        speculation_tracking_id: int | None = None,
        speculative_index: int | None = None,
        verify: str | None = None,
        via_json: str | None = None,
        bypass_rate_limit: bool = False,
        timeout_json: str | None = None,
        json_data: str | None = None,
        files_json: str | None = None,
        auth_json: str | None = None,
        allow_redirects: bool = True,
        proxies_json: str | None = None,
        stream: bool = False,
        cert_json: str | None = None,
        archive_hash_header: str | None = None,
        reseedable: bool | None = None,
        skip_dedup_check: bool = False,
        preresolved_response: PreresolvedResponse | None = None,
    ) -> int:
        """Insert a request inside an existing session (no commit).

        Performs the dedup check and INSERT in the same session so callers
        can compose multiple writes into a single transaction. Pass
        ``skip_dedup_check=True`` when the caller has already checked the
        dedup key against committed rows (e.g. a batched staging flush) to
        avoid a redundant per-row lookup.

        When ``preresolved_response`` is given, the request is inserted with
        its response columns already populated and ``preresolved=True`` — one
        atomic INSERT, so a promoted incidental (see the driver's
        ``Request.incidental`` handling) lands with its body in the same
        transaction that enqueues it. The worker then skips the transport and
        runs the continuation against this stored response.
        """
        if not skip_dedup_check and dedup_key is not None:
            existing = await self._find_by_dedup_key_in_session(
                session, dedup_key
            )
            if existing is not None:
                return existing

        queue_counter = await self._get_next_queue_counter_in_session(session)
        cache_key = compute_cache_key(method, url, body, headers_json)

        # A pre-resolved request lands with its response columns already set
        # and preresolved=True, in this same INSERT — the worker will run the
        # continuation against it without hitting the transport.
        preresolved_fields: dict[str, Any] = {}
        if preresolved_response is not None:
            preresolved_fields = {
                "preresolved": True,
                "response_status_code": preresolved_response.status_code,
                "response_headers_json": preresolved_response.headers_json,
                "response_url": preresolved_response.url,
                "content_compressed": preresolved_response.content_compressed,
                "content_size_original": (
                    preresolved_response.content_size_original
                ),
                "content_size_compressed": (
                    preresolved_response.content_size_compressed
                ),
                "compression_dict_id": (
                    preresolved_response.compression_dict_id
                ),
                "response_created_at": now(),
            }

        req = Request(
            status=RequestStatus.PENDING,
            priority=priority,
            queue_counter=queue_counter,
            request_type=request_type,
            method=method,
            url=url,
            headers_json=headers_json,
            cookies_json=cookies_json,
            body=body,
            continuation=continuation,
            current_location=current_location,
            accumulated_data_json=accumulated_data_json,
            permanent_json=permanent_json,
            expected_type=expected_type,
            deduplication_key=dedup_key,
            parent_request_id=parent_id,
            cache_key=cache_key,
            is_speculative=is_speculative,
            speculation_tracking_id=speculation_tracking_id,
            speculative_index=speculative_index,
            verify=verify,
            via_json=via_json,
            bypass_rate_limit=bypass_rate_limit,
            timeout_json=timeout_json,
            json_data=json_data,
            files_json=files_json,
            auth_json=auth_json,
            allow_redirects=allow_redirects,
            proxies_json=proxies_json,
            stream=stream,
            cert_json=cert_json,
            archive_hash_header=archive_hash_header,
            reseedable=reseedable,
            **preresolved_fields,
        )
        session.add(req)
        await session.flush()
        return req.id

    async def dequeue_next_request(
        self,
    ) -> tuple[Any, ...] | None:
        """Atomically dequeue the next pending request.

        This method atomically selects and marks a request as 'in_progress'
        in a single database operation using UPDATE ... RETURNING. This
        prevents race conditions where multiple workers could select the
        same request.

        Returns:
            Row tuple (the RETURNING columns below, positionally coupled to
            the queue deserializer) or None if the queue is empty.
        """
        async with self._lock, self._session_factory() as session:
            subq = (
                select(Request.id)
                .where(
                    Request.status == RequestStatus.PENDING,
                    or_(
                        Request.started_at.is_(None),
                        Request.started_at <= func.datetime("now"),
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
                    Request.id,
                    Request.request_type,
                    Request.method,
                    Request.url,
                    Request.headers_json,
                    Request.cookies_json,
                    Request.body,
                    Request.continuation,
                    Request.current_location,
                    Request.accumulated_data_json,
                    Request.permanent_json,
                    Request.expected_type,
                    Request.priority,
                    Request.is_speculative,
                    Request.speculation_tracking_id,
                    Request.speculative_index,
                    Request.verify,
                    Request.via_json,
                    Request.bypass_rate_limit,
                    Request.deduplication_key,
                    Request.timeout_json,
                    Request.json_data,
                    Request.files_json,
                    Request.auth_json,
                    Request.allow_redirects,
                    Request.proxies_json,
                    Request.stream,
                    Request.cert_json,
                    Request.archive_hash_header,
                    Request.reseedable,
                    Request.parent_request_id,
                    Request.preresolved,
                )
            )
            result = await session.execute(stmt)
            row = result.first()
            await session.commit()
            return tuple(row) if row else None

    async def mark_request_completed(self, request_id: int) -> None:
        """Mark a request as completed.

        Args:
            request_id: The database ID of the request.
        """
        async with self._lock, self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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

    async def get_retry_state(
        self, request_id: int
    ) -> tuple[int, float] | None:
        """Get retry state for a request.

        Args:
            request_id: The database ID of the request.

        Returns:
            Tuple of (retry_count, cumulative_backoff) or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Request.retry_count, Request.cumulative_backoff).where(
                    Request.id == request_id
                )
            )
            row = result.first()
            if row is None:
                return None
            return (row[0], row[1] or 0.0)

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
                eligible again.
            error: Error message from the current attempt.
        """
        async with self._lock, self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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
        """
        async with self._lock, self._session_factory() as session:
            result = await session.execute(
                select(
                    func.min(
                        case(
                            (
                                or_(
                                    Request.started_at.is_(None),
                                    Request.started_at <= func.datetime("now"),
                                ),
                                0.0,
                            ),
                            else_=(
                                epoch_seconds(Request.started_at)
                                - epoch_seconds(func.datetime("now"))
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
        async with self._lock, self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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
        async with self._session_factory() as session:
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
        async with self._session_factory() as session:
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

        Args:
            threshold: Minimum number of undict-compressed responses
                required before a continuation is returned.

        Returns:
            List of continuation names meeting the threshold.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Request.continuation)
                .where(
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

        Mirrors ``train_compression_dict``'s own filter (a stored
        ``response_status_code`` *and* a non-NULL ``content_compressed``):
        archive requests set a status code but store no body, so counting
        them would seed a compactor that then trains over zero responses.
        Counting only rows with a body keeps the seed and the training set
        consistent.
        """
        async with self._lock, self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(
                    Request.continuation == continuation,
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
        async with self._lock, self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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
        async with self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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
        async with self._lock, self._session_factory() as session:
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
