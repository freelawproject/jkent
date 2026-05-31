"""Request queue operations for SQLManager."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func, or_, select, update

from kent.driver.persistent_driver.models import Request
from kent.driver.persistent_driver.sql_manager._types import compute_cache_key

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession

    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class RequestQueueMixin:
    """Request table database operations."""

    _lock: asyncio.Lock
    _session_factory: ScopedSessionFactory

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

    async def _find_by_dedup_key(self, dedup_key: str) -> int | None:
        """Find an existing request ID by deduplication key.

        Args:
            dedup_key: The deduplication key to look up.

        Returns:
            The request ID if found, None otherwise.
        """
        async with self._session_factory() as session:
            return await self._find_by_dedup_key_in_session(session, dedup_key)

    async def _find_by_dedup_key_in_session(
        self, session: AsyncSession, dedup_key: str
    ) -> int | None:
        """Find a request ID by deduplication key inside an existing session."""
        result = await session.execute(
            select(Request.id).where(Request.deduplication_key == dedup_key)
        )
        return result.scalar()

    async def _get_next_queue_counter_in_session(
        self, session: AsyncSession
    ) -> int:
        """Get the next queue counter inside an existing session."""
        result = await session.execute(select(func.max(Request.queue_counter)))
        return (result.scalar() or 0) + 1

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
                    Request.status.in_(["completed", "in_progress"]),  # type: ignore[attr-defined]
                )
                .order_by(Request.id.desc())  # type: ignore[union-attr]
                .limit(1)
            )
            return result.scalar()

    async def insert_request(
        self,
        priority: int,
        request_type: str,
        method: str,
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
        speculation_id: str | None = None,
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
        hateoas: bool | None = None,
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
            speculation_id: JSON tuple ["func_name", spec_id] for speculative requests.
            bypass_rate_limit: If True, skip rate limiting for this request.

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
                speculation_id=speculation_id,
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
                hateoas=hateoas,
            )
            await session.commit()
            return req_id

    async def insert_request_in_session(
        self,
        session: AsyncSession,
        *,
        priority: int,
        request_type: str,
        method: str,
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
        speculation_id: str | None = None,
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
        hateoas: bool | None = None,
    ) -> int:
        """Insert a request inside an existing session (no commit).

        Performs the dedup check and INSERT in the same session so callers
        can compose multiple writes into a single transaction.
        """
        if dedup_key is not None:
            existing = await self._find_by_dedup_key_in_session(
                session, dedup_key
            )
            if existing is not None:
                return existing

        queue_counter = await self._get_next_queue_counter_in_session(session)
        created_at_ns = time.monotonic_ns()
        cache_key = compute_cache_key(method, url, body, headers_json)

        req = Request(
            status="pending",
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
            created_at_ns=created_at_ns,
            cache_key=cache_key,
            is_speculative=is_speculative,
            speculation_id=speculation_id,
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
            hateoas=hateoas,
        )
        session.add(req)
        await session.flush()
        return req.id  # type: ignore[return-value]

    async def insert_entry_request(
        self,
        priority: int,
        method: str,
        url: str,
        headers_json: str | None,
        cookies_json: str | None,
        body: bytes | None,
        continuation: str,
        current_location: str,
        accumulated_data_json: str | None,
        permanent_json: str | None,
        dedup_key: str | None,
        verify: str | None = None,
        bypass_rate_limit: bool = False,
        request_type: str = "navigating",
        expected_type: str | None = None,
        hateoas: bool | None = None,
    ) -> int:
        """Insert an entry point request.

        Args:
            priority: Request priority.
            method: HTTP method.
            url: Request URL.
            headers_json: JSON-encoded headers.
            cookies_json: JSON-encoded cookies.
            body: Request body bytes.
            continuation: Continuation method name.
            current_location: Current location.
            accumulated_data_json: JSON-encoded accumulated data.
            permanent_json: JSON-encoded permanent data.
            dedup_key: Deduplication key.
            bypass_rate_limit: If True, skip rate limiting for this request.
            request_type: Type of request (navigating, non_navigating, archive).
            expected_type: Expected type for archive requests.

        Returns:
            The ID of the newly inserted request, or the existing ID if
            deduplicated.
        """
        async with self._lock:
            if dedup_key is not None:
                existing = await self._find_by_dedup_key(dedup_key)
                if existing is not None:
                    return existing

            queue_counter = await self._get_next_queue_counter()  # type: ignore[attr-defined]
            created_at_ns = time.monotonic_ns()

            async with self._session_factory() as session:
                req = Request(
                    status="pending",
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
                    deduplication_key=dedup_key,
                    expected_type=expected_type,
                    created_at_ns=created_at_ns,
                    verify=verify,
                    bypass_rate_limit=bypass_rate_limit,
                    hateoas=hateoas,
                )
                session.add(req)
                await session.commit()
                return req.id  # type: ignore[return-value]

    async def get_next_pending_request(
        self,
    ) -> tuple[Any, ...] | None:
        """Get the next pending request from the queue.

        Returns:
            Row tuple or None if queue is empty.

        Note: This method is deprecated for multi-worker scenarios.
        Use dequeue_next_request() instead for atomic dequeue.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
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
                    Request.speculation_id,
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
                )
                .where(
                    Request.status == "pending",
                    or_(
                        Request.started_at.is_(None),  # type: ignore[union-attr]
                        Request.started_at <= func.datetime("now"),
                    ),
                )
                .order_by(
                    Request.priority.asc(),  # type: ignore[attr-defined]
                    Request.queue_counter.asc(),  # type: ignore[attr-defined]
                )
                .limit(1)
            )
            row = result.first()
            return tuple(row) if row else None

    async def dequeue_next_request(
        self,
    ) -> tuple[Any, ...] | None:
        """Atomically dequeue the next pending request.

        This method atomically selects and marks a request as 'in_progress'
        in a single database operation using UPDATE ... RETURNING. This
        prevents race conditions where multiple workers could select the
        same request.

        Returns:
            Row tuple (same columns as get_next_pending_request) or None
            if the queue is empty.
        """
        async with self._lock, self._session_factory() as session:
            started_at_ns = time.monotonic_ns()

            subq = (
                select(Request.id)
                .where(
                    Request.status == "pending",
                    or_(
                        Request.started_at.is_(None),  # type: ignore[union-attr]
                        Request.started_at <= func.datetime("now"),
                    ),
                )
                .order_by(
                    Request.priority.asc(),  # type: ignore[attr-defined]
                    Request.queue_counter.asc(),  # type: ignore[attr-defined]
                )
                .limit(1)
                .scalar_subquery()
            )

            stmt = (
                update(Request)
                .where(Request.id == subq)
                .values(
                    status="in_progress",
                    started_at=func.current_timestamp(),
                    started_at_ns=started_at_ns,
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
                    Request.speculation_id,
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
                    Request.parent_request_id,
                )
            )
            result = await session.execute(stmt)
            row = result.first()
            await session.commit()
            return tuple(row) if row else None

    async def mark_request_in_progress(self, request_id: int) -> None:
        """Mark a request as in progress.

        Args:
            request_id: The database ID of the request.

        Note: This method is deprecated for multi-worker scenarios.
        Use dequeue_next_request() instead for atomic dequeue.
        """
        async with self._lock, self._session_factory() as session:
            started_at_ns = time.monotonic_ns()
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status="in_progress",
                    started_at=func.current_timestamp(),
                    started_at_ns=started_at_ns,
                )
            )
            await session.commit()

    async def restamp_request_start(self, request_id: int) -> None:
        """Update started_at_ns to now (excludes prior wait from duration)."""
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(started_at_ns=time.monotonic_ns())
            )
            await session.commit()

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
        completed_at_ns = time.monotonic_ns()
        await session.execute(
            update(Request)
            .where(Request.id == request_id)
            .values(
                status="completed",
                completed_at=func.current_timestamp(),
                completed_at_ns=completed_at_ns,
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
            completed_at_ns = time.monotonic_ns()
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status="failed",
                    completed_at=func.current_timestamp(),
                    completed_at_ns=completed_at_ns,
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
        next_retry_delay: float,
        error: str,
    ) -> None:
        """Schedule a request for retry with backoff.

        Args:
            request_id: The database ID of the request.
            new_cumulative_backoff: Updated cumulative backoff time.
            next_retry_delay: Delay before next retry.
            error: Error message from the current attempt.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(Request)
                .where(Request.id == request_id)
                .values(
                    status="pending",
                    retry_count=Request.retry_count + 1,
                    cumulative_backoff=new_cumulative_backoff,
                    next_retry_delay=next_retry_delay,
                    last_error=error,
                    started_at=func.datetime(
                        "now",
                        f"+{int(next_retry_delay)} seconds",
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
                .where(Request.status == "pending")
            )
            return result.scalar() or 0

    async def count_active_requests(self) -> int:
        """Count pending and in_progress requests."""
        async with self._lock, self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status.in_(["pending", "in_progress"]))  # type: ignore[attr-defined]
            )
            return result.scalar() or 0

    async def count_in_progress(self) -> int:
        """Count in_progress requests (being processed by workers)."""
        async with self._lock, self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status == "in_progress")
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
        """Average duration of recently completed requests, in seconds.

        Uses the high-precision monotonic timestamps (started_at_ns,
        completed_at_ns) from the last *sample_size* completed requests.

        Returns:
            Average duration in seconds, or None if no completed
            requests with timing data exist.
        """
        subq = (
            select(
                (Request.completed_at_ns - Request.started_at_ns).label(  # type: ignore[union-attr, operator]
                    "duration_ns"
                )
            )
            .where(
                Request.status == "completed",
                Request.started_at_ns.isnot(None),  # type: ignore[union-attr]
                Request.completed_at_ns.isnot(None),  # type: ignore[union-attr]
            )
            .order_by(Request.id.desc())  # type: ignore[union-attr]
            .limit(sample_size)
            .subquery()
        )
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.avg(subq.c.duration_ns))
            )
            avg_ns = result.scalar()
            if avg_ns is None or avg_ns <= 0:
                return None
            return avg_ns / 1_000_000_000

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
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                    Request.content_compressed.isnot(None),  # type: ignore[union-attr]
                    Request.compression_dict_id.is_(None),  # type: ignore[union-attr]
                )
                .group_by(Request.continuation)
                .having(func.count() >= threshold)
            )
            return [row[0] for row in result.all()]

    async def get_next_scheduled_retry_delay(
        self,
    ) -> float | None:
        """Get seconds until the next scheduled retry is ready.

        Returns:
            Seconds until the next pending request becomes available,
            or None if there are no scheduled retries.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.min(
                        sa.case(
                            (
                                Request.started_at > func.datetime("now"),
                                (
                                    func.julianday(Request.started_at)
                                    - func.julianday("now")
                                )
                                * 86400.0,
                            ),
                            else_=sa.null(),
                        )
                    )
                )
                .select_from(Request)
                .where(
                    Request.status == "pending",
                    Request.started_at > func.datetime("now"),
                )
            )
            val = result.scalar()
            return val if val is not None else None

    async def count_scheduled_retries(self) -> int:
        """Count pending requests that are scheduled for later."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(
                    Request.status == "pending",
                    Request.started_at > func.datetime("now"),
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
                    Request.status == "pending",
                    Request.continuation == continuation,
                )
                .values(status="held")
            )
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

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
                    Request.status == "held",
                    Request.continuation == continuation,
                )
                .values(status="pending")
            )
            await session.commit()
            return result.rowcount  # type: ignore[return-value]

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
                .where(Request.status == "held")
            )
            if continuation:
                stmt = stmt.where(Request.continuation == continuation)
            result = await session.execute(stmt)
            return result.scalar() or 0

    async def insert_resume_request(
        self,
        priority: int,
        continuation: str,
        resume_id: str,
        predicate_result: bool,
    ) -> int:
        """Insert a resume request to resume a parked generator.

        Args:
            priority: Request priority.
            continuation: Continuation method name for reference.
            resume_id: ID linking to the parked generator.
            predicate_result: Value to send to the generator (True/False).

        Returns:
            The ID of the newly inserted request.
        """
        async with self._lock:
            queue_counter = await self._get_next_queue_counter()  # type: ignore[attr-defined]
            created_at_ns = time.monotonic_ns()

            async with self._session_factory() as session:
                req = Request(
                    status="pending",
                    priority=priority,
                    queue_counter=queue_counter,
                    request_type="resume",
                    method="GET",
                    url="",
                    continuation=continuation,
                    current_location="",
                    permanent_json=json.dumps(
                        {"predicate_result": predicate_result}
                    ),
                    expected_type=resume_id,
                    created_at_ns=created_at_ns,
                )
                session.add(req)
                await session.commit()
                return req.id  # type: ignore[return-value]

    # --- Request Cancellation ---

    async def cancel_request(self, request_id: int) -> bool:
        """Cancel a pending request.

        Args:
            request_id: The database ID of the request.

        Returns:
            True if cancelled, False if not found or not cancellable.
        """
        async with self._lock, self._session_factory() as session:
            completed_at_ns = time.monotonic_ns()
            result = await session.execute(
                update(Request)
                .where(
                    Request.id == request_id,
                    Request.status.in_(["pending", "held"]),  # type: ignore[attr-defined]
                )
                .values(
                    status="failed",
                    completed_at=func.current_timestamp(),
                    completed_at_ns=completed_at_ns,
                    last_error="Cancelled by user",
                )
            )
            await session.commit()
            return result.rowcount > 0  # type: ignore[return-value]

    async def cancel_requests_by_continuation(self, continuation: str) -> int:
        """Cancel all pending/held requests for a continuation.

        Args:
            continuation: The continuation method name.

        Returns:
            Number of requests cancelled.
        """
        async with self._lock, self._session_factory() as session:
            completed_at_ns = time.monotonic_ns()
            result = await session.execute(
                update(Request)
                .where(
                    Request.continuation == continuation,
                    Request.status.in_(["pending", "held"]),  # type: ignore[attr-defined]
                )
                .values(
                    status="failed",
                    completed_at=func.current_timestamp(),
                    completed_at_ns=completed_at_ns,
                    last_error="Cancelled by user (batch)",
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[return-value]
