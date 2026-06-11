"""Tests for request queue operations (_requests.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.sql_manager import RequestInsert, SQLManager

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


class TestRequestOperations:
    """Tests for request queue operations."""

    async def test_insert_request(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test inserting a new request."""
        request_id = await insert_request(
            url="https://example.com/page",
            headers_json=json.dumps({"Accept": "text/html"}),
            current_location="https://example.com",
            dedup_key="GET:https://example.com/page",
        )

        assert request_id > 0

        # Verify request was inserted
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT url, method, status FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == "https://example.com/page"
        assert row[1] == HttpMethod.GET.code
        assert row[2] == RequestStatus.PENDING.code

    async def test_check_dedup_key_exists(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test deduplication key checking."""
        dedup_key = "GET:https://example.com/unique"

        # Should not exist initially
        assert not await sql_manager.check_dedup_key_exists(dedup_key)

        # Insert request with dedup key
        await insert_request(
            url="https://example.com/unique", dedup_key=dedup_key
        )

        # Should exist now
        assert await sql_manager.check_dedup_key_exists(dedup_key)

    async def test_dequeue_orders_by_priority_and_marks_in_progress(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """dequeue_next_request takes the highest-priority row and claims it."""
        # Insert requests with different priorities
        await insert_request(
            priority=10,  # Lower priority (higher number)
            url="https://example.com/low-priority",
            dedup_key="low",
        )
        await insert_request(
            priority=1,  # Higher priority (lower number)
            url="https://example.com/high-priority",
            dedup_key="high",
        )

        row = await sql_manager.dequeue_next_request()

        assert row is not None
        # Should get high priority request first (priority=1)
        # Column order: id, request_type, method, url, headers_json, ...
        assert row[3] == "https://example.com/high-priority"

        # The dequeued row was atomically claimed.
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, started_at FROM requests WHERE id = :id"
                ),
                {"id": row[0]},
            )
            claimed = result.first()
        assert claimed is not None
        assert claimed[0] == RequestStatus.IN_PROGRESS.code
        assert claimed[1] is not None  # started_at should be set

    async def test_mark_request_completed(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test marking a request as completed."""
        request_id = await insert_request()

        await sql_manager.dequeue_next_request()
        await sql_manager.mark_request_completed(request_id)

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, completed_at FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.COMPLETED.code
        assert row[1] is not None  # completed_at should be set

    async def test_mark_request_failed(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test marking a request as failed."""
        request_id = await insert_request()

        await sql_manager.mark_request_failed(request_id, "Test error")

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, last_error FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.FAILED.code
        assert row[1] == "Test error"

    async def test_restore_queue(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test restore_queue resets in_progress to pending."""
        # Insert and dequeue a request so it is in_progress
        request_id = await insert_request()
        await sql_manager.dequeue_next_request()

        # Verify it's in_progress
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text("SELECT status FROM requests WHERE id = :id"),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.IN_PROGRESS.code

        # Restore queue
        count = await sql_manager.restore_queue()

        # Should be back to pending
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text("SELECT status FROM requests WHERE id = :id"),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.PENDING.code
        assert count == 1

    async def test_count_methods(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test various count methods."""
        # Initially empty
        assert await sql_manager.count_pending_requests() == 0
        assert await sql_manager.count_active_requests() == 0
        assert await sql_manager.count_all_requests() == 0

        # Insert pending request
        req1 = await insert_request(url="https://example.com/1", dedup_key="1")

        assert await sql_manager.count_pending_requests() == 1
        assert await sql_manager.count_active_requests() == 1

        # Dequeue (claims the row as in_progress)
        await sql_manager.dequeue_next_request()

        assert await sql_manager.count_pending_requests() == 0
        assert await sql_manager.count_active_requests() == 1

        # Mark completed
        await sql_manager.mark_request_completed(req1)

        assert await sql_manager.count_pending_requests() == 0
        assert await sql_manager.count_active_requests() == 0
        assert await sql_manager.count_all_requests() == 1


class TestDedupBackstop:
    """The ON CONFLICT IGNORE constraint resolves duplicates, not errors.

    ``skip_dedup_check=True`` promises the constraint "still backstops
    races" — so a duplicate that reaches the INSERT must come back as the
    existing row's id, not surface as an ORM FlushError that poisons the
    session (and, in a staged batch, loses every other row staged with it).
    """

    @staticmethod
    def _params(url: str, dedup_key: str | None) -> RequestInsert:
        return RequestInsert(
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url=url,
            continuation="parse",
            dedup_key=dedup_key,
            priority=1,
        )

    async def test_skipped_check_duplicate_resolves_to_existing_id(
        self, sql_manager: SQLManager
    ) -> None:
        """A committed-row duplicate with skip_dedup_check=True is a no-op."""
        first = await sql_manager.insert_request(
            self._params("https://example.com/a", "docket:1")
        )
        second = await sql_manager.insert_request(
            self._params("https://example.com/a-again", "docket:1"),
            skip_dedup_check=True,
        )
        assert second == first
        assert await sql_manager.count_all_requests() == 1

    async def test_within_batch_duplicate_does_not_poison_the_batch(
        self, sql_manager: SQLManager
    ) -> None:
        """The same key twice inside one staged transaction keeps the batch.

        The batched staging flush only pre-checks *committed* rows, so a
        batch carrying the same dedup key twice (the same opinion PDF linked
        from two rows of a term table) hits the constraint against its own
        flushed sibling — and must not lose the rest of the batch.
        """
        async with sql_manager._write_session() as session:
            a = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/pdf", "opinion:55"),
                skip_dedup_check=True,
            )
            b = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/pdf-dup", "opinion:55"),
                skip_dedup_check=True,
            )
            c = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/other", "opinion:56"),
                skip_dedup_check=True,
            )
            await session.commit()

        assert a == b
        assert c != a
        assert await sql_manager.count_all_requests() == 2


class TestStepControl:
    """Tests for pause/resume step operations."""

    async def test_pause_step(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test pausing requests for a continuation."""
        # Insert requests with different continuations
        await insert_request(
            url="https://example.com/1",
            continuation="parse_listing",
            dedup_key="1",
        )
        await insert_request(
            url="https://example.com/2",
            continuation="parse_listing",
            dedup_key="2",
        )
        await insert_request(
            url="https://example.com/3",
            continuation="parse_detail",
            dedup_key="3",
        )

        # Pause parse_listing
        held_count = await sql_manager.pause_step("parse_listing")
        assert held_count == 2

        # Verify held count
        assert await sql_manager.get_held_count("parse_listing") == 2
        assert await sql_manager.get_held_count("parse_detail") == 0
        assert await sql_manager.get_held_count() == 2

    async def test_resume_step(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test resuming held requests."""
        # Insert and pause
        await insert_request(url="https://example.com/1", dedup_key="1")

        await sql_manager.pause_step("parse")
        assert await sql_manager.get_held_count() == 1

        # Resume
        resumed_count = await sql_manager.resume_step("parse")
        assert resumed_count == 1
        assert await sql_manager.get_held_count() == 0


class TestCancelRequests:
    """Tests for request cancellation."""

    async def test_cancel_request(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test cancelling a single pending request."""
        request_id = await insert_request()

        cancelled = await sql_manager.cancel_request(request_id)
        assert cancelled

        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, last_error FROM requests WHERE id = :id"
                ),
                {"id": request_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.FAILED.code
        assert "Cancelled" in row[1]

    async def test_cancel_request_not_pending(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test that completed requests can't be cancelled."""
        request_id = await insert_request()
        await sql_manager.dequeue_next_request()
        await sql_manager.mark_request_completed(request_id)

        cancelled = await sql_manager.cancel_request(request_id)
        assert not cancelled

    async def test_cancel_requests_by_continuation(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test batch cancelling requests by continuation."""
        # Create multiple requests
        await insert_request(url="https://example.com/1", dedup_key="1")
        await insert_request(url="https://example.com/2", dedup_key="2")
        await insert_request(
            url="https://example.com/3",
            continuation="other",
            dedup_key="3",
        )

        count = await sql_manager.cancel_requests_by_continuation("parse")
        assert count == 2

        # Verify 'other' is still pending
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status FROM requests WHERE continuation = 'other'"
                )
            )
            row = result.first()
        assert row is not None
        assert row[0] == RequestStatus.PENDING.code  # type: ignore[index]


class TestAvgCompletedRequestDuration:
    """Tests for avg_completed_request_duration_s()."""

    async def test_no_completed_requests(
        self, sql_manager: SQLManager
    ) -> None:
        """Returns None when no completed requests exist."""
        result = await sql_manager.avg_completed_request_duration_s()
        assert result is None

    async def test_completed_requests_with_timestamps(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Returns a positive duration via the real dequeue/complete path."""
        request_id = await insert_request()

        # dequeue sets started_at_ns, mark_completed sets completed_at_ns
        await sql_manager.dequeue_next_request()
        await sql_manager.mark_request_completed(request_id)

        # A non-None result *is* the assertion: the function returns None for a
        # missing or non-positive duration (see avg_completed_request_duration_s),
        # so a float here means a positive elapsed time was computed end-to-end.
        result = await sql_manager.avg_completed_request_duration_s()
        assert result is not None

    async def test_sample_size_limits_rows(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """sample_size restricts the average to the most recent N rows."""
        # Three completed requests with known, distinct durations, inserted
        # oldest-first so the newest (highest id) has the largest duration.
        durations_s = [1.0, 3.0, 5.0]
        for i, duration_s in enumerate(durations_s):
            req_id = await insert_request(
                url=f"https://example.com/{i}", dedup_key=f"key-{i}"
            )
            async with sql_manager.session_factory() as session:
                await session.execute(
                    sa.text(
                        "UPDATE requests SET status = :done, "
                        "started_at = '2026-01-01 00:00:00.000', "
                        "completed_at = strftime("
                        "  '%Y-%m-%d %H:%M:%f', '2026-01-01 00:00:00', "
                        "  :offset) "
                        "WHERE id = :id"
                    ),
                    {
                        "done": RequestStatus.COMPLETED.code,
                        "offset": f"+{duration_s} seconds",
                        "id": req_id,
                    },
                )
                await session.commit()

        # sample_size=1 averages only the newest row -> 5.0s.
        newest_only = await sql_manager.avg_completed_request_duration_s(
            sample_size=1
        )
        assert newest_only == pytest.approx(5.0)

        # sample_size=3 averages all three -> (1 + 3 + 5) / 3 = 3.0s.
        all_rows = await sql_manager.avg_completed_request_duration_s(
            sample_size=3
        )
        assert all_rows == pytest.approx(3.0)


class TestContinuationsNeedingCompressionDict:
    """Tests for continuations_needing_compression_dict()."""

    async def _insert_with_response(
        self,
        sql_manager: SQLManager,
        insert_request: InsertRequest,
        url: str,
        continuation: str,
        dedup_key: str,
        *,
        dict_id: int | None = None,
        status: RequestStatus = RequestStatus.COMPLETED,
    ) -> int:
        """Helper: insert a request and stamp it with a response.

        Defaults to COMPLETED because that is what the worker leaves
        behind after a successful fetch, and the compression queries only
        count completed rows.
        """
        req_id = await insert_request(
            url=url, continuation=continuation, dedup_key=dedup_key
        )
        async with sql_manager.session_factory() as session:
            await session.execute(
                sa.text(
                    "UPDATE requests SET "
                    "  response_status_code = 200, "
                    "  content_compressed = X'00', "
                    "  compression_dict_id = :dict_id, "
                    "  status = :status "
                    "WHERE id = :id"
                ),
                {
                    "id": req_id,
                    "dict_id": dict_id,
                    "status": status.code,
                },
            )
            await session.commit()
        return req_id

    async def test_empty_db(self, sql_manager: SQLManager) -> None:
        """Returns empty list when no requests exist."""
        result = await sql_manager.continuations_needing_compression_dict()
        assert result == []

    async def test_below_threshold(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Continuation with fewer than threshold responses is not returned."""
        for i in range(5):
            await self._insert_with_response(
                sql_manager,
                insert_request,
                f"https://example.com/{i}",
                "parse",
                f"k-{i}",
            )
        needing = await sql_manager.continuations_needing_compression_dict(
            threshold=10
        )
        assert needing == []

    async def test_at_threshold(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Continuation at threshold is returned."""
        for i in range(10):
            await self._insert_with_response(
                sql_manager,
                insert_request,
                f"https://example.com/{i}",
                "parse",
                f"k-{i}",
            )
        result = await sql_manager.continuations_needing_compression_dict(
            threshold=10
        )
        assert result == ["parse"]

    async def test_dict_compressed_not_counted(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Responses with a compression_dict_id are excluded."""
        # Create a real compression dict row to satisfy the FK constraint.
        async with sql_manager.session_factory() as session:
            await session.execute(
                sa.text(
                    "INSERT INTO compression_dicts "
                    "(continuation, version, dictionary_data, sample_count) "
                    "VALUES ('parse', 1, X'00', 0)"
                )
            )
            await session.commit()
            result = await session.execute(
                sa.text("SELECT id FROM compression_dicts LIMIT 1")
            )
            real_dict_id = result.scalar_one()

        # 8 without dict, 5 with dict — only 8 count toward threshold
        for i in range(8):
            await self._insert_with_response(
                sql_manager,
                insert_request,
                f"https://example.com/a{i}",
                "parse",
                f"a-{i}",
            )
        for i in range(5):
            await self._insert_with_response(
                sql_manager,
                insert_request,
                f"https://example.com/b{i}",
                "parse",
                f"b-{i}",
                dict_id=real_dict_id,
            )
        needing = await sql_manager.continuations_needing_compression_dict(
            threshold=10
        )
        assert needing == []

    async def test_failed_rows_with_bodies_do_not_count(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A body on a non-completed row is not training material.

        The worker stores a response on rows that did not succeed — a
        transient debug snapshot before a retry, and the observed error
        response on a persistent HTTP failure, which stays on the FAILED
        row permanently. Counting those would push a continuation over the
        compactor threshold on error pages alone.
        """
        for i in range(10):
            await self._insert_with_response(
                sql_manager,
                insert_request,
                f"https://example.com/failed-{i}",
                "parse",
                f"f-{i}",
                status=RequestStatus.FAILED,
            )
        assert (
            await sql_manager.continuations_needing_compression_dict(
                threshold=10
            )
            == []
        )
        assert await sql_manager.resolved_response_count("parse") == 0


class TestScheduleRetryGuard:
    """A delay SQLite cannot render must raise, not silently vanish.

    ``schedule_retry`` expresses the wait by pushing ``started_at`` into
    the future via ``strftime(..., '+N seconds')``. SQLite answers a
    modifier it cannot parse with NULL, and a NULL ``started_at`` is the
    *ready now* sentinel -- so an unrenderable delay would delete the
    backoff and re-dispatch the request in a tight loop.
    """

    @pytest.mark.parametrize("delay", [float("inf"), float("nan"), -1.0])
    async def test_unrenderable_delay_raises(
        self,
        sql_manager: SQLManager,
        insert_request: InsertRequest,
        delay: float,
    ) -> None:
        req_id = await insert_request()
        with pytest.raises(ValueError, match="finite and non-negative"):
            await sql_manager.schedule_retry(req_id, 1.0, delay, "boom")

    async def test_finite_delay_is_scheduled(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """The guard does not disturb an ordinary backoff."""
        req_id = await insert_request()
        await sql_manager.schedule_retry(req_id, 2.5, 30.0, "boom")

        gap = await sql_manager.seconds_until_next_pending()
        assert gap is not None
        assert 25.0 < gap <= 30.0
        state = await sql_manager.get_retry_state(req_id)
        assert state is not None
        assert state.retry_count == 1
        assert state.cumulative_backoff == 2.5


class TestReseedableRoundTrip:
    """reseedable must survive insert -> dequeue -> deserialize (regression).

    The dequeue RETURNING clause and the queue deserializer are positionally
    coupled; reseedable was persisted on insert but previously dropped on the
    way back out, so every dequeued request reset it to None.
    """

    @pytest.mark.parametrize("value", [True, False, None])
    async def test_reseedable_round_trips(
        self,
        sql_manager: SQLManager,
        insert_request: InsertRequest,
        value: bool | None,
    ) -> None:
        await insert_request(
            url="https://example.com/reseedable", reseedable=value
        )

        queue = RequestQueueDB()
        queue.db = sql_manager
        dequeued = await queue.get_next_request()

        assert dequeued is not None
        _request_id, request, _parent_id, _preresolved = dequeued
        assert request.reseedable is value
