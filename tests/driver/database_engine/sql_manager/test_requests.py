"""Tests for request queue operations (_requests.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.models import Request
from jkent.driver.database_engine.sql_manager import (
    InsertResult,
    RequestInsert,
    SQLManager,
)

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
            deduplication_key="GET:https://example.com/page",
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

    async def test_duplicate_key_resolves_to_existing_row(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """A second insert with a used dedup key is the first row, not new."""
        first = await insert_request(
            url="https://example.com/unique", deduplication_key="k"
        )
        second = await sql_manager.insert_request(
            RequestInsert(
                request_type=RequestType.NAVIGATING,
                method=HttpMethod.GET,
                url="https://example.com/unique-again",
                step="parse",
                deduplication_key="k",
            )
        )
        assert second == InsertResult(first, inserted=False)
        assert await sql_manager.count_all_requests() == 1

    async def test_dequeue_orders_by_priority_and_marks_in_progress(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """dequeue_next_request takes the highest-priority row and claims it."""
        # Insert requests with different priorities
        await insert_request(
            priority=10,  # Lower priority (higher number)
            url="https://example.com/low-priority",
            deduplication_key="low",
        )
        await insert_request(
            priority=1,  # Higher priority (lower number)
            url="https://example.com/high-priority",
            deduplication_key="high",
        )

        row = await sql_manager.dequeue_next_request()

        assert row is not None
        # Should get high priority request first (priority=1)
        assert row.url == "https://example.com/high-priority"

        # The dequeued row was atomically claimed.
        async with sql_manager.session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT status, started_at FROM requests WHERE id = :id"
                ),
                {"id": row.id},
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
        req1 = await insert_request(
            url="https://example.com/1", deduplication_key="1"
        )

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

    No lookup precedes the INSERT, so every duplicate reaches the
    constraint — and must come back as the existing row's id, not surface
    as an ORM FlushError that poisons the session (and, in a staged batch,
    loses every other row staged with it).
    """

    @staticmethod
    def _params(url: str, dedup_key: str | None) -> RequestInsert:
        return RequestInsert(
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url=url,
            step="parse",
            deduplication_key=dedup_key,
            priority=1,
        )

    async def test_within_batch_duplicate_does_not_poison_the_batch(
        self, sql_manager: SQLManager
    ) -> None:
        """The same key twice inside one staged transaction keeps the batch.

        A batch carrying the same dedup key twice (the same opinion PDF
        linked from two rows of a term table) hits the constraint against
        its own flushed sibling — and must not lose the rest of the batch.
        """
        async with sql_manager._write_session() as session:
            a = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/pdf", "opinion:55"),
            )
            b = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/pdf-dup", "opinion:55"),
            )
            c = await sql_manager.insert_request_in_session(
                session,
                self._params("https://example.com/other", "opinion:56"),
            )
            await session.commit()

        assert b == InsertResult(a.request_id, inserted=False)
        assert c.inserted and c.request_id != a.request_id
        assert await sql_manager.count_all_requests() == 2


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

    # 3e11 s lands past 9999-12-31; 1e16 formats as ``1e+16``, which the
    # modifier parser rejects outright. Both render NULL.
    @pytest.mark.parametrize("delay", [3e11, 1e16])
    async def test_finite_delay_sqlite_cannot_render_raises(
        self,
        sql_manager: SQLManager,
        insert_request: InsertRequest,
        delay: float,
    ) -> None:
        req_id = await insert_request()
        claimed = await sql_manager.dequeue_next_request()
        assert claimed is not None
        with pytest.raises(ValueError, match="cannot be scheduled"):
            await sql_manager.schedule_retry(req_id, 1.0, delay, "boom")

        # Nothing written: still in progress, not pending-and-ready-now.
        async with sql_manager.session_factory() as session:
            status = (
                await session.execute(
                    sa.select(Request.status).where(Request.id == req_id)
                )
            ).scalar_one()
        assert status == RequestStatus.IN_PROGRESS
        state = await sql_manager.get_retry_state(req_id)
        assert state is not None
        assert state.retry_count == 0

    async def test_finite_delay_is_scheduled(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """An in-flight request goes back to PENDING, with its bookkeeping."""
        req_id = await insert_request()
        claimed = await sql_manager.dequeue_next_request()
        assert claimed is not None
        await sql_manager.schedule_retry(req_id, 2.5, 30.0, "boom")

        async with sql_manager.session_factory() as session:
            status, last_error = (
                await session.execute(
                    sa.select(Request.status, Request.last_error).where(
                        Request.id == req_id
                    )
                )
            ).one()
        assert status == RequestStatus.PENDING
        assert last_error == "boom"

        gap = await sql_manager.seconds_until_next_pending()
        assert gap is not None
        assert 25.0 < gap <= 30.0
        state = await sql_manager.get_retry_state(req_id)
        assert state is not None
        assert state.retry_count == 1
        assert state.cumulative_backoff == 2.5


async def test_two_managers_on_one_file_dequeue_fifo(
    sql_manager: SQLManager,
) -> None:
    """FIFO within a priority holds across two handles on one database."""
    other = SQLManager(sql_manager.engine, sql_manager.session_factory)
    order = []
    # The first handle writes, the second writes past it, then the first
    # writes again: a per-handle counter would slot that last row ahead of
    # the second handle's.
    for i, manager in enumerate(
        [sql_manager, other, other, other, sql_manager]
    ):
        inserted = await manager.insert_request(
            RequestInsert(
                request_type=RequestType.NAVIGATING,
                method=HttpMethod.GET,
                url=f"https://example.com/{i}",
                step="parse",
                priority=5,
            )
        )
        order.append(inserted.request_id)

    dequeued = []
    while (row := await sql_manager.dequeue_next_request()) is not None:
        dequeued.append(row.id)
    assert dequeued == order
