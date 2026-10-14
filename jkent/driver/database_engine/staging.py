"""StagedWrites - per-step write buffer for atomic flush.

Buffers DB writes derived from a parent step's yields (results, queued
requests) so they all land in a single transaction at the end of
the step, or roll back together on exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import (
        RequestInsert,
        ResultInsert,
        SQLManager,
        StoredResponse,
    )

logger = logging.getLogger(__name__)


@dataclass
class _StagedRequest:
    """Already-resolved + serialized request data, ready for INSERT.

    ``request_data`` carries its own ``deduplication_key`` and
    ``parent_request_id``.
    """

    request_data: RequestInsert
    progress_event: dict[str, Any]
    # When set, insert the request pre-resolved: its response columns are
    # populated in the same INSERT and the row is flagged ``preresolved``.
    preresolved_response: StoredResponse | None = None


@dataclass
class StagedWrites:
    """Buffer of DB writes for a single parent step.

    All writes are deferred until ``flush`` is called. If the step raises,
    the buffer is dropped and nothing was committed.
    """

    request_id: int
    results: list[ResultInsert] = field(default_factory=list)
    requests: list[_StagedRequest] = field(default_factory=list)
    # User-visible callbacks (on_data / on_invalid_data) are deferred until
    # after flush so they only fire when their underlying row is durable.
    deferred_callbacks: list[Callable[[], Awaitable[None]]] = field(
        default_factory=list
    )

    def reset(self) -> None:
        """Discard everything staged so far (all buffers + deferred callbacks).

        Used by the autowait retry loop to drop a failed attempt's partial
        yields before re-running the step. Must clear *every* buffer —
        including ``deferred_callbacks`` — or a discarded attempt's
        on_data/on_invalid_data callbacks leak into the successful retry and
        fire against rows that were never committed.
        """
        self.results.clear()
        self.requests.clear()
        self.deferred_callbacks.clear()

    def stage_result(self, result: ResultInsert) -> None:
        self.results.append(result)

    def stage_callback(self, cb: Callable[[], Awaitable[None]]) -> None:
        """Defer a user callback (on_data / on_invalid_data) until post-flush."""
        self.deferred_callbacks.append(cb)

    def stage_request(
        self,
        *,
        request_data: RequestInsert,
        progress_event: dict[str, Any],
        preresolved_response: StoredResponse | None = None,
    ) -> None:
        """Stage a request insert.

        Deduplication — against rows already committed and against earlier
        requests in this same buffer — happens at flush time, inside the
        transaction, by the ``uq_requests_dedup_key`` constraint (see
        ``insert_request_in_session``).

        ``preresolved_response`` carries a promoted incidental's response to
        store in the same INSERT (see ``_StagedRequest.preresolved_response``).
        """
        self.requests.append(
            _StagedRequest(
                request_data=request_data,
                progress_event=progress_event,
                preresolved_response=preresolved_response,
            )
        )

    async def flush(self, db: SQLManager) -> list[dict[str, Any]]:
        """Commit all buffered writes and complete the parent request.

        The results, the queued requests and the parent's ``completed``
        status land in a single transaction.

        Returns the list of progress-event payloads for newly-inserted
        requests, so the caller can fire them post-commit (deduplicated
        inserts are omitted).
        """
        emitted_events: list[dict[str, Any]] = []

        async with db._write_session() as session:
            for result in self.results:
                await db.store_result_in_session(
                    session, self.request_id, result
                )

            for q in self.requests:
                inserted = await db.insert_request_in_session(
                    session,
                    q.request_data,
                    preresolved_response=q.preresolved_response,
                )
                if inserted.inserted:
                    emitted_events.append(q.progress_event)

            await db.mark_request_completed_in_session(
                session, self.request_id
            )

            await session.commit()

        # User callbacks fire after the rows they relate to are durable. The
        # commit has happened, so one raising cannot undo anything: log it
        # and keep going, or the later callbacks and the progress events for
        # children that were inserted are lost.
        for cb in self.deferred_callbacks:
            try:
                await cb()
            except Exception:
                logger.exception(
                    "Deferred callback for request %d raised after commit",
                    self.request_id,
                )

        return emitted_events
