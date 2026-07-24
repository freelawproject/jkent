"""StagedWrites - per-step write buffer for atomic flush.

Buffers DB writes derived from a parent step's yields (results, queued
requests) so they all land in a single transaction at the end of
the step, or roll back together on exception. See [step-staged-writes.md]
for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.database_engine.sql_manager._types import (
        PreresolvedResponse,
    )


@dataclass
class _StagedResult:
    request_id: int
    result_type: str
    data_json: str
    is_valid: bool
    validation_errors_json: str | None


@dataclass
class _StagedRequest:
    """Already-resolved + serialized request data, ready for INSERT."""

    request_data: dict[str, Any]
    dedup_key: str | None
    parent_id: int | None
    progress_event: dict[str, Any]
    # When set, insert the request pre-resolved: its response columns are
    # populated in the same INSERT and the row is flagged ``preresolved``.
    preresolved_response: PreresolvedResponse | None = None


@dataclass
class StagedWrites:
    """Buffer of DB writes for a single parent step.

    All writes are deferred until ``flush`` is called. If the step raises,
    the buffer is dropped and nothing was committed.
    """

    request_id: int
    results: list[_StagedResult] = field(default_factory=list)
    requests: list[_StagedRequest] = field(default_factory=list)
    seen_dedup_keys: set[str] = field(default_factory=set)
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
        self.seen_dedup_keys.clear()
        self.deferred_callbacks.clear()

    def stage_result(
        self,
        *,
        result_type: str,
        data_json: str,
        is_valid: bool = True,
        validation_errors_json: str | None = None,
    ) -> None:
        self.results.append(
            _StagedResult(
                request_id=self.request_id,
                result_type=result_type,
                data_json=data_json,
                is_valid=is_valid,
                validation_errors_json=validation_errors_json,
            )
        )

    def stage_callback(self, cb: Callable[[], Awaitable[None]]) -> None:
        """Defer a user callback (on_data / on_invalid_data) until post-flush."""
        self.deferred_callbacks.append(cb)

    def stage_request(
        self,
        *,
        request_data: dict[str, Any],
        dedup_key: str | None,
        parent_id: int | None,
        progress_event: dict[str, Any],
        preresolved_response: PreresolvedResponse | None = None,
    ) -> bool:
        """Stage a request insert. Returns False if dedup'd against the buffer.

        Intra-step dedup: a second yield with the same ``dedup_key`` as one
        already in this buffer is a no-op. Cross-step dedup against rows
        already committed is checked at flush time inside the transaction.

        ``preresolved_response`` carries a promoted incidental's response to
        store in the same INSERT (see ``_StagedRequest.preresolved_response``).
        """
        if dedup_key is not None:
            if dedup_key in self.seen_dedup_keys:
                return False
            self.seen_dedup_keys.add(dedup_key)

        self.requests.append(
            _StagedRequest(
                request_data=request_data,
                dedup_key=dedup_key,
                parent_id=parent_id,
                progress_event=progress_event,
                preresolved_response=preresolved_response,
            )
        )
        return True

    async def flush(
        self,
        db: SQLManager,
        *,
        mark_completed: bool = True,
    ) -> list[dict[str, Any]]:
        """Commit all buffered writes in a single transaction.

        Returns the list of progress-event payloads for newly-inserted
        requests, so the caller can fire them post-commit (cross-step
        dedup'd inserts are omitted).
        """
        emitted_events: list[dict[str, Any]] = []

        async with db._lock, db._session_factory() as session:
            for r in self.results:
                await db.store_result_in_session(
                    session,
                    request_id=r.request_id,
                    result_type=r.result_type,
                    data_json=r.data_json,
                    is_valid=r.is_valid,
                    validation_errors_json=r.validation_errors_json,
                )

            # Cross-step dedup against already-committed rows, resolved in a
            # single batched query instead of one lookup per staged request.
            staged_keys = [
                q.dedup_key for q in self.requests if q.dedup_key is not None
            ]
            existing_keys = (
                await db._find_existing_dedup_keys_in_session(
                    session, staged_keys
                )
                if staged_keys
                else set()
            )

            for q in self.requests:
                if q.dedup_key is not None and q.dedup_key in existing_keys:
                    continue

                # The batch check above already covered committed rows, and
                # intra-step dedup keys are unique, so skip the per-row lookup.
                await db.insert_request_in_session(
                    session,
                    parent_id=q.parent_id,
                    dedup_key=q.dedup_key,
                    skip_dedup_check=True,
                    preresolved_response=q.preresolved_response,
                    **q.request_data,
                )
                emitted_events.append(q.progress_event)

            if mark_completed:
                await db.mark_request_completed_in_session(
                    session, self.request_id
                )

            await session.commit()

        # User callbacks fire after the rows they relate to are durable.
        for cb in self.deferred_callbacks:
            await cb()

        return emitted_events
