"""StagedWrites - per-step write buffer for atomic flush.

Buffers DB writes derived from a parent step's yields (results, estimates,
queued requests) so they all land in a single transaction at the end of
the step, or roll back together on exception. See [step-staged-writes.md]
for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from kent.driver.persistent_driver.sql_manager import SQLManager


@dataclass
class _StagedResult:
    request_id: int
    result_type: str
    data_json: str
    is_valid: bool
    validation_errors_json: str | None


@dataclass
class _StagedEstimate:
    request_id: int
    expected_types_json: str
    min_count: int
    max_count: int | None


@dataclass
class _StagedRequest:
    """Already-resolved + serialized request data, ready for INSERT."""

    request_data: dict[str, Any]
    dedup_key: str | None
    parent_id: int | None
    progress_event: dict[str, Any]


@dataclass
class StagedWrites:
    """Buffer of DB writes for a single parent step.

    All writes are deferred until ``flush`` is called. If the step raises,
    the buffer is dropped and nothing was committed.
    """

    request_id: int
    results: list[_StagedResult] = field(default_factory=list)
    estimates: list[_StagedEstimate] = field(default_factory=list)
    requests: list[_StagedRequest] = field(default_factory=list)
    seen_dedup_keys: set[str] = field(default_factory=set)
    # User-visible callbacks (on_data / on_invalid_data) are deferred until
    # after flush so they only fire when their underlying row is durable.
    deferred_callbacks: list[Callable[[], Awaitable[None]]] = field(
        default_factory=list
    )

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

    def stage_estimate(
        self,
        *,
        expected_types_json: str,
        min_count: int,
        max_count: int | None,
    ) -> None:
        self.estimates.append(
            _StagedEstimate(
                request_id=self.request_id,
                expected_types_json=expected_types_json,
                min_count=min_count,
                max_count=max_count,
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
    ) -> bool:
        """Stage a request insert. Returns False if dedup'd against the buffer.

        Intra-step dedup: a second yield with the same ``dedup_key`` as one
        already in this buffer is a no-op. Cross-step dedup against rows
        already committed is checked at flush time inside the transaction.
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

            for e in self.estimates:
                await db.store_estimate_in_session(
                    session,
                    request_id=e.request_id,
                    expected_types_json=e.expected_types_json,
                    min_count=e.min_count,
                    max_count=e.max_count,
                )

            for q in self.requests:
                # Cross-step dedup against already-committed rows
                if q.dedup_key is not None:
                    existing = await db._find_by_dedup_key_in_session(
                        session, q.dedup_key
                    )
                    if existing is not None:
                        continue

                await db.insert_request_in_session(
                    session,
                    parent_id=q.parent_id,
                    dedup_key=q.dedup_key,
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
