"""Concrete persistence components for the unified driver.

``RequestQueue`` owns the unified driver's DB-backed queue: it subclasses
``database_engine.queue.RequestQueueDB`` (the shared (de)serialization /
dequeue / staged-enqueue methods) and adds the unified-specific glue — the
``_emit_progress`` hook and the progress-emitting ``enqueue_request``.
``ResponseStorage`` subclasses ``database_engine.storage.ResponseStorageDB``
(the shared lifecycle / response / result storage methods) and adds the
unified-specific ``max_backoff_time`` config; ``ReplayStorage`` adds the
replay-run terminal-state SQL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import text

from jkent.data_types import Request, Response
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.storage import ResponseStorageDB

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager

# Child tables that FK-reference ``requests`` and must be cleaned alongside a
# deleted/pruned request row, as (table, foreign-key column) pairs.
_REQUEST_CHILD_TABLES: tuple[tuple[str, str], ...] = (
    ("results", "request_id"),
    ("errors", "request_id"),
    ("archived_files", "request_id"),
    ("estimates", "request_id"),
    ("incidental_requests", "parent_request_id"),
)


class RequestQueue(RequestQueueDB):
    """DB-backed request queue (enqueue/dequeue/(de)serialize) for the unified driver."""

    def __init__(
        self,
        db: SQLManager,
        *,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
    ) -> None:
        self.db = db
        self._on_progress = on_progress

    async def _emit_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> None:
        """Forward a progress event to the injected callback, if any."""
        if self._on_progress is not None:
            await self._on_progress(event_type, data)

    async def enqueue_request(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None = None,
    ) -> None:
        """Enqueue a new request to the database.

        Persists the request to SQLite.

        Args:
            new_request: The new request to enqueue.
            context: Response or originating request for URL resolution.
            parent_request_id: Optional parent request ID for tracking request relationships.
        """
        # Resolve the request from context
        resolved_request: Request = new_request.resolve_from(context)  # type: ignore[arg-type, assignment]

        # Check for duplicates before inserting
        dedup_key = resolved_request.deduplication_key
        if dedup_key is not None and not isinstance(dedup_key, str):
            # SkipDeduplicationCheck - allow the request
            dedup_key = None

        # Check if this dedup_key already exists
        if dedup_key and await self.db.check_dedup_key_exists(dedup_key):
            # Duplicate found - skip
            return

        # Serialize request data
        request_data = self._serialize_request(resolved_request)

        # Use provided parent_request_id, or look up from context if not provided
        parent_id: int | None = parent_request_id
        if (
            parent_id is None
            and isinstance(context, Response)
            and context.request
        ):
            parent_id = await self.db.find_parent_request_id(
                context.request.request.url
            )

        # Insert the request
        await self.db.insert_request(
            priority=resolved_request.effective_priority,
            request_type=request_data["request_type"],
            method=request_data["method"],
            url=request_data["url"],
            headers_json=request_data["headers_json"],
            cookies_json=request_data["cookies_json"],
            body=request_data["body"],
            continuation=request_data["continuation"],
            current_location=request_data["current_location"],
            accumulated_data_json=request_data["accumulated_data_json"],
            permanent_json=request_data["permanent_json"],
            expected_type=request_data["expected_type"],
            dedup_key=dedup_key,
            parent_id=parent_id,
            is_speculative=request_data["is_speculative"],
            speculation_id=request_data["speculation_id"],
            verify=request_data["verify"],
            via_json=request_data["via_json"],
            bypass_rate_limit=request_data["bypass_rate_limit"],
            timeout_json=request_data["timeout_json"],
            json_data=request_data["json_data"],
            files_json=request_data["files_json"],
            auth_json=request_data["auth_json"],
            allow_redirects=request_data["allow_redirects"],
            proxies_json=request_data["proxies_json"],
            stream=request_data["stream"],
            cert_json=request_data["cert_json"],
            archive_hash_header=request_data["archive_hash_header"],
            reseedable=request_data["reseedable"],
        )

        # Emit progress event
        await self._emit_progress(
            "request_enqueued",
            {
                "url": request_data["url"],
                "continuation": request_data["continuation"],
                "priority": resolved_request.priority,
            },
        )


class ResponseStorage(ResponseStorageDB):
    """Response/result storage and retry/backoff handling for the unified driver."""

    def __init__(
        self,
        db: SQLManager,
        *,
        max_backoff_time: float = 3600.0,
    ) -> None:
        self.db = db
        self.max_backoff_time = max_backoff_time


class ReplayStorage(ResponseStorage):
    """Storage with the replay-run terminal states (stub / delete / finalize).

    A replayed run never carries ``failed`` rows out the back: a missed or
    errored request is either deleted (``--miss skip``) or marked ``stubbed``
    so a downstream ``jkent run`` re-fetches it. These are the DB writes the
    ``ReplayWorker`` and ``ReplayRun`` delegate here (keeping raw SQL out of
    the worker).
    """

    async def stub_request(self, request_id: int) -> None:
        """Mark a single row ``stubbed`` and clear its start timestamps."""
        async with self.db._lock, self.db._session_factory() as session:
            await session.execute(
                text(
                    "UPDATE requests "
                    "SET status = 'stubbed', "
                    "    started_at = NULL, "
                    "    started_at_ns = NULL "
                    "WHERE id = :id"
                ),
                {"id": request_id},
            )
            await session.commit()

    async def stub_with_reseedable_walk(self, request_id: int) -> None:
        """Walk ``parent_request_id`` to a ``reseedable=True`` anchor and stub it.

        Walks the *output* DB (the runtime ancestry of the row that just
        failed), stopping at the first ``reseedable=True`` row or the root.
        Only the anchor is stubbed; :meth:`finalize_stubs` later drops its
        descendants, preserving the invariant that a pending output row has
        no descendants.
        """
        async with self.db._lock, self.db._session_factory() as session:
            anchor_id = request_id
            current_id: int | None = request_id
            seen: set[int] = set()
            while current_id is not None and current_id not in seen:
                seen.add(current_id)
                row = (
                    await session.execute(
                        text(
                            "SELECT id, parent_request_id, reseedable "
                            "FROM requests WHERE id = :id"
                        ),
                        {"id": current_id},
                    )
                ).first()
                if row is None:
                    break
                anchor_id = row[0]
                if row[2]:
                    break
                current_id = row[1]

            await session.execute(
                text(
                    "UPDATE requests "
                    "SET status = 'stubbed', "
                    "    started_at = NULL, "
                    "    started_at_ns = NULL "
                    "WHERE id = :anchor"
                ),
                {"anchor": anchor_id},
            )
            await session.commit()

    async def delete_request_row(self, request_id: int) -> None:
        """Delete one row + its FK-referencing child-table rows (``--miss skip``).

        The row being deleted is whatever the worker was processing at the
        miss, so it has no ``requests`` descendants; we still clean its
        child-table rows to keep referential integrity tight.
        """
        async with self.db._lock, self.db._session_factory() as session:
            for table, fk in _REQUEST_CHILD_TABLES:
                await session.execute(
                    text(f"DELETE FROM {table} WHERE {fk} = :id"),
                    {"id": request_id},
                )
            await session.execute(
                text("DELETE FROM requests WHERE id = :id"),
                {"id": request_id},
            )
            await session.commit()

    async def finalize_stubs(self) -> None:
        """End-of-run: drop descendants of every stub, then stub → pending.

        1. Collect the descendants of every ``stubbed`` row via a recursive
           CTE into a temp table.
        2. Delete those descendants from the FK-child tables, then from
           ``requests`` (``PRAGMA defer_foreign_keys`` lets one DELETE drop a
           whole subtree atomically despite the self-referential FK).
        3. Convert the remaining (anchor-only) ``stubbed`` rows to ``pending``
           so a downstream ``jkent run`` picks them up via ``restore_queue``.
        """
        async with self.db._lock, self.db._session_factory() as session:
            await session.execute(
                text("DROP TABLE IF EXISTS _replay_stub_descendants")
            )
            await session.execute(
                text(
                    "CREATE TEMP TABLE _replay_stub_descendants "
                    "(id INTEGER PRIMARY KEY)"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO _replay_stub_descendants(id) "
                    "WITH RECURSIVE chain(id) AS ("
                    "  SELECT r.id FROM requests r "
                    "  INNER JOIN requests p "
                    "    ON r.parent_request_id = p.id "
                    "  WHERE p.status = 'stubbed' "
                    "  UNION ALL "
                    "  SELECT r.id FROM requests r "
                    "  INNER JOIN chain c "
                    "    ON r.parent_request_id = c.id "
                    ") "
                    "SELECT DISTINCT id FROM chain"
                )
            )
            for table, fk in _REQUEST_CHILD_TABLES:
                await session.execute(
                    text(
                        f"DELETE FROM {table} WHERE {fk} IN "
                        "(SELECT id FROM _replay_stub_descendants)"
                    )
                )
            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            await session.execute(
                text(
                    "DELETE FROM requests WHERE id IN "
                    "(SELECT id FROM _replay_stub_descendants)"
                )
            )
            await session.execute(
                text(
                    "UPDATE requests SET status = 'pending' "
                    "WHERE status = 'stubbed'"
                )
            )
            await session.execute(
                text("DROP TABLE IF EXISTS _replay_stub_descendants")
            )
            await session.commit()
