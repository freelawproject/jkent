"""LocalOnlyDriver: a PersistentDriver subclass that replays from source DBs."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal

from sqlalchemy import text, update

from kent.common.exceptions import (
    RequestFailedHalt,
    TransientException,
)
from kent.data_types import (
    ArchiveResponse,
    BaseRequest,
    Request,
    Response,
)
from kent.driver.local_only_driver.errors import (
    LocalOnlyMiss,
    LocalOnlyScraperMismatchError,
)
from kent.driver.local_only_driver.source_index import (
    FetchedArchive,
    FetchedResponse,
    IndexEntry,
    SourceIndex,
    fallback_replay_key_for_request,
)
from kent.driver.persistent_driver.models import Request as RequestModel
from kent.driver.persistent_driver.persistent_driver import (
    PersistentDriver,
    ScraperReturnDatatype,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from kent.data_types import BaseScraper

logger = logging.getLogger(__name__)


MissPolicy = Literal["raise", "skip", "stub"]
MatchMode = Literal["prev-error-free", "curr-error-free", "desc-error-free"]


class LocalOnlyDriver(
    PersistentDriver[ScraperReturnDatatype],
    Generic[ScraperReturnDatatype],
):
    """Replay a scraper from previous-run databases instead of the network.

    Inherits the full :class:`PersistentDriver` machinery (queue, workers,
    storage, callbacks, retries, async compression) and only swaps out the
    request-resolution step. See ``pdd replay`` for the user-facing CLI.

    Args:
        source_index: Pre-built routing index across all source DBs.
        miss_policy: What to do when a yielded request has no source-DB
            match (``raise`` / ``skip`` / ``stub``).
        mode: Replay mode (``prev-error-free`` / ``curr-error-free`` /
            ``desc-error-free``). Determines which rows the index
            considered fulfillable and which trigger continuation retry.
        trust_subtree_after_retry: When True, children of a retry-eligible
            parent that succeeded on re-execution are looked up normally
            in the source index. When False (the default), they are
            unconditionally treated as misses and stubbed for re-fetch.
    """

    def __init__(
        self,
        *,
        source_index: SourceIndex,
        miss_policy: MissPolicy = "stub",
        mode: MatchMode = "curr-error-free",
        trust_subtree_after_retry: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.source_index = source_index
        self.miss_policy: MissPolicy = miss_policy
        self.mode: MatchMode = mode
        self.trust_subtree_after_retry = trust_subtree_after_retry

        # Output-DB request_ids whose continuation was re-executed because
        # the source row was retry_eligible. Children of these rows get
        # forced-missed unless trust_subtree_after_retry is set.
        self._retry_eligible_parents: set[int] = set()
        self._retry_eligible_lock = asyncio.Lock()

    @classmethod
    @asynccontextmanager
    async def open(  # type: ignore[override]
        cls,
        scraper: BaseScraper[ScraperReturnDatatype],
        db_path: Path,
        *,
        source_db_paths: list[Path],
        miss_policy: MissPolicy = "stub",
        mode: MatchMode = "curr-error-free",
        trust_subtree_after_retry: bool = False,
        index_db_path: Path | None = None,
        num_workers: int = 4,
        **kwargs: Any,
    ) -> AsyncIterator[LocalOnlyDriver[ScraperReturnDatatype]]:
        """Open a LocalOnlyDriver run against one or more source DBs.

        Steps performed before yielding the driver:
        1. Open each source DB read-only and build the SQLite routing
           index. ``mode='desc-error-free'`` additionally computes the
           HATEOAS-aware pruning plan and seeds the output DB with the
           anchor entry requests.
        2. Verify every source DB's recorded ``scraper_name`` (class
           only; versions may differ).
        3. Initialise the output DB exactly the way :class:`PersistentDriver`
           does for ``kent run``, then construct the driver.
        """
        from kent.driver.local_only_driver.error_pruning import (
            compute_pruning_plan,
        )

        # Step 1: build the index. desc-error-free needs a pre-pass to
        # decide which rows to exclude (anchor descendants).
        excluded: dict[int, set[int]] | None = None
        anchors_per_db: dict[int, list[tuple[int, int]]] | None = None
        if mode == "desc-error-free":
            scratch_index = SourceIndex(
                source_db_paths=source_db_paths,
                index_db_path=None,
            )
            try:
                plan = compute_pruning_plan(scratch_index)
                excluded = plan.excluded_request_ids
                anchors_per_db = plan.anchors
            finally:
                scratch_index.close()
        source_index = SourceIndex(
            source_db_paths=source_db_paths,
            index_db_path=index_db_path,
            excluded_request_ids=excluded,
        )

        # Step 2: scraper-class enforcement.
        expected = (
            f"{scraper.__class__.__module__}:{scraper.__class__.__name__}"
        )
        mismatches: list[tuple[Path, str | None]] = []
        for path, found in source_index.all_scraper_names():
            if found is None:
                mismatches.append((path, None))
            else:
                # Versions are stored as ``module:Class`` (no version suffix).
                if found != expected:
                    mismatches.append((path, found))
        if mismatches:
            source_index.close()
            raise LocalOnlyScraperMismatchError(
                expected=expected, mismatches=mismatches
            )

        # prev-error-free: rebuild the index dropping retry-eligible rows
        # so they fall through to the miss policy.
        if mode == "prev-error-free":
            source_index.close()
            source_index = SourceIndex(
                source_db_paths=source_db_paths,
                index_db_path=index_db_path,
                exclude_retry_eligible=True,
            )

        # Step 3: standard PersistentDriver bring-up.
        seed_params = kwargs.pop("seed_params", None)
        max_backoff_time = kwargs.pop("max_backoff_time", 3600.0)
        resume = kwargs.pop("resume", True)
        max_workers = kwargs.pop("max_workers", max(num_workers, 10))
        # Drain kwargs that PersistentDriver.__init__ won't accept; these
        # would only matter for live-network drivers, which we replace.
        kwargs.pop("proxy", None)
        kwargs.pop("timeout", None)
        custom_request_manager = kwargs.pop("request_manager", None)
        engine, sql_manager = await cls._init_db(
            scraper,
            db_path,
            num_workers=num_workers,
            max_backoff_time=max_backoff_time,
            resume=resume,
            seed_params=seed_params,
        )

        # A request_manager is unused by LocalOnlyDriver but PersistentDriver
        # still expects one (it threads it down to AsyncDriver). Provide a
        # stub that raises if anything actually tries to use it.
        if custom_request_manager is not None:
            request_manager = custom_request_manager
        else:
            request_manager = _UnusedRequestManager()

        driver = cls(
            source_index=source_index,
            miss_policy=miss_policy,
            mode=mode,
            trust_subtree_after_retry=trust_subtree_after_retry,
            scraper=scraper,
            db=sql_manager,
            request_manager=request_manager,
            num_workers=num_workers,
            max_workers=max_workers,
            max_backoff_time=max_backoff_time,
            resume=resume,
            rates=None,
            **kwargs,
        )

        # For mode 3, seed the output DB with the hateoas-anchor entry
        # requests harvested from each source DB. These rows enter the
        # queue as status='pending' so the worker dispatch picks them up
        # like any other yielded request — but their dedup_keys won't
        # match the index (we excluded them), so they fall through to the
        # miss policy and end up stubbed in the output for downstream
        # network-replay.
        if mode == "desc-error-free" and anchors_per_db:
            await driver._seed_desc_error_free_anchors(anchors_per_db)

        try:
            yield driver
        finally:
            try:
                await cls._finalize_stubs(sql_manager)
            finally:
                source_index.close()
                await driver.close()

    @staticmethod
    async def _finalize_stubs(sql_manager: Any) -> None:
        """End-of-run cleanup of ``stubbed`` rows.

        Replay's invariant: a pending row in the output DB never has
        descendants — the downstream ``kent run`` re-fetches the row
        and regenerates the subtree fresh. This pass:

        1. Builds the set of descendants of every ``stubbed`` row using
           a recursive CTE.
        2. Deletes from the child tables that FK-reference requests
           (results, errors, archived_files, estimates, incidental_requests).
        3. Deletes the descendant rows from ``requests`` itself.
           ``PRAGMA defer_foreign_keys = ON`` defers the self-referential
           FK check on ``parent_request_id`` to commit time so a single
           DELETE can drop a whole subtree atomically.
        4. Converts the remaining (anchor-only) ``stubbed`` rows to
           ``pending`` so a downstream ``kent run --db <output>``
           picks them up via ``restore_queue()`` as normal entry work.
        """
        async with (
            sql_manager._lock,
            sql_manager._session_factory() as session,
        ):
            # Use a temp table so each DELETE can re-query the descendant
            # set without re-running the recursive CTE.
            await session.execute(
                text("DROP TABLE IF EXISTS _local_only_stub_descendants")
            )
            await session.execute(
                text(
                    "CREATE TEMP TABLE _local_only_stub_descendants "
                    "(id INTEGER PRIMARY KEY)"
                )
            )
            await session.execute(
                text(
                    "INSERT INTO _local_only_stub_descendants(id) "
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

            for table, fk in (
                ("results", "request_id"),
                ("errors", "request_id"),
                ("archived_files", "request_id"),
                ("estimates", "request_id"),
                ("incidental_requests", "parent_request_id"),
            ):
                await session.execute(
                    text(
                        f"DELETE FROM {table} WHERE {fk} IN "
                        "(SELECT id FROM _local_only_stub_descendants)"
                    )
                )

            await session.execute(text("PRAGMA defer_foreign_keys = ON"))
            await session.execute(
                text(
                    "DELETE FROM requests WHERE id IN "
                    "(SELECT id FROM _local_only_stub_descendants)"
                )
            )
            await session.execute(
                text(
                    "UPDATE requests SET status = 'pending' "
                    "WHERE status = 'stubbed'"
                )
            )
            await session.execute(
                text("DROP TABLE IF EXISTS _local_only_stub_descendants")
            )
            await session.commit()

    async def _seed_desc_error_free_anchors(
        self, anchors_per_db: dict[int, list[tuple[int, int]]]
    ) -> None:
        """For mode 3, insert the chosen hateoas anchors as entry requests."""
        for db_idx, anchors in anchors_per_db.items():
            for anchor_id, _depth in anchors:
                row = self.source_index.fetch_entry_request_row(
                    db_idx, anchor_id
                )
                if row is None:
                    continue
                await self.db.insert_entry_request(
                    priority=row["priority"],  # type: ignore[arg-type]
                    method=row["method"],  # type: ignore[arg-type]
                    url=row["url"],  # type: ignore[arg-type]
                    headers_json=row["headers_json"],  # type: ignore[arg-type]
                    cookies_json=row["cookies_json"],  # type: ignore[arg-type]
                    body=row["body"],  # type: ignore[arg-type]
                    continuation=row["continuation"],  # type: ignore[arg-type]
                    current_location=row["current_location"],  # type: ignore[arg-type]
                    accumulated_data_json=row["accumulated_data_json"],  # type: ignore[arg-type]
                    permanent_json=row["permanent_json"],  # type: ignore[arg-type]
                    dedup_key=row["deduplication_key"],  # type: ignore[arg-type]
                    verify=row["verify"],  # type: ignore[arg-type]
                    bypass_rate_limit=bool(row["bypass_rate_limit"]),
                    request_type=row["request_type"],  # type: ignore[arg-type]
                    expected_type=row["expected_type"],  # type: ignore[arg-type]
                )

    # --- The seam: replace network I/O with index lookups ---

    def _lookup_entry(self, request: BaseRequest) -> IndexEntry | None:
        """Probe the index for a yielded Request.

        Tries the request's own deduplication_key first (the override-or-
        auto-generated value). On miss, falls back to a key derived from
        the request's *serialized* URL+body — which matches source rows
        that landed in the DB with a NULL ``deduplication_key``. The
        fallback path is the one that lets us replay older DBs whose
        rows predate (or otherwise bypassed) auto-population.
        """
        dedup_key = _dedup_key_of(request)
        if dedup_key is not None:
            entry = self.source_index.lookup(dedup_key)
            if entry is not None:
                return entry
        fallback = fallback_replay_key_for_request(request.request)
        return self.source_index.lookup(fallback)

    async def resolve_request(self, request: BaseRequest) -> Response:
        """Look up the stored response for ``request`` in the source index.

        Raises:
            LocalOnlyMiss: if no source row matches.
        """
        entry = self._lookup_entry(request)
        if entry is None:
            raise LocalOnlyMiss(
                dedup_key=_dedup_key_of(request), url=request.request.url
            )
        fetched: FetchedResponse = await asyncio.to_thread(
            self.source_index.fetch_response, entry
        )
        return Response(
            status_code=fetched.status_code,
            headers=fetched.headers,
            content=fetched.content,
            text=_safe_decode(fetched.content, fetched.headers),
            url=fetched.url,
            request=request,
        )

    async def resolve_archive_request(  # type: ignore[override]
        self,
        request: Request,
        archive_decision: Any = None,
    ) -> ArchiveResponse:
        """Look up the stored archive for ``request`` in the source index.

        ``archive_decision`` is accepted for signature parity with the
        parent class but ignored: we never run the archive_handler since
        no download happens.

        Raises:
            LocalOnlyMiss: if no source row matches or the matched row
                has no archived_files companion.
        """
        del archive_decision
        entry = self._lookup_entry(request)
        if entry is None:
            raise LocalOnlyMiss(
                dedup_key=_dedup_key_of(request), url=request.request.url
            )
        fetched: FetchedArchive | None = await asyncio.to_thread(
            self.source_index.fetch_archive, entry
        )
        if fetched is None:
            raise LocalOnlyMiss(
                dedup_key=_dedup_key_of(request), url=request.request.url
            )
        return ArchiveResponse(
            status_code=fetched.status_code,
            headers=fetched.headers,
            content=b"",
            text="",
            url=fetched.url,
            request=request,
            file_url=fetched.file_path,
        )

    # --- Worker dispatch: handle retry-eligible matches and miss policies ---

    async def _process_regular_request(  # type: ignore[override]
        self,
        request_id: int,
        request: Request,
        continuation_name: str,
        parent_request_id: int | None = None,
        worker_id: int = 0,
        archive_decision: Any = None,
    ) -> None:
        # First: if this is a forced-miss child of a retried parent, skip.
        if (
            parent_request_id is not None
            and not self.trust_subtree_after_retry
            and await self._is_retry_eligible_parent(parent_request_id)
        ):
            await self._apply_miss_policy(
                request_id,
                request,
                LocalOnlyMiss(
                    dedup_key=_dedup_key_of(request),
                    url=request.request.url,
                ),
            )
            return

        # Resolve from source index.
        try:
            response: Response = (
                await self.resolve_archive_request(request)
                if request.archive
                else await self.resolve_request(request)
            )
        except LocalOnlyMiss as miss:
            await self._apply_miss_policy(request_id, request, miss)
            return

        # If this match is retry_eligible, record the parent_id so any
        # children yielded by the re-executed continuation get
        # forced-missed (unless trust_subtree_after_retry is True).
        dedup_key = _dedup_key_of(request)
        entry = self.source_index.lookup(dedup_key)
        if entry is not None and entry.retry_eligible:
            async with self._retry_eligible_lock:
                self._retry_eligible_parents.add(request_id)

        if request.is_speculative and self._speculation_state:
            await self._track_speculation_outcome(request, response)

        # Exceptions from `_complete_request` route through one of two
        # miss-policy paths so a replayed run never carries `failed`
        # rows out the back:
        #
        # 1. TransientException (network-y errors raised from inside a
        #    step). In a live run these would be retried, but replay
        #    has no network — retrying just serves the same source
        #    response and loops forever. Treat them as a signal that
        #    this row's *subtree* needs to be re-fetched from a clean
        #    re-entry point: with `--miss stub`, walk the output-DB
        #    parent chain to the nearest hateoas=True ancestor (or the
        #    root) and stub that anchor. `_finalize_stubs` cleans up
        #    the now-stale descendants at run close.
        # 2. Anything else (parser/data assumption, generic). Not fixed
        #    by re-fetching — the response is fine, the code or data
        #    shape is the problem. Stub the current request only;
        #    `_finalize_stubs` converts it to pending.
        #
        # In both buckets the policy applies uniformly across replay
        # modes: error-stubs no longer falls through to the worker's
        # `except Exception` (which would mark the row failed).
        try:
            await self._complete_request(
                request_id, response, request, continuation_name
            )
        except RequestFailedHalt:
            raise
        except TransientException as exc:
            await self._apply_transient_miss_policy(request_id, request, exc)
            return
        except Exception as exc:  # noqa: BLE001 - intentionally broad
            logger.warning(
                "Replay continuation error treated as miss: url=%s "
                "mode=%s error=%s: %s",
                request.request.url,
                self.mode,
                type(exc).__name__,
                exc,
            )
            await self._apply_miss_policy(
                request_id,
                request,
                LocalOnlyMiss(
                    dedup_key=_dedup_key_of(request),
                    url=request.request.url,
                ),
            )
            return

        await self._emit_progress(
            "request_completed",
            {
                "request_id": request_id,
                "url": request.request.url,
            },
        )

    async def _is_retry_eligible_parent(self, request_id: int) -> bool:
        async with self._retry_eligible_lock:
            return request_id in self._retry_eligible_parents

    async def _apply_transient_miss_policy(
        self,
        request_id: int,
        request: BaseRequest,
        exc: TransientException,
    ) -> None:
        """Miss-policy handler for transient errors thrown from a step.

        Unlike :meth:`_apply_miss_policy` (one-row stub), the ``stub``
        branch here walks the output-DB parent chain to the nearest
        ``hateoas=True`` ancestor (or the root if none) and stubs that
        anchor — so a downstream ``kent run`` re-fetches from a clean
        re-entry point. ``raise`` halts the run; ``skip`` deletes the
        current row entirely.
        """
        if self.miss_policy == "raise":
            raise RequestFailedHalt(
                f"transient error during replay: {exc}"
            ) from exc
        if self.miss_policy == "skip":
            logger.info(
                "Transient error during replay (skip): url=%s error=%s",
                request.request.url,
                exc,
            )
            await self._delete_request_row(request_id)
            return
        logger.warning(
            "Transient error during replay; walking to hateoas anchor: "
            "url=%s error=%s: %s",
            request.request.url,
            type(exc).__name__,
            exc,
        )
        await self._stub_with_hateoas_walk(request_id)
        await self._emit_progress(
            "request_stubbed",
            {
                "request_id": request_id,
                "url": request.request.url,
                "reason": "transient_error_hateoas_walk",
            },
        )

    async def _stub_with_hateoas_walk(self, request_id: int) -> None:
        """Walk parent_request_id up to a hateoas=True ancestor and stub it.

        Walks the *output DB* (not source) because we need the runtime
        ancestry of the row that just failed. Stops at the first
        ``hateoas=True`` row, or the root (``parent_request_id IS NULL``)
        if no row in the chain is True. Sets *only* the anchor's status
        to ``stubbed``; ``_finalize_stubs`` at the end of the run
        deletes every descendant of every stubbed row, preserving the
        invariant that a pending row in the output never has
        descendants.
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
                            "SELECT id, parent_request_id, hateoas "
                            "FROM requests WHERE id = :id"
                        ),
                        {"id": current_id},
                    )
                ).first()
                if row is None:
                    break
                row_id, parent_id, hateoas = row[0], row[1], row[2]
                anchor_id = row_id
                if hateoas:
                    break
                current_id = parent_id

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

    async def _apply_miss_policy(
        self,
        request_id: int,
        request: BaseRequest,
        miss: LocalOnlyMiss,
    ) -> None:
        """Translate a miss (or non-transient step error) into a final state.

        - ``raise`` → :class:`RequestFailedHalt`, aborts the run.
        - ``skip`` → deletes the row from the output DB. "Log + drop,
          no output row" was the original semantic; a replayed run
          must never carry ``failed`` rows out the back, so we make
          this clean by removing the row entirely instead of marking
          it failed.
        - ``stub`` → sets the row's status to ``stubbed``. The end-of-
          run :meth:`_finalize_stubs` deletes any descendants of the
          stubbed row (none, in practice, for this single-row path)
          and converts it to ``pending`` for downstream resume.
        """
        if self.miss_policy == "raise":
            raise RequestFailedHalt(str(miss)) from miss
        if self.miss_policy == "skip":
            logger.info(
                "LocalOnly miss (skip): url=%s dedup_key=%s",
                request.request.url,
                miss.dedup_key,
            )
            await self._delete_request_row(request_id)
            return
        logger.info(
            "LocalOnly miss (stub): url=%s dedup_key=%s",
            request.request.url,
            miss.dedup_key,
        )
        async with self.db._lock, self.db._session_factory() as session:
            await session.execute(
                update(RequestModel)
                .where(RequestModel.id == request_id)
                .values(
                    status="stubbed",
                    started_at=None,
                    started_at_ns=None,
                )
            )
            await session.commit()
        await self._emit_progress(
            "request_stubbed",
            {
                "request_id": request_id,
                "url": request.request.url,
                "reason": "local_only_miss",
            },
        )

    async def _delete_request_row(self, request_id: int) -> None:
        """Remove a single row + its FK-referencing child-table rows.

        Used by ``--miss skip``. The row being deleted is whichever the
        worker was processing when the miss fired, so it has no
        descendants in ``requests`` (the continuation either never ran
        or threw before yielding). Even so, we clean the immediate
        child-table rows (response gets cleaned up via the requests
        delete; results / errors / etc. are deleted explicitly) to
        keep referential integrity tight.
        """
        async with self.db._lock, self.db._session_factory() as session:
            for table, fk in (
                ("results", "request_id"),
                ("errors", "request_id"),
                ("archived_files", "request_id"),
                ("estimates", "request_id"),
                ("incidental_requests", "parent_request_id"),
            ):
                await session.execute(
                    text(f"DELETE FROM {table} WHERE {fk} = :id"),
                    {"id": request_id},
                )
            await session.execute(
                text("DELETE FROM requests WHERE id = :id"),
                {"id": request_id},
            )
            await session.commit()


def _dedup_key_of(request: BaseRequest) -> str | None:
    """Return the stable string dedup_key, or None for SkipDeduplicationCheck."""
    k = request.deduplication_key
    return k if isinstance(k, str) else None


def _safe_decode(content: bytes, headers: dict[str, str]) -> str:
    """Decode ``content`` as text using the headers' charset; never raise."""
    charset = "utf-8"
    ctype = headers.get("content-type") or headers.get("Content-Type") or ""
    if "charset=" in ctype:
        charset = ctype.split("charset=", 1)[1].split(";", 1)[0].strip()
    try:
        return content.decode(charset, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


class _UnusedRequestManager:
    """A stand-in for ``AsyncRequestManager`` in LocalOnlyDriver.

    PersistentDriver's parent class threads a request manager through its
    constructor. LocalOnlyDriver never actually calls it (every code path
    that would touch the network is overridden), but constructing one is
    required by the inheritance chain. Any unexpected invocation surfaces
    loudly.
    """

    async def __aenter__(self) -> _UnusedRequestManager:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        raise RuntimeError(
            f"LocalOnlyDriver attempted to use the network (request_manager."
            f"{name}). This indicates a bug: every fetch should be served "
            f"from the source-DB index."
        )

    async def close(self) -> None:
        return None
