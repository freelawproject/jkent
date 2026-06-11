"""Concrete :class:`Run` — the outermost lifecycle that wires a scrape.

:class:`ScrapeRun` owns the explicit collaborators a run assembles — transport,
rate limiter, queue, storage, continuation executor, monitor, and the per-step
compactors — and ties their lifetimes together (``open``/``run``/``close``/
``status``/``stop``, plus signal handling and worker spawning).

Cookie persistence on close is out of scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import threading
from typing import TYPE_CHECKING, Any, Literal

import sqlalchemy as sa
from sqlmodel import col

from jkent.data_types import DriverRequirement
from jkent.driver._speculation_support import get_entry_requests
from jkent.driver.database_engine.compression import (
    DEFAULT_DICT_SIZE,
    recompress_responses,
    train_compression_dict,
)
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.errors import store_error
from jkent.driver.database_engine.models import CompressionDict, Request
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.orchestration import Compactor, WorkerMonitor
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.rate_limiter import (
    NoopRateLimiter,
    PyrateRateLimiter,
)
from jkent.driver.unified_driver.speculation import SpeculationManager
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
    lenient_te_for,
)
from jkent.driver.unified_driver.worker import PoolWorker
from jkent.preps import build_provided_preps

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from jkent.common.deferred_validation import DeferredValidation
    from jkent.common.exceptions import ScraperAssumptionException
    from jkent.data_types import BaseScraper
    from jkent.data_types import Request as RequestModel
    from jkent.driver.archive_handler import (
        AsyncArchiveHandler,
        AsyncStreamingArchiveHandler,
    )
    from jkent.driver.unified_driver.rate_limiter import RateLimiter
    from jkent.driver.unified_driver.transport import Transport
    from jkent.preps import RequestPrepProvider

logger = logging.getLogger(__name__)


class ScrapeRun:
    """Concrete :class:`~jkent.driver.unified_driver.orchestration.Run`.

    Owns and wires the transport, rate limiter, queue, storage, continuation
    executor, monitor, and per-step compactors for a single scrape.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        transport: Transport[Any] | None = None,
        num_workers: int = 1,
        max_workers: int = 10,
        max_backoff_time: float = 3600.0,
        timeout: float | None = None,
        resume: bool = True,
        seed_params: list[dict[str, dict[str, Any]]] | None = None,
        proxy: str | None = None,
        request_preps: list[RequestPrepProvider] | None = None,
        archive_handler: AsyncArchiveHandler
        | AsyncStreamingArchiveHandler
        | None = None,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
        on_invalid_data: Callable[[DeferredValidation], Awaitable[None]]
        | None = None,
        on_structural_error: Callable[
            [ScraperAssumptionException], Awaitable[bool]
        ]
        | None = None,
        on_data: Callable[[Any], Awaitable[None]] | None = None,
        on_run_start: Callable[[str], Awaitable[None]] | None = None,
        on_run_complete: Callable[
            [str, str, Exception | None], Awaitable[None]
        ]
        | None = None,
        rate_limited: bool = True,
    ) -> None:
        self.scraper = scraper
        self.db_path = db_path
        # A STRICTLY_SERIAL scraper must be processed one request at a time, in
        # priority order — concurrent workers would interleave a stateful
        # session (e.g. an ASP.NET __VIEWSTATE/postback chain) and defeat the
        # per-step priority ordering. Enforce it here so the contract holds
        # regardless of caller (the CLI caps too, but lower-level callers may
        # not). Mirrors PoolWorker._strictly_serial.
        if DriverRequirement.STRICTLY_SERIAL in getattr(
            scraper, "driver_requirements", []
        ):
            if num_workers != 1 or max_workers != 1:
                logger.warning(
                    "Scraper %s requires STRICTLY_SERIAL; capping "
                    "num_workers/max_workers to 1 (was %d/%d).",
                    scraper.__class__.__name__,
                    num_workers,
                    max_workers,
                )
            num_workers = 1
            max_workers = 1
        self.num_workers = num_workers
        self.max_workers = max_workers
        self.max_backoff_time = max_backoff_time
        self.timeout = timeout
        self.resume = resume
        self.seed_params = seed_params
        self.proxy = proxy
        self.request_preps = request_preps
        self._archive_handler = archive_handler
        self._on_progress = on_progress
        self._on_invalid_data = on_invalid_data
        self._on_structural_error = on_structural_error
        self._on_data = on_data
        self._on_run_start = on_run_start
        self._on_run_complete = on_run_complete
        self._rate_limited = rate_limited

        # Built in open().
        self._transport: Transport[Any] | None = transport
        self._engine: Any | None = None
        self._db: SQLManager | None = None
        self._rate_limiter: RateLimiter | None = None
        self._queue: RequestQueue | None = None
        self._storage: ResponseStorage | None = None
        self._continuation: ContinuationExecutor | None = None
        self._monitor: WorkerMonitor | None = None
        self._compactors: dict[str, Compactor] = {}
        self._provided_preps: dict[str, Callable[..., Any]] = {}
        self._speculation: SpeculationManager | None = None

        # Worker registry.
        self._worker_tasks: dict[int, asyncio.Task[None]] = {}
        self._next_worker_id = 0
        self._monitor_task: asyncio.Task[None] | None = None

        # Lifecycle flags.
        self.stop_event: asyncio.Event = asyncio.Event()
        self._started = False
        self._signals_installed = False

    # --- AsyncLifecycle -------------------------------------------------

    async def open(self, *, setup_signal_handlers: bool = True) -> None:
        """Init DB, bring up the transport, build the limiter, seed compactors."""
        await self._init_db()
        assert self._db is not None

        if self._transport is None:
            self._transport = HttpxTransport(
                timeout=self.timeout,
                scraper=self.scraper,
                ssl_context=self.scraper.get_ssl_context(),
                proxy=self.proxy,
            )
        await self._transport.open()

        # Load persisted browser cookies into transports that support them
        # (Playwright/Camoufox); HTTP/replay lack the hook and are skipped.
        if hasattr(self._transport, "import_cookies"):
            try:
                saved = await self._db.get_browser_cookies()  # type: ignore[misc]
                if saved:
                    await self._transport.import_cookies(saved)  # type: ignore[misc]
            except Exception:
                logger.warning(
                    "Failed to restore browser cookies", exc_info=True
                )

        rate_limiter: RateLimiter
        if self._rate_limited and self.scraper.rate_limits:
            rate_limiter = PyrateRateLimiter(self.scraper.rate_limits)
        else:
            rate_limiter = NoopRateLimiter()
        self._rate_limiter = rate_limiter

        self._queue = RequestQueue(self._db, on_progress=self._on_progress)  # type: ignore[misc]
        self._storage = self._make_storage()
        self._provided_preps = build_provided_preps(
            self.scraper,
            self.request_preps,
            allow_live_page_providers=False,
        )
        self._continuation = ContinuationExecutor(
            self._db,  # type: ignore[misc]
            self.scraper,
            self._queue,  # type: ignore[misc]
            self._storage,  # type: ignore[misc]
            provided_preps=self._provided_preps,
            handle_data=self._on_data,
            on_invalid_data=self._on_invalid_data,
            on_structural_error=self._on_structural_error,
            on_progress=self._on_progress,
        )
        self._monitor = WorkerMonitor(
            self,
            rate_limiter,
            max_workers=self.max_workers,
            pending_requests=self._db.count_pending_requests,  # type: ignore[misc]
            stop_event=self.stop_event,
        )

        await self._seed_compactors()
        await self._setup_speculation()

        if setup_signal_handlers:
            self._setup_signal_handlers()

    async def aclose(self) -> None:
        """Stop the monitor, close the transport, close the DB, restore signals."""
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitor_task  # type: ignore[misc]

        if self._speculation is not None and self._db is not None:
            await self._speculation.persist_all()

        if self._transport is not None:
            # Persist browser cookies before teardown for transports that
            # support it (Playwright/Camoufox); best-effort, never aborts close.
            if self._db is not None and hasattr(
                self._transport, "export_cookies"
            ):
                try:
                    cookies = await self._transport.export_cookies()  # type: ignore[misc]
                    if cookies:
                        await self._db.save_browser_cookies(cookies)  # type: ignore[misc]
                except Exception:
                    logger.warning(
                        "Failed to save browser cookies", exc_info=True
                    )
            await self._transport.aclose()  # type: ignore[misc]

        if self._db is not None:
            await self._db._session_factory.remove_all()
            await self._db.close_run()  # type: ignore[misc]
        if self._engine is not None:
            await self._engine.dispose()

        if self._signals_installed:
            self._restore_signal_handlers()

    # --- Run protocol ---------------------------------------------------

    @property
    def transport(self) -> Transport[Any]:
        """The run-scoped request-execution backend."""
        assert self._transport is not None, "transport accessed before open()"
        return self._transport

    @property
    def active_worker_count(self) -> int:
        """Number of workers currently running."""
        return len(self._worker_tasks)

    def spawn_worker(self) -> int:
        """Create, register, and launch a worker; return its id."""
        worker_id = self._next_worker_id
        self._next_worker_id += 1
        worker = self._make_worker(worker_id)
        task = asyncio.create_task(worker.run())
        self._worker_tasks[worker_id] = task

        def on_done(_: asyncio.Task[None], wid: int = worker_id) -> None:
            self._worker_tasks.pop(wid, None)

        task.add_done_callback(on_done)
        return worker_id

    def _make_storage(self) -> ResponseStorage:
        """Construct the run's storage. Overridable (e.g. replay)."""
        assert self._db is not None
        return ResponseStorage(
            self._db, max_backoff_time=self.max_backoff_time
        )

    def _make_worker(self, worker_id: int) -> PoolWorker:
        """Construct one worker with the run's collaborators. Overridable."""
        assert self._queue is not None
        assert self._transport is not None
        assert self._rate_limiter is not None
        assert self._continuation is not None
        assert self._storage is not None
        assert self._monitor is not None
        return PoolWorker(
            worker_id,
            queue=self._queue,
            transport=self._transport,
            rate_limiter=self._rate_limiter,
            continuation=self._continuation,
            storage=self._storage,
            stop_event=self.stop_event,
            scraper=self.scraper,
            archive_handler=self._archive_handler,
            on_request_duration=self._monitor.record_request_duration,
            compactor_for=self.compactor_for,
            store_error=self._store_error,
            track_speculation=(
                self._speculation.track_outcome
                if self._speculation is not None
                else None
            ),
        )

    async def run(self) -> None:
        """Spawn initial workers + the monitor and drive to completion."""
        assert self._db is not None
        assert self._monitor is not None

        scraper_name = self.scraper.__class__.__name__
        with lenient_te_for(self.scraper):
            self._started = True
            await self._db.update_run_status("running")  # type: ignore[misc]
            await self._emit_progress(
                "run_started", {"scraper_name": scraper_name}
            )
            if self._on_run_start is not None:
                await self._on_run_start(scraper_name)

            status = "completed"
            error: Exception | None = None
            try:
                if self.stop_event.is_set():
                    for _ in range(self.num_workers):
                        self.spawn_worker()
                    await self._drain_workers()
                    return

                for _ in range(self.num_workers):
                    self.spawn_worker()
                self._monitor_task = asyncio.create_task(self._monitor.run())  # type: ignore[misc]
                await self._drain_workers()
            except Exception as e:
                status = "error"
                error = e
                raise
            finally:
                if (
                    self._monitor_task is not None
                    and not self._monitor_task.done()
                ):
                    self._monitor_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._monitor_task  # type: ignore[misc]
                final_status = (
                    "interrupted" if self.stop_event.is_set() else status
                )
                await self._db.finalize_run(  # type: ignore[misc]
                    final_status, str(error) if error else None
                )
                await self._emit_progress(
                    "run_completed",
                    {
                        "scraper_name": scraper_name,
                        "status": final_status,
                        "error": str(error) if error else None,
                    },
                )
                if self._on_run_complete is not None:
                    await self._on_run_complete(
                        scraper_name, final_status, error
                    )

    def stop(self) -> None:
        """Signal graceful shutdown: set the stop event."""
        self.stop_event.set()

    async def status(self) -> Literal["unstarted", "in_progress", "done"]:
        """Derive run state from start flag + queue/worker activity."""
        if not self._started:
            return "unstarted"
        assert self._db is not None
        active = await self._db.count_active_requests()
        if active > 0 or self._worker_tasks:
            return "in_progress"
        return "done"

    # --- Compactors -----------------------------------------------------

    def compactor_for(self, step: str) -> Compactor | None:
        """The compactor tracking ``step``, or None if it needs no compaction."""
        return self._compactors.get(step)

    async def _seed_compactors(self) -> None:
        """Per-step: skip if a dict exists, train at/over threshold, else seed.

        For each scraper step, query the run DB for its count of resolved
        requests (those with a stored response) and whether a compression
        dictionary already exists. A step with a dictionary needs no
        compactor; a step at/over :attr:`Compactor.THRESHOLD` with no
        dictionary is trained now; a below-threshold step gets a
        :class:`Compactor` seeded with its current resolved count.
        """
        assert self._db is not None
        for step_info in self.scraper.list_steps():
            step = step_info.name
            count = await self._resolved_count(step)
            has_dict = await self._has_compression_dict(step)
            if has_dict:
                continue
            if count >= Compactor.THRESHOLD:
                await train_compression_dict(
                    self._db._session_factory,  # type: ignore[misc]
                    step,
                    sample_limit=Compactor.THRESHOLD,
                    dict_size=DEFAULT_DICT_SIZE,
                    db_lock=self._db._lock,  # type: ignore[misc]
                )
                await recompress_responses(
                    self._db._session_factory,
                    step,
                    db_lock=self._db._lock,
                )
                continue
            self._compactors[step] = Compactor(
                step,
                self._db._session_factory,
                db_lock=self._db._lock,
                count=count,
            )

    # --- Speculation ----------------------------------------------------

    async def _setup_speculation(self) -> None:
        """Build the speculation manager, load persisted state, seed probes.

        Discovery reads ``scraper._speculation_templates`` (populated by the
        ``initial_seed`` that ``_init_db`` ran on a fresh queue). On resume the
        templates are empty, but :meth:`SpeculationManager.load` reconstructs
        them from persisted ``template_json``. Composes with the
        non-speculative entry seeding in ``_init_db``: that path skips
        speculative entries, so there is no double-seed.
        """
        assert self._db is not None
        assert self._queue is not None

        manager = SpeculationManager(
            self.scraper,
            self._queue,
            self._db,
            seed_params=self.seed_params,
        )
        manager.discover()
        await manager.load()
        if not manager.has_state:
            return
        self._speculation = manager
        await manager.seed()

    async def _resolved_count(self, step: str) -> int:
        """Count of requests for ``step`` that have a compressible body.

        Mirrors ``train_compression_dict``'s own filter (``content_compressed
        IS NOT NULL``): archive requests set ``response_status_code`` but store
        no body, so counting them would seed a compactor that then trains over
        zero responses. Counting only rows with a body keeps the seed and the
        training set consistent.
        """
        assert self._db is not None
        async with self._db._session_factory() as session:
            result = await session.execute(
                sa.select(sa.func.count())
                .select_from(Request)
                .where(
                    col(Request.continuation) == step,
                    col(Request.response_status_code).isnot(None),
                    col(Request.content_compressed).isnot(None),
                )
            )
            return result.scalar() or 0

    async def _has_compression_dict(self, step: str) -> bool:
        """Whether a compression dictionary already exists for ``step``."""
        assert self._db is not None
        async with self._db._session_factory() as session:
            result = await session.execute(
                sa.select(col(CompressionDict.id))
                .where(col(CompressionDict.continuation) == step)
                .limit(1)
            )
            return result.first() is not None

    # --- Internals ------------------------------------------------------

    async def _emit_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> None:
        """Forward a run-level progress event to the callback, if any."""
        if self._on_progress is not None:
            await self._on_progress(event_type, data)

    async def _init_db(self) -> None:
        """Initialize DB + run metadata, seed entry requests, restore queue."""
        engine, session_factory = await init_database(self.db_path)
        self._engine = engine
        self._db = SQLManager(engine, session_factory)

        scraper_name = (
            f"{self.scraper.__class__.__module__}:"
            f"{self.scraper.__class__.__name__}"
        )
        scraper_version = getattr(self.scraper, "__version__", None)
        await self._db.init_run_metadata(
            scraper_name=scraper_name,
            scraper_version=scraper_version,
            num_workers=self.num_workers,
            max_backoff_time=self.max_backoff_time,
            seed_params=self.seed_params,
        )

        if self.resume:
            pending = await self._db.restore_queue()  # type: ignore[misc]
            if pending > 0:
                logger.info("Restored %d pending requests", pending)

        if not await self._db.has_any_requests():  # type: ignore[misc]
            queue = RequestQueue(self._db, on_progress=self._on_progress)  # type: ignore[misc]
            for entry_request in get_entry_requests(
                self.scraper, self.seed_params
            ):
                await self._enqueue_entry_request(queue, entry_request)

    async def add_seed_params(
        self,
        params: list[dict[str, dict[str, Any]]],
    ) -> None:
        """Run ``params`` through ``initial_seed()`` and enqueue the results.

        Intended for use on already-populated runs — kick off additional
        entries on a resumed DB between :meth:`open` and :meth:`run`.
        Non-speculative entries yield :class:`Request` objects which are inserted
        dedup-aware via :meth:`_enqueue_entry_request`; speculative
        entries store templates on the scraper, which are discovered and
        seeded here (only the templates this call introduced — templates
        the run already tracks are not re-seeded).

        If ``seed_params_json`` is already stored (the run was originally
        seeded with params), the new entries are merged into the stored
        list so speculation filtering keeps templates originating from
        this call. To avoid state-key collisions with the first run's
        rows, the stored speculative invocations are first replayed
        through ``initial_seed()`` so new templates land at the next
        available ``{func_name}:{param_index}`` positions — any
        :class:`Request` objects yielded from that replay are discarded
        (they were already enqueued on the original run).

        Args:
            params: List of ``{entry_name: kwargs}`` invocations, identical
                in shape to the ``--params`` / ``--add-params`` JSON.

        Raises:
            ValueError: If ``params`` is empty or names an unknown entry —
                propagated from :meth:`BaseScraper.initial_seed`.
        """
        db = self._db
        queue = self._queue
        if db is None or queue is None:
            raise RuntimeError(
                "add_seed_params requires an opened run; call open() first"
            )

        stored = await db.get_seed_params()
        if stored:
            # Replay stored invocations so their speculative templates
            # re-populate ``scraper._speculation_templates`` at their
            # original positions; new templates append after them.
            for _ in self.scraper.initial_seed(stored):
                pass

        for entry_request in self.scraper.initial_seed(params):
            await self._enqueue_entry_request(queue, entry_request)

        if stored is not None:
            await db.update_seed_params(stored + params)

        await self._add_speculation_templates(
            stored + params if stored is not None else None
        )

    async def _add_speculation_templates(
        self, seed_params: list[dict[str, dict[str, Any]]] | None
    ) -> None:
        """Discover + seed only speculation templates not already tracked.

        ``open()`` already discovered, loaded, and seeded the run's
        existing templates, and speculative probe inserts carry no dedup
        key — so a state the live manager tracks must not be seeded
        again. A fresh manager discovers over the (replayed + new)
        templates; states already tracked are dropped, the remainder are
        seeded and merged into the live manager.
        """
        assert self._db is not None
        assert self._queue is not None

        manager = SpeculationManager(
            self.scraper, self._queue, self._db, seed_params=seed_params
        )
        manager.discover()
        await manager.load()

        tracked = (
            self._speculation._speculation_state
            if self._speculation is not None
            else {}
        )
        for key in list(manager._speculation_state):
            if key in tracked:
                del manager._speculation_state[key]
        if not manager.has_state:
            return

        await manager.seed()
        if self._speculation is None:
            self._speculation = manager
        else:
            self._speculation._speculation_state.update(
                manager._speculation_state
            )

    async def _enqueue_entry_request(
        self, queue: RequestQueue, entry_request: RequestModel
    ) -> None:
        """Serialize an entry-point request and insert it into the queue."""
        assert self._db is not None
        data = queue._serialize_request(entry_request)
        dedup_key = (
            entry_request.deduplication_key
            if isinstance(entry_request.deduplication_key, str)
            else None
        )
        # Entry-point requests are ordinary queue rows with no parent. The
        # serialized ``data`` keys line up 1:1 with ``insert_request``'s
        # params, so spread them through and supply the row-specific extras.
        await self._db.insert_request(  # type: ignore[misc]
            **data,
            priority=entry_request.effective_priority,
            dedup_key=dedup_key,
            parent_id=None,
        )

    async def _drain_workers(self) -> None:
        """Await the worker pool until the queue is exhausted.

        Only workers are awaited: a worker exits on stop or a durably empty
        queue, so an empty pool means the scrape is drained. The monitor must
        NOT gate completion — it only re-checks its idle-exit on its (long)
        poll cycle — so ``run`` cancels it once this returns.
        """
        while True:
            tasks = [t for t in self._worker_tasks.values() if not t.done()]
            if not tasks:
                return
            done, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]

    async def _store_error(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        """Persist an error row via the shared errors sink."""
        assert self._db is not None
        await store_error(
            self._db._session_factory,
            exc,
            request_id=request_id,
            request_url=request_url,
            db_lock=self._db._lock,
        )

    # --- Signal handling ------------------------------------------------

    def _setup_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers; no-op off the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return

        def handle_signal(signum: int, _frame: Any) -> None:
            logger.info(
                "Received %s, initiating graceful shutdown...",
                signal.Signals(signum).name,
            )
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except (ValueError, AttributeError, OSError):
            return
        self._signals_installed = True

    def _restore_signal_handlers(self) -> None:
        """Restore default SIGINT/SIGTERM handlers."""
        try:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except (ValueError, AttributeError, OSError):
            pass
        self._signals_installed = False
