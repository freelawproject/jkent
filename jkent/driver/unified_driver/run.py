"""The outermost lifecycle that wires a scrape.

:class:`ScrapeRun` owns the explicit collaborators a run assembles — transport,
rate limiter, queue, storage, continuation executor, and the per-step
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

from jkent import observability as obs
from jkent.common.exceptions import TransientException
from jkent.data_types import DriverRequirement
from jkent.driver._speculation_support import get_entry_requests
from jkent.driver.database_engine.compression import (
    DEFAULT_DICT_SIZE,
    recompress_responses,
    train_compression_dict,
)
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.errors import store_error
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import DevDriverStats, get_stats
from jkent.driver.unified_driver.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)
from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.orchestration import Compactor
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

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

    from jkent.common.deferred_validation import DeferredValidation
    from jkent.data_types import BaseScraper
    from jkent.data_types import Request as RequestModel
    from jkent.driver.archive_handler import AsyncStreamingArchiveHandler
    from jkent.driver.unified_driver.rate_limiter import RateLimiter
    from jkent.driver.unified_driver.transport import Transport

logger = logging.getLogger(__name__)


class ScrapeRun(AsyncLifecycle):
    """The outermost lifecycle: owner and supervisor of a single scrape.

    Owns and wires the transport, rate limiter, queue, storage, continuation
    executor, and per-step compactors, and ties their lifetimes to the run.
    The worker pool is pinned: ``num_workers`` workers are spawned up front
    and never replaced — a worker retires only when the queue is drained
    (nothing pending, nothing in flight), so the pool cannot collapse while
    work can still appear. ``open``/``aclose`` (from
    :class:`~jkent.driver.unified_driver.lifecycle.AsyncLifecycle`) bring the
    transport and queue up and down; ``run`` drives the scrape to
    completion; ``stop`` requests a graceful, resumable shutdown.

    The transport is a peer the run owns for the whole scrape — it survives
    individual worker exits and is rebuilt in place on crash, never torn
    down by the run mid-scrape.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        transport: Transport[Any] | None = None,
        num_workers: int = 1,
        max_backoff_time: float = 3600.0,
        timeout: float | None = None,
        resume: bool = True,
        seed_params: list[dict[str, dict[str, Any]]] | None = None,
        proxy: str | None = None,
        archive_handler: AsyncStreamingArchiveHandler | None = None,
        on_progress: Callable[[str, dict[str, Any]], Awaitable[None]]
        | None = None,
        on_invalid_data: Callable[[DeferredValidation], Awaitable[None]]
        | None = None,
        on_data: Callable[[Any], Awaitable[None]] | None = None,
        on_run_start: Callable[[str], Awaitable[None]] | None = None,
        on_run_complete: Callable[
            [str, str, Exception | None], Awaitable[None]
        ]
        | None = None,
        rate_limited: bool = True,
        circuit_breaker_policy: CircuitBreakerPolicy | None = None,
        max_persistent_errors: int | None = None,
    ) -> None:
        self.scraper = scraper
        self.db_path = db_path
        # A STRICTLY_SERIAL scraper must be processed one request at a time, in
        # priority order — concurrent workers would interleave a stateful
        # session (e.g. an ASP.NET __VIEWSTATE/postback chain) and defeat the
        # per-step priority ordering. This is the single enforcement site:
        # every caller (RunBootstrapper included) funnels through this
        # constructor. Mirrors PoolWorker._strictly_serial.
        if DriverRequirement.STRICTLY_SERIAL in getattr(
            scraper, "driver_requirements", []
        ):
            if num_workers != 1:
                logger.warning(
                    "Scraper %s requires STRICTLY_SERIAL; capping "
                    "num_workers to 1 (was %d).",
                    scraper.__class__.__name__,
                    num_workers,
                )
            num_workers = 1
        self.num_workers = num_workers
        self.max_backoff_time = max_backoff_time
        self.timeout = timeout
        self.resume = resume
        self.seed_params = seed_params
        self.proxy = proxy
        self._archive_handler = archive_handler
        self._on_progress = on_progress
        self._on_invalid_data = on_invalid_data
        self._on_data = on_data
        self._on_run_start = on_run_start
        self._on_run_complete = on_run_complete
        self._rate_limited = rate_limited
        self._circuit_breaker_policy = circuit_breaker_policy
        # The run's error budget: stop gracefully once this many
        # never-retried failures have been stored. None = unlimited.
        self.max_persistent_errors = max_persistent_errors
        self._persistent_error_count = 0

        # Built in open().
        self._transport: Transport[Any] | None = transport
        self._engine: Any | None = None
        self._db: SQLManager | None = None
        self._rate_limiter: RateLimiter | None = None
        self._queue: RequestQueue | None = None
        self._storage: ResponseStorage | None = None
        self._continuation: ContinuationExecutor | None = None
        self._compactors: dict[str, Compactor] = {}
        self._speculation: SpeculationManager | None = None
        self._circuit_breaker: CircuitBreaker | None = None

        # Worker registry.
        self._worker_tasks: dict[int, asyncio.Task[None]] = {}
        self._next_worker_id = 0

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

        # One breaker shared by the whole pool — its failure count is a
        # run-level signal. Stop-event-aware so shutdown wakes gated workers.
        self._circuit_breaker = CircuitBreaker(
            self._circuit_breaker_policy, stop_event=self.stop_event
        )

        self._queue = RequestQueue(self._db, on_progress=self._on_progress)  # type: ignore[misc]
        self._storage = self._make_storage()
        self._continuation = ContinuationExecutor(
            self._db,  # type: ignore[misc]
            self.scraper,
            self._queue,  # type: ignore[misc]
            self._storage,  # type: ignore[misc]
            handle_data=self._on_data,
            on_invalid_data=self._on_invalid_data,
            on_progress=self._on_progress,
        )
        await self._seed_compactors()
        await self._setup_speculation()

        if setup_signal_handlers:
            self._setup_signal_handlers()

    async def aclose(self) -> None:
        """Close the transport, close the DB, restore signals."""
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
            await self._db.close_run()  # type: ignore[misc]
        if self._engine is not None:
            await self._engine.dispose()

        if self._signals_installed:
            self._restore_signal_handlers()

    # --- Orchestration surface (read by workers and host integrations) ---

    @property
    def transport(self) -> Transport[Any]:
        """The run-scoped request-execution backend."""
        assert self._transport is not None, "transport accessed before open()"
        return self._transport

    @property
    def persistent_error_count(self) -> int:
        """Never-retried failures stored so far (the error-budget meter).

        Counted whether or not ``max_persistent_errors`` is set, so hosts can
        watch scraper health on unbudgeted runs too.
        """
        return self._persistent_error_count

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """The run's shared circuit breaker (available after ``open()``).

        Hosts reconfigure it mid-run by mutating its policy, e.g.
        ``run.circuit_breaker.policy.recovery_timeout = 60``; changes apply
        at the next state transition.
        """
        assert self._circuit_breaker is not None, (
            "circuit_breaker accessed before open()"
        )
        return self._circuit_breaker

    @property
    def db(self) -> SQLManager:
        """The run's SQL manager (available after ``open()``)."""
        assert self._db is not None, "db accessed before open()"
        return self._db

    @property
    def session_factory(self) -> async_sessionmaker:
        """The run DB's async session factory (available after ``open()``)."""
        return self.db.session_factory

    @property
    def sync_engine(self) -> Any:
        """The run engine's sync facade (available after ``open()``).

        Hosts use this to attach engine-level integrations
        instruments it with OTel's ``SQLAlchemyInstrumentor``.
        """
        assert self._engine is not None, "sync_engine accessed before open()"
        return self._engine.sync_engine

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
        self._publish_worker_active()

        def on_done(_: asyncio.Task[None], wid: int = worker_id) -> None:
            self._worker_tasks.pop(wid, None)
            self._publish_worker_active()

        task.add_done_callback(on_done)
        return worker_id

    def _metric_labels(self) -> dict[str, str]:
        """Per-run metric attributes: scraper name and (if set) run_inst_id."""
        labels = {"scraper": self.scraper.__class__.__name__}
        fid = obs.run_inst_id()
        if fid is not None:
            labels["run_inst_id"] = fid
        return labels

    def _publish_worker_active(self) -> None:
        """Publish the live worker count as a gauge.

        With the pool pinned this flatlines at ``num_workers`` mid-run; its
        value is the drain tail — how long the run limps along on the last
        one or two workers after the rest retire.
        """
        obs.instruments().worker_active.set(
            len(self._worker_tasks), self._metric_labels()
        )

    # How often the queue-backlog gauge samples count_pending_requests. The
    # scaling monitor that used to read pending counts is gone (the pool is
    # pinned), so the gauge polls on its own — but only when a real OTel SDK
    # is installed; otherwise the sampler never starts and the run does no
    # extra DB reads.
    GAUGE_SAMPLE_INTERVAL_S = 5.0

    async def _sample_queue_gauge(self) -> None:
        """Periodically publish the pending-request backlog gauge."""
        assert self._db is not None
        gauge = obs.instruments().queue_pending
        while True:
            pending = await self._db.count_pending_requests()  # type: ignore[misc]
            gauge.set(pending, self._metric_labels())
            await asyncio.sleep(self.GAUGE_SAMPLE_INTERVAL_S)

    async def _cancel_workers(self) -> None:
        """Cancel and await every live worker task.

        Used on shutdown so no worker outlives ``run()`` and keeps issuing
        transport calls or DB writes against collaborators that ``aclose()`` is
        about to tear down. ``return_exceptions=True`` drains each task's result
        — the failure that triggered teardown and the CancelledErrors alike —
        so none is left unretrieved.
        """
        tasks = list(self._worker_tasks.values())
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _make_storage(self) -> ResponseStorage:
        """Construct the run's storage. Overridable (e.g. replay)."""
        assert self._db is not None
        return ResponseStorage(
            self._db, max_backoff_time=self.max_backoff_time
        )

    def _worker_kwargs(self) -> dict[str, Any]:
        """The collaborator kwargs shared by every worker this run builds.

        Subclasses (e.g. jent's ``ReplayRun``) extend this dict rather than
        re-listing the whole constructor, so a new ``PoolWorker`` argument is
        wired in exactly one place.
        """
        assert self._queue is not None
        assert self._transport is not None
        assert self._rate_limiter is not None
        assert self._continuation is not None
        assert self._storage is not None
        assert self._circuit_breaker is not None
        return {
            "queue": self._queue,
            "transport": self._transport,
            "rate_limiter": self._rate_limiter,
            "continuation": self._continuation,
            "storage": self._storage,
            "stop_event": self.stop_event,
            "scraper": self.scraper,
            "archive_handler": self._archive_handler,
            "compactor_for": self.compactor_for,
            "store_error": self._store_error,
            "track_speculation": (
                self._speculation.track_outcome
                if self._speculation is not None
                else None
            ),
            "circuit_breaker": self._circuit_breaker,
        }

    def _make_worker(self, worker_id: int) -> PoolWorker:
        """Construct one worker with the run's collaborators. Overridable."""
        return PoolWorker(worker_id, **self._worker_kwargs())

    async def run(self) -> None:
        """Spawn the pinned worker pool and drive to completion."""
        assert self._db is not None

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
            gauge_sampler: asyncio.Task[None] | None = None
            if obs.sdk_active():
                gauge_sampler = asyncio.create_task(self._sample_queue_gauge())
            try:
                for _ in range(self.num_workers):
                    self.spawn_worker()
                await self._drain_workers()
            except Exception as e:
                status = "error"
                error = e
                raise
            finally:
                if gauge_sampler is not None:
                    gauge_sampler.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await gauge_sampler
                # Tear down any worker still in flight — e.g. a sibling died and
                # _drain_workers re-raised while others were mid-request — so no
                # worker outlives the run and writes to a transport/DB that
                # aclose() is about to close.
                await self._cancel_workers()
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

    async def stats(self) -> DevDriverStats:
        """Aggregate live statistics for this run's database.

        The public progress surface for hosts polling a running scrape
        (queue/result/error counts, run status, throughput) — read-only, so
        it can run alongside the scrape's writers. For post-run reporting on
        a closed database use
        :func:`jkent.driver.database_engine.stats.read_run_summary`.

        Raises:
            RuntimeError: If the run has not been opened.
        """
        if self._db is None:
            raise RuntimeError("run is not open; call open() first")
        return await get_stats(self._db.session_factory)

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
            count = await self._db.resolved_response_count(step)  # type: ignore[misc]
            has_dict = await self._db.has_compression_dict(step)  # type: ignore[misc]
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

    # --- Internals ------------------------------------------------------

    async def _emit_progress(
        self, event_type: str, data: dict[str, Any]
    ) -> None:
        """Forward a run-level progress event to the callback, if any."""
        if self._on_progress is not None:
            await self._on_progress(event_type, data)

    async def _make_engine(self) -> tuple[AsyncEngine, async_sessionmaker]:
        """Build the run's engine + session factory. Overridable (e.g. replay).

        The default is ``init_database``'s ``NullPool`` engine — one fresh
        sqlite connection per session checkout. Hosts whose workload is very
        many short DB calls can override this and forward pool kwargs, e.g.
        ``init_database(self.db_path, poolclass=AsyncAdaptedQueuePool, ...)``.
        """
        return await init_database(self.db_path)

    async def _init_db(self) -> None:
        """Initialize DB + run metadata, seed entry requests, restore queue."""
        engine, session_factory = await self._make_engine()
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

        if self._speculation is None:
            if not manager.has_state:
                return
            await manager.seed()
            self._speculation = manager
            return
        await self._speculation.adopt_untracked(manager)

    async def _enqueue_entry_request(
        self, queue: RequestQueue, entry_request: RequestModel
    ) -> None:
        """Serialize an entry-point request and insert it into the queue."""
        assert self._db is not None
        # Entry requests never come from a parsed form, but a hand-built
        # request can still carry async field resolvers in params/data.
        entry_request = await entry_request.resolve_deferred_fields()
        data = queue.serialize_request(entry_request)
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

        A worker exits on stop or a drained queue (nothing pending, nothing
        in flight), so an empty pool means the scrape is drained.
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
        """Persist an error row; charge non-transient ones to the error budget.

        Every worker failure path funnels through here, so this is the single
        counting site. Transient-exhausted errors are excluded — server
        distress is the circuit breaker's signal; the budget guards against
        *scraper* breakage: persistent HTTP, assumption violations, and
        unclassified exceptions, none of which retrying will fix. Exhausting
        the budget stops the run gracefully (final status ``interrupted``,
        resumable), never mid-request.
        """
        assert self._db is not None
        await store_error(
            self._db._session_factory,
            exc,
            request_id=request_id,
            request_url=request_url,
            db_lock=self._db._lock,
        )
        if isinstance(exc, TransientException):
            return
        self._persistent_error_count += 1
        if (
            self.max_persistent_errors is not None
            and self._persistent_error_count >= self.max_persistent_errors
            and not self.stop_event.is_set()
        ):
            logger.error(
                "Stopping run: %d persistent errors reached the "
                "max_persistent_errors budget (%d)",
                self._persistent_error_count,
                self.max_persistent_errors,
            )
            self.stop()

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
