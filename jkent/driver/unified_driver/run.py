"""The outermost lifecycle that wires a scrape.

:class:`ScrapeRun` assembles a run's collaborators from a
:class:`~jkent.driver.unified_driver.wiring.RunConfig` and ties their
lifetimes together: ``open`` brings the database, transport, queue, storage,
step executor, compactors, and speculation up (each pushed onto one
:class:`~contextlib.AsyncExitStack`, so ``aclose`` — or a failure halfway
through ``open`` — unwinds exactly what was acquired, in reverse); ``run``
drives the pinned worker pool to completion; ``stop`` requests a graceful,
resumable shutdown.

The policies a run used to carry inline live beside it now, each owned by
one object: the worker pool (:mod:`~jkent.driver.unified_driver.pool`), the
error budget (:class:`~jkent.driver.unified_driver.persistence.ErrorBudget`),
entry and speculation seeding (:mod:`~jkent.driver.unified_driver.seeding`),
the compactor registry
(:class:`~jkent.driver.unified_driver.compaction.Compactors`), the
run-level gauges (:mod:`~jkent.driver.unified_driver.gauges`), and signal
handling, which belongs to the process entry point —
:class:`~jkent.driver.unified_driver.bootstrap.RunBootstrapper`.

Cookie persistence on close is best-effort and never aborts teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any, Literal

from jkent.common.rate_limits import RateLimitTable
from jkent.driver.archive_handler import NoArchiveHandler
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RunStatus
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import RunStats, get_stats
from jkent.driver.unified_driver.circuit_breaker import CircuitBreaker
from jkent.driver.unified_driver.compaction import Compactors
from jkent.driver.unified_driver.gauges import RunGauges
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.persistence import (
    ErrorBudget,
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.pool import WorkerPool
from jkent.driver.unified_driver.rate_limiter import RateLimiters
from jkent.driver.unified_driver.seeding import Seeder
from jkent.driver.unified_driver.steps import StepExecutor
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
    lenient_te_for,
)
from jkent.driver.unified_driver.wiring import (
    RunCollaborators,
    RunConfig,
    RunHooks,
)
from jkent.driver.unified_driver.worker import PoolWorker

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )

    from jkent.data_types import BaseScraper
    from jkent.driver.archive_handler import AsyncStreamingArchiveHandler
    from jkent.driver.unified_driver.speculation import SpeculationManager
    from jkent.driver.unified_driver.transport import Transport

logger = logging.getLogger(__name__)


class ScrapeRun(AsyncLifecycle):
    """The outermost lifecycle: owner and supervisor of a single scrape.

    Args:
        scraper: The scraper to run.
        db_path: The run database (created if missing).
        config: The run's knobs; see
            :class:`~jkent.driver.unified_driver.wiring.RunConfig`. The
            ``STRICTLY_SERIAL`` worker cap is applied here, once.
        hooks: The host's callbacks; see
            :class:`~jkent.driver.unified_driver.wiring.RunHooks`.
        transport: The run-scoped request-execution backend. ``None`` builds
            a plain :class:`HttpxTransport` over the scraper — a default,
            not a configuration channel: proxy and timeout are transport
            concerns and belong to whoever builds the transport
            (:class:`~jkent.driver.unified_driver.bootstrap.RunBootstrapper`).
            The transport is a peer the run owns for the whole scrape — it
            survives individual worker exits and is rebuilt in place on
            crash, never torn down mid-scrape.
        archive_handler: Where ``archive=True`` downloads go. ``None`` means
            the run archives nothing, and an archive request is a
            :class:`ScraperConfigError` naming the fix.
        db: An already-open :class:`SQLManager` to use instead of building
            one. Callers that hand a browser transport its own DB handle must
            pass that same manager here: every writer on a SQLite file has
            to share one manager, or the per-manager write lock serializes
            nothing and concurrent writers collide. An injected manager is
            borrowed — the caller disposes its engine.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        config: RunConfig | None = None,
        hooks: RunHooks | None = None,
        transport: Transport[Any] | None = None,
        archive_handler: AsyncStreamingArchiveHandler | None = None,
        db: SQLManager | None = None,
    ) -> None:
        self.scraper = scraper
        self.db_path = db_path
        self.config = (config or RunConfig()).for_scraper(scraper)
        self.hooks = hooks or RunHooks()
        self._archive_handler = archive_handler

        # Set in open(); kept as attributes (not only inside the
        # collaborators object) because subclasses' _make_storage /
        # _make_worker / open overrides read them.
        self._transport: Transport[Any] | None = transport
        self._db: SQLManager | None = db
        self._engine: AsyncEngine | None = (
            db.engine if db is not None else None
        )
        self._owns_engine = db is None
        self._owns_transport = transport is None
        self._storage: ResponseStorage | None = None
        self._speculation: SpeculationManager | None = None
        self._error_budget: ErrorBudget | None = None
        self._circuit_breaker: CircuitBreaker | None = None
        self._collaborators: RunCollaborators | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

        # Lifecycle flags and the pieces that exist before open().
        self._started = False
        #: Set to request a graceful shutdown; shared with every worker.
        self.stop_event: asyncio.Event = asyncio.Event()
        self._gauges = RunGauges(scraper.__class__.__name__)
        self._pool = WorkerPool(
            size=self.config.num_workers,
            ramp_interval=self.config.worker_ramp_interval,
            stop_event=self.stop_event,
            make_worker=self._make_worker,
            on_change=self._gauges.worker_active,
        )

    # --- AsyncLifecycle -------------------------------------------------

    async def open(self) -> None:
        """Bring the database, transport, and every collaborator up.

        Each acquired resource is pushed onto one exit stack as it comes up,
        so a failure partway through unwinds what was acquired and leaks
        nothing — and ``aclose`` is that same unwind.
        """
        if self._exit_stack is not None:
            raise RuntimeError("run is already open")
        stack = contextlib.AsyncExitStack()
        try:
            await self._bring_up(stack)
        except BaseException:
            await stack.aclose()
            raise
        self._exit_stack = stack

    async def aclose(self) -> None:
        """Tear down in reverse order of acquisition.

        Speculation state is persisted, browser cookies exported, the
        transport closed, the run metadata finalized, and (when the run built
        it) the engine disposed. A never-opened run closes as a no-op.

        What ``open`` built is then forgotten, so the accessors refuse as
        they did before ``open`` and a reopen builds afresh; an injected
        ``db`` or transport is the caller's and stays.
        """
        stack = self._exit_stack
        if stack is None:
            return
        self._exit_stack = None
        try:
            await stack.aclose()
        finally:
            self._collaborators = None
            self._storage = None
            self._speculation = None
            self._error_budget = None
            self._circuit_breaker = None
            if self._owns_engine:
                self._db = None
                self._engine = None
            if self._owns_transport:
                self._transport = None

    async def _bring_up(self, stack: contextlib.AsyncExitStack) -> None:
        config = self.config
        hooks = self.hooks
        db = await self._init_db(stack)

        # The scraper's rate-limit lanes: the queue stores a request's lane
        # as its code in this table, and the limiters below are one per lane.
        rate_limits = RateLimitTable.for_scraper(self.scraper)
        queue = RequestQueue(
            db, on_progress=hooks.on_progress, rate_limits=rate_limits
        )
        seeder = Seeder(self.scraper, queue, db)
        # Rows an interrupted run left in progress go back to pending on
        # every open; pending rows are dequeued from the database as is.
        pending = await db.restore_queue()
        if pending > 0:
            logger.info("Restored %d pending requests", pending)
        if not await db.has_any_requests():
            await seeder.seed_entries(config.seed_params)

        transport = self._transport
        if transport is None:
            transport = HttpxTransport(
                scraper=self.scraper,
                ssl_context=self.scraper.get_ssl_context(),
            )
            self._transport = transport
        # Hand over the run's database before the transport opens: a
        # transport injected at construction time predates it.
        transport.bind_run_db(db)
        await transport.open()
        stack.push_async_callback(transport.aclose)
        await self._restore_cookies(db, transport)
        stack.push_async_callback(self._save_cookies, db, transport)

        rate_limiters = RateLimiters.for_table(
            rate_limits,
            adaptive=config.adaptive_rate_limit,
            rate_limited=config.rate_limited,
        )

        # One breaker shared by the whole pool — its failure count is a
        # run-level signal. Stop-event-aware so shutdown wakes gated workers.
        breaker = CircuitBreaker(
            config.circuit_breaker_policy, stop_event=self.stop_event
        )
        self._circuit_breaker = breaker

        storage = self._make_storage()
        self._storage = storage
        executor = StepExecutor(
            db,
            self.scraper,
            queue,
            storage,
            handle_data=hooks.on_data,
            on_invalid_data=hooks.on_invalid_data,
            on_progress=hooks.on_progress,
            captures_incidentals=transport.captures_incidentals,
        )
        compactors = await Compactors.for_scraper(self.scraper, db)
        self._speculation = await seeder.setup_speculation(config.seed_params)
        stack.push_async_callback(self._persist_speculation)

        error_budget = ErrorBudget(
            db,
            max_persistent_errors=config.max_persistent_errors,
            stop=self.stop,
        )
        self._error_budget = error_budget

        self._collaborators = RunCollaborators(
            scraper=self.scraper,
            transport=transport,
            queue=queue,
            storage=storage,
            executor=executor,
            error_sink=error_budget,
            stop_event=self.stop_event,
            rate_limiters=rate_limiters,
            circuit_breaker=breaker,
            archive_handler=(
                self._archive_handler
                if self._archive_handler is not None
                else NoArchiveHandler()
            ),
            compactors=compactors,
            speculation=self._speculation,
        )
        # Armed last so it unwinds first: a worker spawned outside run()
        # must not outlive the transport and engine it is using.
        stack.push_async_callback(self._pool.cancel)

    async def _init_db(self, stack: contextlib.AsyncExitStack) -> SQLManager:
        """Build (or adopt) the manager, write run metadata, arm teardown."""
        db = self._db
        if db is None:
            engine, session_factory = await self._make_engine()
            self._engine = engine
            db = SQLManager(engine, session_factory)
            self._db = db
            # Owned engine: disposed last, after close_run has used it.
            stack.push_async_callback(engine.dispose)
        stack.push_async_callback(db.close_run)

        config = self.config
        await db.init_run_metadata(
            scraper_name=(
                f"{self.scraper.__class__.__module__}:"
                f"{self.scraper.__class__.__name__}"
            ),
            scraper_version=getattr(self.scraper, "__version__", None),
            num_workers=config.num_workers,
            max_backoff_time=config.max_backoff_time,
            jitter=config.retry_jitter,
            seed_params=config.seed_params,
        )
        return db

    async def _restore_cookies(
        self, db: SQLManager, transport: Transport[Any]
    ) -> None:
        """Load persisted browser cookies; the ABC's default import is a no-op."""
        try:
            saved = await db.get_browser_cookies()
            if saved:
                await transport.import_cookies(saved)
        except Exception:
            logger.warning("Failed to restore browser cookies", exc_info=True)

    async def _save_cookies(
        self, db: SQLManager, transport: Transport[Any]
    ) -> None:
        """Persist browser cookies before teardown; best-effort."""
        try:
            cookies = await transport.export_cookies()
            if cookies:
                await db.save_browser_cookies(cookies)
        except Exception:
            logger.warning("Failed to save browser cookies", exc_info=True)

    async def _persist_speculation(self) -> None:
        if self._speculation is not None:
            await self._speculation.persist_all()

    # --- Orchestration surface (read by workers and host integrations) ---

    @property
    def collaborators(self) -> RunCollaborators:
        """What every worker this run spawns works against (after ``open``)."""
        assert self._collaborators is not None, (
            "collaborators accessed before open()"
        )
        return self._collaborators

    @property
    def transport(self) -> Transport[Any]:
        """The run-scoped request-execution backend."""
        assert self._transport is not None, "transport accessed before open()"
        return self._transport

    @property
    def error_budget(self) -> ErrorBudget:
        """The run's error sink and budget meter (available after ``open()``)."""
        assert self._error_budget is not None, (
            "error_budget accessed before open()"
        )
        return self._error_budget

    @property
    def persistent_error_count(self) -> int:
        """Never-retried failures stored so far (the error-budget meter)."""
        return self.error_budget.persistent_error_count

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
    def compactors(self) -> Compactors:
        """Per-step compaction registry (available after ``open()``)."""
        return self.collaborators.compactors

    @property
    def db(self) -> SQLManager:
        """The run's SQL manager (available after ``open()``)."""
        assert self._db is not None, "db accessed before open()"
        return self._db

    @property
    def sync_engine(self) -> Any:
        """The run engine's sync facade (available after ``open()``).

        Hosts use this to attach engine-level integrations, e.g. OTel's
        ``SQLAlchemyInstrumentor``.
        """
        assert self._engine is not None, "sync_engine accessed before open()"
        return self._engine.sync_engine

    @property
    def active_worker_count(self) -> int:
        """Number of workers currently running."""
        return self._pool.active_count

    def spawn_worker(self) -> int:
        """Create, register, and launch a worker; return its id."""
        return self._pool.spawn()

    # --- Subclass seams (a host's replay run overrides these) -------------

    async def _make_engine(
        self,
    ) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
        """Build the run's engine + session factory.

        Not called when the run was constructed with ``db=`` — that manager
        brings its own engine.

        The default is ``init_database``'s ``NullPool`` engine — one fresh
        sqlite connection per session checkout. Hosts whose workload is very
        many short DB calls can override this and forward pool kwargs, e.g.
        ``init_database(self.db_path, poolclass=AsyncAdaptedQueuePool, ...)``.
        """
        return await init_database(self.db_path)

    def _make_storage(self) -> ResponseStorage:
        """Construct the run's storage over ``self._db``."""
        assert self._db is not None
        return ResponseStorage(
            self._db,
            max_backoff_time=self.config.max_backoff_time,
            retry_jitter=self.config.retry_jitter,
        )

    def _make_worker(self, worker_id: int) -> PoolWorker:
        """Construct one worker over the run's collaborators."""
        return PoolWorker(worker_id, self.collaborators)

    # --- Driving ----------------------------------------------------------

    async def run(self) -> None:
        """Spawn the pinned worker pool and drive to completion.

        A run runs once: nothing clears the stop event, so a second call
        would scrape nothing and overwrite the run's ``started_at``.
        """
        if self._started:
            raise RuntimeError("run has already run")
        db = self.db
        hooks = self.hooks
        scraper_name = self.scraper.__class__.__name__
        with lenient_te_for(self.scraper):
            self._started = True
            await db.update_run_status_running()
            await hooks.on_progress(
                "run_started", {"scraper_name": scraper_name}
            )
            if hooks.on_run_start is not None:
                await hooks.on_run_start(scraper_name)

            status = RunStatus.COMPLETED
            error: Exception | None = None
            self._gauges.start_sampler(db)
            try:
                self._pool.start()
                await self._pool.drain()
            except Exception as e:
                status = RunStatus.ERROR
                error = e
                raise
            except BaseException:
                # Cancelled or interrupted (KeyboardInterrupt, a host tearing
                # the task down): the run did not finish, and a COMPLETED row
                # would tell the operator there is nothing to resume.
                status = RunStatus.INTERRUPTED
                raise
            finally:
                await self._gauges.stop_sampler()
                # Tear down any worker still in flight — e.g. a sibling died
                # and drain re-raised while others were mid-request — so no
                # worker outlives the run and writes to a transport/DB that
                # aclose() is about to close.
                await self._pool.cancel()
                # A raise is ERROR even after a stop (the error budget stops
                # on its way out): the stored error must not read as a run
                # stopped on purpose. So is a budget stop that raised
                # nothing: the scraper broke, it was not stopped by hand.
                final_status = status
                message = str(error) if error else None
                budget = self._error_budget
                if error is None and budget is not None and budget.exhausted:
                    final_status = RunStatus.ERROR
                    message = budget.exhaustion_message()
                elif error is None and self.stop_event.is_set():
                    final_status = RunStatus.INTERRUPTED
                await db.finalize_run(final_status, message)
                await hooks.on_progress(
                    "run_completed",
                    {
                        "scraper_name": scraper_name,
                        "status": final_status,
                        "error": message,
                    },
                )
                if hooks.on_run_complete is not None:
                    await hooks.on_run_complete(
                        scraper_name, final_status, error
                    )

    def stop(self) -> None:
        """Signal graceful shutdown: set the stop event."""
        self.stop_event.set()

    async def status(self) -> Literal["unstarted", "in_progress", "done"]:
        """Derive run state from start flag + queue/worker activity."""
        if not self._started:
            return "unstarted"
        active = await self.db.count_active_requests()
        if active > 0 or self._pool.active_count:
            return "in_progress"
        return "done"

    async def stats(self) -> RunStats:
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
