"""What a run is configured with, what it calls back, and what a worker needs.

Three value objects carry a run from the bootstrapper through
:class:`~jkent.driver.unified_driver.run.ScrapeRun` into every worker:

- :class:`RunConfig` — the run's knobs. Frozen; validated once; the
  ``STRICTLY_SERIAL`` worker cap is applied by :meth:`RunConfig.for_scraper`,
  the single enforcement site.
- :class:`RunHooks` — the host's callbacks. ``on_progress`` defaults to a
  no-op; the others default to ``None`` and are skipped when unset.
- :class:`RunCollaborators` — the objects a
  :class:`~jkent.driver.unified_driver.worker.PoolWorker` works against. The
  run builds one at ``open``; a worker is ``PoolWorker(worker_id,
  collaborators)`` and nothing else. Every optional collaborator has a null
  object default (a limiter that never throttles, a breaker that never trips,
  an archive handler that names the missing configuration, an empty compactor
  registry) so the worker never branches on ``None``.

The one deliberate exception is :attr:`RunCollaborators.speculation`, which
stays ``Optional``: "this run has no speculation state" is a fact the worker
must be able to see, because a :class:`SpeculationHTTPFailure` on a probe
with no tracker must be *failed*, not silently recorded as a success (see
``PoolWorker._handle_speculation_http``). A null tracker would erase that
distinction.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from jkent.data_types import DriverRequirement
from jkent.driver.archive_handler import (
    AsyncStreamingArchiveHandler,
    NoArchiveHandler,
)
from jkent.driver.database_engine.storage import DEFAULT_RETRY_JITTER
from jkent.driver.unified_driver.circuit_breaker import (
    Breaker,
    CircuitBreakerPolicy,
    NoopBreaker,
)
from jkent.driver.unified_driver.compaction import Compactors
from jkent.driver.unified_driver.persistence import (
    ErrorSink,
    ProgressCallback,
    RequestQueue,
    ResponseStorage,
    no_progress,
)
from jkent.driver.unified_driver.rate_limiter import RateLimiters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.common.deferred_validation import DeferredValidation
    from jkent.data_types import BaseScraper
    from jkent.driver.unified_driver.speculation import SpeculationManager
    from jkent.driver.unified_driver.steps import StepExecutor
    from jkent.driver.unified_driver.transport import Transport

logger = logging.getLogger(__name__)

#: ``{entry_name: kwargs}`` invocations for ``initial_seed()``.
SeedParams = list[dict[str, dict[str, Any]]]


@dataclass(frozen=True)
class RunConfig:
    """The knobs a run is configured with.

    Attributes:
        num_workers: Pinned worker-pool size — spawned once, never re-grown.
            Capped to 1 for ``STRICTLY_SERIAL`` scrapers by
            :meth:`for_scraper`.
        worker_ramp_interval: Seconds between worker spawns at startup. 0
            spawns the whole pool at once; a positive value staggers it, so
            N workers do not open N browser pages on the first tick. Shapes
            arrival, not capacity.
        max_backoff_time: Give up retrying a transient failure once its
            cumulative backoff would exceed this many seconds.
        retry_jitter: Fractional jitter applied to each retry delay, in
            ``[0, 1)``.
        seed_params: ``{entry: kwargs}`` invocations for ``initial_seed()``
            on a fresh queue; ``None`` seeds every no-arg entry. Persisted
            in run metadata so speculation can rediscover its templates.
        rate_limited: Honor the scraper's ``rate_limits``. Off for replay,
            which does no network I/O.
        adaptive_rate_limit: Make the run's *default* lane an
            :class:`~jkent.driver.unified_driver.rate_limiter.AdaptiveRateLimiter`
            that opens at the scraper's declared ``rate_limits`` and slows
            on server pushback. The scraper's other lanes are unaffected;
            ignored when ``rate_limited`` is off.
        circuit_breaker_policy: Thresholds for the run's shared breaker.
            Mutable on purpose — hosts adjust it mid-run through
            ``run.circuit_breaker.policy``.
        max_persistent_errors: Stop the run (gracefully, resumably) once
            this many never-retried failures have been filed. ``None`` is
            unlimited; the count still runs.
    """

    num_workers: int = 1
    worker_ramp_interval: float = 0.0
    max_backoff_time: float = 3600.0
    retry_jitter: float = DEFAULT_RETRY_JITTER
    seed_params: SeedParams | None = None
    rate_limited: bool = True
    adaptive_rate_limit: bool = False
    circuit_breaker_policy: CircuitBreakerPolicy = field(
        default_factory=CircuitBreakerPolicy
    )
    max_persistent_errors: int | None = None

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.worker_ramp_interval < 0:
            raise ValueError("worker_ramp_interval cannot be negative")
        if self.max_backoff_time <= 0:
            raise ValueError("max_backoff_time must be positive")
        if not 0 <= self.retry_jitter < 1:
            raise ValueError("retry_jitter must be in [0, 1)")
        if (
            self.max_persistent_errors is not None
            and self.max_persistent_errors < 1
        ):
            raise ValueError(
                "max_persistent_errors must be at least 1, or None for "
                "unlimited"
            )

    def for_scraper(self, scraper: BaseScraper[Any]) -> RunConfig:
        """This config as ``scraper`` constrains it.

        A ``STRICTLY_SERIAL`` scraper must be processed one request at a
        time, in priority order — concurrent workers would interleave a
        stateful session (an ASP.NET ``__VIEWSTATE`` postback chain) and
        defeat the per-step priority ordering. This is the single
        enforcement site: every run funnels its config through here.
        """
        if (
            DriverRequirement.STRICTLY_SERIAL
            in getattr(scraper, "driver_requirements", ())
            and self.num_workers != 1
        ):
            logger.warning(
                "Scraper %s requires STRICTLY_SERIAL; capping num_workers "
                "to 1 (was %d).",
                scraper.__class__.__name__,
                self.num_workers,
            )
            return replace(self, num_workers=1)
        return self


@dataclass(frozen=True)
class RunHooks:
    """The host's callbacks into a run.

    Attributes:
        on_progress: ``(event_type, data)`` for every driver event
            (``request_enqueued``, ``run_started`` …). Defaults to a no-op.
        on_data: Awaited with each valid datum a step yields, after the
            yield has been persisted.
        on_invalid_data: Awaited with each :class:`DeferredValidation` that
            failed, in place of raising.
        on_run_start: Awaited with the scraper name as ``run()`` begins.
        on_run_complete: Awaited with ``(scraper_name, final_status,
            error)`` as ``run()`` ends, on every path. ``error`` is what
            ``run()`` raised; an error-budget stop raises nothing, so it
            ends ``"error"`` with ``error=None`` and the reason in the
            run's ``error_message``.
    """

    on_progress: ProgressCallback = no_progress
    on_data: Callable[[Any], Awaitable[None]] | None = None
    on_invalid_data: (
        Callable[[DeferredValidation[Any]], Awaitable[None]] | None
    ) = None
    on_run_start: Callable[[str], Awaitable[None]] | None = None
    on_run_complete: (
        Callable[[str, str, Exception | None], Awaitable[None]] | None
    ) = None


@dataclass(frozen=True, kw_only=True)
class RunCollaborators:
    """Everything a :class:`PoolWorker` works against, in one object.

    The run assembles one at ``open`` and hands it to every worker it spawns;
    a subclass that adds a collaborator adds a field here, and the worker
    constructor never changes.

    Attributes:
        scraper: The scraper being run.
        transport: The run-scoped request-execution backend.
        queue: Dequeue / enqueue over the run database.
        storage: Response, result, and retry-state storage.
        executor: Runs a step against its response and persists the
            yields.
        error_sink: Where failures are reported (the run's error budget).
        stop_event: Set to request a graceful shutdown.
        rate_limiters: The run's limiters by lane, shared across the pool;
            no lane throttles by default.
        circuit_breaker: Shared across the pool; never trips by default.
        archive_handler: Where ``archive=True`` downloads go; by default one
            that names the missing configuration.
        compactors: Per-step compaction registry; empty by default.
        speculation: The speculation tracker, or ``None`` when the run has
            no speculation state (see the module docstring for why this one
            is not a null object).
    """

    scraper: BaseScraper[Any]
    transport: Transport[Any]
    queue: RequestQueue
    storage: ResponseStorage
    executor: StepExecutor
    error_sink: ErrorSink
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    rate_limiters: RateLimiters = field(default_factory=RateLimiters.unlimited)
    circuit_breaker: Breaker = field(default_factory=NoopBreaker)
    archive_handler: AsyncStreamingArchiveHandler = field(
        default_factory=NoArchiveHandler
    )
    compactors: Compactors = field(default_factory=Compactors)
    speculation: SpeculationManager | None = None
