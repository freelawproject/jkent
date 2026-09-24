"""Unified driver — a transport-agnostic driver stack.

The driver core owns orchestration (queue, workers, storage, retries); a
:class:`Transport` owns request execution and the lifecycle of whatever
resource that execution needs.

The package holds the concrete pieces: the transports, the rate limiter,
the orchestration substrate (queue, storage, step executor), and
the compactor.
"""

from __future__ import annotations

from jkent.driver.unified_driver.bootstrap import (
    RunBootstrapper,
    build_transport,
    resolve_browser_profile,
)
from jkent.driver.unified_driver.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)
from jkent.driver.unified_driver.compaction import Compactor, Compactors
from jkent.driver.unified_driver.persistence import (
    ErrorBudget,
    ErrorSink,
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.pool import WorkerPool
from jkent.driver.unified_driver.rate_limiter import (
    DEFAULT_ADAPTIVE_LADDER,
    AdaptiveRateLimiter,
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
    RateLimiters,
)
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.steps import StepExecutor
from jkent.driver.unified_driver.transport import (
    MAX_RETRY_AFTER_S,
    ArchiveStream,
    AwaitCondition,
    FileArchiveStream,
    NoopHandle,
    QueuedRequest,
    StatelessTransport,
    Transport,
    WorkerHandle,
    parse_retry_after,
)
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from jkent.driver.unified_driver.wiring import (
    RunCollaborators,
    RunConfig,
    RunHooks,
)
from jkent.driver.unified_driver.worker import PoolWorker

__all__ = [
    "AdaptiveRateLimiter",
    "ArchiveStream",
    "AwaitCondition",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "Compactor",
    "Compactors",
    "DEFAULT_ADAPTIVE_LADDER",
    "MAX_RETRY_AFTER_S",
    "StepExecutor",
    "ErrorBudget",
    "ErrorSink",
    "FileArchiveStream",
    "HttpxTransport",
    "NoopHandle",
    "NoopRateLimiter",
    "PlaywrightTransport",
    "PoolWorker",
    "PyrateRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "RateLimiters",
    "RequestQueue",
    "ResponseStorage",
    "RunBootstrapper",
    "RunCollaborators",
    "RunConfig",
    "RunHooks",
    "ScrapeRun",
    "StatelessTransport",
    "build_transport",
    "parse_retry_after",
    "resolve_browser_profile",
    "Transport",
    "WorkerHandle",
    "WorkerPool",
]
