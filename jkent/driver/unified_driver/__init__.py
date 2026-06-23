"""Unified driver — a transport-agnostic driver stack.

The driver core owns orchestration (queue, workers, storage, retries); a
:class:`Transport` owns request execution and the lifecycle of whatever
resource that execution needs.

The package holds the concrete pieces: the transports, the rate limiter,
the orchestration substrate (queue, storage, continuation executor), and
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
from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle, Recoverable
from jkent.driver.unified_driver.orchestration import Compactor
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.rate_limiter import (
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
)
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    AwaitCondition,
    QueuedRequest,
    Transport,
    WorkerHandle,
)
from jkent.driver.unified_driver.transport.camoufox_transport import (
    CamoufoxTransport,
)
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from jkent.driver.unified_driver.worker import PoolWorker

__all__ = [
    "ArchiveStream",
    "AsyncLifecycle",
    "AwaitCondition",
    "CamoufoxTransport",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "Compactor",
    "ContinuationExecutor",
    "HttpxTransport",
    "NoopRateLimiter",
    "PlaywrightTransport",
    "PoolWorker",
    "PyrateRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "Recoverable",
    "RequestQueue",
    "ResponseStorage",
    "RunBootstrapper",
    "ScrapeRun",
    "build_transport",
    "resolve_browser_profile",
    "Transport",
    "WorkerHandle",
]
