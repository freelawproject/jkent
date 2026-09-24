"""Unified driver — a transport-agnostic driver stack.

The driver core owns orchestration (queue, workers, storage, retries); a
:class:`Transport` owns request execution and the lifecycle of whatever
resource that execution needs.

The package holds the concrete pieces: the transports, the rate limiter,
the orchestration substrate (queue, storage, step executor), and
the compactor.
"""

from __future__ import annotations

from jkent.driver.unified_driver.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerPolicy,
)
from jkent.driver.unified_driver.compaction import Compactor, Compactors
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
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
from jkent.driver.unified_driver.steps import StepExecutor
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    AwaitCondition,
    QueuedRequest,
    Transport,
    WorkerHandle,
)
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)

__all__ = [
    "AdaptiveRateLimiter",
    "ArchiveStream",
    "AsyncLifecycle",
    "AwaitCondition",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "Compactor",
    "Compactors",
    "DEFAULT_ADAPTIVE_LADDER",
    "StepExecutor",
    "ErrorBudget",
    "ErrorSink",
    "HttpxTransport",
    "NoopRateLimiter",
    "PyrateRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "RateLimiters",
    "RequestQueue",
    "ResponseStorage",
    "Transport",
    "WorkerHandle",
    "WorkerPool",
]
