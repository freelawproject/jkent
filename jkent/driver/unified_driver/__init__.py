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
from jkent.driver.unified_driver.pool import WorkerPool
from jkent.driver.unified_driver.rate_limiter import (
    DEFAULT_ADAPTIVE_LADDER,
    AdaptiveRateLimiter,
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
    RateLimiters,
)
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

__all__ = [
    "AdaptiveRateLimiter",
    "ArchiveStream",
    "AwaitCondition",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "DEFAULT_ADAPTIVE_LADDER",
    "MAX_RETRY_AFTER_S",
    "FileArchiveStream",
    "NoopHandle",
    "NoopRateLimiter",
    "PyrateRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "RateLimiters",
    "StatelessTransport",
    "parse_retry_after",
    "Transport",
    "WorkerHandle",
    "WorkerPool",
]
