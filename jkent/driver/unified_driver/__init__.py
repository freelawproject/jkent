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
from jkent.driver.unified_driver.persistence import (
    ErrorBudget,
    ErrorSink,
    RequestQueue,
    ResponseStorage,
    RowOnlyErrorSink,
)
from jkent.driver.unified_driver.pool import WorkerPool
from jkent.driver.unified_driver.rate_limiter import (
    DEFAULT_ADAPTIVE_LADDER,
    AdaptiveRateLimiter,
    CappedPaceLimiter,
    NoopRateLimiter,
    ProbeReport,
    RateLimiter,
    RateLimiters,
    ReactivePaceLimiter,
    Turn,
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
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)

__all__ = [
    "AdaptiveRateLimiter",
    "ArchiveStream",
    "AwaitCondition",
    "CappedPaceLimiter",
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "Compactor",
    "Compactors",
    "DEFAULT_ADAPTIVE_LADDER",
    "MAX_RETRY_AFTER_S",
    "ErrorBudget",
    "ErrorSink",
    "FileArchiveStream",
    "HttpxTransport",
    "NoopHandle",
    "NoopRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "RateLimiters",
    "RequestQueue",
    "ResponseStorage",
    "RowOnlyErrorSink",
    "StatelessTransport",
    "parse_retry_after",
    "Transport",
    "WorkerHandle",
    "ProbeReport",
    "RateLimiter",
    "RateLimiters",
    "ReactivePaceLimiter",
    "Turn",
    "WorkerPool",
]
