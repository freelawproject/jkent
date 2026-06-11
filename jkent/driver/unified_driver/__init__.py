"""Unified driver — a transport-agnostic driver stack.

The driver core owns orchestration (queue, workers, storage, retries); a
:class:`Transport` owns request execution and the lifecycle of whatever
resource that execution needs.

The package holds the role protocols plus the concrete pieces: the
transports, the rate limiter, the orchestration substrate (queue, storage,
continuation executor), and the monitor/compactor.
"""

from __future__ import annotations

from jkent.driver.unified_driver.bootstrap import (
    RunBootstrapper,
    build_transport,
    resolve_browser_profile,
)
from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle, Recoverable
from jkent.driver.unified_driver.orchestration import (
    Compactor,
    Monitor,
    Run,
    Worker,
    WorkerMonitor,
)
from jkent.driver.unified_driver.persistence import (
    ReplayStorage,
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.rate_limiter import (
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
)
from jkent.driver.unified_driver.replay_run import ReplayRun
from jkent.driver.unified_driver.replay_worker import ReplayWorker
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
from jkent.driver.unified_driver.transport.replay_transport import (
    ReplayMiss,
    ReplayTransport,
)
from jkent.driver.unified_driver.worker import PoolWorker

__all__ = [
    "ArchiveStream",
    "AsyncLifecycle",
    "AwaitCondition",
    "CamoufoxTransport",
    "Compactor",
    "ContinuationExecutor",
    "HttpxTransport",
    "Monitor",
    "NoopRateLimiter",
    "PlaywrightTransport",
    "PoolWorker",
    "PyrateRateLimiter",
    "QueuedRequest",
    "RateLimiter",
    "Recoverable",
    "ReplayMiss",
    "ReplayRun",
    "ReplayStorage",
    "ReplayTransport",
    "ReplayWorker",
    "RequestQueue",
    "ResponseStorage",
    "Run",
    "RunBootstrapper",
    "ScrapeRun",
    "build_transport",
    "resolve_browser_profile",
    "Transport",
    "Worker",
    "WorkerHandle",
    "WorkerMonitor",
]
