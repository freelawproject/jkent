"""The ``Transport`` seam: the one thing the unified driver's backends differ on.

The driver core owns orchestration — the queue, the worker pool, storage,
rate limiting, retries. A ``Transport`` owns the other half: turning a
request into a response, the per-worker resource that work runs on, and
recovery when that resource dies. jkent ships two implementations — HTTP
(httpx) and browser (Playwright/Camoufox) — and hosts add their own: a
replay transport serving responses from previous-run DBs plugs into this
same interface, with no jkent changes.

The interface and its shared bases live in :mod:`.common` and are
re-exported here.
"""

from jkent.driver.unified_driver.transport.common import (
    MAX_RETRY_AFTER_S,
    ArchiveStream,
    AwaitCondition,
    FileArchiveStream,
    HandleT,
    NoopHandle,
    QueuedRequest,
    StatelessTransport,
    Transport,
    WorkerHandle,
    parse_retry_after,
)

__all__ = [
    "MAX_RETRY_AFTER_S",
    "ArchiveStream",
    "AwaitCondition",
    "FileArchiveStream",
    "HandleT",
    "NoopHandle",
    "QueuedRequest",
    "StatelessTransport",
    "Transport",
    "WorkerHandle",
    "parse_retry_after",
]
