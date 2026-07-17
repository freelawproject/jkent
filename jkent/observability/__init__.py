"""jkent observability — OpenTelemetry instrumentation, API-only.

This package uses the OpenTelemetry **API** only; it never configures an SDK,
exporter, or sampler. With no SDK installed (the default, and always the case
when jkent is imported by juriscraper) every span and metric is a no-op, so the
instrumentation costs effectively nothing. The host process owns SDK
setup and exporter policy.

Public surface:

* :func:`instruments` / :func:`labeled` / :func:`current_labels` — metrics.
* :func:`request_span` / :func:`phase` / :func:`run_inst_id` — tracing.
* :class:`LoopLagMonitor` — the shared-loop lag sampler (started by the host).
"""

from __future__ import annotations

from jkent.observability.instrumented_lock import InstrumentedLock
from jkent.observability.loop_monitor import LoopLagMonitor
from jkent.observability.metrics import (
    current_labels,
    instruments,
    labeled,
    sdk_active,
)
from jkent.observability.tracing import (
    phase,
    request_span,
    run_inst_id,
    tracer,
)


def reset() -> None:
    """Drop the cached tracer/instruments so they re-resolve on next use.

    :func:`tracer` and :func:`instruments` memoize their first resolution; if
    either ran before the host installed its SDK providers, the memo pins a
    no-op proxy for the life of the process. Hosts call this right after
    provider installation so the next resolution binds to the real providers.
    """
    tracer.cache_clear()
    instruments.cache_clear()


__all__ = [
    "InstrumentedLock",
    "LoopLagMonitor",
    "current_labels",
    "run_inst_id",
    "instruments",
    "labeled",
    "phase",
    "request_span",
    "reset",
    "sdk_active",
    "tracer",
]
