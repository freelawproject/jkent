"""Metric instruments and the cross-cutting label context.

API-only: instruments come from the *global* MeterProvider via
``opentelemetry.metrics.get_meter``. When no SDK MeterProvider is configured
(the default, and always the case under juriscraper), ``get_meter`` returns a
proxy whose instruments are no-ops — so importing and recording here costs
effectively nothing until the host installs an SDK.

The instruments are built once, lazily, on first use. Building them before the
host sets its MeterProvider is safe: the proxy meter re-binds its instruments
to the real ones when the provider is installed.

``labeled`` / ``current_labels`` carry the low-cardinality dimensions
(``scraper``, ``step``) down to instrumentation sites that have no other way to
learn them — the shared DB lock and the compression functions are generic
infrastructure with no scraper reference in scope. They ride a
:class:`contextvars.ContextVar`, so a value set in the worker's request task is
visible to the synchronous ``compress`` call it later makes on the same task,
and is reset on scope exit.

High-cardinality identifiers (``run_inst_id``) deliberately do NOT go on the
histograms here — only on spans and the per-run gauges — to keep metric
time-series bounded.
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
from collections.abc import Generator
from functools import lru_cache
from typing import TYPE_CHECKING

from opentelemetry.metrics import get_meter, get_meter_provider

if TYPE_CHECKING:
    pass

_METER_NAME = "jkent.driver"


class Phase(str, enum.Enum):
    """The ``phase`` attribute vocabulary (see ``METRICS.md``).

    A ``str`` subclass, so members pass straight through as OTel attribute
    values and interpolate into span names. ``__str__`` is pinned to the
    value because plain ``str, Enum`` renders ``Phase.TOTAL`` on 3.10
    (:class:`enum.StrEnum` does this for us from 3.11).
    """

    __str__ = str.__str__

    TOTAL = "total"
    CIRCUIT_BREAKER_GATE = "circuit_breaker.gate"
    RATE_LIMITER_GATE = "rate_limiter.gate"
    TRANSPORT_RESOLVE = "transport.resolve"
    CONTINUATION = "continuation"
    COMPRESS = "compress"


class Outcome(str, enum.Enum):
    """The ``jkent.outcome`` span attribute vocabulary (see ``METRICS.md``).

    A ``str`` subclass with ``__str__`` pinned, for the same reasons as
    :class:`Phase`.
    """

    __str__ = str.__str__

    OK = "ok"
    HALT = "halt"
    SKIP = "skip"
    TRANSIENT = "transient"
    SPECULATION_HTTP = "speculation_http"
    PERSISTENT_HTTP = "persistent_http"
    ERROR = "error"


class _Instruments:
    """The jkent metric instrument set, created once from the global meter."""

    def __init__(self) -> None:
        meter = get_meter(_METER_NAME)

        # The keystone signal: scheduling latency on the shared event loop.
        # High values mean something is blocking the loop (sync compression /
        # parsing) and every co-resident worker and Run stalls with it.
        self.loop_lag = meter.create_histogram(
            "jkent.event_loop.lag",
            unit="s",
            description="Event-loop scheduling latency (measured drift).",
        )

        # Per-request phase wall time (attrs: scraper, step, phase, outcome).
        self.request_duration = meter.create_histogram(
            "jkent.request.duration",
            unit="s",
            description="Wall-clock duration of a request phase.",
        )
        # On-loop CPU time for the synchronous phases we can measure cleanly
        # (compression today). Separates loop-hogging CPU from awaited I/O.
        self.request_cpu = meter.create_histogram(
            "jkent.request.cpu_time",
            unit="s",
            description="On-loop CPU time of a synchronous request phase.",
        )

        # The single per-run SQLite lock: how long callers wait for it and how
        # long each holder keeps it. This is the "lock getting fought over".
        self.lock_wait = meter.create_histogram(
            "jkent.db.lock.wait",
            unit="s",
            description="Time spent waiting to acquire the run's DB lock.",
        )
        self.lock_hold = meter.create_histogram(
            "jkent.db.lock.hold",
            unit="s",
            description="Time the run's DB lock was held by one caller.",
        )

        # zstd compression (attrs: scraper, step, kind). Currently synchronous
        # on the loop; these prove whether it is a loop-blocker before any fix.
        self.compression_duration = meter.create_histogram(
            "jkent.compression.duration",
            unit="s",
            description="Duration of a zstd compress/decompress operation.",
        )
        self.compression_ratio = meter.create_histogram(
            "jkent.compression.ratio",
            unit="1",
            description="Compressed/original size ratio (lower is better).",
        )
        # The one-shot train-dictionary + recompress burst at the step's
        # threshold — a large, single loop-blocking event worth isolating.
        self.compaction_duration = meter.create_histogram(
            "jkent.compaction.duration",
            unit="s",
            description="Duration of a step's train+recompress compaction.",
        )

        # Worker idle wait: time a worker spent parked because the queue had
        # nothing ready (retry backoff, or a sibling's request still in
        # flight). The pool is pinned, so this is the oversize signal: summed
        # against request.duration{phase=total} it gives per-scraper worker
        # utilization. Mostly-idle workers -> lower num_workers; zero idle
        # with a growing queue.pending -> raise it.
        self.worker_idle = meter.create_histogram(
            "jkent.worker.idle",
            unit="s",
            description="Time a worker waited because no request was ready.",
        )
        # Scheduled retries after transient failures. A retry rate that
        # climbs with worker count is server pushback (429/5xx/timeouts) —
        # evidence against raising num_workers even when loop/lock/gate all
        # have headroom.
        self.request_retries = meter.create_counter(
            "jkent.request.retries",
            unit="1",
            description="Transient-failure retries scheduled.",
        )
        # Circuit-breaker trips: the run-wide escalation of request.retries.
        # A trip means failure_threshold consecutive transient failures
        # pool-wide — the whole pool paused, not one request backing off.
        self.circuit_opens = meter.create_counter(
            "jkent.circuit.opens",
            unit="1",
            description="Circuit-breaker trips (pool-wide transient pile-up).",
        )

        # Per-run state (attrs: scraper, run_inst_id). Gauges, so the last
        # value stands until updated.
        self.worker_active = meter.create_gauge(
            "jkent.worker.active",
            unit="1",
            description="Live continuation-worker count for a run.",
        )
        self.queue_pending = meter.create_gauge(
            "jkent.queue.pending",
            unit="1",
            description="Pending request count for a run.",
        )


@lru_cache(maxsize=1)
def instruments() -> _Instruments:
    """The process-wide jkent instrument set (built once, lazily)."""
    return _Instruments()


def sdk_active() -> bool:
    """Whether a real (SDK) MeterProvider is installed on the global API.

    Recording on a no-op instrument is free, so instrumentation sites never
    need this check. It exists for work done *solely* to obtain a value to
    record — e.g. the queue-backlog sampler's periodic DB count — so that
    work can be skipped entirely when nothing would be emitted. Heuristic:
    a provider from an ``opentelemetry.sdk`` module (the proxy/no-op
    providers live under the API package).
    """
    return "opentelemetry.sdk" in type(get_meter_provider()).__module__


# --- Cross-cutting labels -------------------------------------------------

# Default None (not {}) so the ContextVar holds no shared mutable; readers
# coalesce None to an empty dict.
_labels: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar[dict[str, str] | None](
        "jkent_metric_labels", default=None
    )
)


@contextlib.contextmanager
def labeled(**new: str) -> Generator[None, None, None]:
    """Add ``new`` to the label context.

    Nestable: an inner scope adds to the outer one and restores it on exit.
    Used to carry ``scraper`` / ``step`` to instrumentation sites that cannot
    otherwise see them.
    """
    token = _labels.set({**(_labels.get() or {}), **new})
    try:
        yield
    finally:
        _labels.reset(token)


def current_labels() -> dict[str, str]:
    """A copy of the labels in scope (``scraper`` / ``step`` when set)."""
    return dict(_labels.get() or {})
