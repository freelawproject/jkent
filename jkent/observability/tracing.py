"""Tracing helpers: the request span, per-phase spans, and label plumbing.

API-only, like :mod:`jkent.observability.metrics`: spans come from the global
TracerProvider. With no SDK configured the tracer returns non-recording spans,
so the ``with`` blocks below are cheap no-ops that still run their timing math
(a few ``time.monotonic()`` calls) — negligible per request.

``run_inst_id`` is read from OpenTelemetry **baggage**, which the host sets
before invoking the run (see ``EN_BANC_OTEL.md``). It is attached to spans and
the per-run gauges for cross-run correlation, but never to the high-volume
histograms.
"""

from __future__ import annotations

import contextlib
import time
from functools import lru_cache
from typing import TYPE_CHECKING

from opentelemetry import baggage, trace

from jkent.observability.metrics import current_labels, instruments

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opentelemetry.trace import Span, Tracer

_TRACER_NAME = "jkent.driver"


@lru_cache(maxsize=1)
def tracer() -> Tracer:
    """The process-wide jkent tracer (fetched once)."""
    return trace.get_tracer(_TRACER_NAME)


def run_inst_id() -> str | None:
    """The host-provided run id from baggage, or None if unset."""
    value = baggage.get_baggage("run_inst_id")
    return str(value) if value is not None else None


@contextlib.contextmanager
def request_span(*, scraper: str, step: str | None = None) -> Iterator[Span]:
    """Open the top-level ``jkent.request`` span for one request.

    Stamps ``scraper`` / ``run_inst_id`` (and ``step`` when known) as span
    attributes and records an overall ``phase="total"`` duration on exit. The
    caller sets ``jkent.step`` / ``jkent.outcome`` on the yielded span as those
    become known.
    """
    labels = {"scraper": scraper}
    if step is not None:
        labels["step"] = step
    start = time.monotonic()
    with tracer().start_as_current_span("jkent.request") as span:
        span.set_attribute("jkent.scraper", scraper)
        if step is not None:
            span.set_attribute("jkent.step", step)
        fid = run_inst_id()
        if fid is not None:
            span.set_attribute("jkent.run_inst_id", fid)
        try:
            yield span
        finally:
            instruments().request_duration.record(
                time.monotonic() - start, {**labels, "phase": "total"}
            )


@contextlib.contextmanager
def phase(name: str) -> Iterator[Span]:
    """Time one request phase: a child span plus a ``request.duration`` record.

    Attributes come from the ambient label context (``scraper`` / ``step``)
    plus ``phase=name``. Any auto-instrumented client call inside (httpx,
    SQLAlchemy) nests under this span.
    """
    labels = current_labels()
    start = time.monotonic()
    with tracer().start_as_current_span(f"jkent.{name}") as span:
        try:
            yield span
        finally:
            instruments().request_duration.record(
                time.monotonic() - start, {**labels, "phase": name}
            )
