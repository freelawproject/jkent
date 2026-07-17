"""Run-scoped circuit breaker over transient failures.

One breaker is shared by every worker in a run, so the failure count is a
*run-global* signal: N consecutive transient failures pool-wide mean the server
(not one unlucky request) is in distress. Unlike a classic fail-fast breaker,
an open circuit **blocks** callers at :meth:`CircuitBreaker.gate` instead of
rejecting them — the queue is durable and every rejection would either mark a
request failed or burn its retry backoff toward ``max_backoff_time``, which
is exactly the pile-up the breaker exists to prevent. Workers simply pause;
per-request retry state is untouched.

Recovery is deadline-based: when the cool-down expires, the next caller
through the gate becomes the probe and the deadline is pushed one full window
out, so at most one probe is admitted per ``recovery_timeout`` — even if a
probe's verdict never arrives (e.g. it died to an unclassified error), the
gate self-heals by admitting another a window later. A success observed by
*any* worker (probe or an in-flight straggler) closes the circuit and wakes
every waiter; the rate limiter then re-spaces the released workers.

The policy is deliberately a mutable dataclass: hosts may adjust thresholds
mid-run (``run.circuit_breaker.policy.recovery_timeout = 60``). Mutations
take effect at the next transition — deadlines already set are not recomputed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from jkent import observability as obs
from jkent.contracts import require

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

CircuitState = Literal["closed", "open", "half_open"]

CLOSED: CircuitState = "closed"
OPEN: CircuitState = "open"
HALF_OPEN: CircuitState = "half_open"


@dataclass
class CircuitBreakerPolicy:
    """Tunable knobs for :class:`CircuitBreaker`.

    Attributes:
        failure_threshold: Consecutive pool-wide transient failures that trip
            the circuit. Any success resets the count.
        recovery_timeout: Seconds an open circuit waits before admitting a
            probe; also the spacing between probes while the server stays down.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 300.0


class CircuitBreaker:
    """Blocks the worker pool while the server is failing transiently.

    States: *closed* (normal; counting consecutive transient failures),
    *open* (all callers blocked until the recovery deadline), *half-open*
    (one probe admitted per recovery window; the rest keep waiting).

    ``record_failure`` / ``record_success`` are synchronous — transitions
    happen atomically between awaits on the single event loop, so no lock is
    needed. ``gate`` is stop-event-aware: a graceful shutdown wakes blocked
    workers immediately instead of leaving them parked out the cool-down.
    """

    @require(
        lambda policy: (
            policy is None
            or (policy.failure_threshold > 0 and policy.recovery_timeout > 0)
        ),
        "a positive failure threshold and recovery timeout — zero would trip "
        "on success or busy-spin the gate",
    )
    def __init__(
        self,
        policy: CircuitBreakerPolicy | None = None,
        *,
        stop_event: asyncio.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy if policy is not None else CircuitBreakerPolicy()
        self._stop_event = stop_event
        self._clock = clock
        self._state: CircuitState = CLOSED
        self._failure_count = 0
        self._deadline = 0.0
        # Set (and replaced with a fresh, unset event) when the circuit
        # closes, so waiters from one open period never see a stale set
        # event during the next one.
        self._wake = asyncio.Event()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (closed / open / half_open)."""
        return self._state

    async def gate(self) -> None:
        """Block until the circuit is closed or this caller is the probe.

        Returns immediately when closed, when the stop event is set (a
        shutdown must stay prompt; the worker loop exits on its next check),
        or when the recovery deadline has passed — in which case this caller
        is admitted as the probe and the deadline is pushed a full window out.
        """
        while True:
            if self._state is CLOSED:
                return
            if self._stop_event is not None and self._stop_event.is_set():
                return
            now = self._clock()
            remaining = self._deadline - now
            if remaining <= 0:
                self._state = HALF_OPEN
                self._deadline = now + self.policy.recovery_timeout
                logger.info(
                    "Circuit half-open: admitting one probe request "
                    "(next in %.0fs if no verdict)",
                    self.policy.recovery_timeout,
                )
                return
            await self._wait(remaining)

    def record_failure(self) -> None:
        """Count a transient failure; trip or re-open as warranted."""
        now = self._clock()
        if self._state is CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.policy.failure_threshold:
                self._state = OPEN
                self._deadline = now + self.policy.recovery_timeout
                logger.warning(
                    "Circuit opened after %d consecutive transient failures; "
                    "pausing requests for %.0fs",
                    self._failure_count,
                    self.policy.recovery_timeout,
                )
                obs.instruments().circuit_opens.add(1, obs.current_labels())
            return
        # Open or half-open: the probe failed, or an in-flight straggler
        # from before the trip did. Either way the server is still failing —
        # measure the cool-down from this latest observation.
        self._state = OPEN
        self._deadline = now + self.policy.recovery_timeout

    def record_success(self) -> None:
        """Reset the failure count; close the circuit and wake all waiters.

        A success from *any* worker counts — the probe, or an in-flight
        straggler that resolved while the circuit was open. Both are equal
        evidence the server is answering again.
        """
        self._failure_count = 0
        if self._state is CLOSED:
            return
        self._state = CLOSED
        logger.info("Circuit closed: server answered, resuming requests")
        wake, self._wake = self._wake, asyncio.Event()
        wake.set()

    async def _wait(self, timeout: float) -> None:
        """Park until woken (close / stop) or ``timeout`` elapses."""
        waiters = [asyncio.ensure_future(self._wake.wait())]
        if self._stop_event is not None:
            waiters.append(asyncio.ensure_future(self._stop_event.wait()))
        try:
            await asyncio.wait(
                waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for waiter in waiters:
                waiter.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*waiters, return_exceptions=True)
