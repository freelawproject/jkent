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
out, so at most one probe is admitted per window — even if a probe's verdict
never arrives (e.g. it died to an unclassified error), the gate self-heals by
admitting another a window later. A success observed by *any* worker (probe or
an in-flight straggler) closes the circuit and wakes every waiter; the rate
limiter then re-spaces the released workers.

The cool-down *escalates*: the first window is ``recovery_timeout``, and each
probe that fails while the circuit is already open multiplies it by
``recovery_backoff`` up to ``max_recovery_timeout``. This is what lets the
base window be short. A brief server hiccup — or a single flaky endpoint whose
requeued requests keep being picked as the probe — costs one short window
instead of a fixed long one, while a genuinely down server still backs off to
the same conservative spacing within a few probes. A close resets the window
to the base.

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
            the circuit. Any success resets the count. Sized against the pool:
            it must exceed ``num_workers`` by enough that one bad moment across
            every worker at once does not read as a downed server, or the
            breaker fires on ordinary flakiness.
        recovery_timeout: Seconds an open circuit waits before admitting the
            first probe.
        recovery_backoff: Multiplier applied to the cool-down each time a probe
            (or an in-flight straggler) fails while the circuit is already
            open. ``1.0`` disables escalation, restoring a fixed window.
        max_recovery_timeout: Ceiling on the escalated cool-down. Treated as no
            lower than ``recovery_timeout``, so a policy whose base already
            exceeds the ceiling simply never escalates.
    """

    failure_threshold: int = 8
    recovery_timeout: float = 30.0
    recovery_backoff: float = 2.0
    max_recovery_timeout: float = 300.0


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
            or (
                policy.failure_threshold > 0
                and policy.recovery_timeout > 0
                and policy.recovery_backoff >= 1.0
                and policy.max_recovery_timeout > 0
            )
        ),
        "a positive failure threshold and recovery timeout — zero would trip "
        "on success or busy-spin the gate — and a recovery backoff of at least "
        "1.0, since a shrinking cool-down would probe ever harder at a server "
        "that keeps failing",
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
        # The cool-down currently in force. Re-seeded from the policy on every
        # trip from closed (so mid-run policy edits are picked up) and
        # multiplied by each failed probe; reset by a close.
        self._recovery_window = self.policy.recovery_timeout
        # Set (and replaced with a fresh, unset event) when the circuit
        # closes, so waiters from one open period never see a stale set
        # event during the next one.
        self._wake = asyncio.Event()

    @property
    def state(self) -> CircuitState:
        """Current circuit state (closed / open / half_open)."""
        return self._state

    @property
    def recovery_window(self) -> float:
        """The cool-down currently in force, after any escalation."""
        return self._recovery_window

    def _escalate(self) -> float:
        """Grow the cool-down one step, capped, and return it.

        The ceiling is taken as the larger of the two policy values so a base
        window set above ``max_recovery_timeout`` is honoured rather than
        silently shortened.
        """
        ceiling = max(
            self.policy.recovery_timeout, self.policy.max_recovery_timeout
        )
        self._recovery_window = min(
            self._recovery_window * self.policy.recovery_backoff, ceiling
        )
        return self._recovery_window

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
                self._deadline = now + self._recovery_window
                logger.info(
                    "Circuit half-open: admitting one probe request "
                    "(next in %.0fs if no verdict)",
                    self._recovery_window,
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
                # A fresh trip starts at the base window, re-read from the
                # policy so a mid-run edit takes effect here.
                self._recovery_window = self.policy.recovery_timeout
                self._deadline = now + self._recovery_window
                logger.warning(
                    "Circuit opened after %d consecutive transient failures; "
                    "pausing requests for %.0fs",
                    self._failure_count,
                    self._recovery_window,
                )
                obs.instruments().circuit_opens.add(1, obs.current_labels())
            return
        # Open or half-open: the probe failed, or an in-flight straggler
        # from before the trip did. Either way the server is still failing —
        # measure the cool-down from this latest observation, and lengthen it,
        # so a server that stays down is probed progressively less often
        # instead of paying a fixed long window per attempt.
        self._state = OPEN
        window = self._escalate()
        self._deadline = now + window
        logger.warning(
            "Circuit re-opened: probe failed; next probe in %.0fs", window
        )

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
        # The escalation was evidence-driven; the evidence is gone, so the next
        # unrelated hiccup starts from the short base window again.
        self._recovery_window = self.policy.recovery_timeout
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
