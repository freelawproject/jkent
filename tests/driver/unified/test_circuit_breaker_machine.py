"""Generative rig for ``CircuitBreaker`` (jkent.driver.unified_driver.circuit_breaker).

A Hypothesis ``RuleBasedStateMachine`` drives random ``gate`` /
``record_failure`` / ``record_success`` / ``advance_clock`` sequences
against a real breaker, under a policy drawn per run, on a frozen clock, and
keeps a model of ``(state, consecutive failures, recovery window,
deadline)`` derived from the module's documented contract. After every
step the breaker's observable state must match the model, and these laws
must hold:

- ``state`` is one of the documented literals;
- after ``record_success`` the failure count is 0 and the circuit is
  closed; a *close* (success while open / half-open) resets the window to
  ``policy.recovery_timeout`` — a success while already closed is not a
  transition and leaves the dormant window alone;
- a trip starts at the base window, ``policy.recovery_timeout``;
- the recovery window lies within ``[recovery_timeout,
  max_recovery_timeout]`` and is non-decreasing while non-closed;
- at most one probe is admitted per recovery window: within one open
  period (trip .. close), consecutive probes are separated by at least the
  window in force at the earlier probe;
- failures count only while closed — the count is frozen while open or
  half-open, and reset by any success.

``gate`` is observed without wall-clock waits: the call becomes a task,
the loop is yielded a few times, and done-ness says admitted vs blocked
(a blocked task is cancelled and drained). One event loop per machine
run; rules are sync and await via ``run_until_complete`` because
Hypothesis does not compose with pytest-asyncio. ``time.monotonic`` is
frozen by freezegun per machine run; ``real_asyncio`` leaves the loop on
real time. Frozen time has microsecond resolution, so ``advance_to_deadline``
lands on the deadline only to within a microsecond.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from freezegun import freeze_time
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from typing_extensions import override

from jkent.driver.unified_driver.circuit_breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitState,
)

pytestmark = pytest.mark.generative

_STATES: frozenset[CircuitState] = frozenset({CLOSED, OPEN, HALF_OPEN})

_thresholds = st.integers(min_value=1, max_value=6)
_timeouts = st.floats(
    min_value=1.0, max_value=300.0, allow_nan=False, allow_infinity=False
)
_backoffs = st.floats(
    min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False
)
# (recovery_timeout, max_recovery_timeout): the policy requires base <= ceiling.
_windows = st.tuples(_timeouts, _timeouts).map(sorted)


def _policy(
    threshold: int, backoff: float, windows: list[float]
) -> CircuitBreakerPolicy:
    return CircuitBreakerPolicy(
        failure_threshold=threshold,
        recovery_backoff=backoff,
        recovery_timeout=windows[0],
        max_recovery_timeout=windows[1],
    )


_policies = st.builds(_policy, _thresholds, _backoffs, _windows)


async def _yield_loop(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class BreakerModel:
    """The documented contract, kept independently of the breaker."""

    def __init__(self, policy: CircuitBreakerPolicy) -> None:
        self.policy = policy
        self.state: CircuitState = CLOSED
        self.failures = 0
        self.window = policy.recovery_timeout
        self.deadline = 0.0

    def gate(self, now: float) -> bool:
        """Return True when the caller is admitted (closed, or the probe)."""
        if self.state is CLOSED:
            return True
        if self.deadline - now <= 0:
            self.state = HALF_OPEN
            self.deadline = now + self.window
            return True
        return False

    def record_failure(self, now: float) -> None:
        policy = self.policy
        if self.state is CLOSED:
            self.failures += 1
            if self.failures >= policy.failure_threshold:
                self.state = OPEN
                self.window = policy.recovery_timeout
                self.deadline = now + self.window
            return
        if self.state is HALF_OPEN:
            self.state = OPEN
            self.window = min(
                self.window * policy.recovery_backoff,
                policy.max_recovery_timeout,
            )
        self.deadline = now + self.window

    def record_success(self) -> None:
        self.failures = 0
        if self.state is CLOSED:
            return
        self.state = CLOSED
        self.window = self.policy.recovery_timeout


class CircuitBreakerMachine(RuleBasedStateMachine):
    """Walks breaker sequences, checking the laws after each step.

    The policy is drawn per run by :meth:`start` (an ``initialize`` rule,
    so it always precedes the others) and fixed for the run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.freezer = freeze_time("2026-01-01", real_asyncio=True)
        self.clock = self.freezer.start()
        self.stop_event = asyncio.Event()
        # ``start`` replaces these with the drawn policy; built here so the
        # attributes are initialized for the checkers and never None.
        default = CircuitBreakerPolicy()
        self.breaker = CircuitBreaker(default, stop_event=self.stop_event)
        self.model = BreakerModel(default)
        self.window_floor: float | None = None
        self.last_probe: tuple[float, float] | None = None

    @initialize(policy=_policies)
    def start(self, policy: CircuitBreakerPolicy) -> None:
        self.breaker = CircuitBreaker(policy, stop_event=self.stop_event)
        self.model = BreakerModel(policy)
        # Monotonicity while non-closed, and probe spacing, are tracked per
        # open period.
        self.window_floor = None
        self.last_probe = None

    @property
    def policy(self) -> CircuitBreakerPolicy:
        return self.breaker.policy

    # --- rules ------------------------------------------------------------

    @rule()
    def gate(self) -> None:
        now = time.monotonic()
        was_closed = self.model.state is CLOSED
        expected = self.model.gate(now)

        task = self.loop.create_task(self.breaker.gate())
        self.loop.run_until_complete(_yield_loop())
        admitted = task.done()
        if admitted:
            assert task.exception() is None
        else:
            task.cancel()
            self.loop.run_until_complete(
                asyncio.gather(task, return_exceptions=True)
            )

        assert admitted == expected, (
            f"gate {'admitted' if admitted else 'blocked'} at t={now} but "
            f"the model says {'admitted' if expected else 'blocked'} "
            f"(model state {self.model.state}, deadline {self.model.deadline})"
        )
        if admitted and not was_closed:
            # A probe: at most one per recovery window.
            if self.last_probe is not None:
                probe_at, probe_window = self.last_probe
                assert now >= probe_at + probe_window, (
                    f"second probe admitted at t={now}, inside the window "
                    f"of {probe_window}s opened by the probe at t={probe_at}"
                )
            self.last_probe = (now, self.breaker.recovery_window)

    @rule()
    def record_failure(self) -> None:
        now = time.monotonic()
        was_closed = self.model.state is CLOSED
        self.model.record_failure(now)
        self.breaker.record_failure()

        if was_closed and self.breaker.state is not CLOSED:
            base = self.policy.recovery_timeout
            assert self.breaker.recovery_window == base, (
                f"trip started the window at {self.breaker.recovery_window}, "
                f"not the policy's recovery_timeout {base}"
            )
            self.window_floor = self.breaker.recovery_window
            self.last_probe = None

    @rule()
    def record_success(self) -> None:
        was_closed = self.model.state is CLOSED
        self.model.record_success()
        self.breaker.record_success()

        assert self.breaker.state is CLOSED
        assert self.breaker.failure_count == 0
        if not was_closed:
            base = self.policy.recovery_timeout
            assert self.breaker.recovery_window == base, (
                f"close left the window at {self.breaker.recovery_window}, "
                f"not the policy's recovery_timeout {base}"
            )
        self.window_floor = None
        self.last_probe = None

    @rule(fraction=st.floats(min_value=0.0, max_value=1.0))
    def advance_clock(self, fraction: float) -> None:
        """Advance by a share of the policy's largest window."""
        self.clock.tick(fraction * self.policy.max_recovery_timeout)

    @precondition(lambda self: self.model.state is not CLOSED)
    @rule()
    def advance_to_deadline(self) -> None:
        """Land on the deadline — the boundary the gate tests."""
        remaining = self.model.deadline - time.monotonic()
        if remaining > 0:
            self.clock.tick(remaining)

    # --- invariants ---------------------------------------------------------

    @invariant()
    def state_is_documented(self) -> None:
        assert self.breaker.state in _STATES, (
            f"undocumented state {self.breaker.state!r}"
        )

    @invariant()
    def matches_model(self) -> None:
        assert self.breaker.state == self.model.state, (
            f"state {self.breaker.state} != model {self.model.state}"
        )
        assert self.breaker.failure_count == self.model.failures, (
            f"failure count {self.breaker.failure_count} != model "
            f"{self.model.failures}"
        )
        assert self.breaker.recovery_window == self.model.window, (
            f"window {self.breaker.recovery_window} != model "
            f"{self.model.window}"
        )
        expected_probe = (
            None if self.model.state is CLOSED else self.model.deadline
        )
        assert self.breaker.next_probe_at == expected_probe, (
            f"next probe at {self.breaker.next_probe_at} != model "
            f"{expected_probe}"
        )

    @invariant()
    def window_is_bounded(self) -> None:
        window = self.breaker.recovery_window
        lo, hi = self.policy.recovery_timeout, self.policy.max_recovery_timeout
        assert lo <= window <= hi, (
            f"recovery window {window} outside [{lo}, {hi}]"
        )

    @invariant()
    def window_never_shrinks_while_open(self) -> None:
        if self.breaker.state is CLOSED or self.window_floor is None:
            return
        window = self.breaker.recovery_window
        assert window >= self.window_floor, (
            f"recovery window shrank from {self.window_floor} to {window} "
            "while the circuit was not closed"
        )
        self.window_floor = window

    @invariant()
    def failures_frozen_while_not_closed(self) -> None:
        if self.breaker.state is not CLOSED:
            assert self.breaker.failure_count == self.model.failures

    # --- plumbing -----------------------------------------------------------

    @override
    def teardown(self) -> None:
        try:
            self.stop_event.set()
            self.loop.run_until_complete(_yield_loop())
        finally:
            self.loop.close()
            self.freezer.stop()


def test_circuit_breaker_machine() -> None:
    run_state_machine_as_test(CircuitBreakerMachine)
