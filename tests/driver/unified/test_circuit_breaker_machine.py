"""Generative rig for ``CircuitBreaker`` (jkent.driver.unified_driver.circuit_breaker).

A Hypothesis ``RuleBasedStateMachine`` drives random ``gate`` /
``record_failure`` / ``record_success`` / ``advance_clock`` /
``mutate_policy`` sequences against a real breaker on a fake clock, and
keeps a model of ``(state, consecutive failures, recovery window,
deadline)`` derived from the module's documented contract. After every
step the breaker's observable state must match the model, and these laws
must hold:

- ``state`` is one of the documented literals;
- after ``record_success`` the failure count is 0 and the circuit is
  closed; a *close* (success while open / half-open) resets the window to
  ``policy.recovery_timeout`` — a success while already closed is not a
  transition and leaves the dormant window alone;
- a trip re-seeds the window from ``policy.recovery_timeout`` at that
  moment, so a mid-run policy edit is picked up at the next trip;
- the recovery window lies within ``[recovery_timeout,
  max(recovery_timeout, max_recovery_timeout)]`` and is non-decreasing
  while non-closed. Both are stated relative to the policy in force at
  the last transition: the docstring promises mutations take effect *at
  the next transition*, so a lowered ceiling may shrink the window on the
  next failed probe, and a raised base is not applied until the next trip
  or close. The rig re-baselines these two laws on ``mutate_policy``;
- at most one probe is admitted per recovery window: within one open
  period (trip .. close), consecutive probes are separated by at least the
  window in force at the earlier probe (re-baselined on ``mutate_policy``
  for the same reason);
- failures count only while closed — the count is frozen while open or
  half-open, and reset by any success.

``gate`` is observed without wall-clock waits: the call becomes a task,
the loop is yielded a few times, and done-ness says admitted vs blocked
(a blocked task is cancelled and drained). One event loop per machine
run; rules are sync and await via ``run_until_complete`` because
Hypothesis does not compose with pytest-asyncio.
"""

from __future__ import annotations

import asyncio

import pytest
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
from tests.driver.unified.test_circuit_breaker import Clock

pytestmark = pytest.mark.generative

_STATES: frozenset[str] = frozenset({CLOSED, OPEN, HALF_OPEN})

_thresholds = st.integers(min_value=1, max_value=6)
_timeouts = st.floats(
    min_value=1.0, max_value=300.0, allow_nan=False, allow_infinity=False
)
_backoffs = st.floats(
    min_value=1.0, max_value=4.0, allow_nan=False, allow_infinity=False
)
_policies = st.builds(
    CircuitBreakerPolicy,
    failure_threshold=_thresholds,
    recovery_timeout=_timeouts,
    recovery_backoff=_backoffs,
    max_recovery_timeout=_timeouts,
)


async def _yield_loop(times: int = 5) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


class BreakerModel:
    """The documented contract, kept independently of the breaker."""

    def __init__(self, policy: CircuitBreakerPolicy) -> None:
        self.state: CircuitState = CLOSED
        self.failures = 0
        self.window = policy.recovery_timeout
        self.deadline = 0.0

    @staticmethod
    def _ceiling(policy: CircuitBreakerPolicy) -> float:
        return max(policy.recovery_timeout, policy.max_recovery_timeout)

    def gate(self, now: float, policy: CircuitBreakerPolicy) -> bool:
        """Return True when the caller is admitted (closed, or the probe)."""
        if self.state is CLOSED:
            return True
        if self.deadline - now <= 0:
            self.state = HALF_OPEN
            self.deadline = now + self.window
            return True
        return False

    def record_failure(self, now: float, policy: CircuitBreakerPolicy) -> None:
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
                self.window * policy.recovery_backoff, self._ceiling(policy)
            )
        self.deadline = now + self.window

    def record_success(self, policy: CircuitBreakerPolicy) -> None:
        self.failures = 0
        if self.state is CLOSED:
            return
        self.state = CLOSED
        self.window = policy.recovery_timeout


class CircuitBreakerMachine(RuleBasedStateMachine):
    """Walks breaker sequences, checking the laws after each step.

    The starting policy is drawn per run by :meth:`start` (an
    ``initialize`` rule, so it always precedes the others); later edits
    come through :meth:`mutate_policy`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.clock = Clock()
        self.stop_event = asyncio.Event()
        # ``start`` replaces these with the drawn policy; built here so the
        # attributes are initialized for the checkers and never None.
        default = CircuitBreakerPolicy()
        self.breaker = CircuitBreaker(
            default, stop_event=self.stop_event, clock=self.clock
        )
        self.model = BreakerModel(default)
        self.window_lo = 0.0
        self.window_hi = 0.0
        self.window_floor: float | None = None
        self.last_probe: tuple[float, float] | None = None

    @initialize(policy=_policies)
    def start(self, policy: CircuitBreakerPolicy) -> None:
        self.breaker = CircuitBreaker(
            policy, stop_event=self.stop_event, clock=self.clock
        )
        self.model = BreakerModel(policy)
        # Bounds the window law is checked against: the policy in force at
        # the last window-setting transition (re-baselined on mutation).
        self.window_lo = policy.recovery_timeout
        self.window_hi = BreakerModel._ceiling(policy)
        # Monotonicity while non-closed, and probe spacing, are tracked per
        # open period and re-baselined on mutation.
        self.window_floor = None
        self.last_probe = None

    @property
    def policy(self) -> CircuitBreakerPolicy:
        return self.breaker.policy

    # --- rules ------------------------------------------------------------

    @rule()
    def gate(self) -> None:
        now = self.clock.now
        was_closed = self.model.state is CLOSED
        expected = self.model.gate(now, self.policy)

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
            # A probe: at most one per recovery window. The spacing a probe
            # guarantees is its window, or the policy ceiling if an edit
            # has already lowered that below the (not recomputed) window —
            # a failed verdict re-derives the deadline under the new cap.
            if self.last_probe is not None:
                probe_at, probe_window = self.last_probe
                assert now >= probe_at + probe_window, (
                    f"second probe admitted at t={now}, inside the window "
                    f"of {probe_window}s opened by the probe at t={probe_at}"
                )
            self.last_probe = (
                now,
                min(
                    self.breaker.recovery_window,
                    BreakerModel._ceiling(self.policy),
                ),
            )

    @rule()
    def record_failure(self) -> None:
        now = self.clock.now
        was_closed = self.model.state is CLOSED
        base_now = self.policy.recovery_timeout
        ceiling_now = BreakerModel._ceiling(self.policy)
        self.model.record_failure(now, self.policy)
        self.breaker.record_failure()

        if was_closed and self.breaker.state is not CLOSED:
            # A trip re-seeds the window from the policy at this moment.
            assert self.breaker.recovery_window == base_now, (
                f"trip seeded the window at {self.breaker.recovery_window}, "
                f"not the policy's recovery_timeout {base_now}"
            )
            self.window_lo = base_now
            self.window_hi = ceiling_now
            self.window_floor = self.breaker.recovery_window
            self.last_probe = None
        elif not was_closed:
            # An escalation is capped by the ceiling in force now.
            self.window_hi = max(self.window_hi, ceiling_now)

    @rule()
    def record_success(self) -> None:
        was_closed = self.model.state is CLOSED
        base_now = self.policy.recovery_timeout
        self.model.record_success(self.policy)
        self.breaker.record_success()

        assert self.breaker.state is CLOSED
        assert self.breaker.failure_count == 0
        if not was_closed:
            assert self.breaker.recovery_window == base_now, (
                f"close left the window at {self.breaker.recovery_window}, "
                f"not the policy's recovery_timeout {base_now}"
            )
            self.window_lo = base_now
            self.window_hi = BreakerModel._ceiling(self.policy)
        self.window_floor = None
        self.last_probe = None

    @rule(fraction=st.floats(min_value=0.0, max_value=1.0))
    def advance_clock(self, fraction: float) -> None:
        """Advance by a share of the policy's largest window."""
        self.clock.advance(fraction * BreakerModel._ceiling(self.policy))

    @precondition(lambda self: self.model.state is not CLOSED)
    @rule()
    def advance_to_deadline(self) -> None:
        """Land exactly on the deadline — the boundary the gate tests."""
        remaining = self.model.deadline - self.clock.now
        if remaining > 0:
            self.clock.advance(remaining)

    @rule(
        failure_threshold=_thresholds,
        recovery_timeout=_timeouts,
        recovery_backoff=_backoffs,
        max_recovery_timeout=_timeouts,
    )
    def mutate_policy(
        self,
        failure_threshold: int,
        recovery_timeout: float,
        recovery_backoff: float,
        max_recovery_timeout: float,
    ) -> None:
        """Edit the live policy; effects land at the next transition."""
        self.policy.failure_threshold = failure_threshold
        self.policy.recovery_timeout = recovery_timeout
        self.policy.recovery_backoff = recovery_backoff
        self.policy.max_recovery_timeout = max_recovery_timeout
        # The window in force was computed under the old policy and is not
        # recomputed; the range/monotone/spacing laws restart from here. A
        # lowered ceiling caps the *next* escalation, which may shrink the
        # window to the new ceiling — so the lower bound and the monotone
        # floor drop to it now.
        ceiling = BreakerModel._ceiling(self.policy)
        window = self.breaker.recovery_window
        self.window_lo = min(self.window_lo, window, ceiling)
        self.window_hi = max(self.window_hi, window, ceiling)
        self.window_floor = (
            None if self.model.state is CLOSED else min(window, ceiling)
        )
        self.last_probe = None

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
        assert self.window_lo <= window <= self.window_hi, (
            f"recovery window {window} outside "
            f"[{self.window_lo}, {self.window_hi}]"
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
        # Ratchet, but never above the ceiling the next escalation is capped
        # by — a policy edit may have lowered it below the current window.
        capped = min(window, BreakerModel._ceiling(self.policy))
        self.window_floor = (
            capped
            if self.window_floor is None
            else max(self.window_floor, capped)
        )

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


def test_circuit_breaker_machine() -> None:
    run_state_machine_as_test(CircuitBreakerMachine)
