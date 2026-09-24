"""Generative laws for ``AdaptiveRateLimiter`` (jkent.driver.unified_driver.rate_limiter).

A Hypothesis script — a ladder, an optional ``start_interval``, and a list
of ``(status_code, retry_after, clock_advance)`` steps — is replayed
against the real limiter on an injected fake clock, and a hand-written
model of the docstring's rules (the *oracle*) is stepped alongside it.
After every step the two must agree, and these laws must hold:

- ``current_interval`` is a member of the ladder, never above
  ``ladder[-1]``, and monotonically non-decreasing — descent is for the
  life of the run; one ``record_response`` moves at most one rung;
- a step-down happens at most once per cooldown window: after a step-down
  at time ``t`` to interval ``i``, no further step-down while the clock is
  below ``t + max(i, BUMP_COOLDOWN_FLOOR_S)``;
- ``_resume_at`` only ever extends (never decreases), and is ``None`` or
  at least the clock reading at the moment it was set; a step-down sets it
  to at least one new interval out;
- a status that is not 429 and carries no ``Retry-After`` never changes
  the rung;
- ``start_interval`` picks the first rung at least that slow — the initial
  ``current_interval`` is ``>= start_interval``, or it is ``ladder[-1]``.

The ``gate`` property drives the pause on a virtual clock: the limiter's
``asyncio.sleep`` is swapped for one that parks until the fake clock is
advanced past the requested target, so nothing waits on the wall clock.
The law: a gate that finds a pause set does not resolve while the clock
is below ``_resume_at`` (extensions included), and resolves once the
clock is past it.

The last test is a deterministic regression guard: a NaN ``retry_after``
must not reach ``_resume_at``, where ``gate`` could never clear it — every
gate on that lane would either raise (3.13's ``asyncio.sleep`` rejects NaN)
or spin forever (older Pythons). ``record_response`` drops non-finite
values.
"""

from __future__ import annotations

import asyncio
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.unified_driver.rate_limiter import (
    BUMP_COOLDOWN_FLOOR_S,
    DEFAULT_ADAPTIVE_LADDER,
    AdaptiveRateLimiter,
)

pytestmark = pytest.mark.generative

_TOO_MANY_REQUESTS = 429


def _request() -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com"
        ),
        step="parse",
    )


# --- Virtual time -----------------------------------------------------------


class VirtualClock:
    """A hand-advanced monotonic clock that can also stand in for sleep.

    ``__call__`` is the clock the limiter reads. ``sleep`` is the limiter's
    injected pause wait in the gate property: it parks the caller
    until :meth:`advance` carries the clock to (or past) the target, so a
    300s Retry-After costs no wall time.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
        due = [(t, f) for t, f in self._sleepers if t <= self.now]
        self._sleepers = [(t, f) for t, f in self._sleepers if t > self.now]
        for _, future in due:
            if not future.done():
                future.set_result(None)

    async def sleep(self, delay: float) -> None:
        target = self.now + delay
        if target <= self.now:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._sleepers.append((target, future))
        await future


# --- Strategies ---------------------------------------------------------------

# Rate() asserts a non-zero millisecond interval, so rungs are >= 1ms.
_rung = st.floats(
    min_value=0.001, max_value=120.0, allow_nan=False, allow_infinity=False
)
_ladders = st.one_of(
    st.just(DEFAULT_ADAPTIVE_LADDER),
    st.lists(_rung, min_size=1, max_size=8, unique=True).map(
        lambda xs: tuple(sorted(xs))
    ),
)
_start_intervals = st.none() | st.floats(
    min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False
)
_statuses = st.sampled_from([200, 404, 429, 503])
_retry_afters = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(
        min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    st.just(300.0),
)
_advances = st.floats(
    min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False
)
_steps = st.lists(st.tuples(_statuses, _retry_afters, _advances), max_size=40)


# --- Script property ----------------------------------------------------------


@given(ladder=_ladders, start_interval=_start_intervals, steps=_steps)
def test_adaptive_ladder_laws(
    ladder: tuple[float, ...],
    start_interval: float | None,
    steps: list[tuple[int, float | None, float]],
) -> None:
    clock = VirtualClock()
    limiter = AdaptiveRateLimiter(
        ladder, start_interval=start_interval, clock=clock
    )

    # start_interval selection
    initial = limiter.current_interval
    assert initial in ladder
    if start_interval is not None:
        assert initial >= start_interval or initial == ladder[-1], (
            f"start_interval={start_interval} landed on {initial}, which is "
            f"neither at least that slow nor the slowest rung {ladder[-1]}"
        )
    else:
        assert initial == ladder[0]

    previous_interval = initial
    previous_resume_at: float | None = limiter._resume_at
    last_step_at: float | None = None
    last_step_interval = initial

    for status_code, retry_after, advance in steps:
        now = clock.now
        limiter.record_response(status_code, retry_after=retry_after)

        interval = limiter.current_interval
        resume_at = limiter._resume_at

        # ladder membership and bound
        assert interval in ladder, f"{interval} is not a rung of {ladder}"
        assert interval <= ladder[-1]

        # monotone, one rung at a time
        assert interval >= previous_interval, (
            f"interval went faster: {previous_interval} -> {interval}"
        )
        assert ladder.index(interval) - ladder.index(previous_interval) <= 1, (
            f"one record_response moved more than one rung: "
            f"{previous_interval} -> {interval}"
        )

        # non-bump statuses never move the rung
        if status_code != _TOO_MANY_REQUESTS and retry_after is None:
            assert interval == previous_interval, (
                f"HTTP {status_code} without Retry-After moved the rung "
                f"{previous_interval} -> {interval}"
            )

        # at most one step-down per cooldown window
        cooldown_end = (
            -math.inf
            if last_step_at is None
            else last_step_at + max(last_step_interval, BUMP_COOLDOWN_FLOOR_S)
        )
        if interval != previous_interval:
            assert now >= cooldown_end, (
                f"stepped down at t={now} inside the cooldown that "
                f"ends at t={cooldown_end} (previous step at "
                f"t={last_step_at} to {last_step_interval})"
            )
            last_step_at = now
            last_step_interval = interval
            # a step-down holds the next request one new interval
            assert resume_at is not None and resume_at >= now + interval, (
                f"stepped down to {interval} at t={now} but the pause ends "
                f"at t={resume_at}"
            )
        elif (
            status_code == _TOO_MANY_REQUESTS
            and retry_after is None
            and now >= cooldown_end
            and previous_interval != ladder[-1]
        ):
            # Liveness: every other law here bounds how far the rung may
            # move, so without this one a limiter that never steps down at
            # all satisfies the suite.
            raise AssertionError(
                f"HTTP 429 at t={now}, outside the cooldown ending at "
                f"t={cooldown_end} and not yet at the slowest rung "
                f"{ladder[-1]}, left the interval at {interval}"
            )

        # the pause only ever extends, and never points into the past
        if resume_at is not None:
            assert not math.isnan(resume_at)
            if previous_resume_at is not None:
                assert resume_at >= previous_resume_at, (
                    f"pause shortened: {previous_resume_at} -> {resume_at}"
                )
            if resume_at != previous_resume_at:
                assert resume_at >= now, (
                    f"pause set to {resume_at}, before the clock at {now}"
                )
        else:
            assert previous_resume_at is None, "pause vanished"

        previous_interval = interval
        previous_resume_at = resume_at
        clock.advance(advance)


# --- Gate property --------------------------------------------------------------


async def _yield_loop(times: int = 10) -> None:
    for _ in range(times):
        await asyncio.sleep(0)


_pause_steps = st.lists(
    st.tuples(
        st.floats(
            min_value=0.0,
            max_value=400.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        st.none()
        | st.floats(
            min_value=0.0,
            max_value=300.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    max_size=8,
)


@given(
    retry_after=st.floats(
        min_value=0.01, max_value=300.0, allow_nan=False, allow_infinity=False
    ),
    steps=_pause_steps,
)
async def test_gate_resolves_only_once_the_clock_passes_resume_at(
    retry_after: float, steps: list[tuple[float, float | None]]
) -> None:
    """Each step advances the clock and may extend the pause first."""
    clock = VirtualClock()
    limiter = AdaptiveRateLimiter(clock=clock, sleep=clock.sleep)
    limiter.record_response(429, retry_after=retry_after)
    resume_at = limiter._resume_at
    # The step-down holds every gate one new interval, whichever is later.
    assert resume_at is not None
    assert resume_at == max(retry_after, limiter.current_interval)

    task = asyncio.create_task(limiter.gate(_request()))
    try:
        await _yield_loop()
        assert not task.done(), "gate resolved before the pause started"
        done_at: float | None = None
        for advance, extend in steps:
            if extend is not None and not task.done():
                limiter.record_response(503, retry_after=extend)
                assert limiter._resume_at is not None
                resume_at = max(resume_at, limiter._resume_at)
            clock.advance(advance)
            await _yield_loop()
            if task.done():
                assert task.exception() is None
                if done_at is None:
                    done_at = clock.now
                    assert done_at >= resume_at, (
                        f"gate resolved at t={done_at}, before the pause "
                        f"ends at t={resume_at}"
                    )
            else:
                assert clock.now < resume_at, (
                    f"gate still blocked at t={clock.now}, past the "
                    f"pause end at t={resume_at}"
                )
        # Finally push the clock past every pause: the gate must open.
        clock.advance(resume_at - clock.now + 1.0)
        await _yield_loop()
        assert task.done(), (
            f"gate still blocked at t={clock.now}, past the pause end "
            f"at t={resume_at}"
        )
        assert task.exception() is None
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


# --- Regression: NaN Retry-After must not wedge the lane ---------------------


async def test_nan_retry_after_does_not_block_gate_forever() -> None:
    """``record_response(retry_after=nan)`` must not make ``gate`` unusable.

    A ``nan`` ``_resume_at`` would make the gate's ``wait <= 0`` test always
    False, so the pause would never clear; ``record_response`` drops
    non-finite values instead.
    """
    limiter = AdaptiveRateLimiter()
    limiter.record_response(429, retry_after=float("nan"))
    await asyncio.wait_for(limiter.gate(_request()), timeout=1.0)
