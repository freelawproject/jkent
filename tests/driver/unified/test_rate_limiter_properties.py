"""Generative laws for the pacing limiters (jkent.driver.unified_driver.rate_limiter).

**Capped pace.** A Hypothesis script of arrivals — each a gap before the
next request shows up, and how late its wake lands — is played against a
real ``CappedPaceLimiter`` on a virtual clock. The laws:

- no two starts are ever closer than the interval, however late a wake
  lands (a blocked event loop pushes the next slot back, never forward);
- with punctual wakes, a request starts at ``max(arrival, previous start +
  interval)`` exactly: nothing is added to a wait, and time the lane did
  not use buys no earlier start later.

**Reactive pace.** Requests with generated response times are fetched
through a real ``ReactivePaceLimiter`` by several concurrent callers. The
laws: no two are ever in flight together, and each starts no sooner than
the rest after the one before it finished.

**The probe.** A script of ``(status, retry_after, sent how long ago,
clock advance, gate?)`` steps is replayed against a real
``AdaptiveRateLimiter``, and a hand-written model of the docstring's rules
(the *oracle*) is stepped alongside it. After every step the two must
agree, and these laws must hold:

- ``current_interval`` is a member of the ladder, and never faster than
  the declared pace;
- one ``record_response`` moves at most one rung, and only slower; a
  status that is not 429 and carries no usable ``Retry-After`` moves
  nothing; pushback on a request sent before the last step-down moves no
  rung;
- the pause only ever extends;
- one started request recovers at most one rung, and only after
  ``recover_after`` without counted pushback, time spent paused excluded.

The last test is a deterministic regression guard: a NaN ``retry_after``
must not become a pause, which no clock reading would ever clear.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.unified_driver.rate_limiter import (
    DEFAULT_ADAPTIVE_LADDER,
    AdaptiveRateLimiter,
    CappedPaceLimiter,
    ReactivePaceLimiter,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

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
    """A monotonic clock and a matching sleep, in virtual seconds.

    Sleepers park until the clock reaches their target. :meth:`run` drives
    coroutines to completion, jumping the clock to the next sleeper's
    target (plus ``lag``, a late wake) whenever every task is parked.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.lag = 0.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        future: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self._sleepers.append((self.now + delay, future))
        await future

    def advance(self, seconds: float) -> None:
        self.now += seconds
        due = [f for t, f in self._sleepers if t <= self.now]
        self._sleepers = [(t, f) for t, f in self._sleepers if t > self.now]
        for future in due:
            if not future.done():
                future.set_result(None)

    async def run(self, *coros: Coroutine[Any, Any, Any]) -> None:
        tasks = [asyncio.ensure_future(coro) for coro in coros]
        while True:
            for _ in range(20):
                await asyncio.sleep(0)
            if all(task.done() for task in tasks):
                for task in tasks:
                    task.result()
                return
            targets = [t for t, f in self._sleepers if not f.done()]
            assert targets, "tasks are parked on something other than time"
            self.advance(max(min(targets) - self.now, 0.0) + self.lag)


_seconds = st.floats(
    min_value=0.001, max_value=120.0, allow_nan=False, allow_infinity=False
)
_waits = st.floats(
    min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False
)


# --- Capped pace ---------------------------------------------------------------


@given(
    interval=_seconds,
    arrivals=st.lists(st.tuples(_waits, _waits), min_size=1, max_size=25),
)
async def test_capped_starts_are_never_closer_than_the_interval(
    interval: float, arrivals: list[tuple[float, float]]
) -> None:
    clock = VirtualClock()
    limiter = CappedPaceLimiter(interval, clock=clock, sleep=clock.sleep)
    starts: list[float] = []
    for gap, lag in arrivals:
        clock.advance(gap)
        clock.lag = lag
        await clock.run(limiter.gate(_request()))
        starts.append(clock.now)
    # ``a + interval``, as the limiter reckons the slot: subtracting two
    # clock readings can come out an ulp short of the interval between them.
    assert all(b >= a + interval for a, b in zip(starts, starts[1:])), starts


@given(
    interval=_seconds,
    gaps=st.lists(_waits, min_size=1, max_size=25),
)
async def test_capped_pace_adds_nothing_and_banks_nothing(
    interval: float, gaps: list[float]
) -> None:
    clock = VirtualClock()
    limiter = CappedPaceLimiter(interval, clock=clock, sleep=clock.sleep)
    previous = -math.inf
    for gap in gaps:
        clock.advance(gap)
        arrival = clock.now
        await clock.run(limiter.gate(_request()))
        expected = max(arrival, previous + interval)
        # The wait is computed as a difference of clock readings, so the
        # start can land an ulp or two past the slot, never before it.
        assert expected <= clock.now <= expected + 1e-6
        previous = clock.now


# --- Reactive pace -------------------------------------------------------------


@given(
    rest=st.floats(
        min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False
    ),
    response_times=st.lists(
        st.floats(min_value=0.001, max_value=5.0), min_size=1, max_size=12
    ),
)
async def test_reactive_lane_is_serial_and_rests_after_each_response(
    rest: float, response_times: list[float]
) -> None:
    clock = VirtualClock()
    limiter = ReactivePaceLimiter(rest, clock=clock, sleep=clock.sleep)
    spans: list[tuple[float, float]] = []

    async def fetch(takes: float) -> None:
        turn = await limiter.gate(_request())
        started = clock.now
        await clock.sleep(takes)
        spans.append((started, clock.now))
        turn.finish()

    await clock.run(*(fetch(takes) for takes in response_times))
    spans.sort()
    assert len(spans) == len(response_times)
    for (_, finished), (started, _) in zip(spans, spans[1:]):
        assert started >= finished + rest, spans


# --- The probe ---------------------------------------------------------------

_ladders = st.one_of(
    st.just(DEFAULT_ADAPTIVE_LADDER),
    st.lists(_seconds, min_size=1, max_size=8, unique=True).map(
        lambda xs: tuple(sorted(xs))
    ),
)
_statuses = st.sampled_from([200, 404, 429, 503])
_retry_afters = st.one_of(
    st.none(),
    st.sampled_from([0.0, -1.0, math.nan, math.inf]),
    st.floats(
        min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False
    ),
    st.just(300.0),
)
_steps = st.lists(
    st.tuples(
        _statuses,
        _retry_afters,
        st.none() | _waits,  # how long ago the request was sent
        _waits,  # clock advance before the report
        st.booleans(),  # start a request afterwards
    ),
    max_size=40,
)


def _usable(retry_after: float | None) -> bool:
    return (
        retry_after is not None
        and math.isfinite(retry_after)
        and retry_after > 0
    )


@given(
    ladder=_ladders,
    declared=st.none() | _seconds,
    recover_after=st.none() | _seconds | st.just(300.0),
    steps=_steps,
)
async def test_probe_ladder_laws(
    ladder: tuple[float, ...],
    declared: float | None,
    recover_after: float | None,
    steps: list[tuple[int, float | None, float | None, float, bool]],
) -> None:
    clock = VirtualClock()
    limiter = AdaptiveRateLimiter(
        declared,
        ladder=ladder,
        recover_after=recover_after,
        clock=clock,
        sleep=clock.sleep,
    )
    rungs = limiter._ladder
    if declared is None:
        assert rungs == ladder
    else:
        assert rungs[0] == declared
        assert all(rung > declared for rung in rungs[1:])
    assert limiter.current_interval == rungs[0]

    # The oracle: the docstring's rules, restated.
    rung = 0
    pause_until = -math.inf
    stepped_down_at = -math.inf
    steady_since = 0.0

    for status, retry_after, sent_ago, advance, gate in steps:
        clock.advance(advance)
        now = clock.now
        sent_at = None if sent_ago is None else now - sent_ago
        before = limiter._rung
        limiter.record_response(
            status, retry_after=retry_after, sent_at=sent_at
        )

        pushback = _usable(retry_after) or status == _TOO_MANY_REQUESTS
        if _usable(retry_after):
            assert retry_after is not None
            pause_until = max(pause_until, now + retry_after)
        stale = sent_at is not None and sent_at < stepped_down_at
        if pushback and not stale:
            steady_since = now
            if rung + 1 < len(rungs):
                rung += 1
                stepped_down_at = now

        assert limiter._rung == rung, "limiter and oracle disagree on the rung"
        assert limiter._rung in (before, before + 1)
        if not pushback or stale:
            assert limiter._rung == before
        assert limiter._pause_until == pause_until
        assert limiter.current_interval in rungs
        assert limiter.current_interval >= rungs[0]

        if gate:
            before = limiter._rung
            await clock.run(limiter.gate(_request()))
            # Recovery is judged as the request starts, and a pause is
            # not quiet time.
            started = clock.now
            if (
                rung > 0
                and recover_after is not None
                and started - max(steady_since, pause_until) >= recover_after
            ):
                rung -= 1
                steady_since = started
            assert limiter._rung == rung, "limiter and oracle disagree"
            assert limiter._rung in (before, before - 1)
            assert clock.now >= pause_until

    report = limiter.report()
    assert report.declared_interval == rungs[0]
    assert report.settled_interval == rungs[rung]
    assert report.slowest_interval >= report.settled_interval
    assert report.step_downs - report.recoveries == rung


# --- Regression: NaN Retry-After must not wedge the lane ---------------------


async def test_nan_retry_after_does_not_block_the_lane_forever() -> None:
    """``record_response(retry_after=nan)`` must not make the lane unusable.

    A ``nan`` pause would make every wait comparison False, so it would
    never clear; ``record_response`` drops non-finite values instead.
    """
    limiter = AdaptiveRateLimiter()
    limiter.record_response(429, retry_after=float("nan"))
    await asyncio.wait_for(limiter.gate(_request()), timeout=1.0)
