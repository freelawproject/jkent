"""Tests for the lane limiters and the run's ``RateLimiters`` registry.

Contract under test:

- Every limiter answers ``ready_in``, hands out a ``Turn`` from ``reserve``,
  and takes feedback through ``record_response``; ``gate`` waits for the
  lane, takes a turn and starts it. A turn starts once and finishes any
  number of times.
- ``CappedPaceLimiter`` starts one request per interval with nothing added
  to the wait; stamps the start from the clock after waking, so an overslept
  wake (a blocked event loop) pushes the next slot back instead of letting
  a request through early; banks no credit across idle time; counts a
  reserved turn against the next claim, and gives the slot back when the
  turn ends unsent.
- ``ReactivePaceLimiter`` holds the lane from a turn's claim to its
  response and rests after it, so a slow response stretches the pace; a
  turn that ends unsent frees the lane with no rest.
- ``AdaptiveRateLimiter`` opens at the declared pace; steps one rung slower
  per fresh 429 or Retry-After; ignores pushback on requests sent before
  its last step-down (Retry-After still pauses); climbs one rung per
  ``recover_after`` without counted pushback, never past the declared pace;
  logs reaching the slowest rung as an error; and reports what it saw.
- ``RateLimiters.for_table`` builds the limiter each lane's pace asks for,
  probes only a capped or undeclared default lane, and says which lanes are
  ready.

Time is virtual throughout: the limiters take an injected clock and sleep.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import icontract
import pytest

from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    AdaptivePace,
    BaseScraper,
    CappedPace,
    HttpMethod,
    HTTPRequestParams,
    RateLimitTable,
    ReactivePace,
    Request,
)
from jkent.driver.unified_driver import (
    AdaptiveRateLimiter,
    CappedPaceLimiter,
    NoopRateLimiter,
    RateLimiter,
    RateLimiters,
    ReactivePaceLimiter,
)
from jkent.driver.unified_driver.rate_limiter import (
    DEFAULT_ADAPTIVE_LADDER,
    ProbeReport,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine


def _request(*, rate_limit: str | None = None) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com"
        ),
        step="parse",
        rate_limit=rate_limit,
    )


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


class VirtualTime:
    """A monotonic clock and a matching sleep, in virtual seconds.

    Sleepers park until the clock reaches their target. :meth:`advance`
    moves the clock by hand; :meth:`run` drives coroutines to completion,
    jumping the clock to the next sleeper's target whenever every task is
    parked. ``lag`` is added to each jump — the overshoot of a wake held up
    by a blocked event loop.
    """

    def __init__(self, *, lag: float = 0.0) -> None:
        self.now = 0.0
        self.lag = lag
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
        self._wake()

    def _wake(self) -> None:
        due = [f for t, f in self._sleepers if t <= self.now]
        self._sleepers = [(t, f) for t, f in self._sleepers if t > self.now]
        for future in due:
            if not future.done():
                future.set_result(None)

    async def run(self, *coros: Coroutine[Any, Any, Any]) -> list[Any]:
        tasks = [asyncio.ensure_future(coro) for coro in coros]
        while True:
            await _settle()
            if all(task.done() for task in tasks):
                return [task.result() for task in tasks]
            targets = [t for t, f in self._sleepers if not f.done()]
            assert targets, "tasks are parked on something other than time"
            self.now = max(self.now, min(targets)) + self.lag
            self._wake()


async def _start_times(
    limiter: RateLimiter, vt: VirtualTime, n: int
) -> list[float]:
    """Gate ``n`` concurrent requests; their start times, in order."""
    times: list[float] = []

    async def one() -> None:
        await limiter.gate(_request())
        times.append(vt())

    await vt.run(*(one() for _ in range(n)))
    return times


def _gaps(times: list[float]) -> list[float]:
    return [b - a for a, b in zip(times, times[1:])]


# --- Conformance across the implementations --------------------------------


@pytest.fixture(params=["noop", "capped", "reactive", "adaptive"])
def limiter(request: pytest.FixtureRequest) -> RateLimiter:
    if request.param == "noop":
        return NoopRateLimiter()
    if request.param == "capped":
        return CappedPaceLimiter(0.2)
    if request.param == "reactive":
        return ReactivePaceLimiter(0.2)
    return AdaptiveRateLimiter()


async def test_a_fresh_lane_takes_a_request_at_once(
    limiter: RateLimiter,
) -> None:
    assert limiter.ready_in() == 0
    turn = await limiter.gate(_request())
    assert turn.started
    assert turn.limiter is limiter


async def test_record_response_is_always_safe(limiter: RateLimiter) -> None:
    # The feedback channel exists on every implementation — a no-op default
    # for the static ones — so the worker calls it unconditionally.
    limiter.record_response(429, retry_after=None, url="https://example.com")
    limiter.record_response(503, retry_after=0.0, sent_at=1.0)


async def test_a_turn_starts_once_and_finishes_any_number_of_times(
    limiter: RateLimiter,
) -> None:
    turn = limiter.reserve()
    await turn.start(_request())
    with pytest.raises(RuntimeError, match="starts once"):
        await turn.start(_request())
    turn.finish()
    turn.finish()


# --- CappedPaceLimiter -------------------------------------------------------


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_capped_gap_contract_rejects(seconds: float) -> None:
    with pytest.raises(icontract.ViolationError):
        CappedPaceLimiter(seconds)


async def test_starts_one_request_per_interval_with_nothing_added() -> None:
    vt = VirtualTime()
    rl = CappedPaceLimiter(0.05, clock=vt, sleep=vt.sleep)
    times = await _start_times(rl, vt, 10)
    assert times[0] == 0.0
    assert _gaps(times) == pytest.approx([0.05] * 9)


async def test_a_gap_well_under_a_second_is_kept_exactly() -> None:
    vt = VirtualTime()
    rl = CappedPaceLimiter(1 / 120, clock=vt, sleep=vt.sleep)
    times = await _start_times(rl, vt, 121)
    assert times[-1] == pytest.approx(1.0)


async def test_an_overslept_wake_pushes_the_next_slot_back() -> None:
    # Every wake lands 1s late, as behind a blocked event loop. The slot
    # after it is counted from the late wake, so no gap is ever short.
    vt = VirtualTime(lag=1.0)
    rl = CappedPaceLimiter(0.25, clock=vt, sleep=vt.sleep)
    times = await _start_times(rl, vt, 6)
    assert min(_gaps(times)) >= 0.25


async def test_idle_time_banks_no_burst() -> None:
    vt = VirtualTime()
    rl = CappedPaceLimiter(0.5, clock=vt, sleep=vt.sleep)
    await rl.gate(_request())
    vt.advance(100.0)
    times = await _start_times(rl, vt, 3)
    assert times[0] == 100.0
    assert _gaps(times) == pytest.approx([0.5, 0.5])


async def test_a_cancelled_waiter_takes_no_slot() -> None:
    vt = VirtualTime()
    rl = CappedPaceLimiter(1.0, clock=vt, sleep=vt.sleep)
    await rl.gate(_request())
    parked = asyncio.create_task(rl.gate(_request()))
    await _settle()
    parked.cancel()
    await _settle()
    assert await _start_times(rl, vt, 1) == [1.0]


async def test_a_reserved_turn_holds_the_next_claim_one_interval() -> None:
    # The worker claims a request only for a lane that is ready. A turn
    # taken but not yet started counts, or every worker would claim for
    # the one open slot and queue behind it holding a request.
    vt = VirtualTime()
    rl = CappedPaceLimiter(5.0, clock=vt, sleep=vt.sleep)
    first = rl.reserve()
    assert rl.ready_in() == 5.0
    vt.advance(2.0)
    await first.start(_request())
    assert first.started_at == 2.0
    assert rl.ready_in() == 5.0
    vt.advance(5.0)
    assert rl.ready_in() == 0


async def test_claims_keep_pace_while_starts_lag_behind() -> None:
    # A fast lane whose requests take a while to reach the gate (a handle
    # to lease) may have several turns reserved at once, one per interval.
    vt = VirtualTime()
    rl = CappedPaceLimiter(0.01, clock=vt, sleep=vt.sleep)
    turns = []
    for _ in range(5):
        assert rl.ready_in() == 0
        turns.append(rl.reserve())
        vt.advance(0.01)
    starts: list[float] = []
    for turn in turns:
        await vt.run(turn.start(_request()))
        starts.append(vt())
    assert [turn.started_at for turn in turns] == starts
    assert _gaps(starts) == pytest.approx([0.01] * 4)


def test_a_turn_handed_back_unsent_frees_its_slot() -> None:
    vt = VirtualTime()
    rl = CappedPaceLimiter(5.0, clock=vt, sleep=vt.sleep)
    turn = rl.reserve()
    assert rl.ready_in() == 5.0
    turn.finish()
    assert rl.ready_in() == 0


# --- ReactivePaceLimiter -----------------------------------------------------


@pytest.mark.parametrize("seconds", [-1.0, float("inf"), float("nan")])
def test_reactive_rest_contract_rejects(seconds: float) -> None:
    with pytest.raises(icontract.ViolationError):
        ReactivePaceLimiter(seconds)


async def test_a_reactive_lane_is_held_until_the_response_then_rests() -> None:
    vt = VirtualTime()
    rl = ReactivePaceLimiter(5.0, clock=vt, sleep=vt.sleep)
    turn = rl.reserve()
    assert rl.ready_in() is None
    await turn.start(_request())
    vt.advance(30.0)  # a slow download
    assert rl.ready_in() is None
    turn.finish()
    assert rl.ready_in() == 5.0
    vt.advance(5.0)
    assert rl.ready_in() == 0


def test_a_reactive_turn_handed_back_unsent_frees_the_lane_with_no_rest() -> (
    None
):
    vt = VirtualTime()
    rl = ReactivePaceLimiter(5.0, clock=vt, sleep=vt.sleep)
    rl.reserve().finish()
    assert rl.ready_in() == 0


def test_reserving_a_busy_reactive_lane_is_an_error() -> None:
    rl = ReactivePaceLimiter(5.0)
    rl.reserve()
    with pytest.raises(RuntimeError, match="in flight"):
        rl.reserve()


async def test_a_slow_response_stretches_a_reactive_pace() -> None:
    vt = VirtualTime()
    rl = ReactivePaceLimiter(1.0, clock=vt, sleep=vt.sleep)
    starts: list[float] = []

    async def fetch(takes: float) -> None:
        turn = await rl.gate(_request())
        starts.append(vt())
        await vt.sleep(takes)
        turn.finish()

    await vt.run(fetch(2.0), fetch(4.0), fetch(2.0))
    # One at a time, each a rest after the one before it finished. A gate
    # waiting on a response looks again every BUSY_POLL_INTERVAL_S.
    poll = RateLimiter.BUSY_POLL_INTERVAL_S
    assert starts[0] == 0.0
    assert 3.0 <= starts[1] <= 3.0 + poll
    assert starts[1] + 5.0 <= starts[2] <= starts[1] + 5.0 + poll


async def test_a_reactive_lane_with_no_rest_is_still_one_at_a_time() -> None:
    vt = VirtualTime()
    rl = ReactivePaceLimiter(0.0, clock=vt, sleep=vt.sleep)
    in_flight = 0
    most = 0

    async def fetch() -> None:
        nonlocal in_flight, most
        turn = await rl.gate(_request())
        in_flight += 1
        most = max(most, in_flight)
        await vt.sleep(1.0)
        in_flight -= 1
        turn.finish()

    await vt.run(*(fetch() for _ in range(4)))
    assert most == 1


# --- AdaptiveRateLimiter: ladder ---------------------------------------------


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (5.0, (5.0, 6.0, 10.0, 15.0, 30.0, 60.0)),
        (1 / 3, DEFAULT_ADAPTIVE_LADDER[1:]),
        # Faster than every rung: the declared pace opens the whole ladder.
        (1 / 120, (1 / 120, *DEFAULT_ADAPTIVE_LADDER)),
        # Slower than every rung: the declared pace is the only rung.
        (120.0, (120.0,)),
        (None, DEFAULT_ADAPTIVE_LADDER),
    ],
)
def test_ladder_opens_at_the_declared_pace(
    declared: float | None, expected: tuple[float, ...]
) -> None:
    rl = AdaptiveRateLimiter(declared)
    assert rl._ladder == pytest.approx(expected)
    assert rl.current_interval == pytest.approx(expected[0])


@pytest.mark.parametrize("ladder", [(), (1.0, 1.0), (2.0, 1.0), (0.0, 1.0)])
def test_ladder_contract_rejects(ladder: tuple[float, ...]) -> None:
    with pytest.raises(icontract.ViolationError):
        AdaptiveRateLimiter(ladder=ladder)


# --- AdaptiveRateLimiter: feedback -------------------------------------------


def _adaptive(
    vt: VirtualTime,
    *,
    ladder: tuple[float, ...] = (0.25, 0.5, 1.0),
    **kw: Any,
) -> AdaptiveRateLimiter:
    return AdaptiveRateLimiter(ladder=ladder, clock=vt, sleep=vt.sleep, **kw)


async def test_a_429_steps_down_and_spaces_the_next_request_at_the_new_pace() -> (
    None
):
    vt = VirtualTime()
    rl = _adaptive(vt)
    await rl.gate(_request())
    rl.record_response(429, sent_at=0.0)
    assert rl.current_interval == 0.5
    assert await _start_times(rl, vt, 1) == [0.5]


def test_a_burst_of_429s_from_one_incident_steps_down_once() -> None:
    # Eight requests in flight at 4/s all come back 429, long after the
    # step-down the first one caused. They were sent at the old pace, so
    # they say nothing about the new one.
    vt = VirtualTime()
    rl = _adaptive(vt)
    sent = [i * 0.25 for i in range(8)]
    vt.now = 30.0
    for t in sent:
        rl.record_response(429, sent_at=t)
        vt.now += 5.0
    assert rl.current_interval == 0.5


def test_pushback_on_a_request_sent_after_the_step_down_counts() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    vt.now = 10.0
    rl.record_response(429, sent_at=9.0)
    vt.now = 11.0
    rl.record_response(429, sent_at=10.5)
    assert rl.current_interval == 1.0


def test_pushback_without_sent_at_always_counts() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    rl.record_response(429)
    rl.record_response(429)
    assert rl.current_interval == 1.0


@pytest.mark.parametrize("status", [200, 404, 500, 503])
def test_other_statuses_without_retry_after_do_nothing(status: int) -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    rl.record_response(status, sent_at=0.0)
    assert rl.current_interval == 0.25
    assert rl.ready_in() == 0


@pytest.mark.parametrize(
    "retry_after", [0.0, -5.0, float("nan"), float("inf")]
)
def test_an_unusable_retry_after_is_no_pause(retry_after: float) -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    rl.record_response(503, retry_after=retry_after)
    assert rl.current_interval == 0.25
    assert rl.ready_in() == 0


async def test_retry_after_pauses_every_request_and_steps_down() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    rl.record_response(503, retry_after=30.0, sent_at=0.0)
    assert rl.current_interval == 0.5
    assert rl.ready_in() == 30.0
    assert await _start_times(rl, vt, 2) == pytest.approx([30.0, 30.5])


async def test_a_stale_retry_after_still_pauses_but_steps_nothing() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt)
    vt.now = 10.0
    rl.record_response(429, sent_at=9.0)
    rl.record_response(503, retry_after=20.0, sent_at=9.5)
    assert rl.current_interval == 0.5
    assert await _start_times(rl, vt, 1) == [30.0]


def test_a_shorter_retry_after_does_not_cut_a_pause_short() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(1.0,))
    rl.record_response(503, retry_after=60.0)
    rl.record_response(503, retry_after=5.0)
    assert rl.ready_in() == 60.0


def test_pause_warns(caplog: pytest.LogCaptureFixture) -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(1.0,))
    with caplog.at_level(logging.WARNING):
        rl.record_response(503, retry_after=60.0, url="https://example.com")
    assert "pausing all requests for 60.0s" in caplog.text
    assert "https://example.com" in caplog.text


async def test_a_parked_waiter_honors_a_step_down() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(1.0, 3.0))
    await rl.gate(_request())
    parked = asyncio.create_task(rl.gate(_request()))
    await _settle()
    rl.record_response(429, sent_at=0.0)
    vt.advance(1.0)
    await _settle()
    assert not parked.done(), "started at the pace already left"
    vt.advance(2.0)
    await asyncio.wait_for(parked, timeout=1.0)


async def test_a_parked_waiter_honors_a_pause() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(1.0,))
    await rl.gate(_request())
    parked = asyncio.create_task(rl.gate(_request()))
    await _settle()
    rl.record_response(503, retry_after=300.0)
    vt.advance(1.0)
    await _settle()
    assert not parked.done(), "started mid-pause"
    vt.advance(299.0)
    await asyncio.wait_for(parked, timeout=1.0)


async def test_a_started_turn_parked_at_the_gate_honors_a_pause() -> None:
    # The worker's path: the turn was reserved when the lane was ready,
    # and the pause opened before the request reached the gate.
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(1.0,))
    turn = rl.reserve()
    rl.record_response(503, retry_after=300.0)
    await vt.run(turn.start(_request()))
    assert turn.started_at == 300.0


def test_reaching_the_slowest_rung_is_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(0.25, 0.5))
    with caplog.at_level(logging.WARNING):
        rl.record_response(429)
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert "slowest rung" in caplog.records[0].message


# --- AdaptiveRateLimiter: recovery -------------------------------------------


async def test_recovers_one_rung_per_quiet_window_up_to_the_declared_pace() -> (
    None
):
    vt = VirtualTime()
    rl = _adaptive(vt, recover_after=300.0)
    rl.record_response(429)
    rl.record_response(429)
    assert rl.current_interval == 1.0
    vt.advance(299.0)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 1.0
    vt.advance(1.0)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.5
    vt.advance(300.0)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.25
    vt.advance(3000.0)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.25


async def test_pushback_at_the_slowest_rung_holds_off_recovery() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, ladder=(0.25, 0.5), recover_after=300.0)
    rl.record_response(429)
    vt.now = 200.0
    rl.record_response(429)  # no rung left, but still pushback
    vt.now = 400.0
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.5


async def test_stale_pushback_does_not_hold_off_recovery() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, recover_after=300.0)
    vt.now = 10.0
    rl.record_response(429, sent_at=9.0)
    vt.now = 200.0
    rl.record_response(429, sent_at=9.5)
    vt.now = 310.0
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.25


async def test_a_pause_is_not_quiet_time() -> None:
    # Five minutes of Retry-After is the server asking us to stay away,
    # not five minutes of it putting up with the new pace.
    vt = VirtualTime()
    rl = _adaptive(vt, recover_after=300.0)
    rl.record_response(503, retry_after=300.0)
    await vt.run(rl.gate(_request()))
    assert vt() == 300.0
    assert rl.current_interval == 0.5
    vt.advance(300.0)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.25


async def test_recover_after_none_never_recovers() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, recover_after=None)
    rl.record_response(429)
    vt.advance(1e6)
    await vt.run(rl.gate(_request()))
    assert rl.current_interval == 0.5


# --- AdaptiveRateLimiter: report ---------------------------------------------


async def test_report_says_where_the_lane_went_and_why() -> None:
    vt = VirtualTime()
    rl = _adaptive(vt, recover_after=300.0)
    assert rl.report() == ProbeReport(
        declared_interval=0.25,
        settled_interval=0.25,
        slowest_interval=0.25,
        step_downs=0,
        recoveries=0,
        pushback=0,
        stale_pushback=0,
    )
    vt.now = 10.0
    rl.record_response(429, sent_at=9.0)
    rl.record_response(429, sent_at=9.5)  # stale
    vt.now = 11.0
    rl.record_response(503, retry_after=1.0, sent_at=10.5)
    vt.now = 400.0
    await vt.run(rl.gate(_request()))
    report = rl.report()
    assert report == ProbeReport(
        declared_interval=0.25,
        settled_interval=0.5,
        slowest_interval=1.0,
        step_downs=2,
        recoveries=1,
        pushback=2,
        stale_pushback=1,
    )
    assert "settled at 1 per 0.5s" in report.summary()


# --- RateLimiters: one limiter per lane ------------------------------------


class _LanedScraper(BaseScraper[dict[str, Any]]):
    rate_limits = CappedPace(seconds=0.5)
    named_rate_limits = {
        "downloads": ReactivePace(seconds=5),
        "search": CappedPace(seconds=1),
    }


class _UnlimitedScraper(BaseScraper[dict[str, Any]]):
    """No ``rate_limits``: the default lane must not wait."""


class _ReactiveScraper(BaseScraper[dict[str, Any]]):
    rate_limits = ReactivePace(seconds=2)


def test_for_table_builds_the_limiter_each_pace_asks_for() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper)
    )
    assert list(limiters) == ["default", "none", "downloads", "search"]
    default = limiters[DEFAULT_RATE_LIMIT]
    assert isinstance(default, CappedPaceLimiter)
    assert default.current_interval == 0.5
    assert isinstance(limiters[NO_RATE_LIMIT], NoopRateLimiter)
    assert isinstance(limiters["downloads"], ReactivePaceLimiter)
    assert isinstance(limiters["search"], CappedPaceLimiter)


class _AdaptiveScraper(BaseScraper[dict[str, Any]]):
    rate_limits = AdaptivePace(seconds=0.5)
    named_rate_limits = {
        "search": AdaptivePace(seconds=2),
        "downloads": CappedPace(seconds=1),
    }


def test_only_the_lanes_declared_adaptive_are_probed() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_AdaptiveScraper)
    )
    default, search = limiters[DEFAULT_RATE_LIMIT], limiters["search"]
    assert isinstance(default, AdaptiveRateLimiter)
    assert isinstance(search, AdaptiveRateLimiter)
    # Each opens at its own declared gap and never goes faster.
    assert (default.current_interval, min(default._ladder)) == (0.5, 0.5)
    assert (search.current_interval, min(search._ladder)) == (2, 2)
    assert isinstance(limiters["downloads"], CappedPaceLimiter)
    assert list(limiters.probe_reports()) == [DEFAULT_RATE_LIMIT, "search"]


def test_for_table_unlimited_default_when_scraper_declares_no_pace() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_UnlimitedScraper)
    )
    assert isinstance(limiters[DEFAULT_RATE_LIMIT], NoopRateLimiter)


def test_for_table_replay_makes_every_lane_noop() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper), rate_limited=False
    )
    assert all(isinstance(rl, NoopRateLimiter) for rl in limiters.values())


def test_replay_probes_nothing() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_AdaptiveScraper), rate_limited=False
    )
    assert all(isinstance(rl, NoopRateLimiter) for rl in limiters.values())
    assert limiters.probe_reports() == {}


def test_ready_lanes_are_the_ones_that_can_send_now() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper)
    )
    assert limiters.ready_lanes() == ["default", "none", "downloads", "search"]
    assert limiters.next_ready_in() is None

    limiters["downloads"].reserve()  # in flight: cannot say when
    assert limiters.ready_lanes() == ["default", "none", "search"]
    assert limiters.next_ready_in() is None

    limiters["search"].reserve()
    limiters[DEFAULT_RATE_LIMIT].reserve()
    assert limiters.ready_lanes() == ["none"]
    # The soonest of the lanes that can say: default, half a second out.
    assert limiters.next_ready_in() == pytest.approx(0.5, abs=0.05)


def test_for_request_routes_by_lane_name() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper)
    )
    assert limiters.for_request(_request()) is limiters[DEFAULT_RATE_LIMIT]
    assert (
        limiters.for_request(_request(rate_limit=DEFAULT_RATE_LIMIT))
        is limiters[DEFAULT_RATE_LIMIT]
    )
    assert (
        limiters.for_request(_request(rate_limit=NO_RATE_LIMIT))
        is limiters[NO_RATE_LIMIT]
    )
    assert (
        limiters.for_request(_request(rate_limit="downloads"))
        is limiters["downloads"]
    )


def test_for_request_unknown_lane_raises() -> None:
    limiters = RateLimiters.unlimited()
    with pytest.raises(ValueError, match="downloads"):
        limiters.for_request(_request(rate_limit="downloads"))


def test_unlimited_has_both_framework_lanes() -> None:
    limiters = RateLimiters.unlimited()
    assert set(limiters) == {DEFAULT_RATE_LIMIT, NO_RATE_LIMIT}
    assert all(isinstance(rl, NoopRateLimiter) for rl in limiters.values())
