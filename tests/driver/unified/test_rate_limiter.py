"""Contract tests for ``RateLimiter`` (jkent.driver.unified_driver.rate_limiter).

Three implementations satisfy the interface: ``PyrateRateLimiter`` (a thin
wrapper over ``pyrate_limiter``, configured with ``Rate`` objects),
``NoopRateLimiter`` (replay — never throttles), and ``AdaptiveRateLimiter``
(an even-paced descent ladder driven by 429/Retry-After feedback).

Contract under test:

- ABC conformance: every implementation is a ``RateLimiter`` instance.
- ``gate`` sends a request to the underlying limiter exactly once; there is
  no per-request opt-out inside a limiter — lane selection happens before,
  in ``RateLimiters.for_request``.
- ``record_response`` is safe to call on every implementation (the default is
  a no-op); only ``AdaptiveRateLimiter`` reacts.
- ``NoopRateLimiter`` never throttles.
- ``RateLimiters`` builds one limiter per lane of a ``RateLimitTable``: the
  ``none`` lane and replay never throttle, an adaptive default lane opens at
  the declared rate, and a request is routed by its ``rate_limit`` name.

Throttling is asserted via limiter *consultation* (a spy) or ladder position,
never wall-clock time — except the Retry-After pause tests, which use waits of
a few tens of ms.
"""

import asyncio
import logging
from typing import Any

import icontract
import pytest
from pyrate_limiter import Duration, Limiter, Rate

from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    RateLimitTable,
    Request,
)
from jkent.driver.unified_driver import (
    AdaptiveRateLimiter,
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
    RateLimiters,
)
from jkent.driver.unified_driver.rate_limiter import (
    BUMP_COOLDOWN_FLOOR_S,
    DEFAULT_ADAPTIVE_LADDER,
    adaptive_ladder,
)


def _request(*, rate_limit: str | None = None) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com"
        ),
        step="parse",
        rate_limit=rate_limit,
    )


# --- Conformance across both implementations -----------------------------


@pytest.fixture(params=["noop", "pyrate", "adaptive"])
def limiter(request: pytest.FixtureRequest) -> RateLimiter:
    if request.param == "noop":
        return NoopRateLimiter()
    if request.param == "adaptive":
        return AdaptiveRateLimiter()
    return PyrateRateLimiter([Rate(5, Duration.SECOND)])


async def test_single_request_passes(limiter: RateLimiter) -> None:
    # One request is under any sane limit, so this returns without delay.
    await limiter.gate(_request())


async def test_record_response_is_always_safe(limiter: RateLimiter) -> None:
    # The feedback channel exists on every implementation — a no-op default
    # for the static ones — so the worker calls it unconditionally.
    limiter.record_response(429, retry_after=None, url="https://example.com")
    limiter.record_response(503, retry_after=0.0)
    await limiter.gate(_request())


# --- PyrateRateLimiter: consultation + derivation ------------------------


async def test_gate_consults_limiter_for_normal_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    async def spy(
        self: Limiter,
        name: str = "pyrate",
        weight: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        calls.append((name, weight))
        return True

    monkeypatch.setattr("pyrate_limiter.Limiter.try_acquire_async", spy)
    rl = PyrateRateLimiter([Rate(5, Duration.SECOND)])

    await rl.gate(_request())
    assert len(calls) == 1

    # A limiter has no per-request opt-out: the "none" lane is routed to a
    # NoopRateLimiter by RateLimiters, never to this one.
    await rl.gate(_request(rate_limit=NO_RATE_LIMIT))
    assert len(calls) == 2


def test_no_rates_is_rejected() -> None:
    # "No limit" is NoopRateLimiter; an empty list is a config slip, not a
    # spelling of it (RateLimiters.for_table never builds one).
    with pytest.raises(ValueError, match="at least one Rate"):
        PyrateRateLimiter([])


# --- NoopRateLimiter -----------------------------------------------------


async def test_noop_never_throttles() -> None:
    rl = NoopRateLimiter()
    await rl.gate(_request())
    await rl.gate(_request(rate_limit=NO_RATE_LIMIT))


# --- AdaptiveRateLimiter -------------------------------------------------


class FakeClock:
    """A hand-advanced monotonic clock for cooldown tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_adaptive_starts_at_fastest_rung() -> None:
    assert AdaptiveRateLimiter().current_interval == DEFAULT_ADAPTIVE_LADDER[0]


@pytest.mark.parametrize(
    "ladder",
    [
        # An equal rung makes a step-down that announces a slowdown and
        # changes nothing.
        (1.0, 1.0, 2.0),
        # Below 1ms the bucket's interval rounds to 0 ms, which Rate()
        # refuses — at the first step-down onto that rung, not here.
        (0.0004, 1.0),
    ],
)
def test_adaptive_ladder_contract_rejects(ladder: tuple[float, ...]) -> None:
    with pytest.raises(icontract.ViolationError):
        AdaptiveRateLimiter(ladder=ladder)


def test_adaptive_start_interval_picks_first_rung_at_least_that_slow() -> None:
    # A scraper declaring ~1 req / 0.4s starts at the 0.5s rung, not 0.25s.
    assert AdaptiveRateLimiter(start_interval=0.4).current_interval == 0.5
    # An exact rung match starts there.
    assert AdaptiveRateLimiter(start_interval=1.0).current_interval == 1.0
    # Slower than the whole ladder clamps to the slowest rung.
    assert AdaptiveRateLimiter(start_interval=999).current_interval == 60.0


def test_429_steps_down_one_rung_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rl = AdaptiveRateLimiter(clock=FakeClock())
    with caplog.at_level(logging.WARNING):
        rl.record_response(429, url="https://example.com/list")
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[1]
    assert any("stepped down" in r.message for r in caplog.records)


def test_cooldown_absorbs_the_worker_pool_burst() -> None:
    # One throttling incident reaches N workers as near-simultaneous 429s;
    # only the first may step the ladder.
    clock = FakeClock()
    rl = AdaptiveRateLimiter(clock=clock)
    for _ in range(5):
        rl.record_response(429)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[1]

    # Past the cooldown, the next 429 steps again.
    clock.advance(BUMP_COOLDOWN_FLOOR_S + DEFAULT_ADAPTIVE_LADDER[1])
    rl.record_response(429)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[2]


def test_non_429_without_retry_after_is_not_a_bump() -> None:
    rl = AdaptiveRateLimiter(clock=FakeClock())
    rl.record_response(503)
    rl.record_response(403)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[0]


def test_retry_after_steps_down_whatever_the_status() -> None:
    clock = FakeClock()
    rl = AdaptiveRateLimiter(clock=clock)
    rl.record_response(503, retry_after=30.0)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[1]


@pytest.mark.parametrize("retry_after", [0.0, -5.0])
def test_a_retry_after_already_past_is_no_pushback(retry_after: float) -> None:
    """``Retry-After: 0`` (or a date already past) asks for no wait at all."""
    rl = AdaptiveRateLimiter(clock=FakeClock())
    rl.record_response(503, retry_after=retry_after)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[0]
    # A 429 is still pushback, whatever its Retry-After says.
    rl.record_response(429, retry_after=retry_after)
    assert rl.current_interval == DEFAULT_ADAPTIVE_LADDER[1]


def test_bottom_rung_is_the_floor() -> None:
    clock = FakeClock()
    rl = AdaptiveRateLimiter(ladder=[1.0, 2.0], clock=clock)
    for _ in range(4):
        rl.record_response(429)
        clock.advance(BUMP_COOLDOWN_FLOOR_S + 2.0)
    assert rl.current_interval == 2.0  # descended once, then held


async def test_retry_after_pauses_the_gate_globally() -> None:
    # Real (small) wall-clock wait: the pause is time-based by nature.
    rl = AdaptiveRateLimiter()
    rl.record_response(429, retry_after=0.05)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await rl.gate(_request())
    assert loop.time() - start >= 0.04


async def test_expired_pause_does_not_delay() -> None:
    clock = FakeClock()
    rl = AdaptiveRateLimiter(clock=clock)
    rl.record_response(429, retry_after=10.0)
    clock.advance(11.0)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await rl.gate(_request())
    assert loop.time() - start < 1.0


def test_pause_only_ever_extends() -> None:
    clock = FakeClock()
    rl = AdaptiveRateLimiter(clock=clock)
    rl.record_response(429, retry_after=100.0)
    rl.record_response(429, retry_after=1.0)  # shorter: must not cut it
    assert rl._resume_at == 100.0


def test_pause_warns(caplog: pytest.LogCaptureFixture) -> None:
    rl = AdaptiveRateLimiter(clock=FakeClock())
    with caplog.at_level(logging.WARNING):
        rl.record_response(503, retry_after=15.0)
    assert any("pausing all requests" in r.message for r in caplog.records)


class _VirtualSleep:
    """``sleep`` for a :class:`FakeClock`: parks until the clock passes."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    async def __call__(self, delay: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._sleepers.append((self._clock.now + delay, future))
        await future

    def advance(self, seconds: float) -> None:
        self._clock.advance(seconds)
        for target, future in self._sleepers:
            if target <= self._clock.now and not future.done():
                future.set_result(None)


async def _settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


async def test_a_bare_429_holds_the_next_request_one_new_interval() -> None:
    clock = FakeClock()
    sleep = _VirtualSleep(clock)
    rl = AdaptiveRateLimiter(ladder=(1.0, 2.0), clock=clock, sleep=sleep)
    rl.record_response(429)
    gate = asyncio.create_task(rl.gate(_request()))
    await _settle()
    sleep.advance(1.9)
    await _settle()
    assert not gate.done(), "the first request after a step-down went out"
    sleep.advance(0.2)
    await asyncio.wait_for(gate, timeout=1.0)


class _HeldBucket:
    """A bucket whose one slot is handed out when the test says so."""

    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def try_acquire_async(self, name: str) -> bool:
        await self.release.wait()
        return True


async def test_a_waiter_parked_on_the_old_bucket_waits_at_the_new_pace() -> (
    None
):
    clock = FakeClock()
    sleep = _VirtualSleep(clock)
    rl = AdaptiveRateLimiter(ladder=(1.0, 2.0), clock=clock, sleep=sleep)
    old = _HeldBucket()
    rl._limiter = old  # type: ignore[assignment]
    gate = asyncio.create_task(rl.gate(_request()))
    await _settle()
    rl.record_response(429)
    old.release.set()
    await _settle()
    assert not gate.done(), "a slot on the replaced bucket let it through"
    sleep.advance(2.1)
    await asyncio.wait_for(gate, timeout=1.0)


# --- RateLimiters: one limiter per lane ------------------------------------


class _LanedScraper(BaseScraper[dict[str, Any]]):
    rate_limits = [Rate(2, Duration.SECOND)]
    named_rate_limits = {
        "downloads": [Rate(1, Duration.SECOND * 5)],
        "search": [Rate(1, Duration.SECOND)],
    }


class _UnlimitedScraper(BaseScraper[dict[str, Any]]):
    """No ``rate_limits``: the default lane must not throttle."""


def test_for_table_builds_one_limiter_per_lane() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper)
    )
    assert list(limiters) == ["default", "none", "downloads", "search"]
    assert isinstance(limiters[DEFAULT_RATE_LIMIT], PyrateRateLimiter)
    assert isinstance(limiters[NO_RATE_LIMIT], NoopRateLimiter)
    assert isinstance(limiters["downloads"], PyrateRateLimiter)
    # Distinct instances: a lane's pacing is its own.
    assert limiters["downloads"] is not limiters["search"]


def test_for_table_unlimited_default_when_scraper_declares_no_rates() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_UnlimitedScraper)
    )
    assert isinstance(limiters[DEFAULT_RATE_LIMIT], NoopRateLimiter)


def test_for_table_replay_makes_every_lane_noop() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper), rate_limited=False
    )
    assert all(isinstance(rl, NoopRateLimiter) for rl in limiters.values())


def test_for_table_adaptive_replaces_only_the_default_lane() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper), adaptive=True
    )
    assert isinstance(limiters[DEFAULT_RATE_LIMIT], AdaptiveRateLimiter)
    assert isinstance(limiters["downloads"], PyrateRateLimiter)


def test_for_table_adaptive_is_off_in_replay() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_LanedScraper),
        adaptive=True,
        rate_limited=False,
    )
    assert all(isinstance(rl, NoopRateLimiter) for rl in limiters.values())


class _CourteousScraper(BaseScraper[dict[str, Any]]):
    rate_limits = [Rate(1, Duration.SECOND * 5)]


def test_adaptive_default_lane_never_paces_faster_than_declared() -> None:
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_CourteousScraper), adaptive=True
    )
    adaptive = limiters[DEFAULT_RATE_LIMIT]
    assert isinstance(adaptive, AdaptiveRateLimiter)
    assert adaptive.current_interval == 5.0
    assert min(adaptive._ladder) == 5.0


def test_adaptive_default_lane_without_declared_rates_uses_the_ladder() -> (
    None
):
    limiters = RateLimiters.for_table(
        RateLimitTable.for_scraper(_UnlimitedScraper), adaptive=True
    )
    adaptive = limiters[DEFAULT_RATE_LIMIT]
    assert isinstance(adaptive, AdaptiveRateLimiter)
    assert adaptive._ladder == DEFAULT_ADAPTIVE_LADDER


@pytest.mark.parametrize(
    ("rates", "expected"),
    [
        ([Rate(1, Duration.SECOND * 5)], (5.0, 6.0, 10.0, 15.0, 30.0, 60.0)),
        # Several windows: the slowest implied pace wins.
        (
            [Rate(2, Duration.SECOND), Rate(10, Duration.MINUTE)],
            (6.0, 10.0, 15.0, 30.0, 60.0),
        ),
        # Rounded up to whole ms, never down to a faster pace.
        ([Rate(3, Duration.SECOND)], (0.334, *DEFAULT_ADAPTIVE_LADDER[2:])),
        # Slower than every rung: the declared pace is the only rung.
        ([Rate(1, Duration.MINUTE * 2)], (120.0,)),
        ([], DEFAULT_ADAPTIVE_LADDER),
    ],
)
def test_adaptive_ladder_opens_at_the_declared_pace(
    rates: list[Rate], expected: tuple[float, ...]
) -> None:
    assert adaptive_ladder(rates) == expected


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
