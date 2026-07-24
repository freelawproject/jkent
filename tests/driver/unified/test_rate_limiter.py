"""Contract tests for ``RateLimiter`` (jkent.driver.unified_driver.rate_limiter).

Two implementations satisfy the protocol: ``PyrateRateLimiter`` (a thin wrapper
over ``pyrate_limiter``, configured with ``Rate`` objects) and
``NoopRateLimiter`` (replay — never throttles).

Contract under test (see ``rate_limiter_contract.md``):

- Protocol conformance: both implementations are ``RateLimiter`` instances.
- ``gate`` evaluates ``request.bypass_rate_limit`` itself: a bypassing request
  is never sent to the underlying limiter.
- ``gate`` sends a normal request to the underlying limiter exactly once.
- ``max_rate_per_second`` is the most restrictive configured rate normalized to
  requests/second (``min(limit / interval_seconds)``), or ``None`` when there
  is no limit.
- ``NoopRateLimiter`` never throttles and reports ``max_rate_per_second`` of
  ``None``.

Following the repo convention (see tests/rate_limiting), throttling is asserted
via limiter *consultation* (a spy), never wall-clock time.
"""

import pytest
from pyrate_limiter import Duration, Rate

from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.unified_driver import (
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiter,
)


def _request(*, bypass: bool = False) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com"
        ),
        continuation="parse",
        bypass_rate_limit=bypass,
    )


# --- Conformance across both implementations -----------------------------


@pytest.fixture(params=["noop", "pyrate"])
def limiter(request: pytest.FixtureRequest) -> RateLimiter:
    if request.param == "noop":
        return NoopRateLimiter()
    return PyrateRateLimiter([Rate(5, Duration.SECOND)])


def test_is_a_rate_limiter(limiter: RateLimiter) -> None:
    assert isinstance(limiter, RateLimiter)


def test_max_rate_is_none_or_positive(limiter: RateLimiter) -> None:
    value = limiter.max_rate_per_second
    assert value is None or value > 0


async def test_bypassing_request_passes(limiter: RateLimiter) -> None:
    await limiter.gate(_request(bypass=True))  # returns; never throttles


async def test_single_request_passes(limiter: RateLimiter) -> None:
    # One request is under any sane limit, so this returns without delay.
    await limiter.gate(_request(bypass=False))


# --- PyrateRateLimiter: consultation + derivation ------------------------


async def test_gate_consults_limiter_for_normal_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    async def spy(self, name="pyrate", weight=1, *args, **kwargs) -> bool:
        calls.append((name, weight))
        return True

    monkeypatch.setattr("pyrate_limiter.Limiter.try_acquire_async", spy)
    rl = PyrateRateLimiter([Rate(5, Duration.SECOND)])

    await rl.gate(_request(bypass=False))
    assert len(calls) == 1

    await rl.gate(_request(bypass=True))
    assert len(calls) == 1  # bypass did not consult the limiter


def test_max_rate_per_second_single_rate() -> None:
    # A per-minute cap, normalized to requests/second: 30/60s -> 0.5/s. The
    # expected value is hand-computed (not the implementation's formula) so
    # this actually pins the ms->requests/second normalization.
    rl = PyrateRateLimiter([Rate(30, Duration.MINUTE)])
    assert rl.max_rate_per_second == pytest.approx(0.5)


async def test_no_rates_means_no_limit() -> None:
    # An empty rate list is "no limit": the ceiling is None and gate is a
    # no-op (never consults a limiter, since there isn't one).
    rl = PyrateRateLimiter([])
    assert rl.max_rate_per_second is None
    await rl.gate(_request(bypass=False))


def test_max_rate_per_second_is_most_restrictive() -> None:
    rl = PyrateRateLimiter(
        [Rate(10, Duration.SECOND), Rate(100, Duration.MINUTE)]
    )
    # 10/s vs 100/60s -> the per-minute cap is tighter
    assert rl.max_rate_per_second == pytest.approx(100 / 60)


# --- NoopRateLimiter -----------------------------------------------------


def test_noop_max_rate_is_none() -> None:
    assert NoopRateLimiter().max_rate_per_second is None


async def test_noop_never_throttles() -> None:
    rl = NoopRateLimiter()
    await rl.gate(_request(bypass=False))
    await rl.gate(_request(bypass=True))
