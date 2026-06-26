"""Run-scoped request rate limiting.

A single in-memory limiter is shared by every worker in a run, so the
configured rate is a *global* ceiling, not a per-worker one. Rate state is
not durable — it lives in memory and resets on restart, which is fine: the
limit is a courtesy to the remote server, not a correctness invariant. The
rate itself is static (attached to the scraper); nothing adapts it at runtime.

The canonical implementation is a thin wrapper over ``pyrate_limiter``,
configured with its ``Rate`` objects so any duration it supports (per-second,
-minute, -hour, plus burst windows) is expressible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pyrate_limiter import Limiter

from jkent.contracts import DBC_PROTOCOL_KW, ensure, require

if TYPE_CHECKING:
    from pyrate_limiter import Rate

    from jkent.data_types import Request


@runtime_checkable
# pyre-ignore[31, 39]: pyre parses the ** class keywords as a base
# class. At runtime it's empty (production) or injects the contract
# metaclass (dev); type-wise this is a plain Protocol either way.
class RateLimiter(Protocol, **DBC_PROTOCOL_KW):
    """Gates request frequency for a run; shared across all of its workers.

    Contracts on the method stubs are inherited (dev-time only, see
    ``jkent.contracts``) by implementations that explicitly subclass
    this Protocol — both in-tree implementations do. Purely structural
    conformers still satisfy the type and runtime isinstance checks,
    but get no contract enforcement.
    """

    async def gate(self, request: Request) -> None:
        """Block until ``request`` may proceed.

        Per-request bypasses (e.g. ``request.bypass_rate_limit``) are
        evaluated here, so callers pass the request and need no bypass logic
        of their own. A no-op when the request bypasses or the run has no
        limit (replay). The worker awaits this *before* it starts timing the
        request, so throttle wait stays out of the duration the monitor sizes
        from. One instance serves all workers; implementations coordinate
        concurrent callers internally so the rate stays global to the run.
        """
        ...

    @property
    @ensure(
        lambda result: result is None or result > 0,
        "the ceiling is a positive rate, or None for unlimited — 0 would "
        "zero out the monitor's scaling math",
    )
    def max_rate_per_second(self) -> float | None:
        """Configured ceiling, for the monitor's scaling math; None = unlimited."""
        ...


class PyrateRateLimiter(RateLimiter):
    """Thin in-memory wrapper over ``pyrate_limiter`` — the default limiter.

    Holds one shared ``Limiter`` over an in-memory bucket for the whole run;
    ``gate`` delegates to it. Configured with ``pyrate_limiter.Rate`` objects,
    so arbitrary durations and multi-window limits are expressible.
    ``max_rate_per_second`` is derived from the configured rates (the most
    restrictive, normalized to requests/second) for the monitor.

    """

    @require(
        lambda rates: all(r.limit > 0 and r.interval > 0 for r in rates),
        "every configured rate has a positive limit and interval",
    )
    def __init__(self, rates: list[Rate]) -> None:
        self._rates = list(rates)
        self._limiter = Limiter(self._rates) if self._rates else None

    async def gate(self, request: Request) -> None:
        """Block until ``request`` may proceed; skip on bypass / no rates."""
        if request.bypass_rate_limit or self._limiter is None:
            return
        await self._limiter.try_acquire_async("request")

    @property
    def max_rate_per_second(self) -> float | None:
        """Most restrictive configured rate as requests/second; None = none."""
        if not self._rates:
            return None
        return min(rate.limit / (rate.interval / 1000) for rate in self._rates)


class NoopRateLimiter(RateLimiter):
    """A :class:`RateLimiter` that never throttles — used for replay runs."""

    async def gate(self, request: Request) -> None:
        return None

    @property
    def max_rate_per_second(self) -> float | None:
        return None
