"""Run-scoped request rate limiting.

A run holds one in-memory limiter per rate-limit *lane* (see
:mod:`jkent.common.rate_limits`), each shared by every worker, so a
configured rate is a *global* ceiling for that lane, not a per-worker one.
Rate state is not durable — it lives in memory and resets on restart, which
is fine: the limit is a courtesy to the remote server, not a correctness
invariant.

:class:`RateLimiter` is the interface: :meth:`~RateLimiter.gate` blocks a
request until it may proceed, and :meth:`~RateLimiter.record_response` is
the feedback channel the worker calls with classified HTTP failures so an
implementation *may* adapt. The static implementations inherit the no-op
default and ignore feedback; :class:`AdaptiveRateLimiter` descends a ladder
of ever-slower rates on 429s and honors ``Retry-After`` as a global pause.
:class:`RateLimiters` is the run's registry of lane -> limiter; the worker
picks a request's limiter there once and sends both its gate and its
feedback to that one. Only the ``default`` lane can be adaptive
(:meth:`RateLimiters.for_table`), starting at the scraper's declared rate;
named lanes are static and ignore feedback.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from http import HTTPStatus
from typing import TYPE_CHECKING, Final

from pyrate_limiter import Limiter, Rate
from typing_extensions import override

from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    RateLimitTable,
)
from jkent.contracts import require

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from jkent.data_types import Request

logger = logging.getLogger(__name__)


class RateLimiter(ABC):
    """Gates request frequency for one lane; shared across all workers."""

    @abstractmethod
    async def gate(self, request: Request) -> None:
        """Block until ``request`` may proceed.

        Lane selection has already happened (:meth:`RateLimiters.for_request`)
        by the time a limiter sees a request, so implementations gate
        unconditionally; ``request`` is passed for what a limiter may want
        to key on or log, not for any per-request opt-out. The worker awaits
        this *before* it starts timing the request, so throttle wait stays
        out of the duration the monitor sizes from. One instance serves all
        workers; implementations coordinate concurrent callers internally so
        the rate stays global to the run.
        """
        ...

    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
    ) -> None:
        """Feedback channel: the worker reports classified HTTP failures here.

        Called once per request whose resolve ended in a classified HTTP
        error (transient, persistent, or a failed speculation probe), with
        the observed status and the parsed, clamped ``Retry-After`` where
        the server sent one. The worker reports to the limiter that gated
        the request, so a lane hears only its own pushback. Successful
        responses are not reported. Defaults to a no-op: static limiters
        don't adapt.
        """
        return None


class PyrateRateLimiter(RateLimiter):
    """Thin in-memory wrapper over ``pyrate_limiter`` — the default limiter.

    Holds one shared ``Limiter`` over an in-memory bucket for the whole run;
    ``gate`` delegates to it. Configured with ``pyrate_limiter.Rate`` objects,
    so arbitrary durations and multi-window limits are expressible. A lane
    with no limit is a :class:`NoopRateLimiter`, not an empty rate list.
    """

    @require(
        lambda rates: all(  # pyrefly: ignore[implicit-any-lambda]
            r.limit > 0 and r.interval > 0 for r in rates
        ),
        "every configured rate has a positive limit and interval",
    )
    def __init__(self, rates: list[Rate]) -> None:
        """Build the run-wide limiter over ``rates``.

        Raises:
            ValueError: for an empty ``rates``.
        """
        self._rates = list(rates)
        if not self._rates:
            raise ValueError(
                "PyrateRateLimiter needs at least one Rate; use "
                "NoopRateLimiter for no limit"
            )
        self._limiter = Limiter(self._rates)

    async def gate(self, request: Request) -> None:
        """Block until ``request`` may proceed."""
        await self._limiter.try_acquire_async("request")


class NoopRateLimiter(RateLimiter):
    """A :class:`RateLimiter` that never throttles.

    The ``none`` lane, every lane of a replay run, and the default lane of a
    scraper that declares no ``rate_limits``.
    """

    async def gate(self, request: Request) -> None:
        return None


#: Default descent ladder: seconds between requests, fastest rung first —
#: 4/s down to one per minute. Descent is monotonic for the life of a run;
#: a new run starts fresh at the top (or at the scraper's declared rate).
DEFAULT_ADAPTIVE_LADDER: Final[tuple[float, ...]] = (
    0.25,
    1 / 3,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    10.0,
    15.0,
    30.0,
    60.0,
)

#: The fastest permissible rung. A rung's bucket is one request per
#: ``round(interval * 1000)`` ms, and ``Rate`` refuses a 0 ms interval.
MIN_ADAPTIVE_INTERVAL_S: Final[float] = 0.001

#: Minimum time after a step-down during which further bump signals are
#: ignored. One throttling incident fans out to every in-flight worker as
#: near-simultaneous 429s; without a cooldown a single incident would
#: cascade several rungs instead of one.
BUMP_COOLDOWN_FLOOR_S: Final[float] = 2.0


class AdaptiveRateLimiter(RateLimiter):
    """Even-paced limiter that steps down a ladder on server pushback.

    Each rung is one request per ``interval`` seconds — deliberately even
    pacing, never an N-per-window burst — enforced by a fresh
    ``pyrate_limiter`` bucket per rung. Feedback via
    :meth:`record_response`:

    - HTTP 429 steps one rung slower. A step-down also pauses every
      worker for one new interval, so the first request after it is
      spaced at the new pace.
    - ``Retry-After`` (whatever the status) pauses *all* workers until the
      given time, and also steps one rung slower — the header is a
      statement about the client's overall rate, not about one request.
      Values arrive parsed and clamped by the transport
      (:func:`~jkent.driver.unified_driver.transport.parse_retry_after`),
      so a hostile header slows the run, never stalls it.

    Descent is monotonic for the life of the run and each step-down is
    announced with a warning. A cooldown of ``max(new interval,
    BUMP_COOLDOWN_FLOOR_S)`` after each step absorbs the burst of duplicate
    429s that one throttling incident produces across the worker pool.
    """

    @require(
        lambda ladder: (  # pyrefly: ignore[implicit-any-lambda]
            len(tuple(ladder)) > 0
            and all(interval >= MIN_ADAPTIVE_INTERVAL_S for interval in ladder)
            and all(a < b for a, b in zip(ladder, tuple(ladder)[1:]))
        ),
        "the ladder is a non-empty, strictly ascending sequence of intervals "
        "of at least MIN_ADAPTIVE_INTERVAL_S",
    )
    def __init__(
        self,
        ladder: Sequence[float] = DEFAULT_ADAPTIVE_LADDER,
        *,
        start_interval: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize at the fastest permissible rung.

        Args:
            ladder: Seconds between requests per rung, strictly ascending
                (slower rungs last), none under
                :data:`MIN_ADAPTIVE_INTERVAL_S`.
            start_interval: Start at the first rung at least this slow —
                the way to honor a scraper's declared courtesy rate instead
                of opening at the ladder's fastest. Slower than every rung
                means starting at the slowest. ``None`` starts at the top.
            clock: Monotonic time source; injectable for tests.
            sleep: How a Retry-After pause waits, in ``clock`` seconds;
                injectable alongside ``clock`` so a test's pause costs no
                wall time.
        """
        self._ladder = tuple(ladder)
        self._clock = clock
        self._sleep = sleep
        self._rung = 0
        if start_interval is not None:
            self._rung = next(
                (
                    i
                    for i, interval in enumerate(self._ladder)
                    if interval >= start_interval
                ),
                len(self._ladder) - 1,
            )
        self._limiter = self._bucket_for(self._ladder[self._rung])
        self._cooldown_until = float("-inf")
        self._resume_at: float | None = None

    @staticmethod
    def _bucket_for(interval: float) -> Limiter:
        # One request per interval: even pacing, no burst window. Same
        # construction as PyrateRateLimiter so the two share semantics.
        return Limiter([Rate(1, round(interval * 1000))])

    @property
    def current_interval(self) -> float:
        """Seconds between requests at the current rung."""
        return self._ladder[self._rung]

    async def gate(self, request: Request) -> None:
        """Block until ``request`` may proceed."""
        while True:
            await self._wait_out_pause()
            # Snapshot the bucket: a step-down swaps in a new one while we
            # wait on this one. A slot taken on a replaced (faster) bucket
            # does not count — go back through the pause the step-down set
            # and take one at the new pace.
            limiter = self._limiter
            await limiter.try_acquire_async("request")
            if limiter is self._limiter:
                return

    async def _wait_out_pause(self) -> None:
        # Looped because record_response may extend the pause while we
        # sleep.
        while True:
            resume_at = self._resume_at
            if resume_at is None:
                return
            wait = resume_at - self._clock()
            if wait <= 0:
                # Pause expired; clear it unless a newer one replaced it.
                if self._resume_at == resume_at:
                    self._resume_at = None
                return
            await self._sleep(wait)

    @override
    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
    ) -> None:
        """Step down on 429; pause globally (and step down) on Retry-After."""
        if retry_after is not None and not math.isfinite(retry_after):
            # The transport clamps what it parses, so this is belt and
            # braces: a NaN pause would never expire and every gate on the
            # lane would sleep on it (or raise) for the rest of the run.
            logger.warning(
                "Ignoring non-finite Retry-After %r on HTTP %d%s",
                retry_after,
                status_code,
                f" from {url}" if url else "",
            )
            retry_after = None
        if retry_after is not None and retry_after <= 0:
            # "Retry-After: 0", or a date already past (clock skew): the
            # server asks for no wait, which is not pushback.
            retry_after = None
        if retry_after is not None:
            resume_at = self._clock() + retry_after
            if self._extend_pause(resume_at):
                logger.warning(
                    "Rate limiter pausing all requests for %.1fs "
                    "(Retry-After on HTTP %d%s)",
                    retry_after,
                    status_code,
                    f" from {url}" if url else "",
                )
            self._step_down(
                trigger=f"Retry-After on HTTP {status_code}", url=url
            )
        elif status_code == HTTPStatus.TOO_MANY_REQUESTS:
            self._step_down(trigger="HTTP 429", url=url)

    def _extend_pause(self, resume_at: float) -> bool:
        """Pause every gate until ``resume_at``; True if that extended it.

        Pauses only ever extend — a shorter one must not cut an earlier,
        longer one short.
        """
        if self._resume_at is None or resume_at > self._resume_at:
            self._resume_at = resume_at
            return True
        return False

    def _step_down(self, *, trigger: str, url: str | None) -> None:
        now = self._clock()
        if now < self._cooldown_until:
            return
        if self._rung + 1 >= len(self._ladder):
            # Already at the slowest rung. Debug, not warning: a throttling
            # site can repeat this on every request, and the descent that
            # got us here was already announced.
            logger.debug(
                "Rate limiter already at slowest rung (1 per %.3gs); "
                "ignoring %s",
                self._ladder[self._rung],
                trigger,
            )
            return
        old = self._ladder[self._rung]
        self._rung += 1
        new = self._ladder[self._rung]
        self._limiter = self._bucket_for(new)
        # The fresh bucket is empty, so its first taker would go out at
        # once, unspaced from the last send: hold every arrival one new
        # interval first.
        self._extend_pause(now + new)
        self._cooldown_until = now + max(new, BUMP_COOLDOWN_FLOOR_S)
        logger.warning(
            "Rate limit stepped down: 1 request per %.3gs -> 1 per %.3gs "
            "(trigger: %s%s)",
            old,
            new,
            trigger,
            f" on {url}" if url else "",
        )


def adaptive_ladder(
    rates: Sequence[Rate], base: Sequence[float] = DEFAULT_ADAPTIVE_LADDER
) -> tuple[float, ...]:
    """``base`` cut to start at the even pace ``rates`` declare.

    A declaration of several ``Rate`` s paces at the slowest one they imply
    (``max(interval / limit)``), rounded up to whole milliseconds; the
    ladder opens there and keeps only the ``base`` rungs slower than it.
    No ``rates`` returns ``base`` unchanged.
    """
    if not rates:
        return tuple(base)
    declared_ms = max(math.ceil(r.interval / r.limit) for r in rates)
    declared = max(declared_ms / 1000, MIN_ADAPTIVE_INTERVAL_S)
    return (declared, *(i for i in base if i > declared))


class RateLimiters(Mapping[str, RateLimiter]):
    """A run's limiters by lane name; one instance shared by every worker.

    Always holds the two framework lanes, ``default`` and ``none``. Built
    from a :class:`~jkent.common.rate_limits.RateLimitTable` with
    :meth:`for_table`; :meth:`unlimited` is the null object (every lane a
    :class:`NoopRateLimiter`) for replay and for hosts that construct
    collaborators by hand.
    """

    @require(
        lambda limiters: (  # pyrefly: ignore[implicit-any-lambda]
            DEFAULT_RATE_LIMIT in limiters and NO_RATE_LIMIT in limiters
        ),
        "the default and none lanes are always present",
    )
    def __init__(self, limiters: Mapping[str, RateLimiter]) -> None:
        self._limiters: dict[str, RateLimiter] = dict(limiters)

    @classmethod
    def unlimited(cls) -> RateLimiters:
        """Both framework lanes, neither throttling."""
        return cls(
            {
                DEFAULT_RATE_LIMIT: NoopRateLimiter(),
                NO_RATE_LIMIT: NoopRateLimiter(),
            }
        )

    @classmethod
    def for_table(
        cls,
        table: RateLimitTable,
        *,
        adaptive: bool = False,
        rate_limited: bool = True,
    ) -> RateLimiters:
        """One limiter per lane of ``table``.

        Args:
            table: The scraper's lanes and the rates behind them.
            adaptive: Make the ``default`` lane an
                :class:`AdaptiveRateLimiter` whose ladder starts at the
                lane's declared rate (:func:`adaptive_ladder`), so it never
                paces faster than the scraper asked. Named lanes are
                unaffected.
            rate_limited: Off for replay: every lane is a
                :class:`NoopRateLimiter`, ``adaptive`` or not.
        """
        limiters: dict[str, RateLimiter] = {}
        for name in table.names:
            rates = table.rates.get(name)
            if not rate_limited:
                limiters[name] = NoopRateLimiter()
            elif name == DEFAULT_RATE_LIMIT and adaptive:
                limiters[name] = AdaptiveRateLimiter(
                    adaptive_ladder(rates or ())
                )
            elif not rates:
                limiters[name] = NoopRateLimiter()
            else:
                limiters[name] = PyrateRateLimiter(list(rates))
        return cls(limiters)

    def for_request(self, request: Request) -> RateLimiter:
        """The limiter gating ``request`` (``rate_limit`` None = default).

        Raises:
            ValueError: for a lane this run does not have. The queue rejects
                unknown names at enqueue, so this is reached only by a
                request that never went through it.
        """
        name = request.rate_limit or DEFAULT_RATE_LIMIT
        try:
            return self._limiters[name]
        except KeyError:
            raise ValueError(
                f"Request names rate limit {name!r}, but this run's lanes "
                f"are {list(self._limiters)}"
            ) from None

    def __getitem__(self, name: str) -> RateLimiter:
        return self._limiters[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._limiters)

    def __len__(self) -> int:
        return len(self._limiters)
