"""Run-scoped request pacing.

A run holds one in-memory limiter per rate-limit *lane* (see
:mod:`jkent.common.rate_limits`), each shared by every worker, so a declared
pace is a limit for the whole run on that lane, not a per-worker one. Pacing
state is not durable — it lives in memory and resets on restart, which is
fine: the pace is a courtesy to the remote server, not a correctness
invariant.

:class:`RateLimiter` is the interface, and a request passes through it in
three moments. A worker asks :meth:`~RateLimiter.ready_in` whether the lane
can take a request now, and only claims work for lanes that can.
:meth:`~RateLimiter.reserve` takes the lane's turn for the claimed request
and returns a :class:`Turn`; :meth:`Turn.start` blocks out whatever gap is
left and marks the moment the request is sent; :meth:`Turn.finish` ends the
turn, sent or not. :meth:`~RateLimiter.gate` does all of that for a caller
that already holds a request. :meth:`~RateLimiter.record_response` is the
feedback channel the worker calls with classified HTTP failures.

The implementations follow the two declared paces:
:class:`CappedPaceLimiter` keeps a minimum gap between request starts and
banks no credit, :class:`ReactivePaceLimiter` runs one request at a time
and rests after each response, and :class:`NoopRateLimiter` never waits.
:class:`AdaptiveRateLimiter` is the diagnostic probe: a capped pace that
steps down a ladder of slower ones on 429s and ``Retry-After``, and climbs
back after a quiet spell, never past the declared pace. It runs a lane
declared as an :class:`~jkent.common.rate_limits.AdaptivePace`.

:class:`RateLimiters` is the run's registry of lane -> limiter.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Final

from typing_extensions import override

from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    AdaptivePace,
    CappedPace,
    Pace,
    RateLimitTable,
    ReactivePace,
)
from jkent.contracts import require

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from jkent.data_types import Request

logger = logging.getLogger(__name__)


class Turn:
    """One request's turn in a lane, from its claim to its response.

    Attributes:
        limiter: The lane's limiter, which hears this request's feedback.
        started: Whether :meth:`start` returned, so the request was sent.
        started_at: When it was sent, on the limiter's clock; ``None``
            before :meth:`start`, and for a lane that keeps no time.
    """

    __slots__ = ("_finished", "limiter", "started", "started_at")

    def __init__(self, limiter: RateLimiter) -> None:
        """Hold a turn :meth:`RateLimiter.reserve` just took.

        Args:
            limiter: The limiter the turn was taken on.
        """
        self.limiter = limiter
        self.started = False
        self.started_at: float | None = None
        self._finished = False

    async def start(self, request: Request) -> None:
        """Block until ``request`` may be sent; it goes out on return.

        Args:
            request: The request about to be fetched.

        Raises:
            RuntimeError: If the turn was already started or finished.
        """
        if self.started or self._finished:
            raise RuntimeError("a turn starts once, before it finishes")
        self.started_at = await self.limiter._start(request)
        self.started = True

    def finish(self) -> None:
        """End the turn: the response is in, or the request was never sent.

        Safe to call more than once, so every exit path can call it.
        """
        if self._finished:
            return
        self._finished = True
        self.limiter._finish(started=self.started)


class RateLimiter(ABC):
    """Paces one lane; shared across all workers."""

    #: How long :meth:`gate` sleeps between looks at a lane that cannot say
    #: when it will be free (a reactive lane with a request in flight).
    BUSY_POLL_INTERVAL_S: ClassVar[float] = 0.05

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Bind the limiter's time source.

        Args:
            clock: Monotonic time source; injectable for tests.
            sleep: How a wait waits, in ``clock`` seconds; injectable
                alongside ``clock`` so a test's wait costs no wall time.
        """
        self._clock = clock
        self._sleep = sleep

    @abstractmethod
    def ready_in(self) -> float | None:
        """How long until the lane can take another request.

        Returns:
            ``0.0`` when a request claimed now could be sent now; the
            seconds until that is true; or ``None`` when the lane is busy
            and cannot say (it is waiting on a response).
        """
        ...

    def reserve(self) -> Turn:
        """Take the lane's turn for a request just claimed.

        Call only when :meth:`ready_in` is zero, with no ``await`` between
        the two. The caller owns the returned turn and must
        :meth:`~Turn.finish` it on every exit.

        Returns:
            The turn.
        """
        self._reserve()
        return Turn(self)

    async def gate(self, request: Request) -> Turn:
        """Wait for the lane, take a turn and start it.

        For a caller that already holds ``request`` and must send it on this
        lane however long that takes: a strictly serial worker, or a host
        driving a limiter by hand. The caller still owns
        :meth:`Turn.finish`.

        Args:
            request: The request about to be fetched.

        Returns:
            The started turn.
        """
        while (wait := self.ready_in()) != 0:
            await self._sleep(
                self.BUSY_POLL_INTERVAL_S if wait is None else wait
            )
        turn = self.reserve()
        try:
            await turn.start(request)
        except BaseException:
            turn.finish()
            raise
        return turn

    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
        sent_at: float | None = None,
    ) -> None:
        """Feedback channel: the worker reports classified HTTP failures here.

        Called once per request whose resolve ended in a classified HTTP
        error (transient, persistent, or a failed speculation probe), with
        the observed status and the parsed, clamped ``Retry-After`` where
        the server sent one. ``sent_at`` is the request's
        :attr:`Turn.started_at`, so a limiter that adapts can tell pushback
        on its current pace from pushback on one it already left. The
        worker reports to the limiter that paced the request, so a lane
        hears only its own pushback. Successful responses are not reported.
        Defaults to a no-op: static limiters don't adapt.
        """
        return None

    def _reserve(self) -> None:
        """Count a turn taken. Synchronous, so claiming is atomic."""
        return None

    async def _start(self, request: Request) -> float | None:
        """Wait out the gap left before ``request`` may go.

        Returns:
            When it went, on the limiter's clock, if the lane keeps time.
        """
        return None

    def _finish(self, *, started: bool) -> None:
        """Hear that a turn ended.

        Args:
            started: Whether the request was sent.
        """
        return None


class NoopRateLimiter(RateLimiter):
    """A :class:`RateLimiter` that never waits.

    The ``none`` lane, every lane of a replay run, and the default lane of a
    scraper that declares no ``rate_limits``.
    """

    def ready_in(self) -> float | None:
        """Always ready.

        Returns:
            ``0.0``.
        """
        return 0.0


class _StartPacer(RateLimiter):
    """Keeps a minimum gap between request starts, from one timestamp.

    The state is when the lane last started a request, plus the turns
    reserved but not yet started. Each reserved turn is expected to start
    about when it was reserved, so the lane is next ready one interval after
    the newest of them: workers claim for a lane no faster than it can send,
    and a reserved request is not left holding a worker while it waits its
    turn. :meth:`_start` is the enforcement: starters
    queue on a FIFO lock, and the one at its head stamps the start from the
    clock *after* it wakes, not from when it planned to. A wake delayed by a
    blocked event loop pushes the next slot back instead of letting the next
    request through early, and nothing is added to the wait, so the achieved
    pace is the declared one or slightly slower.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(clock=clock, sleep=sleep)
        self._last_start = float("-inf")
        self._reserved = 0
        # When the newest reserved turn is expected to start; meaningful
        # only while a turn is reserved.
        self._expected_start = float("-inf")
        self._lock = asyncio.Lock()

    @property
    @abstractmethod
    def current_interval(self) -> float:
        """Seconds between request starts."""
        ...

    def _next_at(self) -> float:
        """When the next request may start.

        Returns:
            The earliest start time, on the limiter's clock.
        """
        return self._last_start + self.current_interval

    def ready_in(self) -> float | None:
        """Seconds until a newly claimed request could start.

        Returns:
            The wait until the lane's next slot, which falls one interval
            after the newest reserved turn when there is one.
        """
        return max(self._next_claim_at() - self._clock(), 0.0)

    def _next_claim_at(self) -> float:
        slot = self._next_at()
        if self._reserved:
            slot = max(slot, self._expected_start + self.current_interval)
        return slot

    @override
    def _reserve(self) -> None:
        self._expected_start = max(self._clock(), self._next_claim_at())
        self._reserved += 1

    @override
    async def _start(self, request: Request) -> float | None:
        async with self._lock:
            # Re-read on every wake: feedback may move the slot while we
            # sleep.
            while (wait := self._next_at() - self._clock()) > 0:
                await self._sleep(wait)
            now = self._clock()
            self._last_start = now
            self._reserved -= 1
            return now

    @override
    def _finish(self, *, started: bool) -> None:
        # A turn that never started still holds its reservation.
        if not started:
            self._reserved -= 1


class CappedPaceLimiter(_StartPacer):
    """A :class:`~jkent.common.rate_limits.CappedPace` lane; static.

    At least ``seconds`` between request starts, and no credit for time the
    lane did not use: after any quiet spell, requests still go one per
    interval (see :class:`_StartPacer`).
    """

    @require(
        lambda seconds: seconds > 0 and math.isfinite(seconds),  # pyrefly: ignore[implicit-any-lambda]
        "the gap is a positive, finite number of seconds",
    )
    def __init__(
        self,
        seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Pace starts ``seconds`` apart.

        Args:
            seconds: The minimum gap between request starts.
            clock: Monotonic time source; injectable for tests.
            sleep: How a wait waits, in ``clock`` seconds.
        """
        super().__init__(clock=clock, sleep=sleep)
        self._interval = seconds

    @property
    def current_interval(self) -> float:
        """Seconds between request starts."""
        return self._interval


class ReactivePaceLimiter(RateLimiter):
    """A :class:`~jkent.common.rate_limits.ReactivePace` lane; static.

    One request at a time: a turn holds the lane from its claim until its
    response is in, and the next may be claimed ``seconds`` after that. A
    turn that ends without the request being sent frees the lane at once and
    starts no rest.
    """

    @require(
        lambda seconds: seconds >= 0 and math.isfinite(seconds),  # pyrefly: ignore[implicit-any-lambda]
        "the rest is a non-negative, finite number of seconds",
    )
    def __init__(
        self,
        seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Rest ``seconds`` after each response.

        Args:
            seconds: The rest between one response and the next request.
            clock: Monotonic time source; injectable for tests.
            sleep: How a wait waits, in ``clock`` seconds.
        """
        super().__init__(clock=clock, sleep=sleep)
        self._rest = seconds
        self._busy = False
        self._last_finish = float("-inf")

    def ready_in(self) -> float | None:
        """Seconds of rest left, or ``None`` while a request is in flight.

        Returns:
            The wait, or ``None`` when it depends on a response.
        """
        if self._busy:
            return None
        return max(self._last_finish + self._rest - self._clock(), 0.0)

    @override
    def _reserve(self) -> None:
        if self._busy:
            raise RuntimeError(
                "reactive lane reserved while a request is in flight; "
                "check ready_in() first"
            )
        self._busy = True

    @override
    async def _start(self, request: Request) -> float | None:
        # The reservation is exclusive and was only given once the rest was
        # over, so there is nothing left to wait for.
        return self._clock()

    @override
    def _finish(self, *, started: bool) -> None:
        self._busy = False
        if started:
            self._last_finish = self._clock()


#: Default descent ladder: seconds between requests, fastest rung first —
#: 4/s down to one per minute. A probed lane with a declared pace opens
#: there instead, and keeps only the rungs slower than it.
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

#: How long a lane must go without counted pushback before
#: :class:`AdaptiveRateLimiter` steps one rung back toward its declared
#: pace.
RECOVERY_WINDOW_S: Final[float] = 300.0


def _usable_retry_after(
    retry_after: float | None, status_code: int, url: str | None
) -> float | None:
    """Drop a ``Retry-After`` that asks for no real wait.

    Returns:
        ``retry_after`` if it is a finite, positive wait, else ``None``.
    """
    if retry_after is not None and not math.isfinite(retry_after):
        # The transport clamps what it parses, so this is belt and braces:
        # a NaN pause would never expire and every request on the lane
        # would sleep on it (or raise) for the rest of the run.
        logger.warning(
            "Ignoring non-finite Retry-After %r on HTTP %d%s",
            retry_after,
            status_code,
            f" from {url}" if url else "",
        )
        return None
    if retry_after is not None and retry_after <= 0:
        # "Retry-After: 0", or a date already past (clock skew): the server
        # asks for no wait, which is not pushback.
        return None
    return retry_after


@dataclass(frozen=True)
class ProbeReport:
    """What an :class:`AdaptiveRateLimiter` learned over a run.

    Attributes:
        declared_interval: Seconds between requests the lane opened at.
        settled_interval: Seconds between requests it ended at.
        slowest_interval: The slowest it went.
        step_downs: Times it stepped one rung slower.
        recoveries: Times it stepped one rung back.
        pushback: 429s and ``Retry-After`` responses that counted.
        stale_pushback: Ones ignored because their request was sent before
            the last step-down.
    """

    declared_interval: float
    settled_interval: float
    slowest_interval: float
    step_downs: int
    recoveries: int
    pushback: int
    stale_pushback: int

    def summary(self) -> str:
        """The report as one log line.

        Returns:
            A sentence naming the declared and settled paces and the
            pushback behind the difference.
        """
        return (
            f"declared 1 request per {self.declared_interval:.3g}s, settled "
            f"at 1 per {self.settled_interval:.3g}s (slowest 1 per "
            f"{self.slowest_interval:.3g}s); {self.pushback} pushback "
            f"response(s) counted, {self.stale_pushback} stale, "
            f"{self.step_downs} step-down(s), {self.recoveries} recovery(ies)"
        )


class AdaptiveRateLimiter(_StartPacer):
    """The diagnostic probe: a capped pace that follows server pushback.

    The ladder opens at the declared pace, followed by the ``ladder`` rungs
    slower than it; with no declared pace it is ``ladder`` as given.
    Feedback via :meth:`record_response`:

    - HTTP 429 steps one rung slower. The next request goes one new
      interval after the last one started.
    - ``Retry-After`` (whatever the status) holds every request until then,
      and also steps one rung slower — the header is a statement about the
      client's overall rate, not about one request. Values arrive parsed
      and clamped by the transport
      (:func:`~jkent.driver.unified_driver.transport.parse_retry_after`),
      so a hostile header slows the run, never stalls it.
    - Pushback on a request sent before the last step-down is stale: it
      reports on a pace already left, so it moves no rung (a
      ``Retry-After`` on it still pauses). One throttling incident answers
      every request in flight with a 429; this is what keeps it to one
      rung however long those requests take. It needs ``sent_at``; a
      report without one counts.
    - After ``recover_after`` seconds with no counted pushback, not
      counting time spent paused, the next request to start steps the lane
      one rung faster, never past the declared pace.

    Step-downs are warnings, and the one onto the slowest rung is an error:
    the lane can slow no further, and a site that keeps pushing back there
    needs a person to look at it. :meth:`report` says where the lane ended
    up.
    """

    @require(
        lambda declared: (  # pyrefly: ignore[implicit-any-lambda]
            declared is None or (declared > 0 and math.isfinite(declared))
        ),
        "the declared pace is None or a positive, finite number of seconds",
    )
    @require(
        lambda ladder: (  # pyrefly: ignore[implicit-any-lambda]
            len(tuple(ladder)) > 0
            and all(interval > 0 for interval in ladder)
            and all(a < b for a, b in zip(ladder, tuple(ladder)[1:]))
        ),
        "the ladder is a non-empty, strictly ascending sequence of positive "
        "intervals",
    )
    @require(
        lambda recover_after: (  # pyrefly: ignore[implicit-any-lambda]
            recover_after is None or recover_after > 0
        ),
        "recover_after is None (never recover) or positive",
    )
    def __init__(
        self,
        declared: float | None = None,
        *,
        ladder: Sequence[float] = DEFAULT_ADAPTIVE_LADDER,
        recover_after: float | None = RECOVERY_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Start at the declared pace, the top of the ladder.

        Args:
            declared: The lane's declared gap in seconds; it opens the
                ladder and caps recovery. ``None`` uses ``ladder`` as is.
            ladder: Seconds between requests per rung, strictly ascending
                (slower rungs last).
            recover_after: Seconds without counted pushback before one
                step back up; ``None`` never recovers.
            clock: Monotonic time source; injectable for tests.
            sleep: How a wait waits, in ``clock`` seconds.
        """
        super().__init__(clock=clock, sleep=sleep)
        if declared is not None:
            self._ladder = (declared, *(i for i in ladder if i > declared))
        else:
            self._ladder = tuple(ladder)
        self._rung = 0
        self._slowest_rung = 0
        self._recover_after = recover_after
        self._pause_until = float("-inf")
        self._stepped_down_at = float("-inf")
        self._steady_since = clock()
        self._step_downs = 0
        self._recoveries = 0
        self._pushback = 0
        self._stale_pushback = 0

    @property
    def current_interval(self) -> float:
        """Seconds between request starts at the current rung."""
        return self._ladder[self._rung]

    @override
    def _next_at(self) -> float:
        return max(super()._next_at(), self._pause_until)

    @override
    async def _start(self, request: Request) -> float | None:
        self._recover()
        return await super()._start(request)

    def report(self) -> ProbeReport:
        """What the probe has seen so far.

        Returns:
            The declared and current paces, and the pushback between them.
        """
        return ProbeReport(
            declared_interval=self._ladder[0],
            settled_interval=self.current_interval,
            slowest_interval=self._ladder[self._slowest_rung],
            step_downs=self._step_downs,
            recoveries=self._recoveries,
            pushback=self._pushback,
            stale_pushback=self._stale_pushback,
        )

    @override
    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
        sent_at: float | None = None,
    ) -> None:
        """Step down on fresh 429s and Retry-Afters; pause on Retry-After."""
        retry_after = _usable_retry_after(retry_after, status_code, url)
        now = self._clock()
        if retry_after is not None:
            if now + retry_after > self._pause_until:
                self._pause_until = now + retry_after
                logger.warning(
                    "Rate limiter pausing all requests for %.1fs "
                    "(Retry-After on HTTP %d%s)",
                    retry_after,
                    status_code,
                    f" from {url}" if url else "",
                )
            trigger = f"Retry-After on HTTP {status_code}"
        elif status_code == HTTPStatus.TOO_MANY_REQUESTS:
            trigger = "HTTP 429"
        else:
            return
        if sent_at is not None and sent_at < self._stepped_down_at:
            self._stale_pushback += 1
            logger.debug(
                "Ignoring %s%s: sent before the last step-down",
                trigger,
                f" on {url}" if url else "",
            )
            return
        self._pushback += 1
        self._steady_since = now
        self._step_down(now, trigger=trigger, url=url)

    def _step_down(self, now: float, *, trigger: str, url: str | None) -> None:
        if self._rung + 1 >= len(self._ladder):
            # Already announced, as an error, on the way down.
            logger.debug(
                "Rate limiter already at slowest rung (1 per %.3gs); "
                "ignoring %s",
                self.current_interval,
                trigger,
            )
            return
        old = self.current_interval
        self._rung += 1
        self._slowest_rung = max(self._slowest_rung, self._rung)
        self._step_downs += 1
        self._stepped_down_at = now
        slowest = self._rung + 1 == len(self._ladder)
        logger.log(
            logging.ERROR if slowest else logging.WARNING,
            "Rate limit stepped down: 1 request per %.3gs -> 1 per %.3gs "
            "(trigger: %s%s)%s",
            old,
            self.current_interval,
            trigger,
            f" on {url}" if url else "",
            "; this is the slowest rung" if slowest else "",
        )

    def _recover(self) -> None:
        if self._rung == 0 or self._recover_after is None:
            return
        now = self._clock()
        # Time spent paused is not time the server put up with us.
        quiet_since = max(self._steady_since, self._pause_until)
        if now - quiet_since < self._recover_after:
            return
        old = self.current_interval
        self._rung -= 1
        self._recoveries += 1
        self._steady_since = now
        logger.info(
            "Rate limit recovered: 1 request per %.3gs -> 1 per %.3gs "
            "(%.0fs without pushback)",
            old,
            self.current_interval,
            self._recover_after,
        )


def _limiter_for(pace: Pace) -> RateLimiter:
    """The limiter a declared pace asks for.

    Returns:
        A capped, reactive or adaptive limiter at the pace's ``seconds``.
    """
    match pace:
        case CappedPace():
            return CappedPaceLimiter(pace.seconds)
        case ReactivePace():
            return ReactivePaceLimiter(pace.seconds)
        case AdaptivePace():
            return AdaptiveRateLimiter(pace.seconds)


class RateLimiters(Mapping[str, RateLimiter]):
    """A run's limiters by lane name; one instance shared by every worker.

    Always holds the two framework lanes, ``default`` and ``none``. Built
    from a :class:`~jkent.common.rate_limits.RateLimitTable` with
    :meth:`for_table`; :meth:`unlimited` is the null object (every lane a
    :class:`NoopRateLimiter`) for replay and for hosts that construct
    collaborators by hand.

    Attributes:
        claim_lock: Held by a worker from reading :meth:`ready_lanes` until
            it has reserved a turn for the request it claimed, so two
            workers never claim for the same open slot.
    """

    @require(
        lambda limiters: (  # pyrefly: ignore[implicit-any-lambda]
            DEFAULT_RATE_LIMIT in limiters and NO_RATE_LIMIT in limiters
        ),
        "the default and none lanes are always present",
    )
    def __init__(self, limiters: Mapping[str, RateLimiter]) -> None:
        """Hold a run's limiters.

        Args:
            limiters: Limiter per lane name; must include the ``default``
                and ``none`` framework lanes.
        """
        self._limiters: dict[str, RateLimiter] = dict(limiters)
        self.claim_lock = asyncio.Lock()

    @classmethod
    def unlimited(cls) -> RateLimiters:
        """Build the null object: both framework lanes, neither waiting.

        Returns:
            Limiters in which every lane is always ready.
        """
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
        rate_limited: bool = True,
    ) -> RateLimiters:
        """One limiter per lane of ``table``.

        Args:
            table: The scraper's lanes and the pace behind each.
            rate_limited: Off for replay: every lane is a
                :class:`NoopRateLimiter`, whatever it declares.

        Returns:
            One limiter per lane of ``table``.
        """
        limiters: dict[str, RateLimiter] = {}
        for name in table.names:
            pace = table.paces.get(name)
            if pace is None or not rate_limited:
                limiters[name] = NoopRateLimiter()
            else:
                limiters[name] = _limiter_for(pace)
        return cls(limiters)

    def for_request(self, request: Request) -> RateLimiter:
        """The limiter pacing ``request`` (``rate_limit`` None = default).

        Args:
            request: The request about to be fetched.

        Returns:
            The limiter for the lane the request names.

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

    def ready_lanes(self) -> list[str]:
        """The lanes that can take a request now.

        Returns:
            Their names, in lane order.
        """
        return [
            name
            for name, limiter in self._limiters.items()
            if limiter.ready_in() == 0
        ]

    def next_ready_in(self) -> float | None:
        """How long until a lane that is not ready becomes so.

        Returns:
            The shortest wait any waiting lane reports; ``None`` when
            none can say (each is waiting on a response), or none is
            waiting.
        """
        waits = [
            wait
            for limiter in self._limiters.values()
            if (wait := limiter.ready_in())
        ]
        return min(waits, default=None)

    def probe_reports(self) -> dict[str, ProbeReport]:
        """What each probed lane learned.

        Returns:
            A report per lane running an :class:`AdaptiveRateLimiter`.
        """
        return {
            name: limiter.report()
            for name, limiter in self._limiters.items()
            if isinstance(limiter, AdaptiveRateLimiter)
        }

    def __getitem__(self, name: str) -> RateLimiter:
        """Look up a lane by name.

        Args:
            name: The lane name.

        Returns:
            That lane's limiter.
        """
        return self._limiters[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate the lanes.

        Returns:
            An iterator over the run's lane names.
        """
        return iter(self._limiters)

    def __len__(self) -> int:
        """Count the lanes.

        Returns:
            How many lanes this run has.
        """
        return len(self._limiters)
