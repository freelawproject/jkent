"""Conformance suite for the worker role.

The worker surface is intentionally thin — ``worker_id: int`` plus
``async def run()`` — with the rich behavior living inside ``run()``. Because
the contract is collaborator-driven, this suite drives a fully-wired runnable
worker over in-memory fakes (a queue, a transport, a retry policy) and
asserts the *observable* outcomes rather than poking at internals.

Contract under test:

- Identity: ``worker_id`` is an ``int``.
- Drain: ``run()`` processes every queued request and returns once the queue is
  idle.
- Transient retry: a request whose ``resolve`` raises ``TransientException``
  is retried (re-processed) while the retry policy grants a delay, and
  ultimately completes; once the policy answers ``None`` (backoff exhausted —
  the real ``ResponseStorage.handle_retry`` contract) the request is marked
  failed instead.
- Persistent failure: a ``PersistentException`` (here the HTTP flavour) is
  never retried; the request is marked failed on the first occurrence.
- Halt propagates: a ``RequestFailedHalt`` raised while resolving propagates out
  of ``run()`` and stops the worker.
- Stop signal: setting the stop event causes ``run()`` to exit.

The reference fake worker below implements exactly that documented loop over the
fake collaborators and is exercised through ``TestReferenceWorker``; per-item
failures are scripted via the fake transport. ``test_worker.py`` binds the
suite to the real ``PoolWorker``.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import pytest
from hypothesis import given
from hypothesis import strategies as st
from typing_extensions import override

from jkent.common.exceptions import (
    PersistentException,
    PersistentHTTPResponseException,
    RequestFailedHalt,
    TransientException,
    TransientKind,
)

# --- In-memory fake collaborators ----------------------------------------


@dataclass
class FakeQueue:
    """A trivial FIFO of request ids that re-enqueues retried items."""

    _items: deque[int] = field(default_factory=deque)

    def put(self, request_id: int) -> None:
        """Append a request id (used for both initial fill and retries)."""
        self._items.append(request_id)

    def get(self) -> int | None:
        """Pop the next request id, or ``None`` when durably idle."""
        return self._items.popleft() if self._items else None

    def pending_ids(self) -> list[int]:
        """The ids still queued, in dequeue order (harness observability)."""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class FakeTransport:
    """Resolves requests, raising scripted per-item failures in sequence.

    ``failures`` maps a request id to the exceptions its successive resolves
    raise, oldest first; each fires once. A transient scripted ``times=k``
    fails the first k attempts and then succeeds; a persistent failure or a
    halt is terminal for that id, so one entry is all the worker ever sees.
    """

    failures: dict[int, deque[Exception]] = field(default_factory=dict)
    acquired: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)
    acquire_count: int = 0
    release_count: int = 0

    def fail(self, request_id: int, exc: Exception, *, times: int = 1) -> None:
        """Script ``exc`` for the next ``times`` resolves of ``request_id``."""
        self.failures.setdefault(request_id, deque()).extend([exc] * times)

    def acquire(self, worker_id: int) -> None:
        """Lease this worker's handle (a no-op beyond bookkeeping here)."""
        self.acquired.add(worker_id)
        self.acquire_count += 1

    def release(self, worker_id: int) -> None:
        """Release this worker's handle on exit."""
        self.released.add(worker_id)
        self.release_count += 1

    async def resolve(self, request_id: int) -> int:
        """Return the resolved id, raising the next scripted failure first."""
        # A real transport does I/O and suspends here; yield so a concurrent
        # stopper can observe partial progress and fire a genuine mid-run stop
        # (otherwise run() drains the whole queue before the stopper is ever
        # scheduled, and the stop-loses-nothing law is never exercised).
        await asyncio.sleep(0)
        pending = self.failures.get(request_id)
        if pending:
            raise pending.popleft()
        return request_id


class Requeues(Protocol):
    """What the retry policy needs from a queue: a way to schedule a retry."""

    def put(self, request_id: int) -> None: ...


@dataclass
class FakeRetryPolicy:
    """Stand-in for ``ResponseStorage.handle_retry``'s decision.

    Mirrors the real return contract: the delay (seconds) the retry was
    scheduled with, or ``None`` when the request's retry budget is spent and
    the worker must mark it failed. Like the real storage, a granted retry
    is *scheduled here* (re-enqueued) — the worker only learns the delay.

    ``budgets`` is retries remaining per id (an unscripted id gets
    ``default_budget``); every grant reports ``delay``, which the workers
    under this suite do not act on. Every grant is logged to ``retried``, so
    ``len(retried)`` is the number of attempts beyond the first across the
    run.
    """

    queue: Requeues
    retried: list[int]
    budgets: dict[int, int] = field(default_factory=dict)
    default_budget: int = 1
    delay: float = 1.0

    def handle_retry(self, request_id: int) -> float | None:
        remaining = self.budgets.get(request_id, self.default_budget)
        if remaining <= 0:
            return None
        self.budgets[request_id] = remaining - 1
        self.retried.append(request_id)
        self.queue.put(request_id)
        return self.delay


@dataclass
class WorkerHarness:
    """Observable surface a conformance test inspects after driving ``run()``.

    ``worker`` is anything with the worker surface: ``worker_id: int`` plus
    ``async def run()``.
    """

    worker: Any
    queue: FakeQueue
    transport: FakeTransport
    stop_event: asyncio.Event
    processed: list[int]
    retried: list[int]
    failed: list[int]
    retry_policy: FakeRetryPolicy
    halt_id: int | None = None


# --- Generative-rig strategies and script plumbing ------------------------


@dataclass(frozen=True)
class Transient:
    """A request whose first ``fails`` resolves raise ``TransientException``.

    ``budget`` is how many retries the policy grants before answering
    ``None``. With ``budget >= fails`` the request completes on attempt
    ``fails + 1``; otherwise it is marked failed on attempt ``budget + 1``.
    """

    fails: int
    budget: int


#: One request's scripted outcome.
Script = str | Transient

# The alphabet every request can draw from. "transient" is the original
# fails-once-then-succeeds letter; ``Transient`` generalizes it to k
# consecutive failures against a drawn retry budget. Halts are drawn
# separately (a position), never as a letter, so the no-halt scenarios keep
# their closed-form per-id expectations.
_SCRIPT: st.SearchStrategy[Script] = st.sampled_from(
    ["ok", "transient", "persistent"]
) | st.builds(
    Transient,
    fails=st.integers(1, 3),
    budget=st.integers(0, 3),
)


@st.composite
def _alphabet_scenarios(
    draw: st.DrawFn,
) -> tuple[list[Script], int | None]:
    """A full-alphabet script list plus an optional halting position."""
    scripts = draw(st.lists(_SCRIPT, max_size=10))
    halt_pos: int | None = None
    if scripts:
        halt_pos = draw(st.none() | st.integers(0, len(scripts) - 1))
    return scripts, halt_pos


@st.composite
def _stop_scenarios(draw: st.DrawFn) -> tuple[int, int]:
    """A queue size plus the completion count after which stop fires."""
    total = draw(st.integers(0, 8))
    stop_after = draw(st.integers(0, total))
    return total, stop_after


_PERSISTENT_URL = "https://example.com/p"


def _enqueue_scripts(
    harness: WorkerHarness,
    scripts: list[Script],
    *,
    halt_pos: int | None = None,
) -> int | None:
    """Fill the harness queue with ids 1..N and script their failures.

    Returns the id that will halt (the request at ``halt_pos``), or None.
    """
    halt_id: int | None = None
    for position, script in enumerate(scripts):
        request_id = position + 1
        harness.queue.put(request_id)
        if position == halt_pos:
            harness.transport.fail(request_id, RequestFailedHalt())
            halt_id = request_id
        elif script == "transient":
            harness.transport.fail(
                request_id,
                TransientException("flaky", kind=TransientKind.NETWORK),
            )
            harness.retry_policy.budgets[request_id] = 1
        elif script == "persistent":
            harness.transport.fail(
                request_id,
                PersistentHTTPResponseException(403, _PERSISTENT_URL),
            )
        elif isinstance(script, Transient):
            harness.transport.fail(
                request_id,
                TransientException("flaky", kind=TransientKind.NETWORK),
                times=script.fails,
            )
            harness.retry_policy.budgets[request_id] = script.budget
    return halt_id


def _expected_fate(script: Script) -> tuple[str, int]:
    """``(fate, attempts)`` for one script in an uninterrupted run.

    ``fate`` is ``"processed"`` or ``"failed"``; ``attempts`` counts every
    resolve, so ``attempts - 1`` is the retries the id must log.
    """
    if script == "ok":
        return "processed", 1
    if script == "persistent":
        return "failed", 1
    if script == "transient":
        return "processed", 2
    assert isinstance(script, Transient)
    if script.budget >= script.fails:
        return "processed", script.fails + 1
    return "failed", script.budget + 1


# --- Reference fake worker -----------------------------------------------


class ReferenceWorker:
    """Minimal worker implementing the documented loop over the fakes.

    Pulls one id at a time, resolves it via the transport, and routes failures
    by the taxonomy: transient → ask the retry policy (a delay means it was
    re-enqueued; ``None`` means mark failed), persistent → mark failed, halt
    → propagate. Exits on the stop event or a durably empty queue.
    """

    def __init__(
        self,
        worker_id: int,
        queue: FakeQueue,
        transport: FakeTransport,
        retry_policy: FakeRetryPolicy,
        stop_event: asyncio.Event,
        processed: list[int],
        failed: list[int],
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._transport = transport
        self._retry_policy = retry_policy
        self._stop_event = stop_event
        self._processed = processed
        self._failed = failed

    async def run(self) -> None:
        """Drain the queue, routing failures, until stop or durable idle."""
        try:
            while not self._stop_event.is_set():
                request_id = self._queue.get()
                if request_id is None:
                    return  # durably idle
                self._transport.acquire(self.worker_id)
                try:
                    resolved = await self._transport.resolve(request_id)
                except RequestFailedHalt:
                    raise  # propagate, stops the worker
                except TransientException:
                    if self._retry_policy.handle_retry(request_id) is None:
                        self._failed.append(request_id)  # backoff exhausted
                    continue
                except PersistentException:
                    self._failed.append(request_id)  # never retried
                    continue
                self._processed.append(resolved)  # persist + mark complete
        finally:
            self._transport.release(self.worker_id)


# --- Reusable conformance base -------------------------------------------


class WorkerConformance:
    """Reusable contract assertions for any ``Worker`` implementation.

    Subclass and override :meth:`make_harness` to return a runnable worker
    plus a :class:`WorkerHarness` exposing the queue, stop event, and outcome
    logs. The generative tests build a fresh harness per Hypothesis example
    (function-scoped fixtures are NOT reset between examples), so the factory
    — not the ``subject`` fixture — is the override point.
    """

    def make_harness(self) -> WorkerHarness:
        """Build a fresh runnable worker and its observable harness."""
        raise NotImplementedError

    @pytest.fixture
    def subject(self) -> WorkerHarness:
        """Return a runnable worker and its observable harness."""
        return self.make_harness()

    async def test_drains_queue_and_returns_when_idle(
        self, subject: WorkerHarness
    ) -> None:
        ids = [1, 2, 3, 4, 5]
        for request_id in ids:
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.processed == ids
        assert len(subject.queue) == 0

    async def test_transient_failure_is_retried_and_completes(
        self, subject: WorkerHarness
    ) -> None:
        subject.transport.fail(
            2, TransientException("flaky", kind=TransientKind.NETWORK)
        )
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.retried == [2]  # first attempt at 2 was retried
        assert sorted(subject.processed) == [1, 2, 3]  # all completed
        assert len(subject.queue) == 0

    async def test_transient_exhausting_its_budget_is_failed(
        self, subject: WorkerHarness
    ) -> None:
        """``handle_retry`` -> None: no retry, the request is marked failed."""
        subject.transport.fail(
            2, TransientException("flaky", kind=TransientKind.NETWORK), times=2
        )
        subject.retry_policy.budgets[2] = 1
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.retried == [2]  # the one retry the budget allowed
        assert subject.failed == [2]
        assert sorted(subject.processed) == [1, 3]
        assert len(subject.queue) == 0

    async def test_persistent_failure_is_not_retried(
        self, subject: WorkerHarness
    ) -> None:
        subject.transport.fail(
            2, PersistentHTTPResponseException(403, _PERSISTENT_URL)
        )
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.retried == []
        assert subject.failed == [2]
        assert subject.processed == [1, 3]
        assert len(subject.queue) == 0

    async def test_halt_propagates(self, subject: WorkerHarness) -> None:
        subject.transport.fail(2, RequestFailedHalt())
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        with pytest.raises(RequestFailedHalt):
            await subject.worker.run()

        assert subject.processed == [1]  # stopped at the halting request

    async def test_stop_signal_exits(self, subject: WorkerHarness) -> None:
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)
        subject.stop_event.set()

        await subject.worker.run()

        assert subject.processed == []  # exited before processing anything

    # --- Generative rig bodies ---------------------------------------------
    #
    # The example tests above pin one instance of each contract clause; these
    # drive Hypothesis-drawn failure scripts and stop timings through the same
    # harness and assert the conservation laws that hold for EVERY script.
    # Sync + asyncio.run because @given does not compose with async def under
    # pytest-asyncio (same pattern as test_recoverable_conformance).
    #
    # The bodies live here undecorated; every binding declares its own thin
    # ``@given`` wrappers (see ``TestReferenceWorker``) because a ``@given``
    # method shared across two subclasses trips hypothesis's
    # ``differing_executors`` health check.

    def check_failure_alphabet_conserves_every_request(
        self, scenario: tuple[list[Script], int | None]
    ) -> None:
        """The full alphabet — ok / transient×k / persistent / halt — conserves.

        Laws, for every script and whether or not a halt interrupts it:

        * every id ends in exactly one of processed / failed /
          pending-after-halt (/ the halted one): nothing lost, nothing
          duplicated;
        * every attempt ends in exactly one of processed / retried / failed
          / halted, so ``acquire_count`` equals their sum — the handle is
          leased once per attempt;
        * the handle is released exactly once.

        Without a halt the per-id fate is closed-form (:func:`_expected_fate`):
        a transient completes iff its budget covers its failures, its retry
        count is its attempts beyond the first, a persistent is failed on
        attempt one with no retry, and the queue drains.
        """
        scripts, halt_pos = scenario

        async def drive() -> WorkerHarness:
            harness = self.make_harness()
            halt_id = _enqueue_scripts(harness, scripts, halt_pos=halt_pos)
            if halt_id is None:
                await harness.worker.run()
            else:
                with pytest.raises(RequestFailedHalt):
                    await harness.worker.run()
            harness.halt_id = halt_id
            return harness

        harness = asyncio.run(drive())

        ids = list(range(1, len(scripts) + 1))
        halted = [harness.halt_id] if harness.halt_id is not None else []
        fates = (
            list(harness.processed)
            + list(harness.failed)
            + harness.queue.pending_ids()
            + halted
        )
        assert sorted(fates) == ids  # exactly-once conservation
        attempts = (
            len(harness.processed)
            + len(harness.retried)
            + len(harness.failed)
            + len(halted)
        )
        assert harness.transport.acquire_count == attempts
        assert harness.transport.release_count == 1

        if halted:
            return  # a halt cuts scripts short; no closed form per id
        assert harness.queue.pending_ids() == []
        expected = {
            rid: _expected_fate(script) for rid, script in zip(ids, scripts)
        }
        assert sorted(harness.processed) == [
            rid for rid, (fate, _) in expected.items() if fate == "processed"
        ]
        assert sorted(harness.failed) == [
            rid for rid, (fate, _) in expected.items() if fate == "failed"
        ]
        assert Counter(harness.retried) == Counter(
            {rid: n - 1 for rid, (_, n) in expected.items() if n > 1}
        )
        assert harness.transport.acquire_count == sum(
            n for _, n in expected.values()
        )

    def check_stop_never_loses_or_duplicates_work(
        self, scenario: tuple[int, int]
    ) -> None:
        """Stopping after any number of completions keeps the queue exact.

        Laws: ``run()`` returns; processed is a prefix of the dequeue order
        of length >= the stop point; processed + still-queued is exactly the
        original set; the handle is released.
        """
        total, stop_after = scenario

        async def drive() -> WorkerHarness:
            harness = self.make_harness()
            ids = list(range(1, total + 1))
            for request_id in ids:
                harness.queue.put(request_id)

            async def stopper() -> None:
                while len(harness.processed) < stop_after:
                    await asyncio.sleep(0)
                harness.stop_event.set()

            await asyncio.wait_for(
                asyncio.gather(harness.worker.run(), stopper()), timeout=10
            )
            return harness

        harness = asyncio.run(drive())

        ids = list(range(1, total + 1))
        done = len(harness.processed)
        assert stop_after <= done <= total
        assert harness.processed == ids[:done]  # a prefix: in order, no dups
        assert harness.queue.pending_ids() == ids[done:]  # nothing lost
        assert harness.transport.release_count == 1


# --- Reference implementation under the suite ----------------------------


class TestReferenceWorker(WorkerConformance):
    """Runs the conformance suite against the reference fake worker."""

    @pytest.mark.generative
    @given(scenario=_alphabet_scenarios())
    def test_failure_alphabet_conserves_every_request(
        self, scenario: tuple[list[Script], int | None]
    ) -> None:
        self.check_failure_alphabet_conserves_every_request(scenario)

    @pytest.mark.generative
    @given(scenario=_stop_scenarios())
    def test_stop_never_loses_or_duplicates_work(
        self, scenario: tuple[int, int]
    ) -> None:
        self.check_stop_never_loses_or_duplicates_work(scenario)

    @override
    def make_harness(self) -> WorkerHarness:
        queue = FakeQueue()
        transport = FakeTransport()
        stop_event = asyncio.Event()
        processed: list[int] = []
        retried: list[int] = []
        failed: list[int] = []
        retry_policy = FakeRetryPolicy(queue=queue, retried=retried)
        worker = ReferenceWorker(
            worker_id=1,
            queue=queue,
            transport=transport,
            retry_policy=retry_policy,
            stop_event=stop_event,
            processed=processed,
            failed=failed,
        )
        return WorkerHarness(
            worker=worker,
            queue=queue,
            transport=transport,
            stop_event=stop_event,
            processed=processed,
            retried=retried,
            failed=failed,
            retry_policy=retry_policy,
        )
