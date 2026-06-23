"""Conformance suite for the worker role (``PoolWorker`` and stand-ins).

The worker surface is intentionally thin — ``worker_id: int`` plus
``async def run()`` — with the rich behavior living inside ``run()``. Because
the contract is collaborator-driven, this suite drives a fully-wired runnable
worker over in-memory fakes (a queue + a transport) and asserts the
*observable* outcomes rather than poking at internals.

Contract under test:

- Identity: ``worker_id`` is an ``int``.
- Drain: ``run()`` processes every queued request and returns once the queue is
  idle.
- Transient retry: a request whose first ``resolve`` raises ``TransientException``
  is retried (re-processed) and ultimately completes.
- Halt propagates: a ``RequestFailedHalt`` raised while resolving propagates out
  of ``run()`` and stops the worker.
- Skip continues: a ``RequestFailedSkip`` marks the request failed (not retried)
  and the worker carries on with the rest of the queue.
- Stop signal: setting the stop event causes ``run()`` to exit.

The reference fake worker below implements exactly that documented loop over the
fake collaborators and is exercised through ``TestReferenceWorker`` so the file
runs green; per-item failures are scripted via the fake transport.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jkent.common.exceptions import (
    RequestFailedHalt,
    RequestFailedSkip,
    TransientException,
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
    """Resolves requests, raising scripted per-item failures on first attempt.

    ``failures`` maps a request id to the exception its *first* resolve should
    raise; transient failures are popped after firing so the retry succeeds,
    while halt/skip failures are terminal for that id.
    """

    failures: dict[int, Exception] = field(default_factory=dict)
    acquired: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)
    acquire_count: int = 0
    release_count: int = 0

    def acquire(self, worker_id: int) -> None:
        """Lease this worker's handle (a no-op beyond bookkeeping here)."""
        self.acquired.add(worker_id)
        self.acquire_count += 1

    def release(self, worker_id: int) -> None:
        """Release this worker's handle on exit."""
        self.released.add(worker_id)
        self.release_count += 1

    async def resolve(self, request_id: int) -> int:
        """Return the resolved id, raising any scripted failure first."""
        # A real transport does I/O and suspends here; yield so a concurrent
        # stopper can observe partial progress and fire a genuine mid-run stop
        # (otherwise run() drains the whole queue before the stopper is ever
        # scheduled, and the stop-loses-nothing law is never exercised).
        await asyncio.sleep(0)
        exc = self.failures.get(request_id)
        if exc is not None:
            if isinstance(exc, TransientException):
                del self.failures[request_id]  # transient clears on first hit
            raise exc
        return request_id


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
    skipped: list[int]
    halt_id: int | None = None


# --- Generative-rig strategies and script plumbing ------------------------

# Per-request outcome script. "transient" raises once then succeeds on the
# retry; "skip" is terminal for the id; halts are drawn separately so the
# no-halt soup keeps its stronger partition law.
_SCRIPT = st.sampled_from(["ok", "transient", "skip"])


@st.composite
def _halt_scenarios(draw: st.DrawFn) -> tuple[list[str], int]:
    """A failure-script list plus the position whose request halts."""
    scripts = draw(st.lists(_SCRIPT, min_size=1, max_size=10))
    halt_pos = draw(st.integers(0, len(scripts) - 1))
    return scripts, halt_pos


@st.composite
def _stop_scenarios(draw: st.DrawFn) -> tuple[int, int]:
    """A queue size plus the completion count after which stop fires."""
    total = draw(st.integers(0, 8))
    stop_after = draw(st.integers(0, total))
    return total, stop_after


def _enqueue_scripts(
    harness: WorkerHarness,
    scripts: list[str],
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
            harness.transport.failures[request_id] = RequestFailedHalt()
            halt_id = request_id
        elif script == "transient":
            harness.transport.failures[request_id] = TransientException(
                "flaky"
            )
        elif script == "skip":
            harness.transport.failures[request_id] = RequestFailedSkip()
    return halt_id


# --- Reference fake worker -----------------------------------------------


class ReferenceWorker:
    """Minimal worker implementing the documented loop over the fakes.

    Pulls one id at a time, resolves it via the transport, and routes failures
    by the taxonomy: transient → re-enqueue (retry), halt → propagate, skip →
    mark and continue. Exits on the stop event or a durably empty queue.
    """

    def __init__(
        self,
        worker_id: int,
        queue: FakeQueue,
        transport: FakeTransport,
        stop_event: asyncio.Event,
        processed: list[int],
        retried: list[int],
        skipped: list[int],
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._transport = transport
        self._stop_event = stop_event
        self._processed = processed
        self._retried = retried
        self._skipped = skipped

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
                except RequestFailedSkip:
                    self._skipped.append(request_id)  # mark failed, continue
                    continue
                except TransientException:
                    self._retried.append(request_id)
                    self._queue.put(request_id)  # schedule a retry
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

    def test_worker_id_is_int(self, subject: WorkerHarness) -> None:
        assert isinstance(subject.worker.worker_id, int)

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
        subject.transport.failures[2] = TransientException("flaky")
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.retried == [2]  # first attempt at 2 was retried
        assert sorted(subject.processed) == [1, 2, 3]  # all completed
        assert len(subject.queue) == 0

    async def test_halt_propagates(self, subject: WorkerHarness) -> None:
        subject.transport.failures[2] = RequestFailedHalt()
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        with pytest.raises(RequestFailedHalt):
            await subject.worker.run()

        assert subject.processed == [1]  # stopped at the halting request

    async def test_skip_is_marked_and_worker_continues(
        self, subject: WorkerHarness
    ) -> None:
        subject.transport.failures[2] = RequestFailedSkip()
        for request_id in (1, 2, 3):
            subject.queue.put(request_id)

        await subject.worker.run()

        assert subject.skipped == [2]  # marked, not retried
        assert subject.retried == []
        assert subject.processed == [1, 3]  # continued past the skip

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

    def check_failure_soup_partitions_the_queue(
        self, scripts: list[str]
    ) -> None:
        """Any mix of ok/transient/skip outcomes conserves every request.

        Laws: the worker returns; processed and skipped partition the ids
        (transients land in processed after their retry); each transient is
        retried exactly once; the queue drains; the handle is leased once per
        attempt and released exactly once.
        """

        async def drive() -> WorkerHarness:
            harness = self.make_harness()
            _enqueue_scripts(harness, scripts)
            await harness.worker.run()
            return harness

        harness = asyncio.run(drive())

        ids = list(range(1, len(scripts) + 1))
        ok_or_transient = [rid for rid, s in zip(ids, scripts) if s != "skip"]
        skip_ids = [rid for rid, s in zip(ids, scripts) if s == "skip"]
        transient_ids = [
            rid for rid, s in zip(ids, scripts) if s == "transient"
        ]

        assert sorted(harness.processed) == ok_or_transient
        assert sorted(harness.skipped) == skip_ids
        assert sorted(harness.retried) == transient_ids
        assert len(harness.queue) == 0
        attempts = (
            len(harness.processed)
            + len(harness.skipped)
            + len(harness.retried)
        )
        assert harness.transport.acquire_count == attempts
        assert harness.transport.release_count == 1

    def check_halt_accounts_for_every_request(
        self, scenario: tuple[list[str], int]
    ) -> None:
        """A halt anywhere in any script leaves a resumable, lossless queue.

        Laws: ``run()`` raises; every id ends in exactly one of
        {processed, skipped, still-queued, the-halted-one} — nothing is lost
        or duplicated, whatever mix of retries and skips preceded the halt —
        and the handle is still released on the raising path.
        """
        scripts, halt_pos = scenario

        async def drive() -> WorkerHarness:
            harness = self.make_harness()
            halt_id = _enqueue_scripts(harness, scripts, halt_pos=halt_pos)
            with pytest.raises(RequestFailedHalt):
                await harness.worker.run()
            assert halt_id is not None
            harness.halt_id = halt_id
            return harness

        harness = asyncio.run(drive())

        ids = list(range(1, len(scripts) + 1))
        halt_id = harness.halt_id
        assert halt_id is not None
        outcomes = (
            list(harness.processed)
            + list(harness.skipped)
            + harness.queue.pending_ids()
            + [halt_id]
        )
        assert sorted(outcomes) == ids  # exactly-once conservation
        attempts = (
            len(harness.processed)
            + len(harness.skipped)
            + len(harness.retried)
            + 1  # the halting attempt itself
        )
        assert harness.transport.acquire_count == attempts
        assert harness.transport.release_count == 1  # released despite raise

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
    @given(scripts=st.lists(_SCRIPT, max_size=10))
    def test_failure_soup_partitions_the_queue(
        self, scripts: list[str]
    ) -> None:
        self.check_failure_soup_partitions_the_queue(scripts)

    @pytest.mark.generative
    @given(scenario=_halt_scenarios())
    def test_halt_accounts_for_every_request(
        self, scenario: tuple[list[str], int]
    ) -> None:
        self.check_halt_accounts_for_every_request(scenario)

    @pytest.mark.generative
    @given(scenario=_stop_scenarios())
    def test_stop_never_loses_or_duplicates_work(
        self, scenario: tuple[int, int]
    ) -> None:
        self.check_stop_never_loses_or_duplicates_work(scenario)

    def make_harness(self) -> WorkerHarness:
        queue = FakeQueue()
        transport = FakeTransport()
        stop_event = asyncio.Event()
        processed: list[int] = []
        retried: list[int] = []
        skipped: list[int] = []
        worker = ReferenceWorker(
            worker_id=1,
            queue=queue,
            transport=transport,
            stop_event=stop_event,
            processed=processed,
            retried=retried,
            skipped=skipped,
        )
        return WorkerHarness(
            worker=worker,
            queue=queue,
            transport=transport,
            stop_event=stop_event,
            processed=processed,
            retried=retried,
            skipped=skipped,
        )
