"""Conformance suite for the ``Worker`` protocol (orchestration.py).

``Worker`` is intentionally thin — ``worker_id: int`` plus ``async def run()``
— with the rich behavior living inside ``run()`` and described in
``worker_contract.md``. Because the contract is collaborator-driven, this suite
drives a fully-wired runnable worker over in-memory fakes (a queue + a
transport) and asserts the *observable* outcomes rather than poking at
internals.

Contract under test (see ``worker_contract.md``):

- Identity: ``worker_id`` is an ``int`` and the worker is a ``Worker`` instance.
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

import pytest

from jkent.common.exceptions import (
    RequestFailedHalt,
    RequestFailedSkip,
    TransientException,
)
from jkent.driver.unified_driver import Worker

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

    def acquire(self, worker_id: int) -> None:
        """Lease this worker's handle (a no-op beyond bookkeeping here)."""
        self.acquired.add(worker_id)

    def release(self, worker_id: int) -> None:
        """Release this worker's handle on exit."""
        self.released.add(worker_id)

    async def resolve(self, request_id: int) -> int:
        """Return the resolved id, raising any scripted failure first."""
        exc = self.failures.get(request_id)
        if exc is not None:
            if isinstance(exc, TransientException):
                del self.failures[request_id]  # transient clears on first hit
            raise exc
        return request_id


@dataclass
class WorkerHarness:
    """Observable surface a conformance test inspects after driving ``run()``."""

    worker: Worker
    queue: FakeQueue
    transport: FakeTransport
    stop_event: asyncio.Event
    processed: list[int]
    retried: list[int]
    skipped: list[int]


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

    Subclass and override ``subject`` to return a runnable worker plus a
    :class:`WorkerHarness` exposing the queue, stop event, and outcome logs.
    """

    @pytest.fixture
    def subject(self) -> WorkerHarness:
        """Return a runnable worker and its observable harness."""
        raise NotImplementedError

    def test_worker_id_is_int(self, subject: WorkerHarness) -> None:
        assert isinstance(subject.worker.worker_id, int)

    def test_is_a_worker(self, subject: WorkerHarness) -> None:
        assert isinstance(subject.worker, Worker)

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


# --- Reference implementation under the suite ----------------------------


class TestReferenceWorker(WorkerConformance):
    """Runs the conformance suite against the reference fake worker."""

    @pytest.fixture
    def subject(self) -> WorkerHarness:
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
