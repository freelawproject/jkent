"""Reusable conformance suite for the run role (``ScrapeRun`` and stand-ins).

A run is the outermost lifecycle: the owner and supervisor of a single
scrape. It composes ``AsyncLifecycle`` (``open``/``aclose``) and owns the
transport, the worker registry, the compactors, and the rate limiter. This
suite pins only the *observable* surface, not the full orchestration wiring.

Contract under test:

- Lifecycle: ``open`` and ``aclose`` are awaitable and bracket usage; the
  transport is brought up in ``open`` and torn down in ``aclose``.
- Transport ownership: ``transport`` is exposed and non-None once open.
- ``status()`` returns one of ``"unstarted" | "in_progress" | "done"`` and
  starts at ``"unstarted"``.
- ``spawn_worker()`` returns an int id, increments ``active_worker_count``, and
  distinct calls return distinct ids.
- ``stop()`` signals a graceful shutdown, observable as ``status()`` reaching
  ``"done"`` after the run drives to completion.

Timing is kept out of the contract: "done" is modeled as an empty/exhausted
work set, never wall-clock. With nothing to do, a stopped and an unstopped run
look the same, so the suite cannot tell whether ``stop`` did anything — only
that calling it does not keep the run from finishing.

Below the suite lives a minimal in-memory reference run and
``TestReferenceRun`` so this file runs green on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, get_args

import pytest
from typing_extensions import override

if TYPE_CHECKING:
    from jkent.driver.unified_driver.transport import Transport


class Run(Protocol):
    """The run surface the suite drives (structural, test-only)."""

    async def open(self) -> None: ...

    async def aclose(self) -> None: ...

    @property
    def transport(self) -> Transport[Any]: ...

    @property
    def active_worker_count(self) -> int: ...

    def spawn_worker(self) -> int: ...

    async def run(self) -> None: ...

    def stop(self) -> None: ...

    async def status(self) -> Literal["unstarted", "in_progress", "done"]: ...


class RunConformance:
    """Reusable contract tests for any run implementation.

    Subclass and override :meth:`subject` to return the implementation under
    test. The tests assert only the documented observable surface.
    """

    @pytest.fixture
    def subject(self) -> Run:
        """The run implementation under test."""
        raise NotImplementedError

    async def test_open_and_aclose_bracket_usage(self, subject: Run) -> None:
        """``open``/``aclose`` are awaitable and complete cleanly."""
        await subject.open()
        try:
            assert await subject.status() in get_args(_Status)
        finally:
            await subject.aclose()

    async def test_transport_exposed_after_open(self, subject: Run) -> None:
        """The transport is owned and non-None once the run is open."""
        await subject.open()
        try:
            assert subject.transport is not None
        finally:
            await subject.aclose()

    async def test_status_starts_unstarted(self, subject: Run) -> None:
        """A freshly opened run reports ``"unstarted"``."""
        await subject.open()
        try:
            assert await subject.status() == "unstarted"
        finally:
            await subject.aclose()

    async def test_status_is_a_known_literal(self, subject: Run) -> None:
        """``status`` only ever returns one of the three documented states."""
        await subject.open()
        try:
            assert await subject.status() in get_args(_Status)
        finally:
            await subject.aclose()

    async def test_spawn_worker_returns_int_and_counts(
        self, subject: Run
    ) -> None:
        """``spawn_worker`` returns an int id and bumps ``active_worker_count``."""
        await subject.open()
        try:
            before = subject.active_worker_count
            worker_id = subject.spawn_worker()
            assert isinstance(worker_id, int)
            assert subject.active_worker_count == before + 1
        finally:
            await subject.aclose()

    async def test_spawn_worker_ids_are_distinct(self, subject: Run) -> None:
        """Distinct ``spawn_worker`` calls return distinct ids."""
        await subject.open()
        try:
            ids = [subject.spawn_worker() for _ in range(3)]
            assert len(set(ids)) == len(ids)
            assert subject.active_worker_count >= len(ids)
        finally:
            await subject.aclose()

    async def test_run_drives_status_to_done(self, subject: Run) -> None:
        """``run`` drives a run with no pending work to ``"done"``.

        Completion is not gated on ``stop``: a run that drives its work set to
        exhaustion reaches ``"done"`` on its own.
        """
        await subject.open()
        try:
            await subject.run()
            assert await subject.status() == "done"
        finally:
            await subject.aclose()

    async def test_stop_drives_status_to_done(self, subject: Run) -> None:
        """``stop`` signals shutdown; ``run`` then drives ``status`` to done."""
        await subject.open()
        try:
            subject.stop()
            await subject.run()
            assert await subject.status() == "done"
        finally:
            await subject.aclose()


# Mirror of ``ScrapeRun.status``'s return literal, to validate return values.
_Status = Literal["unstarted", "in_progress", "done"]


# --- Reference fake run --------------------------------------------------


class _FakeTransport:
    """A trivial run-scoped transport peer: tracks open/closed state only."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False

    async def open(self) -> None:
        self.opened = True

    async def aclose(self) -> None:
        self.closed = True


class _ReferenceRun:
    """Minimal in-memory run over trivial fakes; no DB, no real workers.

    ``status`` derives from whether the run has started and whether the
    (empty) work set has drained.
    """

    def __init__(self) -> None:
        self._transport: _FakeTransport | None = None
        self._workers: list[int] = []
        self._next_id = 0
        self._started = False
        self._finished = False

    async def open(self) -> None:
        """Bring up the run-scoped transport."""
        self._transport = _FakeTransport()
        await self._transport.open()

    async def aclose(self) -> None:
        """Tear down the transport after workers have exited."""
        if self._transport is not None:
            await self._transport.aclose()

    @property
    def transport(self) -> Transport[Any]:
        """The run-scoped transport peer."""
        assert self._transport is not None, "transport accessed before open()"
        return cast("Transport[Any]", self._transport)

    @property
    def active_worker_count(self) -> int:
        """Number of registered workers."""
        return len(self._workers)

    def spawn_worker(self) -> int:
        """Register a worker and return its fresh id."""
        worker_id = self._next_id
        self._next_id += 1
        self._workers.append(worker_id)
        return worker_id

    async def run(self) -> None:
        """Drive the scrape to completion, whether or not ``stop`` was called.

        With an empty work set the run finishes immediately; a prior ``stop``
        only means workers exit at their next idle check. Either path reaches
        completion, mirroring the real ``ScrapeRun`` whose ``status`` is
        "done" once no work is active and no workers remain — completion is
        not gated on a stop having been signalled.
        """
        self._started = True
        self._workers.clear()
        self._finished = True

    def stop(self) -> None:
        """Signal graceful shutdown: a no-op with no work to leave undone."""

    async def status(self) -> Literal["unstarted", "in_progress", "done"]:
        """Derive run state from start + completion (not from stop)."""
        if not self._started:
            return "unstarted"
        if self._finished and not self._workers:
            return "done"
        return "in_progress"


class TestReferenceRun(RunConformance):
    """Run the conformance suite against the in-memory reference run."""

    @pytest.fixture
    @override
    def subject(self) -> _ReferenceRun:
        return _ReferenceRun()
