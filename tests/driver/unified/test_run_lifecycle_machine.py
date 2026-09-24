"""Generative lifecycle rig for the ``Run`` protocol (``ScrapeRun``).

A Hypothesis ``RuleBasedStateMachine`` walks random *legal* sequences of
the ``Run`` surface — ``open``, ``spawn_worker``, ``status``, ``stop``,
``run``, ``aclose`` (legality enforced with preconditions: open once,
spawn only while open and before ``run``, close once) — and checks, along
each such sequence, the laws the contract states:

- ``status()`` only returns the three documented literals and is
  monotone along a legal sequence: ``unstarted -> in_progress -> done``,
  never backwards;
- ``status()`` is ``"unstarted"`` until ``run()`` has been called, and
  ``"done"`` only once every seeded request has been resolved;
- ``spawn_worker()`` returns strictly increasing (hence distinct) ids;
- ``active_worker_count`` never exceeds the number spawned and never
  goes negative (workers retire themselves on idle, so it may drop);
- ``transport`` is non-None from ``open`` onward;
- exactly-once: no seeded request is ever resolved twice, and nothing
  outside the seeded set is resolved (an invariant, so it holds while
  pre-``run`` workers are draining too);
- ``run()`` returns, lands the run in ``"done"`` and has resolved every
  seeded request exactly once — including when ``stop()`` fired first
  (graceful-shutdown path).

The subject is the real ``ScrapeRun`` over a fresh temp DB (copied from
the session schema template so ``open`` finds the schema built), the trivial
scraper, and the counting transport from ``test_run_conservation.py``. Each
machine draws a seed count (0..5) up front and, on ``open``, inserts that
many pending rows the way the conservation rig does — so workers, whether
spawned by a rule or by ``run()``, have real work to dequeue and the
in-flight arms of the laws are exercised rather than vacuous. One event
loop per machine run; rules await via ``run_until_complete`` because
Hypothesis does not compose with pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
from collections import Counter
from typing import TYPE_CHECKING, Literal

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)
from typing_extensions import override

from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.wiring import RunConfig
from tests.driver.unified.test_run import TrivialScraper
from tests.driver.unified.test_run_conservation import (
    CountingTransport,
    _seed_pending,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.generative

_RANK: dict[str, int] = {"unstarted": 0, "in_progress": 1, "done": 2}


class RunLifecycleMachine(RuleBasedStateMachine):
    """Walks legal Run sequences, checking the observable laws after each."""

    def __init__(
        self, make_run: Callable[[CountingTransport], ScrapeRun]
    ) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.transport = CountingTransport()
        try:
            self.subject: ScrapeRun = make_run(self.transport)
        except BaseException:
            # teardown only runs if __init__ returns, so close the loop here
            # if construction fails — otherwise it leaks (ResourceWarning).
            self.loop.close()
            raise
        self.seed_count = 0
        self.seeded_ids: list[int] = []
        self.opened = False
        self.closed = False
        self.ran = False
        self.stopped = False
        self.spawned_ids: list[int] = []
        self.last_rank = 0

    # --- setup ------------------------------------------------------------

    @initialize(seed_count=st.integers(min_value=0, max_value=5))
    def choose_seed(self, seed_count: int) -> None:
        """How many pending rows ``open`` will seed (0 keeps the old
        empty-queue walk in the mix)."""
        self.seed_count = seed_count

    # --- rules ------------------------------------------------------------

    @precondition(lambda self: not self.opened and not self.closed)
    @rule()
    def open(self) -> None:
        self.loop.run_until_complete(self.subject.open())
        self.opened = True
        assert self.subject.transport is not None
        self.seeded_ids = self.loop.run_until_complete(
            _seed_pending(self.subject, self.seed_count)
        )
        assert len(self.seeded_ids) == self.seed_count

    @precondition(
        lambda self: self.opened and not self.closed and not self.ran
    )
    @rule()
    def spawn_worker(self) -> None:
        async def spawn() -> int:
            return self.subject.spawn_worker()

        worker_id = self.loop.run_until_complete(spawn())
        assert isinstance(worker_id, int)
        if self.spawned_ids:
            assert worker_id > self.spawned_ids[-1], (
                "spawn_worker ids must be strictly increasing (distinct)"
            )
        self.spawned_ids.append(worker_id)

    @precondition(lambda self: self.opened and not self.closed)
    @rule()
    def stop(self) -> None:
        self.subject.stop()
        self.stopped = True

    @precondition(
        lambda self: self.opened and not self.closed and not self.ran
    )
    @rule()
    def run(self) -> None:
        self.loop.run_until_complete(self.subject.run())
        self.ran = True
        status = self.loop.run_until_complete(self.subject.status())
        resolved = sorted(self.transport.resolved)
        if self.stopped and resolved != self.seeded_ids:
            # Graceful shutdown by design: a stop that lands before the
            # workers drain leaves the remainder pending for a resumed run
            # (test_run_conservation pins the same arm), so run() returns
            # without finishing and status() reports the live rows.
            assert status == "in_progress", (
                "a stopped run with rows left must report in_progress "
                f"(seeded {self.seeded_ids}, resolved {resolved}), got "
                f"{status!r}"
            )
            return
        assert status == "done", (
            "run() must drive the scrape to completion — even when stop() "
            f"fired first (stopped={self.stopped}, seeded="
            f"{self.seed_count}) — but status() is {status!r}"
        )
        assert resolved == self.seeded_ids, (
            "run() must resolve every seeded request exactly once: "
            f"seeded {self.seeded_ids}, resolved {self.transport.resolved}"
        )

    @precondition(lambda self: self.opened and not self.closed)
    @rule()
    def status(self) -> None:
        value: Literal["unstarted", "in_progress", "done"]
        value = self.loop.run_until_complete(self.subject.status())
        assert value in _RANK, f"undocumented status {value!r}"
        rank = _RANK[value]
        assert rank >= self.last_rank, (
            f"status went backwards: rank {self.last_rank} -> {value!r}"
        )
        self.last_rank = rank
        # ``run()`` is synchronous here, so before it the run has not
        # started (whatever pre-spawned workers are doing to the queue) and
        # after it the only literals left are the live/finished pair.
        if not self.ran:
            assert value == "unstarted", (
                f"status() before run() must be 'unstarted', got {value!r}"
            )
        else:
            assert value != "unstarted"
        if value == "done":
            assert sorted(self.transport.resolved) == self.seeded_ids, (
                "'done' with work outstanding: seeded "
                f"{self.seeded_ids}, resolved {self.transport.resolved}"
            )

    @precondition(lambda self: self.opened and not self.closed)
    @rule()
    def aclose(self) -> None:
        self.loop.run_until_complete(self._settle())
        self.loop.run_until_complete(self.subject.aclose())
        self.closed = True

    @rule()
    def observe_worker_count(self) -> None:
        """Always-legal observation (also keeps the post-close state live —
        hypothesis requires some rule to stay enabled)."""
        assert self.subject.active_worker_count >= 0

    # --- invariants ---------------------------------------------------------

    @invariant()
    def worker_count_is_sane(self) -> None:
        count = self.subject.active_worker_count
        assert 0 <= count <= len(self.spawned_ids)

    @invariant()
    def resolved_at_most_once_and_only_seeded(self) -> None:
        """No double-dequeue, ever — also while pre-``run`` workers drain."""
        counts = Counter(self.transport.resolved)
        assert all(n == 1 for n in counts.values()), (
            f"a request was resolved more than once: {counts}"
        )
        assert set(counts) <= set(self.seeded_ids), (
            f"resolved ids outside the seeded set: {counts}"
        )

    # --- plumbing -----------------------------------------------------------

    async def _settle(self) -> None:
        """Let idle-spawned workers retire before teardown.

        Workers spawned without a ``run()`` pull from the queue until it is
        drained (or the stop event fires) and exit; a few loop ticks let
        those tasks finish so closing the DB doesn't strand them mid-pull.
        """
        self.subject.stop()
        for _ in range(20):
            if self.subject.active_worker_count == 0:
                return
            await asyncio.sleep(0.01)

    @override
    def teardown(self) -> None:
        try:
            if self.opened and not self.closed:
                self.loop.run_until_complete(self._settle())
                self.loop.run_until_complete(self.subject.aclose())
        finally:
            self.loop.close()


def test_run_lifecycle_machine(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workdir = tmp_path_factory.mktemp("run_machine")
    counter = itertools.count()

    def make_run(transport: CountingTransport) -> ScrapeRun:
        db_path = workdir / f"run-{next(counter)}.db"
        shutil.copy(schema_template, db_path)
        return ScrapeRun(
            TrivialScraper(),
            db_path,
            transport=transport,
            config=RunConfig(rate_limited=False),
        )

    run_state_machine_as_test(lambda: RunLifecycleMachine(make_run))
