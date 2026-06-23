"""Generative lifecycle rig for the ``Run`` protocol (``ScrapeRun``).

A Hypothesis ``RuleBasedStateMachine`` walks random *legal* sequences of
the ``Run`` surface — ``open``, ``spawn_worker``, ``status``, ``stop``,
``run``, ``aclose`` (legality enforced with preconditions: open once,
spawn only while open and before ``run``, close once) — and checks the
laws the contract states for every ordering:

- ``status()`` only returns the three documented literals and is
  monotone along a legal sequence: ``unstarted -> in_progress -> done``,
  never backwards;
- ``spawn_worker()`` returns strictly increasing (hence distinct) ids;
- ``active_worker_count`` never exceeds the number spawned and never
  goes negative (workers retire themselves on idle, so it may drop);
- ``transport`` is non-None from ``open`` onward;
- ``run()`` returns and lands the run in ``"done"`` — including when
  ``stop()`` fired first (graceful-shutdown path).

The subject is the real ``ScrapeRun`` over a fresh temp DB (copied from
the session schema template so ``open`` finds the schema built), the trivial
scraper, and the spy transport from ``test_run.py`` — the queue stays
empty, so a hit ``resolve`` is an assertion failure. One event loop per
machine run; rules await via ``run_until_complete`` because Hypothesis
does not compose with pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
from typing import TYPE_CHECKING, Literal

import pytest
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from tests.driver.unified.test_run import (
    SpyTransport,
    TrivialScraper,
    _NoSignalScrapeRun,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jkent.driver.unified_driver.run import ScrapeRun

pytestmark = pytest.mark.generative

_RANK: dict[str, int] = {"unstarted": 0, "in_progress": 1, "done": 2}


class RunLifecycleMachine(RuleBasedStateMachine):
    """Walks legal Run sequences, checking the observable laws after each."""

    def __init__(self, make_run: object) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        try:
            self.subject: ScrapeRun = make_run()  # type: ignore[operator]
        except BaseException:
            # teardown only runs if __init__ returns, so close the loop here
            # if construction fails — otherwise it leaks (ResourceWarning).
            self.loop.close()
            raise
        self.opened = False
        self.closed = False
        self.ran = False
        self.spawned_ids: list[int] = []
        self.last_rank = 0

    # --- rules ------------------------------------------------------------

    @precondition(lambda self: not self.opened and not self.closed)
    @rule()
    def open(self) -> None:
        self.loop.run_until_complete(self.subject.open())
        self.opened = True
        assert self.subject.transport is not None

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

    @precondition(
        lambda self: self.opened and not self.closed and not self.ran
    )
    @rule()
    def run(self) -> None:
        self.loop.run_until_complete(self.subject.run())
        self.ran = True
        status = self.loop.run_until_complete(self.subject.status())
        assert status == "done", (
            "run() must drive the scrape to completion — even when stop() "
            f"fired first — but status() is {status!r}"
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

    # --- plumbing -----------------------------------------------------------

    async def _settle(self) -> None:
        """Let idle-spawned workers retire before teardown.

        Workers spawned without a ``run()`` pull once from the empty
        queue and exit; a few loop ticks let those tasks finish so
        closing the DB doesn't strand them mid-pull.
        """
        self.subject.stop()
        for _ in range(20):
            if self.subject.active_worker_count == 0:
                return
            await asyncio.sleep(0.01)

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

    def make_run() -> ScrapeRun:
        db_path = workdir / f"run-{next(counter)}.db"
        shutil.copy(schema_template, db_path)
        return _NoSignalScrapeRun(
            TrivialScraper(),
            db_path,
            transport=SpyTransport(),
            rate_limited=False,
            resume=False,
        )

    run_state_machine_as_test(lambda: RunLifecycleMachine(make_run))
