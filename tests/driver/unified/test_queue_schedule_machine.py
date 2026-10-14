"""Generative schedule rig for the SQLite request queue.

A Hypothesis ``RuleBasedStateMachine`` drives the real ``SQLManager`` queue
surface — ``insert_request``, ``dequeue_next_request``, ``schedule_retry``,
``seconds_until_next_pending`` — over one fresh DB file per machine, against
a model of the pending rows (``priority``, insertion counter, eligibility
window) and checks, after every step:

- ``seconds_until_next_pending()`` is None iff no pending row exists;
- it is ``0.0`` iff a dequeue right now succeeds (a ready row), and never
  larger than the soonest pending row's remaining delay;
- a dequeue returns exactly the model's minimum by ``(priority asc,
  insertion order asc)`` among the eligible rows — a retried row keeps its
  original insertion slot;
- no id is dequeued twice without an intervening ``schedule_retry``;
- a dequeued row is ``IN_PROGRESS`` with ``started_at`` stamped;
- a retry with delay ``d`` is not dequeued before ``d`` has elapsed.

Time. The queue gates on ``datetime('now','subsec')`` against a
``started_at`` written as ``strftime(..., 'now', '+d seconds')`` — SQLite's
wall clock, millisecond resolution, read inside each statement. The model
cannot see that instant, only the Python clock before and after the call,
so every retried row carries an eligibility *window*: it is certainly not
eligible before ``t_before + d - TOL`` and certainly eligible after
``t_after + d + TOL``, where ``TOL`` covers the millisecond rounding of the
two stamps. Laws are stated against the window: a dequeued row must be
possibly eligible, and nothing ordered before it may be certainly eligible;
a None result means no row was certainly eligible. Rows that were never
retried have no window (``started_at`` NULL), so for them — the majority —
the ordering law is exact. The ``wait`` rule sleeps for real, only when a
retried row is still inside its delay, and only until it is certainly
eligible.

Determinism. Hypothesis replays a failing sequence and requires generation
to be identical, so nothing the generator sees may depend on the clock or
on a time-dependent DB answer: every rule is always enabled (a rule with
nothing to act on is a no-op), draws come from fixed strategies (a retry
picks its row by index modulo the in-progress set), and ``wait`` settles
retried rows in the order they were scheduled rather than by who is
soonest. The laws themselves are checked against the real clock.
"""

from __future__ import annotations

import asyncio
import itertools
import math
import shutil
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)
from typing_extensions import override

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.sql_manager import RequestInsert, SQLManager
from tests.db_queries import get_request_row

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

pytestmark = pytest.mark.generative

#: Slack for the two millisecond-rounded stamps (schedule and gate), plus
#: the Python-vs-SQLite clock read skew. Seconds.
TOL = 0.003

_DELAYS = [0.0, 0.05, 0.2, 1.0]


@dataclass(frozen=True)
class _Row:
    """A pending row as the model sees it.

    ``not_before`` / ``ready_by`` bound the instant (``time.monotonic``)
    the queue's gate opens for the row: ``-inf`` for a never-retried row
    (``started_at`` NULL, eligible at once).
    """

    priority: int
    seq: int
    not_before: float = -math.inf
    ready_by: float = -math.inf
    delay: float | None = None
    #: Order in which retries were scheduled (``wait`` settles in this order).
    retry_seq: int | None = None
    #: ``wait`` has slept past ``ready_by`` for this row.
    settled: bool = False

    @property
    def key(self) -> tuple[int, int]:
        return (self.priority, self.seq)

    def certainly_eligible(self, at: float) -> bool:
        return self.ready_by <= at

    def possibly_eligible(self, at: float) -> bool:
        return self.not_before <= at


class QueueScheduleMachine(RuleBasedStateMachine):
    """Random insert / dequeue / retry / wait sequences over a real queue."""

    def __init__(self, make_db_path: Callable[[], Path]) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        try:
            engine, factory = self.loop.run_until_complete(
                init_database(make_db_path())
            )
        except BaseException:
            self.loop.close()
            raise
        self.engine = engine
        self.db = SQLManager(engine, factory)
        self.pending: dict[int, _Row] = {}
        self.in_progress: dict[int, _Row] = {}
        self.seq = itertools.count()
        self.urls = itertools.count()
        self.retry_seq = itertools.count()

    # --- model helpers ------------------------------------------------------

    def _certainly_eligible(self, at: float) -> list[tuple[int, _Row]]:
        return sorted(
            (
                (rid, row)
                for rid, row in self.pending.items()
                if row.certainly_eligible(at)
            ),
            key=lambda item: item[1].key,
        )

    # --- rules ----------------------------------------------------------------

    @rule(priority=st.integers(min_value=0, max_value=9))
    def insert(self, priority: int) -> None:
        params = RequestInsert(
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url=f"https://schedule.test/{next(self.urls)}",
            step="parse",
            priority=priority,
        )
        result = self.loop.run_until_complete(self.db.insert_request(params))
        assert result.inserted, "distinct URLs, no dedup key: must be new"
        assert result.request_id not in self.pending
        assert result.request_id not in self.in_progress
        self.pending[result.request_id] = _Row(priority, next(self.seq))

    @rule()
    def dequeue(self) -> None:
        t0 = time.monotonic()
        wait = self.loop.run_until_complete(
            self.db.seconds_until_next_pending()
        )
        t1 = time.monotonic()
        row = self.loop.run_until_complete(self.db.dequeue_next_request())
        t2 = time.monotonic()

        # seconds_until_next_pending vs. the dequeue that follows it.
        if wait is None:
            assert not self.pending
            assert row is None, f"None reported, yet dequeued {row.id}"
        elif wait == 0.0:
            assert row is not None, "0.0 reported, yet dequeue found nothing"
        elif row is not None:
            # A positive wait then a hit: the row must have crossed its
            # gate between the two statements, so the reported wait is at
            # most the time that passed.
            assert wait <= (t2 - t0) + TOL, (
                f"reported {wait}s to next pending, but {row.id} was "
                f"dequeued {t2 - t0:.4f}s later"
            )

        if row is None:
            certain = self._certainly_eligible(t1)
            assert not certain, (
                f"dequeue found nothing, but {certain[0][0]} "
                f"{certain[0][1]} was eligible"
            )
            return

        assert row.id in self.pending, (
            f"{row.id} dequeued while not pending "
            f"(in_progress={sorted(self.in_progress)})"
        )
        model = self.pending.pop(row.id)
        assert row.priority == model.priority
        assert model.possibly_eligible(t2), (
            f"{row.id} dequeued {model.not_before - t2:.4f}s before its "
            f"{model.delay}s retry delay elapsed"
        )
        ahead = [
            (rid, other)
            for rid, other in self._certainly_eligible(t1)
            if other.key < model.key
        ]
        assert not ahead, (
            f"dequeued {row.id} {model.key} ahead of eligible "
            f"{ahead[0][0]} {ahead[0][1].key}"
        )

        stored = self.loop.run_until_complete(get_request_row(self.db, row.id))
        assert stored is not None
        assert stored.status == RequestStatus.IN_PROGRESS
        assert stored.started_at is not None
        self.in_progress[row.id] = model

    @rule(
        index=st.integers(min_value=0, max_value=63),
        delay=st.sampled_from(_DELAYS),
    )
    def schedule_retry(self, index: int, delay: float) -> None:
        """Retry the ``index``-th (mod count) in-progress row; no-op if none."""
        claimed = sorted(self.in_progress)
        if not claimed:
            return
        request_id = claimed[index % len(claimed)]
        t0 = time.monotonic()
        self.loop.run_until_complete(
            self.db.schedule_retry(request_id, delay, delay, "boom")
        )
        t1 = time.monotonic()
        model = self.in_progress.pop(request_id)
        self.pending[request_id] = replace(
            model,
            not_before=t0 + delay - TOL,
            ready_by=t1 + delay + TOL,
            delay=delay,
            retry_seq=next(self.retry_seq),
            settled=False,
        )

    @rule()
    def wait(self) -> None:
        """Sleep until the earliest-scheduled unsettled retry is certainly
        eligible; no-op when every pending retry is already settled."""
        unsettled = [
            (rid, row)
            for rid, row in self.pending.items()
            if row.retry_seq is not None and not row.settled
        ]
        if not unsettled:
            return
        rid, row = min(unsettled, key=lambda item: item[1].retry_seq or 0)
        remaining = row.ready_by - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.pending[rid] = replace(row, settled=True)

    # --- invariants -----------------------------------------------------------

    @invariant()
    def next_pending_matches_model(self) -> None:
        t0 = time.monotonic()
        wait = self.loop.run_until_complete(
            self.db.seconds_until_next_pending()
        )
        t1 = time.monotonic()

        if not self.pending:
            assert wait is None, f"no pending rows, but reported {wait}"
            return
        assert wait is not None, "pending rows exist, but reported None"
        assert wait >= 0.0

        if self._certainly_eligible(t0):
            assert wait == 0.0, f"an eligible row exists, but reported {wait}s"
        if wait == 0.0:
            assert any(
                row.possibly_eligible(t1) for row in self.pending.values()
            ), "0.0 reported while every pending row is inside its delay"
        # Never later than the soonest row's remaining delay.
        bound = min(row.ready_by - t0 for row in self.pending.values())
        assert wait <= max(0.0, bound), (
            f"reported {wait}s, but a row is ready within {bound:.4f}s"
        )

    @invariant()
    def partition_is_disjoint(self) -> None:
        assert not (self.pending.keys() & self.in_progress.keys())

    # --- plumbing -------------------------------------------------------------

    @override
    def teardown(self) -> None:
        try:
            self.loop.run_until_complete(self.engine.dispose())
        finally:
            self.loop.close()


def test_queue_schedule_machine(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    workdir = tmp_path_factory.mktemp("queue_machine")
    counter = itertools.count()

    def make_db_path() -> Path:
        db_path = workdir / f"queue-{next(counter)}.db"
        shutil.copy(schema_template, db_path)
        return db_path

    run_state_machine_as_test(lambda: QueueScheduleMachine(make_db_path))
