"""Generative laws for the shared speculation engine.

Subject: :mod:`jkent.driver._speculation_support` — ``compute_seed_plan``
(the resume-aware seed plan) and the ``AsyncSpeculationSupport`` mixin's
``seed`` / ``_extend_speculation`` / ``track_outcome``.

The template is a stub :class:`~jkent.common.speculative.Speculative`
(a pydantic model, as ``@entry`` requires) parameterised by the seed range
``[start, stop)``, the ``should_advance`` policy and ``max_gap`` in 0..8.
``start`` is drawn from 0 upward, and a state's ``current_ceiling`` is
None (never seeded) or any ID at or above -1.

Which path drives ``track_outcome``: the mixin's own extension point. The
laws are about the seed/extend/track state machine, whose only external
effects are the two abstract hooks (``_enqueue_speculative``, ``_persist``).
An in-memory subclass records both, so every enqueued probe id and every
persisted snapshot is directly observable, and the real ``track_outcome``
and ``_extend_speculation`` bodies run unmodified. The DB-backed hooks of
``SpeculationManager`` are covered by ``test_speculation.py``; they add
nothing to these laws and would add a SQLite file per example.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from jkent.common.decorators import entry, step
from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver._speculation_support import (
    AsyncSpeculationSupport,
    SpeculationState,
    build_speculative_request,
    compute_seed_plan,
)

pytestmark = pytest.mark.generative

_MAX_GAP = 8


# --- Stub template + scraper -------------------------------------------------


class _Probe(BaseModel, Speculative):
    """Stub speculative id: seed ``[start, stop)``, window of ``gap``."""

    n: int = 0
    start: int = 1
    stop: int = 1
    should_advance: bool = True
    gap: int = 2

    def seed_range(self) -> range:
        return range(self.start, self.stop)

    def from_int(self, n: int) -> _Probe:
        return self.model_copy(update={"n": n})

    def max_gap(self) -> int:
        return self.gap


class _ProbeScraper(BaseScraper[dict[str, Any]]):
    """One speculative entry so ``build_speculative_request`` resolves."""

    @entry(dict)
    def probe(self, pid: _Probe) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"http://127.0.0.1/probe/{pid.n}"
            ),
            step="parse_probe",
        )

    @step
    def parse_probe(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={})


@st.composite
def _templates(draw: st.DrawFn) -> _Probe:
    return _Probe(
        start=draw(st.integers(min_value=0, max_value=12)),
        # May be <= start: an empty seed range is valid ("rely on the
        # advance window").
        stop=draw(st.integers(min_value=0, max_value=20)),
        should_advance=draw(st.booleans()),
        gap=draw(st.integers(min_value=0, max_value=_MAX_GAP)),
    )


def _state(
    template: _Probe, current_ceiling: int | None = None
) -> SpeculationState:
    return SpeculationState(
        func_name="probe:0",
        template=template,
        param_index=0,
        base_func_name="probe",
        current_ceiling=current_ceiling,
    )


# --- compute_seed_plan --------------------------------------------------------


_ceilings = st.none() | st.integers(min_value=-1, max_value=30)


@given(template=_templates(), current_ceiling=_ceilings)
def test_seed_plan_shape(
    template: _Probe, current_ceiling: int | None
) -> None:
    """ids strictly increasing and above the ceiling (at or above the seed
    start when unseeded); the new ceiling never regresses and equals the
    last id planned; stopped iff no window."""
    spec_state = _state(template, current_ceiling)
    ids, new_ceiling, stopped = compute_seed_plan(spec_state)

    assert all(a < b for a, b in zip(ids, ids[1:], strict=False)), ids
    floor = template.start if current_ceiling is None else current_ceiling + 1
    assert all(n >= floor for n in ids), (ids, current_ceiling)
    assert new_ceiling >= floor - 1
    if ids:
        assert new_ceiling == max(ids)
    has_window = template.should_advance and template.gap > 0
    assert stopped == (not has_window)
    # The plan is a pure function of the state: it must not touch it.
    assert spec_state.current_ceiling == current_ceiling
    assert spec_state.stopped is False


@given(
    template=_templates(),
    current_ceiling=_ceilings,
    rounds=st.integers(min_value=2, max_value=6),
)
def test_reseed_chain_never_replans_an_id(
    template: _Probe, current_ceiling: int | None, rounds: int
) -> None:
    """Feeding ``new_ceiling`` back as ``current_ceiling`` and replanning
    never yields an id an earlier round already planned (no double probe on
    resume), across a chain of re-seeds.

    The ceiling never falls and moves only as far as the plan: to its
    highest id, or — when nothing is planned — to a fixed point that
    replans to itself (it may first snap up to just below the template's
    range, whose lower ids are never probed).
    """
    seen: set[int] = set()
    ceiling: int | None = current_ceiling
    for _ in range(rounds):
        ids, new_ceiling, _stopped = compute_seed_plan(
            _state(template, ceiling)
        )
        clash = seen & set(ids)
        assert not clash, (
            f"re-seed from ceiling {ceiling} replanned {sorted(clash)}"
        )
        assert ceiling is None or new_ceiling >= ceiling
        if ids:
            assert new_ceiling == max(ids)
        else:
            replan = compute_seed_plan(_state(template, new_ceiling))
            assert replan[:2] == ([], new_ceiling)
        seen.update(ids)
        ceiling = new_ceiling


@given(
    template=_templates(),
    current_ceiling=_ceilings,
    highest=st.none() | st.integers(min_value=-1, max_value=30),
)
def test_replan_without_new_outcomes_plans_nothing(
    template: _Probe, current_ceiling: int | None, highest: int | None
) -> None:
    """Restarting with no outcome since the last plan probes nothing new.

    The seeded ceiling is persisted before any outcome, so a run killed and
    reopened N times must not push the window N×gap past the frontier: the
    window is anchored to the seed range and the highest success, not to the
    ceiling it already reached.
    """
    spec_state = _state(template, current_ceiling)
    spec_state.highest_successful_id = highest
    _ids, new_ceiling, _stopped = compute_seed_plan(spec_state)

    resumed = _state(template, new_ceiling)
    resumed.highest_successful_id = highest
    ids, ceiling, _stopped = compute_seed_plan(resumed)
    assert ids == []
    assert ceiling == new_ceiling


def test_window_follows_highest_success_on_resume() -> None:
    """A resume whose highest success is near the ceiling tops the window up
    to ``gap`` past it, probing only ids above the ceiling."""
    template = _Probe(start=0, stop=2, gap=3)
    spec_state = _state(template, current_ceiling=4)
    spec_state.highest_successful_id = 4
    ids, ceiling, stopped = compute_seed_plan(spec_state)
    assert ids == [5, 6, 7]
    assert ceiling == 7
    assert stopped is False


@pytest.mark.parametrize(
    "template",
    [
        pytest.param(
            _Probe(start=0, stop=1, should_advance=False), id="seed-only-0"
        ),
        pytest.param(_Probe(start=0, stop=0, gap=1), id="window-only-0"),
    ],
)
def test_resume_after_seeding_only_id_zero_plans_nothing_new(
    template: _Probe,
) -> None:
    """A resumed state whose ceiling is 0 does not re-plan ID 0."""
    ids, ceiling, _stopped = compute_seed_plan(_state(template))
    assert ids == [0]
    assert ceiling == 0

    resumed_ids, _ceiling, _stopped = compute_seed_plan(
        _state(template, ceiling)
    )
    assert 0 not in resumed_ids


# --- track_outcome ------------------------------------------------------------


@dataclass(frozen=True)
class _Snapshot:
    highest: int | None
    failures: int
    ceiling: int | None
    stopped: bool

    @classmethod
    def of(cls, spec_state: SpeculationState) -> _Snapshot:
        return cls(
            spec_state.highest_successful_id,
            spec_state.consecutive_failures,
            spec_state.current_ceiling,
            spec_state.stopped,
        )


class _MemorySpeculation(AsyncSpeculationSupport):
    """The mixin over recording hooks: probes to a list, persists to a log."""

    def __init__(self, spec_state: SpeculationState) -> None:
        self.scraper = _ProbeScraper()
        self._speculation_state = {spec_state.func_name: spec_state}
        self._speculation_lock = asyncio.Lock()
        self.enqueued: list[int] = []
        self.persisted: list[_Snapshot] = []

    async def _enqueue_speculative(self, request: Request) -> None:
        assert request.is_speculative
        assert request.speculative_index is not None
        self.enqueued.append(request.speculative_index)

    async def _persist(self, spec_state: SpeculationState) -> None:
        spec_state.tracking_id = 1
        self.persisted.append(_Snapshot.of(spec_state))


def _response(req: Request, status: int) -> Response:
    return Response(
        status_code=status,
        headers={},
        content=b"",
        text="",
        url=req.request.url,
        request=req,
    )


# Each script step picks the ``rank``-th (mod count) outstanding probe —
# enqueued, not yet reported — so delivery order is arbitrary relative to
# enqueue order, every reported id is one the engine actually probed, and
# each probe is reported at most once. Shrinks toward rank 0 / failure.
_scripts = st.lists(
    st.tuples(st.integers(min_value=0, max_value=63), st.booleans()),
    max_size=40,
)


async def _drive(template: _Probe, script: list[tuple[int, bool]]) -> None:
    spec_state = _state(template)
    support = _MemorySpeculation(spec_state)
    gap = template.gap

    await support.seed()
    plan_ids, plan_ceiling, plan_stopped = compute_seed_plan(_state(template))
    assert support.enqueued == plan_ids
    assert (spec_state.current_ceiling, spec_state.stopped) == (
        plan_ceiling,
        plan_stopped,
    )
    seed_count = len(plan_ids)
    # The seeded ceiling is persisted before any outcome arrives.
    assert support.persisted[-1] == _Snapshot.of(spec_state)
    assert spec_state.tracking_id is not None
    tracking_id = spec_state.tracking_id

    reported: set[int] = set()
    successes: set[int] = set()
    before = _Snapshot.of(spec_state)
    for rank, ok in script:
        outstanding = sorted(set(support.enqueued) - reported)
        if not outstanding:
            break
        n = outstanding[rank % len(outstanding)]
        reported.add(n)
        if ok:
            successes.add(n)

        enqueued_before = len(support.enqueued)
        req = build_speculative_request(support.scraper, spec_state, n)
        assert req.speculation_tracking_id == tracking_id
        await support.track_outcome(req, _response(req, 200 if ok else 404))
        after = _Snapshot.of(spec_state)
        # Seeding set the ceiling, and nothing after it clears one.
        assert before.ceiling is not None
        assert after.ceiling is not None
        enqueued_after = len(support.enqueued)

        # highest_successful_id: monotone, and exactly the max success seen
        # (None until the first success).
        assert after.highest == max(successes, default=None)
        if before.highest is not None:
            assert after.highest is not None
            assert after.highest >= before.highest

        # consecutive_failures: any success resets; a failure counts only
        # when it lands above the watermark that held before it.
        if ok:
            assert after.failures == 0
        elif before.highest is None or n > before.highest:
            assert after.failures == before.failures + 1
        else:
            assert after.failures == before.failures

        # stopped is absorbing, and gap failures force it.
        if before.stopped:
            assert after.stopped
        if after.failures >= gap:
            assert after.stopped

        # ceiling monotone; probes only ever open above it and never twice.
        assert after.ceiling >= before.ceiling
        new_ids = support.enqueued[enqueued_before:]
        assert all(n > before.ceiling for n in new_ids), (new_ids, before)
        assert all(n <= after.ceiling for n in support.enqueued)
        assert len(support.enqueued) == len(set(support.enqueued))

        # Termination: a failure never opens probes; a success opens at
        # most one window of ``gap``; the ceiling stays within two windows
        # of the watermark, so the total is bounded by the successes.
        if not ok:
            assert enqueued_after == enqueued_before
        else:
            assert enqueued_after - enqueued_before in (0, gap)
        if before.stopped:
            assert enqueued_after == enqueued_before
        if after.highest is None:
            assert after.ceiling == plan_ceiling
        else:
            assert after.ceiling <= max(plan_ceiling, after.highest + 2 * gap)
        assert len(support.enqueued) <= seed_count + len(successes) * gap

        # Every outcome is persisted, as the state stands after tracking.
        assert support.persisted[-1] == after

        before = after

    # One persist before planning, one after, then one per outcome.
    assert len(support.persisted) == 2 + len(reported)


@given(template=_templates(), script=_scripts)
def test_track_outcome_laws(
    template: _Probe, script: list[tuple[int, bool]]
) -> None:
    """Outcomes delivered in arbitrary order keep the tracking laws."""
    asyncio.run(_drive(template, script))
