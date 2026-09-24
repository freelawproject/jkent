"""Speculation end-to-end for the unified driver.

A ``@speculate`` scraper probes a sequential id against ``/spec/{n}`` (the
mock server returns 200 for n <= 3, then a persistent 404). This exercises:

- ``SpeculationManager`` discovery/seed/track/persist in isolation against a
  real in-memory ``SQLManager`` + ``RequestQueue`` (no transport);
- a full ``ScrapeRun`` against the live server: the right probes are attempted,
  results land, and ``speculation_tracking`` reflects the final state.
"""

from __future__ import annotations

import asyncio
import sqlite3
import string
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel
from sqlalchemy import text as sqlalchemy_text

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
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import (
    RequestStatus,
    SpeculationOutcome,
)
from jkent.driver.database_engine.sql_manager import (
    SpeculationStateRecord,
    SQLManager,
)
from jkent.driver.unified_driver import ScrapeRun
from jkent.driver.unified_driver.persistence import RequestQueue
from jkent.driver.unified_driver.speculation import SpeculationManager
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks

# --- A speculative scraper -----------------------------------------------


class _SpecId(BaseModel, Speculative):
    """Speculative id: seed empty, advance a window of ``gap``."""

    n: int
    soft_max: int = 0
    should_advance: bool = True
    gap: int = 2

    def seed_range(self) -> range:
        return range(self.n, self.soft_max)

    def from_int(self, n: int) -> _SpecId:
        return _SpecId(
            n=n,
            soft_max=self.soft_max,
            should_advance=self.should_advance,
            gap=self.gap,
        )

    def max_gap(self) -> int:
        return self.gap


class _SpecScraper(BaseScraper[dict[str, Any]]):
    """entry probes /spec/{n}; parse records the surviving id."""

    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_spec(self, sid: _SpecId) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/spec/{sid.n}"
            ),
            step="parse_spec",
        )

    @step
    def parse_spec(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        n = int(response.url.rsplit("/", 1)[-1])
        yield ParsedData(data={"n": n})


_SEED = [{"fetch_spec": {"sid": {"n": 1, "gap": 2}}}]


def _make_scraper(server_url: str) -> _SpecScraper:
    scraper = _SpecScraper()
    scraper.base = server_url
    return scraper


# --- Unit-ish: SpeculationManager against a real in-memory DB ------------


async def _build_manager(
    db_path: Path, seed_params: list[dict[str, dict[str, Any]]]
) -> tuple[SpeculationManager, _SpecScraper, SQLManager]:
    engine, factory = await init_database(db_path)
    db = SQLManager(engine, factory)
    scraper = _SpecScraper()
    # initial_seed populates _speculation_templates for discovery.
    list(scraper.initial_seed(seed_params))
    queue = RequestQueue(db)
    manager = SpeculationManager(scraper, queue, db, seed_params=seed_params)
    manager.discover()
    await manager.load()
    return manager, scraper, db


def _spec_request(
    scraper: _SpecScraper, n: int, tracking_id: int = 1
) -> Request:
    req = scraper.fetch_spec(_SpecId(n=n, gap=2))
    return req.speculative(tracking_id, n)


def _response(req: Request, status: int) -> Response:
    return Response(
        status_code=status,
        headers={},
        content=b"",
        text="",
        url=req.request.url,
        request=req,
    )


async def test_seed_enqueues_initial_window(tmp_path: Path) -> None:
    """seed() enqueues the advance window of speculative probes."""
    manager, _scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    assert manager.has_state

    await manager.seed()

    pending = await db.count_pending_requests()
    # seed_range(1, 0) empty; advance window gap=2 → [1, 2].
    assert pending == 2
    state = manager._speculation_state["fetch_spec:0"]
    assert state.current_ceiling == 2


# Round-trippable field strategies. We only vary fields the queue's
# serialize/deserialize is designed to preserve by value (see _retuple's
# docstring); the lossy ones (params folded into the URL, data/json/files
# re-encoded) are left at their defaults so equality on ``request`` is exact.
_word = st.text(
    alphabet=string.ascii_letters + string.digits, min_size=1, max_size=8
)
_str_dict = st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6),
    _word,
    min_size=1,
    max_size=3,
)
# Finite floats only: json round-trips them exactly; NaN/inf would break
# equality (NaN != NaN) or the float repr.
_finite = st.floats(
    min_value=0, max_value=10_000, allow_nan=False, allow_infinity=False
)


@st.composite
def _round_trippable_requests(draw: st.DrawFn) -> Request:
    """A speculative ``Request`` exercising every by-value-preserved field."""
    params = HTTPRequestParams(
        method=draw(st.sampled_from(HttpMethod)),
        url="https://example.com/" + draw(_word),
        headers=draw(st.none() | _str_dict),
        cookies=draw(st.none() | _str_dict),
        timeout=draw(st.none() | _finite | st.tuples(_finite, _finite)),
        # verify: True | False | a CA-bundle path. The literal "false" is
        # rejected by the serializer (ambiguous with verify=False), so exclude it.
        verify=draw(st.booleans() | _word.filter(lambda s: s != "false")),
    )
    req = Request(
        request=params,
        step="parse_spec",
        current_location=draw(st.sampled_from(["", "https://example.com"])),
        rate_limit=draw(st.sampled_from([None, "none"])),
        reseedable=draw(st.none() | st.booleans()),
    )
    return req.speculative(
        # Placeholder tracking id; _enqueue_then_dequeue re-stamps it with the
        # real row id once it has created the tracking row.
        0,
        draw(st.integers(min_value=1, max_value=10_000)),
    )


async def _enqueue_then_dequeue(req: Request) -> tuple[Request, Request]:
    """Enqueue ``req`` as a speculative probe and read it back from a fresh DB."""
    with tempfile.TemporaryDirectory() as d:
        engine, factory = await init_database(Path(d) / "rt.db")
        try:
            db = SQLManager(engine, factory)
            manager = SpeculationManager(_SpecScraper(), RequestQueue(db), db)
            # A probe row carries a foreign key to its tracking row, so the
            # row has to exist before the insert — as it does in the real
            # seed path, where seed() persists the state first.
            tracking_id = await db.save_speculation_state(
                SpeculationStateRecord(func_name="fetch_spec:0")
            )
            assert req.speculative_index is not None
            req = req.speculative(tracking_id, req.speculative_index)
            await manager._enqueue_speculative(req)
            dequeued = await RequestQueue(db).get_next_request()
            assert dequeued is not None
            return req, dequeued[1]
        finally:
            await engine.dispose()


@pytest.mark.generative
@settings(deadline=None)
@given(req=_round_trippable_requests())
def test_enqueue_speculative_preserves_all_request_fields(
    req: Request,
) -> None:
    """A speculative probe preserves every serialized field on enqueue.

    Regression: ``_enqueue_speculative`` once hand-listed a subset of the
    serialized keys, silently dropping fields like ``timeout`` and
    ``rate_limit``. It now spreads the full serialized key set, so any
    by-value-preserved field must survive insert + dequeue. Hypothesis varies
    them all so a future re-narrowing of that spread fails here.
    """
    req, restored = asyncio.run(_enqueue_then_dequeue(req))
    # Full HTTPRequestParams equality covers method, url, headers, cookies,
    # timeout, and verify at once.
    assert restored.request == req.request
    assert restored.current_location == req.current_location
    assert restored.rate_limit == req.rate_limit
    assert restored.reseedable == req.reseedable
    assert restored.is_speculative is True
    assert restored.speculation_tracking_id == req.speculation_tracking_id
    assert restored.speculative_index == req.speculative_index


async def test_speculative_probes_are_never_deduplicated(
    tmp_path: Path,
) -> None:
    """Two identical probes both land: a deduplicated probe would never run,
    so its outcome would never reach the tracker."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    before = await db.count_pending_requests()
    state = manager._speculation_state["fetch_spec:0"]
    assert state.tracking_id is not None

    probe = _spec_request(scraper, 1, state.tracking_id)
    await manager._enqueue_speculative(probe)
    await manager._enqueue_speculative(probe)

    assert await db.count_pending_requests() == before + 2


async def test_success_advances_and_extends(tmp_path: Path) -> None:
    """A successful probe bumps highest_successful_id and extends the window."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    assert state.current_ceiling == 2
    assert state.tracking_id is not None

    # Success at 2 (== ceiling, within gap of ceiling) → extend to 4.
    probe = _spec_request(scraper, 2, state.tracking_id)
    outcome = await manager.track_outcome(probe, _response(probe, 200))

    assert outcome is SpeculationOutcome.HIT

    assert state.highest_successful_id == 2
    assert state.consecutive_failures == 0
    assert state.current_ceiling == 4
    # Persisted to DB.
    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].highest_successful_id == 2
    assert saved["fetch_spec:0"].current_ceiling == 4


async def test_failure_stops_after_max_gap(tmp_path: Path) -> None:
    """SpeculationHTTPFailure outcomes record failures and stop extension."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    state.highest_successful_id = 3
    # Bound to a local: the awaits below can in principle reassign the
    # attribute, so the narrowing does not survive them.
    tracking_id = state.tracking_id
    assert tracking_id is not None

    # Two consecutive failures beyond watermark (gap=2) → stopped.
    probe = _spec_request(scraper, 4, tracking_id)
    outcome = await manager.track_outcome(probe, _response(probe, 404))
    assert outcome is SpeculationOutcome.MISS
    assert state.consecutive_failures == 1
    assert state.stopped is False
    assert manager.has_stopped(probe) is False

    probe = _spec_request(scraper, 5, tracking_id)
    outcome = await manager.track_outcome(probe, _response(probe, 404))
    assert outcome is SpeculationOutcome.STOPPED
    assert state.consecutive_failures == 2
    assert state.stopped is True
    # type-checkers carry the earlier `stopped is False` narrowing past
    # track_outcome(), which really does flip it.
    assert manager.has_stopped(_spec_request(scraper, 6, tracking_id))  # type: ignore[unreachable]

    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].stopped is True
    assert saved["fetch_spec:0"].consecutive_failures == 2

    # A later miss is a plain miss: only one probe stops the template.
    probe = _spec_request(scraper, 6, tracking_id)
    outcome = await manager.track_outcome(probe, _response(probe, 404))
    assert outcome is SpeculationOutcome.MISS


async def test_outcome_for_an_untracked_state_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A probe whose tracking row no state owns is not dropped silently.

    The worker marks the probe completed either way, so the log line is the
    only trace that its outcome never reached the speculation window.
    """
    manager, scraper, _db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    assert state.tracking_id is not None

    probe = _spec_request(scraper, 2, state.tracking_id + 99)
    with caplog.at_level("WARNING"):
        outcome = await manager.track_outcome(probe, _response(probe, 200))

    assert outcome is None
    assert manager.has_stopped(probe) is False

    assert state.highest_successful_id is None
    assert any(
        str(state.tracking_id + 99) in r.getMessage() for r in caplog.records
    )


async def test_soft_failure_2xx_counts_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2xx the scraper rejects via actually_successful is a failure."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    assert state.tracking_id is not None

    def _reject(_response: Response) -> bool:
        return False

    monkeypatch.setattr(scraper, "actually_successful", _reject)
    pending_before = await db.count_pending_requests()

    probe = _spec_request(scraper, 2, state.tracking_id)
    outcome = await manager.track_outcome(probe, _response(probe, 200))

    assert outcome is SpeculationOutcome.MISS
    assert state.highest_successful_id is None
    assert state.consecutive_failures == 1
    assert state.current_ceiling == 2
    assert await db.count_pending_requests() == pending_before


_SEED_FROM_ZERO = [{"fetch_spec": {"sid": {"n": 0, "gap": 2}}}]


async def test_fresh_state_has_no_success(tmp_path: Path) -> None:
    """Before any probe succeeds, the state and its row record no success."""
    manager, _scraper, db = await _build_manager(
        tmp_path / "u.db", _SEED_FROM_ZERO
    )
    await manager.seed()

    assert (
        manager._speculation_state["fetch_spec:0"].highest_successful_id
        is None
    )
    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].highest_successful_id is None


@pytest.mark.parametrize(
    ("status", "highest", "failures"),
    [
        pytest.param(200, 0, 0, id="success"),
        pytest.param(404, None, 1, id="failure"),
    ],
)
async def test_outcome_at_id_zero_is_counted(
    tmp_path: Path, status: int, highest: int | None, failures: int
) -> None:
    """A range starting at 0 counts ID 0's outcome like any other ID's."""
    manager, scraper, db = await _build_manager(
        tmp_path / "u.db", _SEED_FROM_ZERO
    )
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    tracking_id = state.tracking_id
    assert tracking_id is not None

    probe = _spec_request(scraper, 0, tracking_id)
    await manager.track_outcome(probe, _response(probe, status))

    assert state.highest_successful_id == highest
    assert state.consecutive_failures == failures
    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].highest_successful_id == highest
    assert saved["fetch_spec:0"].consecutive_failures == failures


async def test_probes_point_at_their_tracking_row(tmp_path: Path) -> None:
    """Seeding creates the tracking row before the probes that reference it.

    ``requests.speculation_tracking_id`` is a real foreign key and the engine
    opens databases with ``PRAGMA foreign_keys=ON``, so a probe inserted ahead
    of its ``speculation_tracking`` row does not merely lose provenance — the
    insert fails. This pins the persist-before-probe ordering ``seed`` establishes.
    """
    manager, _scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()

    state = manager._speculation_state["fetch_spec:0"]
    assert state.tracking_id is not None

    async with db.session_factory() as session:
        rows = (
            await session.execute(
                sqlalchemy_text(
                    "SELECT speculation_tracking_id, speculative_index "
                    "FROM requests WHERE is_speculative = 1 "
                    "ORDER BY speculative_index"
                )
            )
        ).all()
    # The advance window [1, 2], each pointing at the one tracking row.
    assert rows == [(state.tracking_id, 1), (state.tracking_id, 2)]

    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].id == state.tracking_id


async def test_resume_after_seed_does_not_reseed(tmp_path: Path) -> None:
    """A run that seeded and stopped before any outcome resumes with nothing
    new to probe: its window is already queued, and no outcome has moved the
    frontier."""
    db_path = tmp_path / "u.db"
    manager, _scraper, _db = await _build_manager(db_path, _SEED)
    await manager.seed()

    resumed, _scraper2, db2 = await _build_manager(db_path, _SEED)
    await resumed.seed()

    async with db2.session_factory() as session:
        indices = (
            (
                await session.execute(
                    sqlalchemy_text(
                        "SELECT speculative_index FROM requests "
                        "WHERE is_speculative = 1 ORDER BY speculative_index"
                    )
                )
            )
            .scalars()
            .all()
        )
    # First seed: window [1, 2]; the resume adds nothing past it.
    assert indices == [1, 2]


async def test_resume_rehydrates_tracking_id(tmp_path: Path) -> None:
    """A resumed manager reuses the persisted tracking row for new probes.

    Without this the resumed run would either re-derive an id (orphaning the
    first run's probes from their template) or have none to write.
    """
    db_path = tmp_path / "u.db"
    manager, _scraper, db = await _build_manager(db_path, _SEED)
    await manager.seed()
    first_id = manager._speculation_state["fetch_spec:0"].tracking_id
    assert first_id is not None

    resumed, scraper, db2 = await _build_manager(db_path, _SEED)
    state = resumed._speculation_state["fetch_spec:0"]
    assert state.tracking_id == first_id

    # Extending after a success on the resumed manager writes probes against
    # that same row rather than a second one. The first seed persisted its
    # ceiling of 2, so a success at 2 extends the window to 4.
    probe = _spec_request(scraper, 2, first_id)
    await resumed.track_outcome(probe, _response(probe, 200))
    assert state.current_ceiling == 4

    saved = await db2.load_all_speculation_states()
    assert len(saved) == 1
    async with db2.session_factory() as session:
        distinct = (
            (
                await session.execute(
                    sqlalchemy_text(
                        "SELECT DISTINCT speculation_tracking_id FROM requests "
                        "WHERE is_speculative = 1"
                    )
                )
            )
            .scalars()
            .all()
        )
    assert distinct == [first_id]


async def test_seed_value_round_trips_through_state(tmp_path: Path) -> None:
    """The raw seed value persists with the state and survives a reload.

    Hosts map state rows back to their cursor store via this value, so both the
    dict-seeded and string-seeded shapes must round-trip verbatim.
    """
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    state = manager._speculation_state["fetch_spec:0"]
    assert state.seed_value == {"n": 1, "gap": 2}

    await manager.persist_all()
    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"].seed_value_json == '{"n":1,"gap":2}'

    # A fresh manager with no discovered templates (resume) reconstructs the
    # state from the DB, seed_value included.
    fresh_scraper = _SpecScraper()
    queue = RequestQueue(db)
    fresh = SpeculationManager(fresh_scraper, queue, db, seed_params=_SEED)
    fresh.discover()
    await fresh.load()
    assert fresh._speculation_state["fetch_spec:0"].seed_value == {
        "n": 1,
        "gap": 2,
    }


async def _resume_with_saved(
    db_path: Path, record: SpeculationStateRecord
) -> SpeculationManager:
    """Store *record*, then load it into a fresh (resumed) manager."""
    engine, factory = await init_database(db_path)
    db = SQLManager(engine, factory)
    await db.save_speculation_state(record)
    manager = SpeculationManager(
        _SpecScraper(), RequestQueue(db), db, seed_params=_SEED
    )
    manager.discover()
    await manager.load()
    return manager


async def test_load_warns_when_a_saved_state_has_no_speculative_entry(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A state whose entry is gone (renamed, un-speculated) is not dropped
    silently: resuming would otherwise stop probing it with no trace.
    """
    record = SpeculationStateRecord(
        func_name="renamed_entry:0",
        template_json=_SpecId(n=1).model_dump_json(),
    )
    with caplog.at_level("WARNING"):
        manager = await _resume_with_saved(tmp_path / "u.db", record)

    assert "renamed_entry:0" not in manager._speculation_state
    assert any("renamed_entry:0" in r.getMessage() for r in caplog.records)


async def test_load_logs_the_template_error_it_skips(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A template that no longer validates is skipped with its traceback."""
    record = SpeculationStateRecord(
        func_name="fetch_spec:7", template_json='{"n": "not an int"}'
    )
    with caplog.at_level("WARNING"):
        manager = await _resume_with_saved(tmp_path / "u.db", record)

    assert "fetch_spec:7" not in manager._speculation_state
    (warning,) = [
        r for r in caplog.records if "fetch_spec:7" in r.getMessage()
    ]
    assert warning.exc_info is not None


# --- End-to-end through ScrapeRun ----------------------------------------


def _counts(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        q = conn.execute
        return {
            "completed": q(
                "SELECT COUNT(*) FROM requests WHERE status = ?",
                (RequestStatus.COMPLETED.code,),
            ).fetchone()[0],
            "results": [
                r[0] for r in q("SELECT data_json FROM results").fetchall()
            ],
            "errors": q("SELECT COUNT(*) FROM errors").fetchone()[0],
            "spec": q(
                "SELECT func_name, highest_successful_id, stopped "
                "FROM speculation_tracking"
            ).fetchall(),
            "probe_urls": [
                r[0]
                for r in q(
                    "SELECT url FROM requests WHERE is_speculative=1"
                ).fetchall()
            ],
        }
    finally:
        conn.close()


async def test_end_to_end_speculation(server_url: str, tmp_path: Path) -> None:
    """A speculative scrape walks /spec until the persistent 404 and stops."""
    results: list[dict[str, Any]] = []

    async def on_data(data: Any) -> None:
        results.append(data)

    db_path = tmp_path / "run.db"
    run = ScrapeRun(
        _make_scraper(server_url),
        db_path,
        config=RunConfig(num_workers=2, seed_params=_SEED, rate_limited=False),
        hooks=RunHooks(on_data=on_data),
    )
    await run.open()
    assert run._speculation is not None
    try:
        await run.run()
        assert await run.status() == "done"
    finally:
        await run.aclose()

    counts = _counts(db_path)
    # /spec/1,2,3 succeed → 3 results; ids 4,5 fail (gap=2) → stop.
    got = sorted(d["n"] for d in results)
    assert got == [1, 2, 3]
    assert counts["errors"] == 0
    # speculation_tracking reflects the final state.
    assert len(counts["spec"]) == 1
    func_name, highest, stopped = counts["spec"][0]
    assert func_name == "fetch_spec:0"
    assert highest == 3
    assert stopped == 1
    # Probes attempted: at least 1..5 (3 ok + the 2 failures needed to hit
    # max_gap). A success extends only while the ceiling is within one gap of
    # the watermark, so however outcomes interleave across workers no probe
    # lands more than two gaps past the highest success.
    gap = _SEED[0]["fetch_spec"]["sid"]["gap"]
    probed = sorted(int(u.rsplit("/", 1)[-1]) for u in counts["probe_urls"])
    assert probed[:5] == [1, 2, 3, 4, 5]
    assert max(probed) <= highest + 2 * gap
