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
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver import ScrapeRun
from jkent.driver.unified_driver.persistence import RequestQueue
from jkent.driver.unified_driver.speculation import SpeculationManager

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


class _SpecScraper(BaseScraper[dict]):
    """entry probes /spec/{n}; parse records the surviving id."""

    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_spec(self, sid: _SpecId) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/spec/{sid.n}"
            ),
            continuation="parse_spec",
        )

    @step
    def parse_spec(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
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


def _spec_request(scraper: _SpecScraper, n: int) -> Request:
    req = scraper.fetch_spec(_SpecId(n=n, gap=2))
    return req.speculative("fetch_spec:0", 0, n)


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
        method=draw(st.sampled_from(list(HttpMethod))),
        url="https://example.com/" + draw(_word),
        headers=draw(st.none() | _str_dict),
        cookies=draw(st.none() | _str_dict),
        auth=draw(st.none() | st.tuples(_word, _word)),
        timeout=draw(st.none() | _finite | st.tuples(_finite, _finite)),
        allow_redirects=draw(st.booleans()),
        proxies=draw(st.none() | _str_dict),
        # verify: True | False | a CA-bundle path. The literal "false" is
        # rejected by the serializer (ambiguous with verify=False), so exclude it.
        verify=draw(st.booleans() | _word.filter(lambda s: s != "false")),
        stream=draw(st.booleans()),
        cert=draw(st.none() | _word | st.tuples(_word, _word)),
    )
    req = Request(
        request=params,
        continuation="parse_spec",
        current_location=draw(st.sampled_from(["", "https://example.com"])),
        bypass_rate_limit=draw(st.booleans()),
        reseedable=draw(st.none() | st.booleans()),
    )
    return req.speculative(
        "fetch_spec:0", 0, draw(st.integers(min_value=1, max_value=10_000))
    )


async def _enqueue_then_dequeue(req: Request) -> Request:
    """Enqueue ``req`` as a speculative probe and read it back from a fresh DB."""
    with tempfile.TemporaryDirectory() as d:
        engine, factory = await init_database(Path(d) / "rt.db")
        try:
            db = SQLManager(engine, factory)
            manager = SpeculationManager(_SpecScraper(), RequestQueue(db), db)
            await manager._enqueue_speculative(req)
            dequeued = await RequestQueue(db).get_next_request()
            assert dequeued is not None
            return dequeued[1]
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
    serialized keys, silently dropping fields like ``allow_redirects`` and
    ``bypass_rate_limit``. It now spreads the full serialized key set, so any
    by-value-preserved field must survive insert + dequeue. Hypothesis varies
    them all so a future re-narrowing of that spread fails here.
    """
    restored = asyncio.run(_enqueue_then_dequeue(req))
    # Full HTTPRequestParams equality covers method, url, headers, cookies,
    # auth, timeout, allow_redirects, proxies, verify, stream, and cert at once.
    assert restored.request == req.request
    assert restored.current_location == req.current_location
    assert restored.bypass_rate_limit == req.bypass_rate_limit
    assert restored.reseedable == req.reseedable
    assert restored.is_speculative is True
    assert restored.speculation_id == req.speculation_id


async def test_success_advances_and_extends(tmp_path: Path) -> None:
    """A successful probe bumps highest_successful_id and extends the window."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    assert state.current_ceiling == 2

    # Success at 2 (== ceiling, within gap of ceiling) → extend to 4.
    await manager.track_outcome(
        _spec_request(scraper, 2), _response(_spec_request(scraper, 2), 200)
    )

    assert state.highest_successful_id == 2
    assert state.consecutive_failures == 0
    assert state.current_ceiling == 4
    # Persisted to DB.
    saved = await db.load_all_speculation_states()
    assert saved["fetch_spec:0"]["highest_successful_id"] == 2
    assert saved["fetch_spec:0"]["current_ceiling"] == 4


async def test_failure_stops_after_max_gap(tmp_path: Path) -> None:
    """SpeculationHTTPFailure outcomes record failures and stop extension."""
    manager, scraper, db = await _build_manager(tmp_path / "u.db", _SEED)
    await manager.seed()
    state = manager._speculation_state["fetch_spec:0"]
    state.highest_successful_id = 3

    # Two consecutive failures beyond watermark (gap=2) → stopped.
    await manager.track_outcome(
        _spec_request(scraper, 4), _response(_spec_request(scraper, 4), 404)
    )
    assert state.consecutive_failures == 1
    assert state.stopped is False

    await manager.track_outcome(
        _spec_request(scraper, 5), _response(_spec_request(scraper, 5), 404)
    )
    assert state.consecutive_failures == 2
    assert state.stopped is True

    # type-checkers carry the earlier `stopped is False` narrowing past
    # track_outcome(), which really does flip it.
    saved = await db.load_all_speculation_states()  # type: ignore[unreachable]
    assert saved["fetch_spec:0"]["stopped"] is True
    assert saved["fetch_spec:0"]["consecutive_failures"] == 2


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
    assert saved["fetch_spec:0"]["seed_value_json"] == '{"n": 1, "gap": 2}'

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


# --- End-to-end through ScrapeRun ----------------------------------------


def _counts(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        q = conn.execute
        return {
            "completed": q(
                "SELECT COUNT(*) FROM requests WHERE status='completed'"
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
        num_workers=2,
        seed_params=_SEED,
        on_data=on_data,
        rate_limited=False,
    )
    await run.open(setup_signal_handlers=False)
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
    # max_gap). Extension may seed a few probes past the watermark before the
    # stop registers; the exact upper bound is concurrency-dependent and is
    # pinned for parity in test_speculation_differential.
    probed = sorted(int(u.rsplit("/", 1)[-1]) for u in counts["probe_urls"])
    assert probed[:5] == [1, 2, 3, 4, 5]
    assert max(probed) <= 7
