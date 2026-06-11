"""Speculation end-to-end for the unified driver.

A ``@speculate`` scraper probes a sequential id against ``/spec/{n}`` (the
mock server returns 200 for n <= 3, then a persistent 404). This exercises:

- ``SpeculationManager`` discovery/seed/track/persist in isolation against a
  real in-memory ``SQLManager`` + ``RequestQueue`` (no transport);
- a full ``ScrapeRun`` against the live server: the right probes are attempted,
  results land, and ``speculation_tracking`` reflects the final state.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

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
