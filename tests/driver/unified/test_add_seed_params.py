"""``ScrapeRun.add_seed_params`` — layering new entries onto an existing run.

Mirrors the old driver's contract (tests/demo/test_add_params_non_advancing.py,
which stays with the excluded demo fixture) with inline fixtures:

1. First run seeds a non-advancing speculative range [1, 6) → cases 1-5.
2. Second run resumes the same DB and ``add_seed_params`` a second range
   [6, 11) → cases 6-10 are fetched, cases 1-5 are not re-fetched.
3. Final state: all 10 cases exactly once, no errors, and the stored
   ``seed_params_json`` reflects the merged invocations.

Also pins the non-speculative path: re-adding an already-enqueued entry
dedups; a new entry is enqueued.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from jkent.common.decorators import entry, step
from jkent.common.param_models import SpeculativeRange
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.unified_driver import ScrapeRun

# --- A ten-case court served inline ---------------------------------------


@pytest.fixture
async def case_server_url() -> Any:
    """GET /case/{n} → a docket body for 1 <= n <= 10, else 404."""

    async def handle_case(request: web.Request) -> web.Response:
        n = int(request.match_info["n"])
        if 1 <= n <= 10:
            return web.Response(text=f"BCC-2024-{n:03d}")
        return web.Response(status=404)

    app = web.Application()
    app.router.add_get("/case/{n}", handle_case)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][0], runner.addresses[0][1]
    try:
        yield f"http://{host}:{port}"
    finally:
        await runner.cleanup()


class _CaseScraper(BaseScraper[dict]):
    """Speculative entry probing /case/{n}; parse records the docket."""

    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_case(self, case_id: SpeculativeRange) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/case/{case_id.min}"
            ),
            continuation="parse_case",
        )

    @step
    def parse_case(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"docket": response.text})


def _make_scraper(server_url: str) -> _CaseScraper:
    scraper = _CaseScraper()
    scraper.base = server_url
    return scraper


def _range_params(lo: int, hi: int) -> list[dict[str, dict[str, Any]]]:
    return [
        {
            "fetch_case": {
                "case_id": {
                    "min": lo,
                    "soft_max": hi,
                    "should_advance": False,
                }
            }
        }
    ]


def _db_state(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path))
    try:
        q = conn.execute
        return {
            "dockets": [
                r[0]
                for r in q(
                    "SELECT json_extract(data_json, '$.docket') FROM results"
                ).fetchall()
            ],
            "errors": q("SELECT COUNT(*) FROM errors").fetchone()[0],
            "seed_params_json": q(
                "SELECT seed_params_json FROM run_metadata"
            ).fetchone()[0],
        }
    finally:
        conn.close()


async def test_add_params_non_advancing(
    case_server_url: str, tmp_path: Path
) -> None:
    db_path = tmp_path / "run.db"

    # Run 1: seed cases [1, 6).
    run = ScrapeRun(
        _make_scraper(case_server_url),
        db_path,
        seed_params=_range_params(1, 6),
        rate_limited=False,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await run.run()
    finally:
        await run.aclose()

    first = _db_state(db_path)
    assert sorted(first["dockets"]) == [
        f"BCC-2024-{n:03d}" for n in range(1, 6)
    ]
    assert first["errors"] == 0

    # Run 2: resume (fresh scraper instance = fresh process) and add [6, 11).
    run = ScrapeRun(
        _make_scraper(case_server_url),
        db_path,
        rate_limited=False,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await run.add_seed_params(_range_params(6, 11))
        await run.run()
    finally:
        await run.aclose()

    final = _db_state(db_path)
    # All ten cases exactly once — run 1's cases were not re-fetched.
    assert sorted(final["dockets"]) == [
        f"BCC-2024-{n:03d}" for n in range(1, 11)
    ]
    assert final["errors"] == 0
    # Stored intent reflects the merge, for any further resume.
    stored = json.loads(final["seed_params_json"])
    assert stored == _range_params(1, 6) + _range_params(6, 11)


async def test_add_params_requires_open(tmp_path: Path) -> None:
    run = ScrapeRun(_CaseScraper(), tmp_path / "run.db", rate_limited=False)
    with pytest.raises(RuntimeError, match="open"):
        await run.add_seed_params(_range_params(1, 2))


# --- Non-speculative path: entry dedup -------------------------------------


class _PlainScraper(BaseScraper[dict]):
    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_page(self, page_id: int) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/page/{page_id}"
            ),
            continuation="parse_page",
        )

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"url": response.url})


def _request_urls(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            r[0] for r in conn.execute("SELECT url FROM requests").fetchall()
        ]
    finally:
        conn.close()


async def test_add_params_dedups_existing_entries(tmp_path: Path) -> None:
    """Re-adding an already-enqueued invocation dedups; new ones enqueue."""
    db_path = tmp_path / "run.db"

    run = ScrapeRun(
        _PlainScraper(),
        db_path,
        seed_params=[{"fetch_page": {"page_id": 1}}],
        rate_limited=False,
    )
    await run.open(setup_signal_handlers=False)
    try:
        assert len(_request_urls(db_path)) == 1
        # Same invocation again → dedup; a new page → enqueued.
        await run.add_seed_params([{"fetch_page": {"page_id": 1}}])
        assert len(_request_urls(db_path)) == 1
        await run.add_seed_params([{"fetch_page": {"page_id": 2}}])
        urls = _request_urls(db_path)
        assert len(urls) == 2
        assert any(u.endswith("/page/2") for u in urls)
    finally:
        await run.aclose()
