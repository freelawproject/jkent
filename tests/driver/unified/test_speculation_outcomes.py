"""What a speculative probe records, end to end through ``ScrapeRun``.

One worker walks ``/probe/{n}`` with ``gap=3`` against a server where 1 is
a hit, 2 is a 200 the scraper's ``actually_successful`` rejects, and every
later id is a 404 with a body. With one worker the queue runs in id order:

====  ===========================  ==================  ========  =========
id    server                       outcome             stored    step ran
====  ===========================  ==================  ========  =========
1     200                          ``hit``             yes       yes
2     200, "no such record"        ``miss``            body      no
3     404, body                    ``miss``            body      no
4     404, body                    ``stopped``         body      no
5, 6  never asked                  ``terminated_early``  nothing   no
====  ===========================  ==================  ========  =========

The hit at 1 extends the window to 6 before the misses stop it, so 5 and 6
are already queued when the template stops. Run over both transports: each
must hand the worker the miss's real body.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from pydantic import BaseModel
from typing_extensions import override

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
from jkent.driver.database_engine.compression import decompress
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.unified_driver import ScrapeRun
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks
from tests.servers import start_app

if TYPE_CHECKING:
    from pathlib import Path

    from jkent.driver.unified_driver.transport import Transport


class _ProbeId(BaseModel, Speculative):
    n: int
    should_advance: bool = True
    gap: int = 3

    def seed_range(self) -> range:
        return range(self.n, 0)

    def from_int(self, n: int) -> _ProbeId:
        return _ProbeId(n=n, gap=self.gap)

    def max_gap(self) -> int:
        return self.gap


class _ProbeScraper(BaseScraper[dict[str, Any]]):
    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_probe(self, pid: _ProbeId) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/probe/{pid.n}"
            ),
            step="parse_probe",
        )

    @step
    def parse_probe(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={"n": int(response.url.rsplit("/", 1)[-1])})

    @override
    def actually_successful(self, response: Response) -> bool:
        return b"no such record" not in response.content


def _app(asked: list[int]) -> web.Application:
    async def probe(request: web.Request) -> web.Response:
        n = int(request.match_info["n"])
        asked.append(n)
        if n == 1:
            text, status = "record 1", 200
        elif n == 2:
            text, status = "no such record", 200
        else:
            text, status = f"not found {n}", 404
        return web.Response(
            text=f"<html><body>{text}</body></html>",
            status=status,
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/probe/{n}", probe)
    return app


def _rows(db_path: Path) -> dict[int, tuple[Any, ...]]:
    """``{probe id: (outcome, status code, body)}`` from the run db."""
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT url, speculation_outcome, response_status_code, "
            "content_compressed FROM requests WHERE is_speculative = 1"
        ).fetchall()
    finally:
        conn.close()
    return {
        int(url.rsplit("/", 1)[-1]): (
            None if outcome is None else SpeculationOutcome.from_code(outcome),
            status,
            None if body is None else decompress(body),
        )
        for url, outcome, status, body in rows
    }


def _httpx(_scraper: _ProbeScraper) -> Transport[Any] | None:
    return None  # ScrapeRun's default


def _playwright(scraper: _ProbeScraper) -> Transport[Any] | None:
    return PlaywrightTransport(scraper, headless=True)


@pytest.fixture(params=[_httpx, _playwright], ids=["httpx", "playwright"])
def transport_for(request: pytest.FixtureRequest) -> Any:
    if request.param is _playwright:
        request.getfixturevalue("require_browser")
    return request.param


async def test_each_probe_records_what_it_found(
    tmp_path: Path, transport_for: Any
) -> None:
    asked: list[int] = []
    server = await start_app(_app(asked))
    results: list[dict[str, int]] = []

    async def on_data(data: Any) -> None:
        results.append(data)

    try:
        scraper = _ProbeScraper()
        scraper.base = server.base_url
        db_path = tmp_path / "run.db"
        run = ScrapeRun(
            scraper,
            db_path,
            transport=transport_for(scraper),
            config=RunConfig(
                num_workers=1,
                rate_limited=False,
                seed_params=[{"fetch_probe": {"pid": {"n": 1}}}],
            ),
            hooks=RunHooks(on_data=on_data),
        )
        await run.open()
        try:
            await run.run()
        finally:
            await run.aclose()
    finally:
        await server.aclose()

    rows = _rows(db_path)
    assert sorted(asked) == [1, 2, 3, 4], (
        "a stopped template's probe was fetched"
    )
    assert results == [{"n": 1}], "a step ran on a miss"
    outcomes = {n: outcome for n, (outcome, _, _) in rows.items()}
    assert outcomes == {
        1: SpeculationOutcome.HIT,
        2: SpeculationOutcome.MISS,
        3: SpeculationOutcome.MISS,
        4: SpeculationOutcome.STOPPED,
        5: SpeculationOutcome.TERMINATED_EARLY,
        6: SpeculationOutcome.TERMINATED_EARLY,
    }
    for n, (_, status, body) in rows.items():
        if n == 1:
            assert status == 200 and b"record 1" in body
        elif n == 2:
            assert status == 200 and b"no such record" in body
        elif n in (3, 4):
            assert status == 404 and f"not found {n}".encode() in body
        else:
            assert (status, body) == (None, None)
