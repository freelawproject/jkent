"""Browser-gated end-to-end proof for the Playwright continuation path.

Drives a tiny scrape through a real ``ScrapeRun`` wired to a real
``PlaywrightTransport`` against a local aiohttp server, proving the
worker -> PlaywrightTransport.resolve -> ContinuationExecutor.complete_request
chain composes once the page/autowait branch is unblocked. The worker threads
the live ``WorkerPage.page`` into ``complete_request``; the served HTML is
parsed by a decorated ``@step`` and its ``ParsedData`` lands in the DB.

Skipped cleanly when no browser engine can launch (via the shared
``has_browser`` fixture in ``conftest``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest
from aiohttp import web

from jkent.common.decorators import entry, step
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
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.conftest import StartedServer, start_app


class _PWScraper(BaseScraper[dict]):
    """entry -> /page; parse the served HTML into one datum."""

    base = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/page"
            ),
            continuation="parse_page",
        )

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
        # Pull the served marker straight out of the snapshot text.
        assert "served-by-playwright" in response.text
        yield ParsedData(data={"marker": "served-by-playwright"})


async def _start_server() -> StartedServer:
    html = "<html><body><h1 id='ok'>served-by-playwright</h1></body></html>"

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    return await start_app(app)


def _result_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    finally:
        conn.close()


async def test_playwright_scrape_lands_result(
    has_browser: bool, tmp_path: Path
) -> None:
    """A real Playwright scrape resolves, parses, and persists via ScrapeRun.

    Proves the worker passes the live page and the
    resolve -> complete_request cycle stores a result.
    """
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")

    server = await _start_server()
    try:
        db_path = tmp_path / "run.db"
        # The transport needs a DB reference for incidental/parent reads; it
        # shares the same on-disk DB file ScrapeRun initializes for the run.
        engine, session_factory = await init_database(db_path)
        transport_db = SQLManager(engine, session_factory)

        scraper = _PWScraper()
        scraper.base = server.base_url

        results: list[dict] = []

        async def on_data(data: dict) -> None:
            results.append(data)

        transport = PlaywrightTransport(
            scraper, headless=True, db=transport_db
        )
        run = ScrapeRun(
            scraper,
            db_path,
            transport=transport,
            num_workers=1,
            on_data=on_data,
            rate_limited=False,
        )
        await run.open(setup_signal_handlers=False)
        try:
            await run.run()
            assert await run.status() == "done"
        finally:
            await run.aclose()
            await engine.dispose()

        # The served datum reached on_data and a results row.
        assert results == [{"marker": "served-by-playwright"}]
        assert _result_count(db_path) == 1
    finally:
        await server.runner.cleanup()
