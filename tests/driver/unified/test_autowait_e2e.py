"""Autowait against a real browser: a step that fails on a selector waits for it.

A ``@step(auto_await_timeout=...)`` whose query misses on the first DOM
snapshot has the executor wait for that selector on the live page, re-snapshot,
and run the step again. The served page adds its heading on a timer, after
``domcontentloaded``, so the first snapshot never contains it.

Skipped cleanly when no browser engine can launch (via the shared
``require_browser`` fixture in ``conftest``).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Selector,
)
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver import ScrapeRun
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from jkent.driver.unified_driver.wiring import RunConfig
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from jkent.common.page_element import PageElement

#: How long after ``domcontentloaded`` the page adds its heading.
_LATE_MS = 1_500

_LATE_HTML = f"""<html><body><script>
document.addEventListener("DOMContentLoaded", () => setTimeout(() => {{
  const h = document.createElement("h1");
  h.id = "late";
  h.textContent = "arrived";
  document.body.appendChild(h);
}}, {_LATE_MS}));
</script></body></html>"""

_NEVER_HTML = "<html><body><p>no heading here</p></body></html>"

#: One row at load, the other two after ``_LATE_MS``: a query for three
#: already matches something, so a plain wait for the selector returns at once.
_PARTIAL_HTML = f"""<html><body><ul><li>1</li></ul><script>
document.addEventListener("DOMContentLoaded", () => setTimeout(() => {{
  for (const n of ["2", "3"]) {{
    const li = document.createElement("li");
    li.textContent = n;
    document.querySelector("ul").appendChild(li);
  }}
}}, {_LATE_MS}));
</script></body></html>"""

#: (selector, min_count, max_count) the step queries.
_HEADING = ("//h1[@id='late']", 1, 1)
_THREE_ROWS = ("//li", 3, None)


def _scraper(
    auto_await_timeout: int | None,
    query: tuple[str, int, int | None] = _HEADING,
) -> BaseScraper[dict[str, Any]]:
    xpath, min_count, max_count = query

    class _Scraper(BaseScraper[dict[str, Any]]):
        base = "http://127.0.0.1"

        @entry(dict)
        def start(self) -> Generator[Request, None, None]:
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=f"{self.base}/page"
                ),
                step="parse_page",
            )

        @step(auto_await_timeout=auto_await_timeout)
        def parse_page(
            self, page: PageElement
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            found = page.query(
                Selector.XPath(xpath), "late content", min_count, max_count
            )
            yield ParsedData(data={"found": len(found)})

    return _Scraper()


async def _start_server(html: str) -> StartedServer:
    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    return await start_app(app)


def _outcome(db_path: Path) -> tuple[int, int]:
    """(results rows, the request's status code) after the run."""
    conn = sqlite3.connect(str(db_path))
    try:
        results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        (status,) = conn.execute("SELECT status FROM requests").fetchone()
        return results, status
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("html", "query", "auto_await_timeout", "results", "status"),
    [
        pytest.param(
            _LATE_HTML,
            _HEADING,
            10_000,
            1,
            RequestStatus.COMPLETED,
            id="waits",
        ),
        pytest.param(
            _LATE_HTML,
            _HEADING,
            None,
            0,
            RequestStatus.FAILED,
            id="no-autowait",
        ),
        pytest.param(
            _NEVER_HTML,
            _HEADING,
            300,
            0,
            RequestStatus.FAILED,
            id="times-out",
        ),
        pytest.param(
            _PARTIAL_HTML,
            _THREE_ROWS,
            10_000,
            1,
            RequestStatus.COMPLETED,
            id="waits-for-the-rest",
        ),
    ],
)
async def test_autowait_against_a_live_page(
    require_browser: None,
    tmp_path: Path,
    html: str,
    query: tuple[str, int, int | None],
    auto_await_timeout: int | None,
    results: int,
    status: RequestStatus,
) -> None:
    server = await _start_server(html)
    try:
        db_path = tmp_path / "run.db"
        engine, session_factory = await init_database(db_path)
        scraper = _scraper(auto_await_timeout, query)
        scraper.base = server.base_url  # type: ignore[attr-defined]
        transport = PlaywrightTransport(
            scraper,
            headless=True,
            db=SQLManager(engine, session_factory),
        )
        run = ScrapeRun(
            scraper,
            db_path,
            transport=transport,
            config=RunConfig(num_workers=1, rate_limited=False),
        )
        await run.open()
        try:
            await run.run()
        finally:
            await run.aclose()
            await engine.dispose()

        assert _outcome(db_path) == (results, status.code)
    finally:
        await server.aclose()
