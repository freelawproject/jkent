"""Scraper ``default_headers`` and per-request ``cookies`` through
``PlaywrightTransport``.

``HttpxTransport`` sends a scraper's ``default_headers`` on every request
(a per-request header of the same name, any case, overrides) and merges
per-request ``cookies`` into the wire — see ``test_httpx_default_headers``.
A browser transport must put the same headers on the wire, or adding a
browser requirement (``JS_EVAL``, say) to a scraper silently changes what
the server sees. Both the navigation and the archive-download paths are
covered. Browser-gated; skips cleanly without a launchable engine.
"""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING, ClassVar

import pytest
from aiohttp import web
from lxml import html as lxml_html

from jkent.common.page_element import ViaLink
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
)
from jkent.driver.unified_driver import QueuedRequest
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.db_staging import insert_request_row
from tests.driver.unified.test_playwright_transport import (
    _Scraper,
    _sql_manager,
)
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _HeaderedScraper(_Scraper):
    default_headers: ClassVar[dict[str, str]] = {
        "X-Court-Client": "jkent",
        "X-Default-Only": "kept",
    }


# The server reports every header it received as a list of ``[name, value]``
# pairs, so a default sent *alongside* a differently-cased override shows up
# as two entries rather than being collapsed by a dict.
def _pairs(request: web.Request) -> str:
    return json.dumps([[k, v] for k, v in request.headers.items()])


def _echo_app() -> web.Application:
    async def echo(request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<html><body><pre id='h'>"
                f"{escape(_pairs(request))}"
                "</pre></body></html>"
            ),
            content_type="text/html",
        )

    async def parent(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body><a id='dl' href='/file.bin'>dl</a></body></html>",
            content_type="text/html",
        )

    async def file_(request: web.Request) -> web.Response:
        return web.Response(
            body=_pairs(request).encode(),
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="file.bin"',
            },
        )

    app = web.Application()
    app.router.add_get("/echo", echo)
    app.router.add_get("/parent", parent)
    app.router.add_get("/file.bin", file_)
    return app


def _by_name(pairs_json: str) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    for name, value in json.loads(pairs_json):
        seen.setdefault(name.lower(), []).append(value)
    return seen


@pytest.fixture
async def server() -> AsyncIterator[StartedServer]:
    started = await start_app(_echo_app())
    try:
        yield started
    finally:
        await started.aclose()


@pytest.fixture
async def transport(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
):
    subject = PlaywrightTransport(
        _HeaderedScraper(),
        headless=True,
        db=_sql_manager(memory_session_factory),
    )
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


async def _wire_headers(
    transport: PlaywrightTransport,
    sf: async_sessionmaker[AsyncSession],
    url: str,
    *,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    handle = await transport.acquire(0)
    rid = await insert_request_row(sf, url)
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url=url, headers=headers, cookies=cookies
        ),
        step="parse",
    )
    response = await transport.resolve(
        handle, QueuedRequest(request=request, request_id=rid)
    )
    pre = lxml_html.fromstring(response.content).get_element_by_id("h")
    return _by_name(pre.text or "")


class TestNavigationHeaders:
    async def test_defaults_sent_when_request_has_no_headers(
        self,
        transport: PlaywrightTransport,
        memory_session_factory: async_sessionmaker[AsyncSession],
        server: StartedServer,
    ) -> None:
        seen = await _wire_headers(
            transport, memory_session_factory, f"{server.base_url}/echo"
        )
        assert seen["x-court-client"] == ["jkent"]
        assert seen["x-default-only"] == ["kept"]

    async def test_per_request_header_overrides_default_case_insensitively(
        self,
        transport: PlaywrightTransport,
        memory_session_factory: async_sessionmaker[AsyncSession],
        server: StartedServer,
    ) -> None:
        seen = await _wire_headers(
            transport,
            memory_session_factory,
            f"{server.base_url}/echo",
            headers={"x-court-client": "special"},
        )
        # Exactly one value on the wire: the default must not ride along
        # under its own casing.
        assert seen["x-court-client"] == ["special"]
        assert seen["x-default-only"] == ["kept"]

    async def test_per_request_cookies_reach_the_server(
        self,
        transport: PlaywrightTransport,
        memory_session_factory: async_sessionmaker[AsyncSession],
        server: StartedServer,
    ) -> None:
        seen = await _wire_headers(
            transport,
            memory_session_factory,
            f"{server.base_url}/echo",
            cookies={"session": "abc123"},
        )
        assert "session=abc123" in " ".join(seen.get("cookie", []))


async def test_archive_download_carries_defaults(
    transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    server: StartedServer,
) -> None:
    """The download click sends the defaults too, with the request's own
    headers on top — and nothing left over from the previous navigation."""
    previous = await _wire_headers(
        transport,
        memory_session_factory,
        f"{server.base_url}/echo",
        headers={"X-Nav-Only": "yes"},
    )
    assert previous["x-nav-only"] == ["yes"]
    handle = await transport.acquire(0)
    await handle.page.goto(
        f"{server.base_url}/parent", wait_until="domcontentloaded"
    )
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url=f"{server.base_url}/file.bin",
            headers={"X-Archive-Only": "yes"},
            timeout=30,
        ),
        step="parse",
        archive=True,
        via=ViaLink(selector=Selector.CSS("#dl"), description="download"),
    )
    stream = await transport.resolve_archive(
        handle, QueuedRequest(request=request, request_id=1)
    )
    try:
        body = b"".join([chunk async for chunk in stream])
    finally:
        await transport.finish_archiving(stream)
    seen = _by_name(body.decode())
    assert seen["x-court-client"] == ["jkent"]
    assert seen["x-archive-only"] == ["yes"]
    assert "x-nav-only" not in seen
