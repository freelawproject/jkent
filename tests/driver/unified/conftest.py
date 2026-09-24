"""Shared helpers for unified-driver tests.

The DB and server *fixtures* live in the root ``conftest``; this module holds
the unified-specific helpers — the trivial HTTP scraper, the browser gates,
and the archive/inline-render servers.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Generator
from typing import Any

import pytest
from aiohttp import web
from playwright.async_api import Error as PlaywrightError

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.servers import StartedServer, start_app


class HttpPageScraper(BaseScraper[dict[str, Any]]):
    """Plain-HTTP scraper: fetch ``{base}/page/{page_id}``, record the body."""

    base = "http://127.0.0.1"

    @entry(dict)
    def fetch_page(self, page_id: int) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/page/{page_id}"
            ),
            step="parse_page",
        )

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={"body": response.text})


# --- Shared browser-launch gate ------------------------------------------


async def _browser_launch_failure() -> str | None:
    """Why a real browser context can't be brought up here, or None if it can.

    Uses a requirement-free scraper so engine selection matches the standard
    (chromium) path the browser-gated tests exercise. Only Playwright's own
    launch error (e.g. "Executable doesn't exist") means "no usable engine";
    anything else is a bug in the launch path and propagates.
    """
    transport = PlaywrightTransport(HttpPageScraper(), headless=True)
    try:
        await transport.open()
    except PlaywrightError as exc:
        await transport.aclose()
        return f"{type(exc).__name__}: {exc}".splitlines()[0]
    await transport.aclose()
    return None


@pytest.fixture(scope="session")
def browser_launch_failure() -> str | None:
    """Session-wide: why no browser engine launches in this env, or None.

    When none does, says why once in the warnings summary, so a run where
    every browser-gated test skipped is not mistaken for a green one.
    """
    reason = asyncio.run(_browser_launch_failure())
    if reason is not None:
        warnings.warn(
            f"browser-gated tests skip: {reason}", RuntimeWarning, stacklevel=1
        )
    return reason


@pytest.fixture(scope="session")
def has_browser(browser_launch_failure: str | None) -> bool:
    """Session-wide flag: True iff a browser engine launches in this env."""
    return browser_launch_failure is None


@pytest.fixture
def require_browser(browser_launch_failure: str | None) -> None:
    """Skip the requesting test unless a browser engine launches here."""
    if browser_launch_failure is not None:
        pytest.skip(f"no launchable browser engine: {browser_launch_failure}")


async def serve_archive_download(body: bytes) -> StartedServer:
    """Serve a parent page with a download link + an attachment endpoint.

    ``/parent`` links to ``/file.bin``; ``/file.bin`` returns ``body`` with a
    ``Content-Disposition: attachment`` header so a browser click downloads it.
    The caller owns teardown via ``await server.aclose()`` and keeps the
    ``body`` it passed in for any equality check.
    """

    async def parent(_req: web.Request) -> web.Response:
        html = (
            "<html><body>"
            "<a id='dl' href='/file.bin'>download</a>"
            "</body></html>"
        )
        return web.Response(status=200, body=html, content_type="text/html")

    async def file_(_req: web.Request) -> web.Response:
        return web.Response(
            status=200,
            body=body,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="file.bin"',
            },
        )

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_get("/file.bin", file_)
    return await start_app(app)


async def serve_pdf(body: bytes) -> StartedServer:
    """Serve ``body`` at ``/doc.pdf`` as a real ``application/pdf``.

    No ``Content-Disposition``: whether the browser downloads it (headless
    Chromium) or opens it in its viewer (Firefox's pdf.js) is the browser's
    call, which is the case an archive request must survive.
    """

    async def doc(_req: web.Request) -> web.Response:
        return web.Response(
            status=200, body=body, content_type="application/pdf"
        )

    app = web.Application()
    app.router.add_get("/doc.pdf", doc)
    return await start_app(app)


async def serve_inline_render(body: bytes) -> StartedServer:
    """Serve a parent page whose link renders inline instead of downloading.

    ``/parent`` links to ``/doc.pdf``; ``/doc.pdf`` returns ``body`` as
    ``text/html`` with no ``Content-Disposition``, so every browser *navigates*
    to it rather than firing a ``download`` event — the same shape as Firefox
    opening a real PDF in pdf.js. The ``.pdf`` path matches the transport's
    file-response capture heuristic, making the inline-archive fallback
    deterministic across engines. Teardown is the caller's, via
    ``await server.aclose()``.
    """

    async def parent(_req: web.Request) -> web.Response:
        html = "<html><body><a id='dl' href='/doc.pdf'>view</a></body></html>"
        return web.Response(status=200, body=html, content_type="text/html")

    async def doc(_req: web.Request) -> web.Response:
        return web.Response(status=200, body=body, content_type="text/html")

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_get("/doc.pdf", doc)
    return await start_app(app)
