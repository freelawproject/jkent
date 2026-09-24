"""Ephemeral aiohttp servers for tests, all in the test's own event loop.

``start_app`` is the one place the ``AppRunner``/``TCPSite`` boilerplate
lives. The ``serve_routes`` / ``bug_court_server`` fixtures in the root
``conftest`` build on it; the hypothesis rigs that stand up a server inside
their own ``asyncio.run`` per example call it directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from aiohttp import web

#: A GET handler suitable for :func:`aiohttp.web.UrlDispatcher.add_get`.
RouteHandler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@dataclass
class StartedServer:
    """A running aiohttp server on an ephemeral port; ``runner`` tears it down."""

    runner: web.AppRunner
    base_url: str

    async def aclose(self) -> None:
        await self.runner.cleanup()


async def start_app(app: web.Application) -> StartedServer:
    """Start ``app`` on an ephemeral 127.0.0.1 port and return its handle+URL.

    The caller owns teardown via ``await server.aclose()``.
    """
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][0], runner.addresses[0][1]
    return StartedServer(runner=runner, base_url=f"http://{host}:{port}")


def single_page_app(html: str) -> web.Application:
    """A GET-only ``web.Application`` serving ``html`` at ``/page``.

    The transport conformance fixtures need a real server with no
    subresources; this centralizes the one-route app they would otherwise
    each inline.
    """

    async def page(_request: web.Request) -> web.Response:
        return web.Response(status=200, content_type="text/html", text=html)

    app = web.Application()
    app.router.add_get("/page", page)
    return app


def status_app(status: int, body: bytes) -> web.Application:
    """A ``web.Application`` answering every route with ``status`` + ``body``.

    Served as ``text/html`` so browser transports render (and snapshot) the
    body even on error statuses. The classification conformance bindings need
    a server surfacing an arbitrary status code; this centralizes the app
    they would otherwise each inline.
    """

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=status, body=body, content_type="text/html")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app
