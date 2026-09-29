"""Shared fixtures for the test suite.

Two families, each with one home:

- **Servers** — ``serve_routes`` (a factory: ``await serve({path: handler})``
  → base URL) and ``bug_court_server`` / ``server_url`` (the mock court
  site). Both run in the test's own event loop via :mod:`tests.servers`; a
  test that needs one is therefore ``async``.
- **Hypothesis** profiles.
"""

import os

# Contracts (jkent.contracts) gate at decoration time, so the
# toggle must be set before anything below imports a jkent module. The
# whole test run enforces contracts; production leaves them off.
os.environ.setdefault("JKENT_ENFORCE_CONTRACTS", "1")

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

import pytest
from aiohttp import web
from hypothesis import settings as _hyp_settings

from tests.mock_server import (
    create_app,
    generate_cases_html,
)
from tests.servers import RouteHandler, StartedServer, start_app

# Hypothesis profiles — select with ``--hypothesis-profile NAME`` or
# ``HYPOTHESIS_PROFILE=NAME``. Tests that pin their own ``max_examples`` are
# unaffected; the unpinned ones (the unified-driver rigs/conformance) follow
# whichever profile is loaded. All keep ``deadline=None`` so the I/O-heavy
# rigs (live servers, SQLite files) aren't failed on per-example timing.
_hyp_settings.register_profile("dev", max_examples=25, deadline=None)
_hyp_settings.register_profile("ci", max_examples=200, deadline=None)
_hyp_settings.register_profile("thorough", max_examples=2000, deadline=None)
_hyp_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def cases_html() -> str:
    """Generate the case list HTML.

    Returns:
        HTML string containing all Bug Civil Court cases.
    """
    return generate_cases_html()


# =============================================================================
# Servers
# =============================================================================


@pytest.fixture
async def serve_routes() -> AsyncIterator[
    Callable[[dict[str, RouteHandler]], Awaitable[str]]
]:
    """Factory that starts an ephemeral-port aiohttp server per call.

    Yields an async ``serve({path: handler})`` returning the server's base
    URL; every server it starts is torn down at fixture teardown.
    """
    servers: list[StartedServer] = []

    async def _serve(routes: dict[str, RouteHandler]) -> str:
        app = web.Application()
        for path, handler in routes.items():
            app.router.add_get(path, handler)
        server = await start_app(app)
        servers.append(server)
        return server.base_url

    try:
        yield _serve
    finally:
        results = await asyncio.gather(
            *(server.aclose() for server in servers), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result


@pytest.fixture
async def bug_court_server() -> AsyncIterator[StartedServer]:
    """The Bug Court mock site, served on an ephemeral port in this loop."""
    server = await start_app(create_app())
    try:
        yield server
    finally:
        await server.aclose()


@pytest.fixture
def server_url(bug_court_server: StartedServer) -> str:
    """Base URL of the Bug Court server (e.g. ``http://127.0.0.1:54321``)."""
    return bug_court_server.base_url
