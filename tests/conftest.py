"""Shared fixtures for the test suite."""

import os

# Contracts (jkent.contracts) gate at decoration time, so the
# toggle must be set before anything below imports a jkent module. The
# whole test run enforces contracts; production leaves them off.
os.environ.setdefault("JKENT_ENFORCE_CONTRACTS", "1")

import asyncio
import socket
import threading
import time
from collections.abc import Generator
from contextlib import closing, suppress

import pytest
from aiohttp import web
from hypothesis import settings as _hyp_settings

from tests.mock_server import (
    create_app,
    generate_cases_html,
)

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
# Step 2: aiohttp test server fixtures
# =============================================================================


def find_free_port() -> int:
    """Find a free port on localhost.

    Returns:
        An available port number.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class AioHttpTestServer:
    """Wrapper to run aiohttp server in a background thread."""

    def __init__(self, app: web.Application, port: int) -> None:
        self.app = app
        self.port = port
        self.host = "127.0.0.1"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Get the base URL of the server."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        """Start the server in a background thread."""
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        # Give the server time to start
        time.sleep(0.1)

    def _run_server(self) -> None:
        """Run the server in an asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def start() -> None:
            self._runner = web.AppRunner(self.app)
            await self._runner.setup()
            site = web.TCPSite(self._runner, self.host, self.port)  # type: ignore
            await site.start()

        self._loop.run_until_complete(start())  # type: ignore
        self._loop.run_forever()  # type: ignore

    def stop(self) -> None:
        """Stop the server and clean up resources."""
        if self._loop and self._runner:
            # Schedule cleanup in the event loop
            async def cleanup() -> None:
                await (
                    self._runner.cleanup()
                ) if self._runner is not None else None

            future = asyncio.run_coroutine_threadsafe(cleanup(), self._loop)  # type: ignore
            with suppress(Exception):  # Best effort cleanup
                future.result(timeout=2.0)

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=2.0)


@pytest.fixture
def bug_court_server() -> Generator[AioHttpTestServer, None, None]:
    """Create and start an aiohttp test server running the Bug Court app.

    This fixture starts a real HTTP server on a random port that can be
    used for integration testing with real HTTP requests.

    Yields:
        AioHttpTestServer instance with the Bug Court app running.
    """
    app = create_app()
    port = find_free_port()
    server = AioHttpTestServer(app, port)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def server_url(bug_court_server: AioHttpTestServer) -> str:
    """Get the base URL of the test server.

    Args:
        bug_court_server: The test server fixture.

    Returns:
        The base URL string (e.g., "http://127.0.0.1:8080").
    """
    return bug_court_server.url
