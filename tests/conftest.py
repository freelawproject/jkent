"""Shared fixtures for the test suite.

Three families, each with one home:

- **Servers** — ``serve_routes`` (a factory: ``await serve({path: handler})``
  → base URL) and ``bug_court_server`` / ``server_url`` (the mock court
  site). Both run in the test's own event loop via :mod:`tests.servers`; a
  test that needs one is therefore ``async``.
- **Databases** — ``db_path`` / ``initialized_db`` / ``sql_manager`` /
  ``insert_request`` for a real SQLite *file* (what the run and the replay
  ``SourceIndex`` open); ``memory_session_factory`` for an in-memory
  StaticPool schema when a test only needs sessions; ``schema_template`` for
  a once-built empty DB file the generative rigs copy per example.
- **Hypothesis** profiles.
"""

import os

# Contracts (jkent.contracts) gate at decoration time, so the
# toggle must be set before anything below imports a jkent module. The
# whole test run enforces contracts; production leaves them off.
os.environ.setdefault("JKENT_ENFORCE_CONTRACTS", "1")

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from hypothesis import settings as _hyp_settings
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import (
    create_engine_and_init,
    get_session_factory,
    init_database,
)
from jkent.driver.database_engine.enums import RequestType
from jkent.driver.database_engine.sql_manager import RequestInsert, SQLManager
from tests.mock_server import (
    create_app,
    generate_cases_html,
)
from tests.servers import RouteHandler, StartedServer, start_app

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

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


# =============================================================================
# Databases
# =============================================================================

# What the initialized_db fixture resolves to for its consumers.
_InitializedDB = tuple["AsyncEngine", async_sessionmaker[AsyncSession]]


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
async def initialized_db(db_path: Path) -> AsyncIterator[_InitializedDB]:
    """An initialized (schema-built) engine + session factory on a file."""
    engine, session_factory = await init_database(db_path)
    yield engine, session_factory
    await engine.dispose()


@pytest.fixture
async def sql_manager(initialized_db: _InitializedDB) -> SQLManager:
    """A :class:`SQLManager` over ``initialized_db``."""
    engine, session_factory = initialized_db
    return SQLManager(engine, session_factory)


@pytest.fixture
def insert_request(
    sql_manager: SQLManager,
) -> Callable[..., Awaitable[int]]:
    """Factory that inserts a request with sensible defaults.

    Most tests only vary ``url``/``deduplication_key``/``priority``/
    ``step``; override only what the test cares about::

        req_id = await insert_request(
            url="https://example.com/1", deduplication_key="1"
        )
    """

    async def _insert(**overrides: Any) -> int:
        params: dict[str, Any] = {
            "priority": 5,
            "request_type": RequestType.NAVIGATING,
            "method": HttpMethod.GET,
            "url": "https://example.com/test",
            "step": "parse",
        }
        params.update(overrides)
        inserted = await sql_manager.insert_request(RequestInsert(**params))
        return inserted.request_id

    return _insert


@pytest.fixture
async def memory_session_factory() -> AsyncIterator[
    async_sessionmaker[AsyncSession]
]:
    """An initialized in-memory SQLite DB, shared across sessions.

    Built by :func:`create_engine_and_init`, so it has production's
    connection pragmas and ``BEGIN IMMEDIATE`` writer transactions.
    """
    engine = await create_engine_and_init(
        Path(":memory:"), poolclass=StaticPool
    )

    try:
        yield get_session_factory(engine)
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def schema_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A once-built, empty, fully-migrated DB file to copy per example.

    The replay/archive rigs copy it per hypothesis example (the replay
    ``SourceIndex`` opens source DBs read-only, so they must be real files).
    """
    path = tmp_path_factory.mktemp("schema_template") / "template.db"

    async def build() -> None:
        engine, _ = await init_database(path)
        await engine.dispose()

    asyncio.run(build())
    return path
