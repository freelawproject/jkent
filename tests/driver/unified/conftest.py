"""Shared fixtures for unified-driver tests.

``memory_session_factory`` stands up a real, fully-migrated SQLite schema
entirely in memory (StaticPool so every session shares the one connection),
giving DB-backed contract tests a fast, isolated database with no temp files.

``schema_template`` is a once-built, empty, fully-migrated DB *file* that the
replay/archive rigs copy per hypothesis example (the replay ``SourceIndex``
opens source DBs read-only, so they must be real files).
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from jkent.common.decorators import entry, step
from jkent.data_types import (
    ArchiveResponse,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.database_engine.database import (
    get_session_factory,
    init_database,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

#: A GET handler suitable for :func:`aiohttp.web.UrlDispatcher.add_get`.
RouteHandler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@pytest.fixture(scope="session")
def schema_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A once-built, empty, fully-migrated DB file to copy per example."""
    path = tmp_path_factory.mktemp("schema_template") / "template.db"

    async def build() -> None:
        engine, _ = await init_database(path)
        await engine.dispose()

    asyncio.run(build())
    return path


# --- Source-DB materialization (via the production storage path) ---------
#
# The replay/archive rigs need source DBs that look exactly like a real run's
# output. Rather than hand-roll ``INSERT``s (which silently drift from
# production — e.g. missing the empty-body → NULL ``content_compressed`` rule),
# these helpers drive the same ``SQLManager`` + ``ResponseStorage`` path the
# unified worker uses on its success branch.

_DERIVE_DEDUP = object()


async def store_completed_request(
    db: SQLManager,
    storage: ResponseStorage,
    request: Request,
    response: Response,
    *,
    dedup_key: Any = _DERIVE_DEDUP,
) -> int:
    """Insert ``request`` and store ``response`` exactly as the worker would.

    Mirrors the worker success path: serialize + insert the request row, store
    the response through ``ResponseStorage`` (so content compression — and the
    empty-body NULL rule — match production), then mark it completed.

    ``dedup_key`` defaults to the request's own key; pass ``None`` to force a
    NULL ``deduplication_key`` column (the SkipDeduplicationCheck / unpopulated
    path that the URL+body fallback covers).
    """
    if dedup_key is _DERIVE_DEDUP:
        key = request.deduplication_key
        dedup_key = key if isinstance(key, str) else None
    request_data = RequestQueue(db).serialize_request(request)
    request_id = await db.insert_request(
        priority=0,
        dedup_key=dedup_key,
        parent_id=None,
        skip_dedup_check=True,
        **request_data,
    )
    # serialize_request already normalized a callable continuation to its name.
    await storage.store_response(
        request_id, response, request_data["continuation"]
    )
    await db.mark_request_completed(request_id)
    return request_id


def checkpoint_wal(dest: Path) -> None:
    """Fold the WAL into the main DB so a read-only ``SourceIndex`` sees writes."""
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


async def build_source_db(
    template: Path,
    dest: Path,
    rows: list[tuple[Request, Response]],
    *,
    dedup_key: Any = _DERIVE_DEDUP,
) -> None:
    """Copy the schema template and persist ``(request, response)`` rows.

    Each row goes through :func:`store_completed_request`, so the resulting DB
    is byte-for-byte what a real run would produce for those responses. Pass
    ``dedup_key=None`` to force every row's ``deduplication_key`` column NULL.
    """
    shutil.copy(template, dest)
    async with SQLManager.open(dest) as db:
        storage = ResponseStorage(db)
        for request, response in rows:
            await store_completed_request(
                db, storage, request, response, dedup_key=dedup_key
            )
    checkpoint_wal(dest)


@dataclass
class ArchiveSpec:
    """One archived file to materialize: an archive ``request`` whose body was
    written to ``file_path`` on disk."""

    request: Request
    file_path: Path
    content: bytes


async def build_archive_source_db(
    template: Path, dest: Path, specs: list[ArchiveSpec]
) -> None:
    """Write each archive's bytes to disk and record it via the storage path.

    Stores an :class:`ArchiveResponse` through ``ResponseStorage`` (the same
    call the worker makes), so the ``requests`` + ``archived_files`` rows match
    a real archive download. Each ``spec.request`` must be ``archive=True``.
    """
    shutil.copy(template, dest)
    async with SQLManager.open(dest) as db:
        storage = ResponseStorage(db)
        for spec in specs:
            spec.file_path.write_bytes(spec.content)
            url = spec.request.request.url
            response = ArchiveResponse(
                status_code=200,
                headers={},
                content=spec.content,
                text="",
                url=url,
                request=spec.request,
                file_url=str(spec.file_path),
            )
            await store_completed_request(db, storage, spec.request, response)
    checkpoint_wal(dest)


@pytest.fixture
async def memory_session_factory() -> AsyncIterator[async_sessionmaker]:
    """An initialized in-memory SQLite DB, shared across sessions."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    try:
        yield get_session_factory(engine)
    finally:
        await engine.dispose()


@dataclass
class StartedServer:
    """A running aiohttp server on an ephemeral port; ``runner`` tears it down."""

    runner: web.AppRunner
    base_url: str


async def start_app(app: web.Application) -> StartedServer:
    """Start ``app`` on an ephemeral 127.0.0.1 port and return its handle+URL.

    Centralizes the ``AppRunner``/``TCPSite`` boilerplate for HTTP tests that
    need a full ``web.Application`` (custom methods/routes) and so can't use the
    GET-only ``serve_routes`` fixture — e.g. the hypothesis rigs that stand up a
    server inside their own ``asyncio.run`` per example. The caller owns
    teardown via ``await server.runner.cleanup()``.
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


@pytest.fixture
async def serve_routes() -> AsyncIterator[
    Callable[[dict[str, RouteHandler]], Awaitable[str]]
]:
    """Factory that starts an ephemeral-port aiohttp server per call.

    Yields an async ``serve({path: handler})`` returning the server's base
    URL; every server it starts is torn down at fixture teardown. Centralizes
    the ``AppRunner``/``TCPSite`` boilerplate the unified HTTP tests would
    otherwise each repeat.
    """
    runners: list[web.AppRunner] = []

    async def _serve(routes: dict[str, RouteHandler]) -> str:
        app = web.Application()
        for path, handler in routes.items():
            app.router.add_get(path, handler)
        server = await start_app(app)
        runners.append(server.runner)
        return server.base_url

    try:
        yield _serve
    finally:
        for runner in runners:
            await runner.cleanup()


class HttpPageScraper(BaseScraper[dict]):
    """Plain-HTTP scraper: fetch ``{base}/page/{page_id}``, record the body."""

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
        yield ParsedData(data={"body": response.text})


# --- Shared browser-launch gate ------------------------------------------


async def _browser_launches() -> bool:
    """Probe whether a real browser context can be brought up + torn down.

    Uses a requirement-free scraper so engine selection matches the standard
    (chromium) path the browser-gated tests exercise; any launch/teardown
    failure means no usable engine here, so the gated suites skip cleanly.
    """
    transport = PlaywrightTransport(HttpPageScraper(), headless=True)
    try:
        await transport.open()
    except Exception:
        try:
            await transport.aclose()
        except Exception:
            pass
        return False
    await transport.aclose()
    return True


@pytest.fixture(scope="session")
def has_browser() -> bool:
    """Session-wide flag: True iff a browser engine launches in this env."""
    return asyncio.run(_browser_launches())


async def serve_archive_download(body: bytes) -> StartedServer:
    """Serve a parent page with a download link + an attachment endpoint.

    ``/parent`` links to ``/file.bin``; ``/file.bin`` returns ``body`` with a
    ``Content-Disposition: attachment`` header so a browser click downloads it.
    The caller owns teardown via ``await server.runner.cleanup()`` and keeps the
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


async def serve_inline_render(body: bytes) -> StartedServer:
    """Serve a parent page whose link renders inline instead of downloading.

    ``/parent`` links to ``/doc.pdf``; ``/doc.pdf`` returns ``body`` as
    ``text/html`` with no ``Content-Disposition``, so every browser *navigates*
    to it rather than firing a ``download`` event — the same shape as Firefox
    opening a real PDF in pdf.js. The ``.pdf`` path matches the transport's
    file-response capture heuristic, making the inline-archive fallback
    deterministic across engines. Teardown is the caller's, via
    ``await server.runner.cleanup()``.
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
