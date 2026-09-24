"""Tests for ``PlaywrightTransport``.

Covers engine + page lifecycle (``open``/``aclose``, ``acquire``/
``release``), the navigation path (``resolve``: navigate, apply
``await_conditions``, snapshot the DOM, persist incidental sub-requests
against the request's row id, stage a forked tab from a parent's cached
response), crash recovery, and the shared ``Transport`` conformance suite.
Archive download is tested in ``test_playwright_archive.py``.

Browser-dependent tests are skipped cleanly when no browser engine can launch
(camoufox/chromium may be absent here); the structural-conformance checks run
without a browser. The navigation tests stand up a local aiohttp server and a
real in-memory DB.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa
from aiohttp import web
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from pyrate_limiter import Duration, Limiter, Rate
from typing_extensions import override

from jkent.common.decorators import step
from jkent.common.exceptions import TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.common.response import decode_text
from jkent.common.via import FieldResolver, FieldValue
from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    Selector,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.driver.browser_engine.worker_page import WorkerPage
from jkent.driver.database_engine.compression import (
    compress,
    recompress_responses,
    train_compression_dict,
)
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import (
    IncidentalCapture,
    SQLManager,
)
from jkent.driver.unified_driver.interstitials import InterstitialHandler
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.persistence import RowOnlyErrorSink
from jkent.driver.unified_driver.rate_limiter import (
    NoopRateLimiter,
    PyrateRateLimiter,
    RateLimiters,
)
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    QueuedRequest,
    WorkerHandle,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
    ResolveTimeout,
)
from jkent.driver.unified_driver.wiring import RunCollaborators
from jkent.driver.unified_driver.worker import PoolWorker
from tests.db_staging import (
    insert_request_row as _insert_request_row,
)
from tests.db_staging import (
    insert_staged_parent as _insert_staged_parent,
)
from tests.driver.unified.conftest import (
    serve_archive_download,
)
from tests.driver.unified.test_async_lifecycle_conformance import (
    AsyncLifecycleConformance,
)
from tests.driver.unified.test_recoverable_conformance import (
    RecoverableConformance,
)
from tests.driver.unified.test_transport_conformance import (
    ClassificationConformance,
    TransportConformance,
)
from tests.servers import (
    StartedServer,
    single_page_app,
    start_app,
    status_app,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Sequence
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _Scraper(BaseScraper[None]):
    """Minimal scraper (no CFCAP requirement -> standard playwright engine)."""

    @step
    def parse(self, response: Response) -> Generator[Request, None, None]:
        """No-op step so requests can reference ``step='parse'``.

        Worker tests drive a mocked step executor, but the worker still
        resolves this step's (empty) await_list before fetching, so it must be a
        discoverable step.
        """
        yield from ()


@pytest.fixture
async def transport(require_browser: None):
    subject = PlaywrightTransport(_Scraper(), headless=True)
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


def _sql_manager(sf: async_sessionmaker[AsyncSession]) -> SQLManager:
    """Build an ``SQLManager`` over a test ``async_sessionmaker[AsyncSession]``.

    The parent-response read + incidental write touch only the session
    factory and lock; the engine is taken off the factory's bind.
    """
    engine = sf.kw["bind"]
    return SQLManager(engine, sf)


async def test_each_run_db_gets_its_own_browser_data_root(
    tmp_path: Path,
) -> None:
    """Two runs of one scraper never share a browser profile."""
    roots = []
    for name in ("a.db", "b.db"):
        engine, sf = await init_database(tmp_path / name)
        try:
            transport = PlaywrightTransport(
                _Scraper(), db=SQLManager(engine, sf)
            )
            roots.append(transport._build_engine()._user_data_root)
        finally:
            await engine.dispose()
    assert roots == [
        tmp_path / "a.db.browser-data",
        tmp_path / "b.db.browser-data",
    ]


# --- local server --------------------------------------------------------


def _queued_for(url: str) -> QueuedRequest:
    """A minimal QueuedRequest — ``resolve`` reads its url to label failures."""
    return QueuedRequest(request=_request_for(url), request_id=1)


def _request_for(url: str) -> Request:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        step="parse",
    )


# --- Always-runnable structural conformance (no browser) ------------------


def test_exposes_full_transport_surface() -> None:
    """``PlaywrightTransport`` has the whole ``Transport`` method surface.

    Defined on the class itself, not inherited: ``Transport`` supplies
    defaults for several of these, and a missing override would silently fall
    back to them.
    """
    assert not PlaywrightTransport.__abstractmethods__
    for name in (
        "open",
        "aclose",
        "acquire",
        "release",
        "resolve",
        "resolve_archive",
        "finish_archiving",
    ):
        assert name in PlaywrightTransport.__dict__, name


async def test_export_cookies_none_without_context() -> None:
    """``export_cookies`` returns None before a context exists (no browser)."""
    subject = PlaywrightTransport(_Scraper())
    assert await subject.export_cookies() is None


async def test_cookie_round_trip_through_context(
    transport: PlaywrightTransport,
) -> None:
    """Cookies exported from the context re-import to the same cookie set."""
    context = transport._require_context()
    await context.add_cookies(
        [
            {
                "name": "sid",
                "value": "round-trip",
                "domain": "example.com",
                "path": "/",
            }
        ]
    )
    exported = await transport.export_cookies()
    assert exported is not None
    [cookie] = json.loads(exported)
    assert (cookie["name"], cookie["value"], cookie["domain"]) == (
        "sid",
        "round-trip",
        "example.com",
    )

    await context.clear_cookies()
    assert await transport.export_cookies() == "[]"
    await transport.import_cookies(exported)
    reexported = await transport.export_cookies()
    assert reexported is not None
    assert json.loads(reexported) == json.loads(exported)


async def test_resolve_without_db_raises() -> None:
    """``resolve`` needs a DB reference; without one it's a clear error."""
    subject = PlaywrightTransport(_Scraper())
    with pytest.raises(RuntimeError, match="DB reference"):
        await subject.resolve(None, None)  # type: ignore[arg-type]


async def test_acquire_before_open_raises() -> None:
    """Using the transport before ``open`` is a programming error."""
    subject = PlaywrightTransport(_Scraper())
    with pytest.raises(RuntimeError):
        await subject.acquire(0)


async def test_aclose_forgets_the_engine_even_when_teardown_raises() -> None:
    """A failing engine exit still leaves the transport closed, not half-open.

    Otherwise a second ``aclose`` re-exits a context manager that already
    ran, and ``acquire`` keeps handing out pages on a torn-down context.
    """

    class _FailingExit:
        def __init__(self) -> None:
            self.exits = 0

        async def __aexit__(self, *_exc: object) -> None:
            self.exits += 1
            raise RuntimeError("browser close failed")

    subject = PlaywrightTransport(_Scraper())
    cm = _FailingExit()
    subject._engine_cm = cm
    subject._engine = object()  # type: ignore[assignment]
    subject._context = object()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="browser close failed"):
        await subject.aclose()
    assert (subject._engine_cm, subject._engine, subject._context) == (
        None,
        None,
        None,
    )
    await subject.aclose()
    assert cm.exits == 1


# --- Browser-dependent lifecycle (skipped cleanly w/o a browser) ----------


async def test_open_then_aclose_clears_refs(require_browser: None) -> None:
    """``aclose`` releases everything ``open`` acquired (no leak)."""
    subject = PlaywrightTransport(_Scraper(), headless=True)
    await subject.open()
    assert subject._engine is not None
    assert subject._context is not None
    await subject.aclose()
    assert subject._engine is None
    # type-checkers keep the narrowing from the pre-close asserts; aclose() really does reset
    assert subject._engine_cm is None  # type: ignore[unreachable]
    assert subject._context is None
    assert subject._handles == {}


class TestPlaywrightTransportLifecycle(AsyncLifecycleConformance):
    """``PlaywrightTransport`` honors the open -> use -> aclose lifecycle."""

    @override
    @pytest.fixture
    async def subject(self, require_browser: None):
        # Yield + aclose in teardown: the base suite's
        # ``test_open_awaits_to_none`` opens without closing, which would leak
        # a live browser (and, for camoufox, deadlock the profile lock for the
        # next test). The teardown guarantees cleanup after every case.
        transport = PlaywrightTransport(_Scraper(), headless=True)
        try:
            yield transport
        finally:
            await transport.aclose()

    @override
    def live_resources(self, subject: AsyncLifecycle) -> int:
        """Engine, engine context-manager, context, and page handles."""
        assert isinstance(subject, PlaywrightTransport)
        return (
            (subject._engine is not None)
            + (subject._engine_cm is not None)
            + (subject._context is not None)
            + len(subject._handles)
        )


async def test_acquire_returns_worker_handle(
    transport: PlaywrightTransport,
) -> None:
    """``acquire`` yields a ``WorkerHandle`` with no-throw reset/close."""
    handle = await transport.acquire(0)
    assert isinstance(handle, WorkerHandle)
    await handle.reset_for_reuse()
    await handle.close()


async def test_acquire_stable_per_worker(
    transport: PlaywrightTransport,
) -> None:
    """Two acquires for the same worker id return the same handle."""
    first = await transport.acquire(1)
    second = await transport.acquire(1)
    assert first is second


async def test_release_then_acquire_is_fresh(
    transport: PlaywrightTransport,
) -> None:
    """After ``release`` the next ``acquire`` builds a fresh handle."""
    first = await transport.acquire(2)
    await transport.release(2)
    second = await transport.acquire(2)
    assert first is not second


async def test_release_unknown_worker_is_noop(
    transport: PlaywrightTransport,
) -> None:
    """Releasing a worker that never acquired is a no-op."""
    await transport.release(999)


# --- Navigation (skipped cleanly w/o a browser) ---------------------------


@pytest.fixture
async def nav_transport(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
):
    subject = PlaywrightTransport(
        _Scraper(),
        headless=True,
        db=_sql_manager(memory_session_factory),
    )
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


async def test_resolve_returns_served_response(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``resolve`` navigates and returns the served HTML, status, and URL."""
    html = "<html><body><h1 id='ok'>served</h1></body></html>"

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    server = await start_app(app)
    try:
        handle = await nav_transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url)
        request = _request_for(url)
        queued = QueuedRequest(request=request, request_id=rid)
        resp = await nav_transport.resolve(handle, queued)

        assert resp.status_code == 200
        assert "served" in resp.text
        assert resp.url == url
        # The Response carries the exact request object it was given.
        assert resp.request is queued.request
    finally:
        await server.runner.cleanup()


async def test_resolve_applies_await_conditions(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``await_conditions`` are applied before snapshotting (load + selector).

    The ``.ready`` div is inserted by a timer after load, so a snapshot taken
    without waiting for the selector does not contain it.
    """
    html = (
        "<html><body><script>"
        "window.addEventListener('load', () => setTimeout(() => {"
        "const d = document.createElement('div');"
        "d.className = 'ready'; d.textContent = 'here';"
        "document.body.appendChild(d);"
        "}, 300));"
        "</script></body></html>"
    )

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    server = await start_app(app)
    try:
        handle = await nav_transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url)
        request = _request_for(url)
        queued = QueuedRequest(request=request, request_id=rid)
        resp = await nav_transport.resolve(
            handle,
            queued,
            await_conditions=(
                WaitForLoadState(state="load", timeout=5000),
                WaitForSelector(selector=".ready", timeout=5000),
            ),
        )
        # The selector the await waited for is present in the snapshot.
        assert '<div class="ready">here</div>' in resp.text
        assert resp.status_code == 200
    finally:
        await server.runner.cleanup()


async def test_resolve_persists_incidentals(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A sub-request (fetch) is captured and persisted against request_id.

    ``/sub.json`` sends its headers at once and its body later, and the page
    marks the DOM as soon as ``fetch`` resolves (on headers). The resolve's
    await ends on that marker, so the body read is still in flight at
    snapshot time and lands in the row only if resolve drains the captures.
    """
    page_html = (
        "<html><body><script>"
        "fetch('/sub.json').then(r => {"
        "const d = document.createElement('div');"
        "d.id = 'fetched'; document.body.appendChild(d);"
        "return r.text();"
        "});"
        "</script></body></html>"
    )

    async def page_handler(_request: web.Request) -> web.Response:
        return web.Response(
            status=200, body=page_html, content_type="text/html"
        )

    async def sub_handler(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200, headers={"content-type": "application/json"}
        )
        await response.prepare(request)
        await asyncio.sleep(0.5)
        await response.write(b'{"k": "v"}')
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_get("/page", page_handler)
    app.router.add_get("/sub.json", sub_handler)
    server = await start_app(app)
    try:
        handle = await nav_transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url)
        request = _request_for(url)
        queued = QueuedRequest(request=request, request_id=rid)
        await nav_transport.resolve(
            handle,
            queued,
            await_conditions=(
                WaitForSelector(
                    selector="#fetched", state="attached", timeout=5000
                ),
            ),
        )

        sf = memory_session_factory
        async with sf() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT ir.parent_request_id, ir.url, ir.status_code, "
                        "s.content_compressed IS NOT NULL "
                        "FROM incidental_requests ir "
                        "LEFT JOIN incidental_request_storage s "
                        "ON s.id = ir.storage_id"
                    )
                )
            ).all()
        # Exactly the navigation and its fetch, both tagged to the navigating
        # request id, both with the status and body stored (the body only
        # lands if resolve drained the capture tasks before persisting).
        assert sorted(tuple(row) for row in rows) == [
            (rid, url, 200, True),
            (rid, f"{server.base_url}/sub.json", 200, True),
        ]
    finally:
        await server.runner.cleanup()


async def test_resolve_stages_parent_then_via_navigates(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A via child stages the cached parent, then clicks through to the child.

    The cached parent body (served from the DB via route-intercept) carries a
    link to the real child; staging loads the parent, then via-navigation
    clicks the link and the snapshot is the *child* page.
    """

    async def child(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body><h1 id='child'>child-page</h1></body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/child", child)
    server = await start_app(app)
    try:
        child_url = f"{server.base_url}/child"
        staged_url = "https://staged.example/parent"
        staged_body = (
            f"<html><body><a id='go' href='{child_url}'>go</a></body></html>"
        ).encode()
        compressed = compress(staged_body)

        sf = memory_session_factory
        async with sf() as session:
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location, response_status_code,
                        response_url, response_headers_json, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES (:status, 9, :method, :url, 'parse', '', 200,
                        :url, NULL, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": staged_url,
                    "compressed": compressed,
                    "osize": len(staged_body),
                    "csize": len(compressed),
                },
            )
            await session.commit()
            parent_id = (
                await session.execute(
                    sa.text("SELECT id FROM requests WHERE url = :url"),
                    {"url": staged_url},
                )
            ).scalar_one()

        handle = await nav_transport.acquire(0)
        child_id = await _insert_request_row(sf, child_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=child_url),
            step="parse",
            via=ViaLink(selector=Selector.CSS("#go"), description="to child"),
        )
        queued = QueuedRequest(
            request=request,
            request_id=child_id,
            parent_request_id=int(parent_id),
        )
        resp = await nav_transport.resolve(handle, queued)

        assert resp.url.endswith("/child")
        assert "child-page" in resp.text
        assert resp.request is queued.request
    finally:
        await server.runner.cleanup()


async def test_resolve_no_via_child_navigates_to_own_url(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A child with a parent only for lineage (no via) goes to its OWN url.

    Regression guard: staging must NOT fire for a no-via child, even with a
    parent_request_id — it would otherwise snapshot the parent (the fixed bug).
    """

    async def own(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body id='own'>own-page</body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/own", own)
    server = await start_app(app)
    try:
        sf = memory_session_factory
        # A parent row exists but is never staged (the child has no via).
        parent_id = await _insert_request_row(
            sf, "https://staged.example/parent"
        )
        child_url = f"{server.base_url}/own"
        child_id = await _insert_request_row(sf, child_url)

        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=_request_for(child_url),
            request_id=child_id,
            parent_request_id=parent_id,
        )
        resp = await nav_transport.resolve(handle, queued)

        assert resp.url.endswith("/own")
        assert "own-page" in resp.text
    finally:
        await server.runner.cleanup()


async def test_resolve_via_child_of_unstored_parent_navigates_to_own_url(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    serve_routes: Any,
) -> None:
    """A via child whose parent has no stored response goes to its own url."""

    async def own(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body id='own'>own-page</body></html>",
            content_type="text/html",
        )

    base = await serve_routes({"/own": own})
    sf = memory_session_factory
    parent_id = await _insert_request_row(sf, "https://staged.example/parent")
    child_url = f"{base}/own"
    child_id = await _insert_request_row(sf, child_url)

    handle = await nav_transport.acquire(0)
    request = Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=child_url),
        step="parse",
        via=ViaLink(selector=Selector.CSS("#go"), description="to child"),
    )
    queued = QueuedRequest(
        request=request, request_id=child_id, parent_request_id=parent_id
    )
    resp = await nav_transport.resolve(handle, queued)

    assert resp.url == child_url
    assert "own-page" in resp.text


#: Enough similar pages for zstd to train a dictionary on.
_DICT_CORPUS = [
    (
        "<html><head><title>Opinion {n}</title></head><body>"
        "<div class='case-header'><h1>Case Number: {n}</h1></div>"
        "<div class='opinion'><p>The court finds that the defendant in "
        "matter {n} is liable for damages. The plaintiff's motion for "
        "summary judgment is granted.</p></div></body></html>"
    )
    .replace("{n}", str(n))
    .encode()
    for n in range(20)
]


async def test_stage_parent_tab_serves_dictionary_compressed_body(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A parent recompressed onto a trained dictionary stages its real body."""
    sf = memory_session_factory
    ids = [
        await _insert_staged_parent(
            sf, url=f"https://staged.example/p/{n}", body=body
        )
        for n, body in enumerate(_DICT_CORPUS)
    ]
    db = nav_transport._db
    assert db is not None
    dict_id = await train_compression_dict(
        db, "parse", sample_limit=len(ids), dict_size=4096
    )
    await recompress_responses(db, "parse", dict_id=dict_id)
    parent = await db.get_stored_response(ids[7])
    assert parent is not None
    assert parent.compression_dict_id == dict_id

    handle = await nav_transport.acquire(0)
    staged = await nav_transport._stage_parent_tab(
        handle.page, ids[7], timeout_ms=10_000
    )

    assert staged is True
    assert handle.page.url == "https://staged.example/p/7"
    assert "Case Number: 7" in await handle.page.content()


class _ClearingHandler(InterstitialHandler):
    """Detects ``#challenge``; clearing it adds ``#content`` after a delay."""

    def __init__(self) -> None:
        self.cleared = 0

    def waitlist(self) -> list[Any]:
        return [
            WaitForSelector("#challenge", state="attached", timeout=30_000)
        ]

    async def navigate_through(self, page: Any) -> None:
        self.cleared += 1
        await page.evaluate(
            """() => {
                document.body.innerHTML = '<p>cleared</p>';
                setTimeout(() => {
                    const d = document.createElement('div');
                    d.id = 'content';
                    d.textContent = 'real-content';
                    document.body.appendChild(d);
                }, 500);
            }"""
        )


async def test_resolve_interstitial_win_navigates_through_then_awaits(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    serve_routes: Any,
) -> None:
    """A winning handler clears the page, then the scraper's awaits apply.

    The challenge's own 403 describes a document that is gone, so the
    response falls back to 200 instead of claiming it.
    """

    async def challenge(_request: web.Request) -> web.Response:
        return web.Response(
            status=403,
            text="<html><body><div id='challenge'>wait</div></body></html>",
            content_type="text/html",
        )

    base = await serve_routes({"/page": challenge})
    handler = _ClearingHandler()
    nav_transport._interstitial_handlers = [handler]
    url = f"{base}/page"
    rid = await _insert_request_row(memory_session_factory, url)

    handle = await nav_transport.acquire(0)
    resp = await nav_transport.resolve(
        handle,
        QueuedRequest(request=_request_for(url), request_id=rid),
        await_conditions=(WaitForSelector("#content", timeout=30_000),),
    )

    assert handler.cleared == 1
    assert "real-content" in resp.text
    assert "challenge" not in resp.text
    assert resp.status_code == 200


async def test_resolve_honours_wait_for_url(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    serve_routes: Any,
) -> None:
    """A ``WaitForURL`` holds the snapshot until the page reaches the URL."""

    async def start(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<html><body><script>"
                "setTimeout(() => history.pushState({}, '', '/done'), 300);"
                "</script></body></html>"
            ),
            content_type="text/html",
        )

    base = await serve_routes({"/start": start})
    url = f"{base}/start"
    rid = await _insert_request_row(memory_session_factory, url)

    handle = await nav_transport.acquire(0)
    resp = await nav_transport.resolve(
        handle,
        QueuedRequest(request=_request_for(url), request_id=rid),
        await_conditions=(WaitForURL("**/done", timeout=5000),),
    )

    assert resp.url == f"{base}/done"


async def test_resolve_honours_wait_for_timeout(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    serve_routes: Any,
) -> None:
    """A ``WaitForTimeout`` delays the snapshot by at least its timeout."""

    async def page(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body>page</body></html>", content_type="text/html"
        )

    base = await serve_routes({"/page": page})
    url = f"{base}/page"
    rid = await _insert_request_row(memory_session_factory, url)

    handle = await nav_transport.acquire(0)
    loop = asyncio.get_running_loop()
    started = loop.time()
    await nav_transport.resolve(
        handle,
        QueuedRequest(request=_request_for(url), request_id=rid),
        await_conditions=(WaitForTimeout(400),),
    )

    assert loop.time() - started >= 0.4


# --- ViaFormSubmit navigation --------------------------------------------


async def test_resolve_via_form_submit_navigates_with_submitted_data(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A ``ViaFormSubmit`` child fills a staged form and submits to the result.

    The parent (the form page) is staged from cache; the via fills a visible
    text field and clicks submit; the GET result page echoes the submitted
    value, proving the form's data reached the server.
    """
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        q = request.query.get("case_type", "")
        return web.Response(
            text=(f"<html><body><div id='echo'>{q}</div></body></html>"),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/search"
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            "<input type='text' name='case_type'>"
            "<button type='submit' id='go'>Search</button>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector="#go",
                field_data={"case_type": "Property Dispute"},
                description="case search",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        assert "Property Dispute" in resp.text
        assert "case_type=Property+Dispute" in resp.url
    finally:
        await server.runner.cleanup()


async def test_resolve_via_form_submit_hidden_radio_select_fields(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``ViaFormSubmit`` fills hidden, invisible, radio, and select fields.

    Regression guard for the form-fill subset: the result page echoes each
    field's submitted value, proving hidden/invisible inputs were assigned
    (not skipped), the radio was checked, and the select was chosen.
    """
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        q = request.query
        return web.Response(
            text=(
                "<html><body>"
                f"<div id='hidden'>{q.get('vs', '')}</div>"
                f"<div id='invisible'>{q.get('iso', '')}</div>"
                f"<div id='category'>{q.get('category', '')}</div>"
                f"<div id='case_type'>{q.get('case_type', '')}</div>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/complex"
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            "<input type='hidden' name='vs' value=''>"
            "<input name='iso' style='display:none' value=''>"
            "<input type='radio' name='category' value='civil'>"
            "<input type='radio' name='category' value='criminal' checked>"
            "<select name='case_type'>"
            "<option value='Contract Dispute'>Contract</option>"
            "<option value='Defamation' selected>Defamation</option>"
            "</select>"
            "<button type='submit' id='go'>Search</button>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector="#go",
                field_data={
                    "vs": "viewstate-token",
                    "iso": "2024-01-01",
                    "category": "civil",
                    "case_type": "Contract Dispute",
                },
                description="complex search",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        # Each echoed field proves its value reached the server (the GET query
        # is also URL-encoded into resp.url).
        assert '<div id="hidden">viewstate-token</div>' in resp.text
        assert '<div id="invisible">2024-01-01</div>' in resp.text
        assert '<div id="category">civil</div>' in resp.text  # radio switched
        assert (
            '<div id="case_type">Contract Dispute</div>' in resp.text
        )  # select chosen
        assert "vs=viewstate-token" in resp.url
        assert "category=civil" in resp.url
        assert "case_type=Contract+Dispute" in resp.url
    finally:
        await server.runner.cleanup()


async def test_resolve_via_form_submit_repeated_field_values(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A list ``field_data`` value replays a checkbox group and multi-select.

    Repeated keys (list values) must reach the server as repeated names, just
    as the browser POSTs them: every matching checkbox is checked and every
    matching ``<select multiple>`` option is selected.
    """
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        # ``query.getall`` collects repeated keys, matching what the browser
        # sends for a checkbox group / multi-select.
        cats = ",".join(request.query.getall("category", []))
        types = ",".join(request.query.getall("case_type", []))
        return web.Response(
            text=(
                "<html><body>"
                f"<div id='category'>{cats}</div>"
                f"<div id='case_type'>{types}</div>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/repeated"
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            "<input type='checkbox' name='category' value='civil'>"
            "<input type='checkbox' name='category' value='criminal'>"
            "<input type='checkbox' name='category' value='family'>"
            "<select name='case_type' multiple>"
            "<option value='Contract'>Contract</option>"
            "<option value='Defamation'>Defamation</option>"
            "<option value='Tort'>Tort</option>"
            "</select>"
            "<button type='submit' id='go'>Search</button>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector="#go",
                field_data={
                    "category": ["civil", "family"],
                    "case_type": ["Contract", "Tort"],
                },
                description="repeated-key search",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        # Both checked boxes and both selected options reached the server as
        # repeated keys (and the unchecked 'criminal'/'Defamation' did not).
        assert '<div id="category">civil,family</div>' in resp.text
        assert '<div id="case_type">Contract,Tort</div>' in resp.text
        assert "category=civil" in resp.url
        assert "category=family" in resp.url
        assert "criminal" not in resp.url
    finally:
        await server.runner.cleanup()


@pytest.mark.parametrize(
    ("controls", "field_data", "expected"),
    [
        pytest.param(
            "<input type='radio' name='c' value='civil'>"
            "<input type='radio' name='c' value='criminal' checked>",
            {"c": "family"},
            [("c", "family")],
            id="radio-value-matches-nothing",
        ),
        pytest.param(
            "<input type='checkbox' name='c' value='a' checked>"
            "<input type='checkbox' name='c' value='b'>",
            {"c": "b"},
            [("c", "b")],
            id="checkbox-scalar-replaces-rendered",
        ),
        pytest.param(
            "<input type='checkbox' name='c' value='a' checked>"
            "<input type='checkbox' name='c' value='b' checked>",
            {"c": []},
            [],
            id="checkbox-group-emptied",
        ),
        pytest.param(
            "<input type='checkbox' name='c' value='a' checked>"
            "<input type='checkbox' name='c' value='b'>",
            {"c": ["b", "z"]},
            [("c", "b"), ("c", "z")],
            id="checkbox-list-replaces-rendered",
        ),
        pytest.param(
            "<select name='c'><option value='a' selected>A</option></select>",
            {"c": "nope"},
            [("c", "nope")],
            id="select-value-matches-no-option",
        ),
        pytest.param(
            "<select name='c' multiple><option value='a'>A</option>"
            "<option value='b' selected>B</option></select>",
            {"c": ["a", "nope"]},
            [("c", "a"), ("c", "nope")],
            id="multi-select-value-matches-no-option",
        ),
    ],
)
async def test_resolve_via_form_submit_choice_submits_exactly_field_data(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    controls: str,
    field_data: dict[str, FieldValue | FieldResolver],
    expected: list[tuple[str, str]],
) -> None:
    """A radio/checkbox group or select submits exactly its ``field_data``.

    ``field_data`` is what the HTTP transport posts, so the browser must
    uncheck rendered boxes the override dropped and deselect options it left
    out, rather than leaving the rendered state in place. A value no box
    carries goes as a hidden input; one no option carries gets an injected
    ``<option>`` (``select_option`` alone would time out).
    """
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        pairs = [(k, v) for k, v in request.query.items() if k != "go"]
        return web.Response(
            text=f"<html><body><pre id='q'>{pairs!r}</pre></body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/groups"
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            f"{controls}"
            "<button type='submit' id='go'>Search</button>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )
        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector="#go",
                field_data=field_data,
                description="group search",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        assert html.escape(repr(expected), quote=False) in resp.text
    finally:
        await server.runner.cleanup()


async def test_resolve_via_form_submit_event_target_postback(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An ``__EVENTTARGET`` form submits programmatically (ASP.NET postback)."""
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        q = request.query
        return web.Response(
            text=(
                "<html><body>"
                f"<div id='target'>{q.get('__EVENTTARGET', '')}</div>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/postback"
        # No submit button: the __EVENTTARGET path submits the form via JS.
        # The form needs a visible element so wait_for_selector resolves.
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            "<input type='hidden' name='__EVENTTARGET' value=''>"
            "<input type='text' name='q' value='x'>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector=None,
                field_data={"__EVENTTARGET": "btnNext"},
                description="postback",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        assert "__EVENTTARGET=btnNext" in resp.url
        assert "btnNext" in resp.text
    finally:
        await server.runner.cleanup()


async def test_resolve_via_form_submit_injects_absent_fields(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """``field_data`` names with no rendered control are submitted anyway.

    ViaFormSubmit can carry fields the form never showed (merged overrides a
    scraper passed to ``Form.submit``). The fill path injects a hidden input for
    each, so they reach the server exactly as the HTTP transport sends them —
    both a scalar and a repeated (list) absent key.
    """
    sf = memory_session_factory

    async def results(request: web.Request) -> web.Response:
        q = request.query
        extra_list = ",".join(q.getall("absent_list", []))
        return web.Response(
            text=(
                "<html><body>"
                f"<div id='present'>{q.get('q', '')}</div>"
                f"<div id='absent'>{q.get('absent', '')}</div>"
                f"<div id='absent_list'>{extra_list}</div>"
                "</body></html>"
            ),
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/results", results)
    server = await start_app(app)
    try:
        form_url = "https://staged.example/inject"
        # The form renders only ``q``; ``absent`` / ``absent_list`` are not here.
        form_body = (
            "<html><body>"
            f"<form id='f' method='get' action='{server.base_url}/results'>"
            "<input type='text' name='q'>"
            "<button type='submit' id='go'>Search</button>"
            "</form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            step="parse",
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector="#go",
                field_data={
                    "q": "rendered",
                    "absent": "injected",
                    "absent_list": ["a", "b"],
                },
                description="inject absent fields",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await nav_transport.resolve(handle, queued)

        # The rendered field and both absent fields reached the server.
        assert '<div id="present">rendered</div>' in resp.text
        assert '<div id="absent">injected</div>' in resp.text
        assert '<div id="absent_list">a,b</div>' in resp.text
        assert "absent=injected" in resp.url
        assert "absent_list=a" in resp.url
        assert "absent_list=b" in resp.url
    finally:
        await server.runner.cleanup()


async def test_resolve_archive_form_submit_swallowed_button_downloads(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A download whose submit control was swallowed by bad HTML still fires.

    Regression: Court-PASS emits an unclosed ``<style>`` that turns the
    ``gvFiles`` download ``<input type=submit>`` into raw text, so the live DOM
    has no such element. ``_fill_form_fields`` then synthesizes a hidden input
    carrying the button's name/value; the submit_selector resolves to that
    hidden input, which cannot be ``click()``-ed (it is never visible). The
    download path must fall back to a bare ``form.submit()`` — its name=value is
    already on the form — so the POST reaches the server and the file downloads.
    """
    sf = memory_session_factory
    received: dict[str, str] = {}

    async def download(request: web.Request) -> web.Response:
        received.update(
            {
                k: v
                for k, v in (await request.post()).items()
                if isinstance(v, str)
            }
        )
        return web.Response(
            body=b"%PDF-1.7\nfake-pdf\n",
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="file.pdf"',
            },
        )

    app = web.Application()
    app.router.add_post("/dl", download)
    server = await start_app(app)
    try:
        button = "ctl00$cphMain$gvFiles$ctl02$bttnDownload"
        form_url = "https://staged.example/filing"
        # The download button lives *inside* an unclosed <style>, so the browser
        # parses it as raw text — there is no <input> element in the live DOM.
        form_body = (
            "<html><body>"
            f"<form id='Form2' method='post' action='{server.base_url}/dl'>"
            "<h1>Filing detail</h1>"
            "<input type='hidden' name='__VIEWSTATE' value=''>"
            "<style pdffontname='Times-Roman'> citation leakage text "
            f"<input type='submit' name='{button}' value='Download PDF'>"
            "</style></form></body></html>"
        ).encode()
        parent_id = await _insert_staged_parent(
            sf, url=form_url, body=form_body
        )

        file_url = f"{server.base_url}/dl"
        child_id = await _insert_request_row(sf, file_url)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST, url=file_url, timeout=30
            ),
            step="collect",
            archive=True,
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#Form2"),
                submit_selector=f"input[name='{button}']",
                field_data={"__VIEWSTATE": "vs-token", button: "Download PDF"},
                description="file download",
            ),
        )
        handle = await nav_transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        stream = await nav_transport.resolve_archive(handle, queued)
        chunks = b"".join([chunk async for chunk in stream])

        assert chunks.startswith(b"%PDF")
        # The swallowed button's name=value and the real hidden field both
        # reached the server via the fallback form.submit().
        assert received.get(button) == "Download PDF"
        assert received.get("__VIEWSTATE") == "vs-token"
    finally:
        await server.runner.cleanup()


# --- Crash recovery (browser-free) ----------------------------------------


class _RecoveryTransport(PlaywrightTransport):
    """Transport with a browser-free rebuild step for recovery tests.

    Overrides the single browser-touching method ``_rebuild_context`` with a
    counter that yields once (as the real rebuild awaits the engine), so the
    generation / single-flight logic is exercisable without launching a
    browser.
    """

    def __init__(self) -> None:
        super().__init__(_Scraper())
        self.rebuild_count = 0
        # A non-None sentinel so _require_context() / acquire don't trip the
        # "used before open()" guard during recovery tests.
        self._context = object()  # type: ignore[assignment]

    @override
    async def _rebuild_context(self) -> None:
        self.rebuild_count += 1
        await asyncio.sleep(0)
        self._context = object()  # type: ignore[assignment]


class TestPlaywrightTransportRecoverable(RecoverableConformance):
    """Run the crash-recovery conformance suite against a browser-free subject."""

    @pytest.fixture
    @override
    def subject(self) -> _RecoveryTransport:
        return _RecoveryTransport()

    @override
    def make_subject(self) -> _RecoveryTransport:
        return _RecoveryTransport()

    @override
    def dead_exc(self) -> BaseException:
        return Exception("Connection closed")


async def test_resolve_remaps_dead_connection_and_poisons_handle() -> None:
    """A dead-connection in _resolve poisons the handle + raises transient."""

    class _StubPage:
        def is_closed(self) -> bool:
            return False

        async def close(self) -> None:
            return None

    class _StubHandle:
        def __init__(self) -> None:
            self.page = _StubPage()
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    t = PlaywrightTransport(_Scraper(), db=object())  # type: ignore[arg-type]
    handle = _StubHandle()
    t._handles[0] = handle  # type: ignore[assignment]

    async def _boom(
        _handle: object,
        _queued: object,
        _await: Sequence[object],
    ) -> Response:
        raise Exception("Connection closed")

    t._resolve = _boom  # type: ignore[assignment, method-assign]

    with pytest.raises(TransientException):
        await t.resolve(handle, _queued_for("https://example.com/p"))  # type: ignore[arg-type]

    # The handle was poisoned (removed from the cache) and closed.
    assert 0 not in t._handles
    assert handle.closed is True


async def test_resolve_remaps_playwright_timeout_to_transient() -> None:
    """A Playwright timeout in _resolve becomes a transient (handle kept).

    A slow page load or an await_list selector that never appears is
    retryable, not a hard/structural failure — so the worker retries with
    backoff rather than marking the request failed. The page is still alive,
    so the handle must NOT be poisoned.
    """
    t = PlaywrightTransport(_Scraper(), db=object())  # type: ignore[arg-type]
    sentinel = object()
    t._handles[0] = sentinel  # type: ignore[assignment]

    async def _timeout(
        _handle: object,
        _queued: object,
        _await: Sequence[object],
    ) -> Response:
        raise PlaywrightTimeoutError("Timeout 15000ms exceeded")

    t._resolve = _timeout  # type: ignore[assignment, method-assign]

    with pytest.raises(TransientException, match="timeout"):
        await t.resolve(sentinel, _queued_for("https://example.com/p"))  # type: ignore[arg-type]
    # Handle untouched: a timeout doesn't kill the connection.
    assert t._handles[0] is sentinel


async def test_resolve_snapshots_dom_on_timeout() -> None:
    """An await timeout still snapshots the DOM, carried on ResolveTimeout.

    Mirrors the old driver: capture the (partial) page on timeout so the
    failed attempt is inspectable, then raise a (transient) ResolveTimeout
    carrying that snapshot for the worker to persist before retrying.
    """

    class _FakePage:
        url = "https://example.com/x"

        def __init__(self) -> None:
            self.calls: list[str] = []

        async def set_extra_http_headers(self, _headers: object) -> None:
            return None

        async def goto(
            self,
            _url: str,
            wait_until: object = None,
            timeout: object = None,
        ) -> None:
            return None

        async def wait_for_selector(
            self, _selector: str, state: object = None, timeout: object = None
        ) -> None:
            raise PlaywrightTimeoutError(f"Timeout {timeout}ms exceeded")

        async def evaluate(self, script: str) -> None:
            self.calls.append(script)

        async def content(self) -> str:
            self.calls.append("content")
            return "<html>partial</html>"

    class _FakeHandle:
        def __init__(self) -> None:
            self.page = _FakePage()
            self.incidental_requests: list[IncidentalCapture] = []
            self.current_parent_request_id: int | None = None

        def clear_request_state(self) -> None:
            self.incidental_requests = []

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        def __init__(self) -> None:
            self.replaced: list[tuple[object, ...]] = []

        async def replace_incidental_requests(self, *a: object) -> list[int]:
            self.replaced.append(a)
            return []

    db = _FakeDB()
    t = PlaywrightTransport(_Scraper(), db=db)  # type: ignore[arg-type]
    queued = QueuedRequest(
        request=_request_for("https://example.com/x"), request_id=7
    )

    handle = _FakeHandle()
    with pytest.raises(ResolveTimeout) as excinfo:
        await t._resolve(
            handle,  # type: ignore[arg-type]
            queued,
            [WaitForSelector("#missing")],
        )
    snapshot = excinfo.value.debug_response
    assert snapshot is not None
    assert snapshot.text == "<html>partial</html>"
    assert snapshot.url == "https://example.com/x"
    # The runaway load is stopped before the snapshot: Playwright's timeout
    # only ends the wait, so window.stop() must abort the browser-side load.
    assert handle.page.calls == ["window.stop()", "content"]
    # Nothing captured still replaces: an earlier attempt's captures go.
    assert db.replaced == [(7, [])]


async def test_resolve_goto_honors_request_timeout() -> None:
    """The request's timeout reaches page.goto as milliseconds.

    A ``(connect, read)`` tuple uses the read element; an unset timeout
    falls back to the transport's own timeout (never Playwright's default).
    """

    class _FakePage:
        url = "https://example.com/x"

        def __init__(self) -> None:
            self.goto_kwargs: dict[str, object] = {}

        async def set_extra_http_headers(self, _headers: object) -> None:
            return None

        async def goto(self, _url: str, **kwargs: object) -> None:
            self.goto_kwargs = kwargs

        async def content(self) -> str:
            return "<html>ok</html>"

    class _FakeHandle:
        def __init__(self) -> None:
            self.page = _FakePage()
            self.incidental_requests: list[IncidentalCapture] = []

        def clear_request_state(self) -> None:
            self.incidental_requests = []

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        async def replace_incidental_requests(self, *_a: object) -> list[int]:
            return []

    t = PlaywrightTransport(
        _Scraper(),
        db=_FakeDB(),  # type: ignore[arg-type]
        timeout=7.5,
    )

    cases: tuple[tuple[dict[str, Any], float], ...] = (
        ({"timeout": 45}, 45000.0),
        ({"timeout": (5, 90)}, 90000.0),
        ({}, 7500.0),  # unset -> the transport's timeout
    )
    for extra, expected in cases:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com/x",
                **extra,
            ),
            step="parse",
        )
        handle = _FakeHandle()
        await t._resolve(
            handle,  # type: ignore[arg-type]
            QueuedRequest(request=request, request_id=7),
            [],
        )
        assert handle.page.goto_kwargs.get("timeout") == expected


@pytest.mark.parametrize("with_handler", [False, True], ids=["plain", "race"])
async def test_await_conditions_without_a_timeout_use_the_requests(
    with_handler: bool,
) -> None:
    """An await condition with no ``timeout`` waits the request's, not 30 s.

    Its own ``timeout`` still wins. Both paths apply conditions: the plain
    loop, and the race against an interstitial handler's waitlist.
    """

    class _FakePage:
        url = "https://example.com/x"

        def __init__(self) -> None:
            self.waits: dict[str, object] = {}

        async def set_extra_http_headers(self, _headers: object) -> None:
            return None

        async def goto(self, _url: str, **_kwargs: object) -> None:
            return None

        async def wait_for_selector(
            self, selector: str, **kwargs: Any
        ) -> None:
            if selector == ".interstitial":
                # The handler's marker never appears: it loses the race.
                raise PlaywrightTimeoutError("absent")
            self.waits[selector] = kwargs["timeout"]

        async def wait_for_load_state(self, state: str, **kwargs: Any) -> None:
            self.waits[state] = kwargs["timeout"]

        async def wait_for_url(self, url: str, **kwargs: Any) -> None:
            self.waits[url] = kwargs["timeout"]

        async def content(self) -> str:
            return "<html>ok</html>"

    class _FakeHandle:
        def __init__(self) -> None:
            self.page = _FakePage()
            self.incidental_requests: list[IncidentalCapture] = []

        def clear_request_state(self) -> None:
            return None

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        async def replace_incidental_requests(self, *_a: object) -> list[int]:
            return []

    class _AbsentHandler(InterstitialHandler):
        def waitlist(self) -> list[Any]:
            return [WaitForSelector(".interstitial", timeout=1)]

        async def navigate_through(self, page: Any) -> None:
            raise AssertionError("never present")

    t = PlaywrightTransport(_Scraper(), db=_FakeDB())  # type: ignore[arg-type]
    if with_handler:
        t._interstitial_handlers = [_AbsentHandler()]
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/x", timeout=12
        ),
        step="parse",
    )
    handle = _FakeHandle()
    await t._resolve(
        handle,  # type: ignore[arg-type]
        QueuedRequest(request=request, request_id=7),
        [
            WaitForSelector(".content"),
            WaitForSelector(".own", timeout=500),
            WaitForLoadState("networkidle"),
            WaitForURL("**/x"),
        ],
    )
    assert handle.page.waits == {
        ".content": 12000.0,
        ".own": 500,
        "networkidle": 12000.0,
        "**/x": 12000.0,
    }


async def test_resolve_snapshot_bytes_declare_their_utf8() -> None:
    """The stored DOM snapshot re-decodes as the text it was taken from.

    ``page.content()`` keeps the page's own ``<meta charset>``; the snapshot
    is UTF-8, and ``decode_text`` trusts the document over the header, so a
    stale windows-1252 declaration garbles every later read of the bytes.
    """
    html = (
        '<html><head><meta charset="windows-1252"></head>'
        "<body>caf\xe9</body></html>"
    )

    class _FakePage:
        url = "https://example.com/x"

        async def set_extra_http_headers(self, _headers: object) -> None:
            return None

        async def goto(self, _url: str, **_kwargs: object) -> None:
            return None

        async def content(self) -> str:
            return html

    class _FakeHandle:
        def __init__(self) -> None:
            self.page = _FakePage()
            self.incidental_requests: list[IncidentalCapture] = []

        def clear_request_state(self) -> None:
            self.incidental_requests = []

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        async def replace_incidental_requests(self, *_a: object) -> list[int]:
            return []

    t = PlaywrightTransport(_Scraper(), db=_FakeDB())  # type: ignore[arg-type]
    response = await t._resolve(
        _FakeHandle(),  # type: ignore[arg-type]
        QueuedRequest(
            request=_request_for("https://example.com/x"), request_id=7
        ),
        [],
    )
    assert "caf\xe9" in response.text
    assert decode_text(response.content, response.headers) == response.text


async def test_resolve_propagates_non_dead_error_unwrapped() -> None:
    """A non-dead error from _resolve is not re-mapped and not poisoned."""
    t = PlaywrightTransport(_Scraper(), db=object())  # type: ignore[arg-type]
    sentinel = object()
    t._handles[0] = sentinel  # type: ignore[assignment]

    async def _boom(
        _handle: object,
        _queued: object,
        _await: Sequence[object],
    ) -> Response:
        raise ValueError("ordinary parse failure")

    t._resolve = _boom  # type: ignore[assignment, method-assign]

    with pytest.raises(ValueError, match="ordinary parse failure"):
        await t.resolve(sentinel, _queued_for("https://example.com/p"))  # type: ignore[arg-type]
    # Handle untouched: only dead-connection errors poison.
    assert t._handles[0] is sentinel


async def test_reset_for_reuse_stops_inflight_navigation() -> None:
    """``reset_for_reuse`` aborts any in-flight navigation before its goto.

    A timed-out/abandoned goto keeps navigating in the browser after
    Playwright stops waiting; window.stop() must fire before the about:blank
    goto so the two don't race ("interrupted by another navigation"). A
    failing stop (e.g. the execution context died as the navigation
    committed) must not break the reset.
    """

    class _RecordingPage:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def on(self, _event: str, _listener: object) -> None:
            return None

        async def evaluate(self, script: str) -> None:
            self.calls.append(script)

        async def goto(self, url: str, wait_until: object = None) -> None:
            self.calls.append(f"goto:{url}")

    page = _RecordingPage()
    await WorkerPage(page, set()).reset_for_reuse()  # type: ignore[arg-type]
    assert page.calls == ["window.stop()", "goto:about:blank"]

    class _StopDiesPage(_RecordingPage):
        @override
        async def evaluate(self, script: str) -> None:
            raise PlaywrightError("Execution context was destroyed")

    dying = _StopDiesPage()
    await WorkerPage(dying, set()).reset_for_reuse()  # type: ignore[arg-type]
    assert dying.calls == ["goto:about:blank"]


async def test_acquire_rebuilds_page_when_reset_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Any reset_for_reuse failure discards the page and builds a fresh one.

    Seen in the wild when a prior slow/timed-out request leaves the reused
    page mid-navigation: the reset's about:blank goto raises ``interrupted
    by another navigation`` / ``NS_BINDING_ABORTED``, which matches none of
    ``should_restart``'s dead-connection strings. That must yield a fresh
    page (with a warning logged), not a hard request failure.
    """

    class _StalePage:
        """Reports itself open but refuses the reset navigation."""

        def __init__(self) -> None:
            self.closed = False

        def is_closed(self) -> bool:
            return False

        def on(self, _event: str, _listener: object) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def goto(self, _url: str, wait_until: object = None) -> None:
            raise PlaywrightError(
                'Page.goto: Navigation to "about:blank" is interrupted '
                'by another navigation to "about:blank"'
            )

    class _FreshPage:
        def is_closed(self) -> bool:
            return False

        def on(self, _event: str, _listener: object) -> None:
            return None

        async def route(self, _url: object, _handler: object) -> None:
            return None

    class _Context:
        async def new_page(self) -> object:
            return _FreshPage()

    t = PlaywrightTransport(_Scraper())
    t._context = _Context()  # type: ignore[assignment]
    stale_page = _StalePage()
    stale = WorkerPage(stale_page, set())  # type: ignore[arg-type]
    t._handles[0] = stale

    with caplog.at_level(
        "WARNING",
        logger="jkent.driver.unified_driver.transport.playwright_transport",
    ):
        handle = await t.acquire(0)

    assert handle is not stale
    # ``__class__``, not ``isinstance``: ``handle.page`` is typed ``Page`` and
    # the stub does not subclass it, so a type checker reads the isinstance
    # narrowing as unreachable.
    assert handle.page.__class__ is _FreshPage
    assert t._handles[0] is handle
    # The unusable page was closed (cancelling its orphan navigation).
    assert stale_page.closed
    # The swallowed failure stays observable.
    assert any("reset_for_reuse" in r.message for r in caplog.records)


async def test_acquire_escalates_to_single_flight_restart() -> None:
    """A dead new_page() escalates to restart and retries once."""

    class _DeadOnceContext:
        """Fails new_page with a dead-connection error until rebuilt."""

        def __init__(self) -> None:
            self.alive = False

        async def new_page(self) -> object:
            if not self.alive:
                raise Exception("Connection closed")
            return object()

    class _EscalatingTransport(PlaywrightTransport):
        def __init__(self) -> None:
            super().__init__(_Scraper())
            self._dead_once = _DeadOnceContext()
            self._context = self._dead_once  # type: ignore[assignment]
            self.rebuild_count = 0

        @override
        async def _rebuild_context(self) -> None:
            self.rebuild_count += 1
            self._dead_once.alive = True

    t = _EscalatingTransport()
    # new_page raises dead -> restart() rebuilds -> retry succeeds.
    page = await t._new_page()
    assert page is not None
    assert t.rebuild_count == 1
    assert t.generation == 1


async def test_a_late_dead_new_page_does_not_restart_twice() -> None:
    """A failure that lands after a racer's restart does not restart again.

    Worker B's ``new_page`` is in flight on the dead context when worker A's
    fails, restarts and bumps the generation; B's failure then arrives. B
    saw generation 0 when it started, so its restart must be the no-op one.
    """
    b_may_fail = asyncio.Event()

    class _DeadContext:
        def __init__(self) -> None:
            self.calls = 0

        async def new_page(self) -> object:
            self.calls += 1
            if self.calls == 1:
                # B: still waiting on the dead connection.
                await b_may_fail.wait()
            raise Exception("Connection closed")

    class _LiveContext:
        async def new_page(self) -> object:
            return object()

    class _CountingTransport(PlaywrightTransport):
        def __init__(self) -> None:
            super().__init__(_Scraper())
            self._context = _DeadContext()  # type: ignore[assignment]
            self.rebuild_count = 0

        @override
        async def _rebuild_context(self) -> None:
            self.rebuild_count += 1
            self._context = _LiveContext()  # type: ignore[assignment]
            b_may_fail.set()

    t = _CountingTransport()
    worker_b = asyncio.create_task(t._new_page())
    await asyncio.sleep(0)  # B is now parked inside the dead new_page
    await t._new_page()  # A: fails, restarts, retries on the live context
    await worker_b
    assert t.rebuild_count == 1
    assert t.generation == 1


async def test_restart_closes_the_handles_it_drops() -> None:
    """Every cached page is closed, not just forgotten, even if a close fails.

    A restart that then cannot rebuild (a persistent context) leaves the old
    browser up, so a forgotten page would stay open in it.
    """

    class _Handle:
        def __init__(self, *, fails: bool) -> None:
            self.fails = fails
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            if self.fails:
                raise Exception("Connection closed")

    t = _RecoveryTransport()
    handles = [_Handle(fails=True), _Handle(fails=False)]
    t._handles = cast("dict[int, WorkerPage]", dict(enumerate(handles)))
    await t.restart(t.generation)
    assert t._handles == {}
    assert [h.closed for h in handles] == [True, True]
    assert t.rebuild_count == 1


async def test_acquire_transient_when_restart_cannot_rebuild() -> None:
    """If new_page stays dead after restart, surface a TransientException."""

    class _AlwaysDeadContext:
        async def new_page(self) -> object:
            raise Exception("Connection closed")

    class _NoEngineTransport(PlaywrightTransport):
        def __init__(self) -> None:
            super().__init__(_Scraper())
            self._context = _AlwaysDeadContext()  # type: ignore[assignment]

        @override
        async def _rebuild_context(self) -> None:
            # No-op rebuild: the context stays dead, mirroring an engine
            # that "restarted" but the connection is still gone.
            return None

    t = _NoEngineTransport()
    with pytest.raises(TransientException):
        await t._new_page()


# --- Crash recovery (browser-gated) ---------------------------------------


async def test_real_restart_reassigns_context_and_clears_handles(
    transport: PlaywrightTransport,
) -> None:
    """A real engine restart rebuilds the context and clears all handles."""
    await transport.acquire(0)
    old_context = transport._context
    assert transport._handles
    await transport.restart(transport.generation)
    assert transport._handles == {}
    assert transport._context is not old_context
    assert transport.generation == 1


# --- Full Transport conformance over a real browser -----------------------


class PlaywrightArchiveConformance(TransportConformance):
    """``TransportConformance`` with the archive methods a Playwright binding needs.

    ``resolve_archive`` on Playwright needs a ``via``-driven download, which
    the base suite's plain GET cannot trigger, so these stand up an
    attachment endpoint and a parent page and drive the download through a
    ``ViaLink``. Shared by the chromium and camoufox bindings: the download
    event is the engine's, not the transport's.
    """

    async def test_resolve_archive_metadata_and_chunks(  # type: ignore[override]
        self, subject: PlaywrightTransport
    ) -> None:
        """``resolve_archive`` yields valid metadata then ``bytes`` chunks.

        Overridden: Playwright needs a ``via``-driven download, so this stands
        up an attachment endpoint + parent page, lands the worker on the
        parent, and drives the download via a ``ViaLink``.
        """
        server, queued, body = await self._stage_archive()
        try:
            await subject.open()
            handle = await subject.acquire(0)
            await handle.page.goto(
                f"{server.base_url}/parent", wait_until="domcontentloaded"
            )
            stream = await subject.resolve_archive(handle, queued)
            try:
                assert isinstance(stream, ArchiveStream)
                assert isinstance(stream.status_code, int)
                assert isinstance(stream.headers, dict)
                assert isinstance(stream.url, str)

                chunks = [chunk async for chunk in stream]
                assert all(isinstance(chunk, bytes) for chunk in chunks)
                assert b"".join(chunks) == body
            finally:
                await subject.finish_archiving(stream)
            await subject.aclose()
        finally:
            await server.runner.cleanup()

    async def test_finish_archiving_is_no_throw(  # type: ignore[override]
        self, subject: PlaywrightTransport
    ) -> None:
        """``finish_archiving`` is no-throw and removes the staged temp file."""
        server, queued, body = await self._stage_archive()
        try:
            await subject.open()
            handle = await subject.acquire(0)
            await handle.page.goto(
                f"{server.base_url}/parent", wait_until="domcontentloaded"
            )
            stream = await subject.resolve_archive(handle, queued)
            staged_path = stream.file_path  # type: ignore[attr-defined]
            async for _ in stream:
                pass
            assert os.path.exists(staged_path)
            await subject.finish_archiving(stream)
            assert not os.path.exists(staged_path)
            await subject.aclose()
        finally:
            await server.runner.cleanup()

    @staticmethod
    async def _stage_archive() -> tuple[StartedServer, QueuedRequest, bytes]:
        """Serve an attachment endpoint + parent page; build the archive request.

        The body spans several of ``FileArchiveStream``'s 64 KiB chunks, so a
        stream that stops early is visible in the joined bytes.
        """
        body = b"%PDF-1.7\nconformance-pdf\n" + bytes(range(256)) * 512
        server = await serve_archive_download(body)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}/file.bin",
                timeout=30,
            ),
            step="collect",
            archive=True,
            via=ViaLink(
                selector=Selector.CSS("#dl"),
                description="download link",
            ),
        )
        # No parent_request_id: the worker is landed on the parent page
        # directly, so resolve_archive does not touch the DB to stage.
        return server, QueuedRequest(request=request, request_id=1), body


class TestPlaywrightTransportConformance(PlaywrightArchiveConformance):
    """Run the shared ``Transport`` conformance suite over a real browser.

    Playwright is tested against a real headless chromium browser, not a
    stubbed engine: the engine/page lifecycle and navigation are the
    behavior, so a stub would assert nothing meaningful. These tests skip
    cleanly where no browser is installed.

    Response fidelity (served HTML round-trips, await-conditions applied,
    incidentals captured, parent staging) is covered by the navigation tests
    above, downloaded-bytes fidelity by ``test_playwright_archive.py``, and
    crash recovery by the ``RecoverableConformance`` binding. This
    conformance binding pins the shared ``Transport`` protocol
    surface — lifecycle, ``acquire`` stability/freshness, ``resolve``, and
    archive streaming — over the real engine.

    ``subject`` serves a minimal no-subresource page (a plain ``resolve``
    produces no incidentals, so no FK surprises) and inserts the ``requests``
    row that ``make_queued`` references; the conformance tests drive
    ``open``/``aclose`` themselves. The archive methods come from
    :class:`PlaywrightArchiveConformance`.
    """

    @override
    @pytest.fixture
    async def subject(
        self,
        require_browser: None,
        memory_session_factory: async_sessionmaker[AsyncSession],
    ) -> AsyncIterator[PlaywrightTransport]:
        html = "<html><body><h1 id='ok'>conformance</h1></body></html>"
        server = await start_app(single_page_app(html))
        self._base_url = server.base_url
        # Insert the FK target row for incidentals; record its id so
        # make_queued references exactly this request row.
        self._request_id = await _insert_request_row(
            memory_session_factory, f"{server.base_url}/page"
        )
        transport = PlaywrightTransport(
            _Scraper(),
            headless=True,
            db=_sql_manager(memory_session_factory),
        )
        try:
            yield transport
        finally:
            # The conformance tests drive open()/aclose() themselves, but a
            # failure between them would skip their aclose() and leak the live
            # browser (camoufox would then deadlock the profile lock for the
            # next test). aclose() is idempotent, so guarantee teardown here.
            await transport.aclose()
            await server.runner.cleanup()

    @override
    def make_queued(self, *, request_id: int | None = None) -> QueuedRequest:
        rid = self._request_id if request_id is None else request_id
        return QueuedRequest(
            request=_request_for(f"{self._base_url}/page"),
            request_id=rid,
        )


# --- Playwright + rate-limiter gate --------------------------------------


@dataclass
class _GateSpyLimiter:
    """A ``RateLimiter`` recording every ``gate`` call (consult count)."""

    gate_calls: list[str | None] = field(default_factory=list)

    async def gate(self, request: Request) -> None:
        self.gate_calls.append(getattr(request, "rate_limit", None))


@dataclass
class _RecordingQueue:
    """Single-pass queue keyed by request id (the worker drains it once)."""

    items: list[tuple[int, Request, int | None, bool]] = field(
        default_factory=list
    )

    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
        if not self.items:
            return None
        self._in_flight += 1
        return self.items.pop(0)

    async def seconds_until_next_pending(self) -> float | None:
        return None

    async def restamp_request_start(self, request_id: int) -> None:
        return None

    _in_flight: int = 0

    @property
    def in_flight_count(self) -> int:
        return self._in_flight

    def request_done(self) -> None:
        self._in_flight -= 1


@dataclass
class _RecordingExecutor:
    """Step that records completed request ids."""

    completed: list[int] = field(default_factory=list)

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        **_: Any,
    ) -> None:
        self.completed.append(request_id)


@dataclass
class _RecordingStorage:
    """Storage that records failures (none expected on the happy path)."""

    failed: list[tuple[int, str]] = field(default_factory=list)

    async def handle_retry(self, request_id: int, error: Exception) -> None:
        return None

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id: int) -> None:
        return None


def _pool_worker(
    *,
    queue: Any,
    transport: PlaywrightTransport,
    rate_limiter: Any,
    executor: Any,
    storage: Any,
) -> PoolWorker:
    # ``queue``/``rate_limiter``/``executor``/``storage`` take the duck-typed
    # recording fakes above, which are not subclasses of the real classes.
    # ``rate_limiter`` gates the default lane; the "none" lane never
    # throttles, as in a real run.
    return PoolWorker(
        0,
        RunCollaborators(
            scraper=_Scraper(),
            transport=transport,
            queue=queue,
            storage=storage,
            executor=executor,
            error_sink=RowOnlyErrorSink(storage),
            rate_limiters=RateLimiters(
                {
                    DEFAULT_RATE_LIMIT: rate_limiter,
                    NO_RATE_LIMIT: NoopRateLimiter(),
                }
            ),
        ),
    )


async def test_pool_worker_gates_each_playwright_resolve(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The worker consults the limiter once per default-lane Playwright resolve."""
    html = "<html><body><h1 id='ok'>gated</h1></body></html>"

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    server = await start_app(app)
    try:
        sf = memory_session_factory
        n = 3
        items: list[tuple[int, Request, int | None, bool]] = []
        for i in range(n):
            url = f"{server.base_url}/page?i={i}"
            rid = await _insert_request_row(sf, url)
            items.append((rid, _request_for(url), None, False))

        limiter = _GateSpyLimiter()
        executor = _RecordingExecutor()
        storage = _RecordingStorage()
        worker = _pool_worker(
            queue=_RecordingQueue(items=items),
            transport=nav_transport,
            rate_limiter=limiter,
            executor=executor,
            storage=storage,
        )

        await worker.run()

        assert storage.failed == []
        assert len(executor.completed) == n  # all resolved + completed
        # Gate consulted exactly once per request, all in the default lane.
        assert limiter.gate_calls == [None] * n  # all in the default lane
    finally:
        await server.runner.cleanup()


async def test_pool_worker_none_lane_skips_token_acquire(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``rate_limit="none"`` requests never reach the default lane's limiter.

    The default lane is a real ``PyrateRateLimiter``; the worker routes the
    request to the "none" lane's ``NoopRateLimiter`` instead. A spy on
    ``Limiter.try_acquire_async`` proves no token was consumed, while the
    Playwright resolve still runs to completion.
    """
    acquires: list[str] = []

    async def fake_acquire(
        self: object, name: str = "pyrate", weight: int = 1, **_: Any
    ) -> bool:
        acquires.append(name)
        return True

    monkeypatch.setattr(Limiter, "try_acquire_async", fake_acquire)

    html = "<html><body><h1 id='ok'>bypassed</h1></body></html>"

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    server = await start_app(app)
    try:
        sf = memory_session_factory
        n = 3
        items: list[tuple[int, Request, int | None, bool]] = []
        for i in range(n):
            url = f"{server.base_url}/page?i={i}"
            rid = await _insert_request_row(sf, url)
            request = Request(
                request=HTTPRequestParams(method=HttpMethod.GET, url=url),
                step="parse",
                rate_limit="none",
            )
            items.append((rid, request, None, False))

        executor = _RecordingExecutor()
        storage = _RecordingStorage()
        worker = _pool_worker(
            queue=_RecordingQueue(items=items),
            transport=nav_transport,
            rate_limiter=PyrateRateLimiter([Rate(1, Duration.SECOND)]),
            executor=executor,
            storage=storage,
        )

        await worker.run()

        assert storage.failed == []
        assert len(executor.completed) == n
        # "none"-lane requests never consumed a default-lane token.
        assert acquires == []
    finally:
        await server.runner.cleanup()


# --- Classification conformance over a real browser ------------------------


class TestPlaywrightTransportClassification(ClassificationConformance):
    """Status classification over a real chromium navigation.

    The classifier sees what the browser can surface — the navigation status
    (``page.goto``'s response), the DOM snapshot as the body, and synthesized
    headers — so this binding pins that a non-200 navigation actually reaches
    the scraper's classifier. The sharpest regression this guards: a
    speculative probe landing on an error page must raise
    ``SpeculationHTTPFailure``, or speculation over a browser transport never
    records a failure and probes forever.
    """

    # Bound by the autouse ``_bind_env`` fixture before every test.
    _sf: async_sessionmaker[AsyncSession]

    @pytest.fixture(autouse=True)
    async def _bind_env(
        self,
        require_browser: None,
        memory_session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._sf = memory_session_factory

    @override
    async def classify(
        self,
        scraper: type[BaseScraper[Any]],
        code: int,
        body: bytes,
        *,
        speculative: bool = False,
    ) -> Response:
        server = await start_app(status_app(code, body))
        request_id = await _insert_request_row(
            self._sf, f"{server.base_url}/page"
        )
        transport = PlaywrightTransport(
            scraper(), headless=True, db=_sql_manager(self._sf)
        )
        try:
            await transport.open()
            handle = await transport.acquire(0)
            request = Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=f"{server.base_url}/page"
                ),
                step="parse",
                is_speculative=speculative,
            )
            return await transport.resolve(
                handle, QueuedRequest(request=request, request_id=request_id)
            )
        finally:
            await transport.aclose()
            await server.runner.cleanup()


def _blockable_page_html(server_base: str) -> str:
    return (
        "<html><head>"
        f"<link rel='stylesheet' href='{server_base}/style.css'>"
        "</head><body>"
        f"<img src='{server_base}/pic.png'>"
        "<div id='ok'>served</div>"
        "</body></html>"
    )


async def _resolve_with_blocking(
    memory_session_factory: async_sessionmaker[AsyncSession],
    blocked: set[str] | None,
) -> tuple[list[str], list[str]]:
    """Serve a page with an image + stylesheet; return (hits, incidental urls).

    ``hits`` is what the *server* actually saw, which is the only proof that a
    request was dropped before the network rather than merely unrecorded.
    """
    hits: list[str] = []

    async def page_handler(_request: web.Request) -> web.Response:
        return web.Response(
            status=200,
            body=_blockable_page_html(server.base_url),
            content_type="text/html",
        )

    async def css_handler(_request: web.Request) -> web.Response:
        hits.append("/style.css")
        return web.Response(
            status=200, body="#ok{color:red}", content_type="text/css"
        )

    async def png_handler(_request: web.Request) -> web.Response:
        hits.append("/pic.png")
        # A PNG signature is enough; only the request matters.
        return web.Response(
            status=200, body=b"\x89PNG\r\n\x1a\n", content_type="image/png"
        )

    app = web.Application()
    app.router.add_get("/page", page_handler)
    app.router.add_get("/style.css", css_handler)
    app.router.add_get("/pic.png", png_handler)
    server = await start_app(app)

    transport = PlaywrightTransport(
        _Scraper(),
        headless=True,
        blocked_resource_types=blocked,
        db=_sql_manager(memory_session_factory),
    )
    await transport.open()
    try:
        handle = await transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url)
        queued = QueuedRequest(request=_request_for(url), request_id=rid)
        await transport.resolve(handle, queued)
        incidentals = [i.url for i in handle.incidental_requests]
        return hits, incidentals
    finally:
        await transport.aclose()
        await server.runner.cleanup()


async def test_blocked_resource_types_never_reach_the_network(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Images are aborted pre-flight; stylesheets still load (visibility)."""

    hits, incidentals = await _resolve_with_blocking(
        memory_session_factory,
        blocked=None,  # default: image/media/font
    )

    # The server never saw the image at all.
    assert "/pic.png" not in hits
    # Stylesheets must keep loading: state="visible" waits resolve against
    # computed layout, so blocking CSS would break selector await conditions.
    assert "/style.css" in hits
    # And a blocked request leaves no half-written incidental row behind.
    assert not any(u.endswith("/pic.png") for u in incidentals)
    assert any(u.endswith("/style.css") for u in incidentals)


async def test_empty_blocked_resource_types_restores_fetching(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The single switch: an empty set fetches blocked types again."""

    hits, incidentals = await _resolve_with_blocking(
        memory_session_factory, blocked=set()
    )

    assert "/pic.png" in hits
    assert any(u.endswith("/pic.png") for u in incidentals)


async def test_page_recycled_after_configured_uses(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The worker's page is rebuilt once it has been reused N times.

    ``uses`` counts reuses, so a page recycled at N has served N + 1 requests;
    the log line has to report that, not N.
    """

    transport = PlaywrightTransport(
        _Scraper(),
        headless=True,
        page_recycle_after=2,
        db=_sql_manager(memory_session_factory),
    )
    await transport.open()
    try:
        first = await transport.acquire(0)
        assert first.uses == 0
        # Reuses below the threshold keep the same page.
        assert await transport.acquire(0) is first
        assert await transport.acquire(0) is first
        assert first.uses == 2
        # The next acquire crosses it and rebuilds.
        with caplog.at_level("INFO", logger=PlaywrightTransport.__module__):
            recycled = await transport.acquire(0)
        assert recycled is not first
        assert recycled.uses == 0
        assert first.page.is_closed()
        assert "recycled after 3 requests (2 reuses)" in caplog.text
    finally:
        await transport.aclose()


async def test_page_recycling_preserves_session_cookies(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A recycled page keeps the context's cookies (the ASP.NET session).

    Recycling would be unusable on a session-backed site if it dropped
    cookies, so this pins the context-vs-page scoping.
    """

    seen_cookies: list[str] = []

    async def set_handler(_request: web.Request) -> web.Response:
        resp = web.Response(
            status=200,
            body="<html><body>set</body></html>",
            content_type="text/html",
        )
        resp.set_cookie("SESSIONID", "abc123")
        return resp

    async def echo_handler(request: web.Request) -> web.Response:
        seen_cookies.append(request.headers.get("Cookie", ""))
        return web.Response(
            status=200,
            body="<html><body>echo</body></html>",
            content_type="text/html",
        )

    app = web.Application()
    app.router.add_get("/set", set_handler)
    app.router.add_get("/echo", echo_handler)
    server = await start_app(app)

    transport = PlaywrightTransport(
        _Scraper(),
        headless=True,
        page_recycle_after=1,
        db=_sql_manager(memory_session_factory),
    )
    await transport.open()
    try:
        handle = await transport.acquire(0)
        set_url = f"{server.base_url}/set"
        rid = await _insert_request_row(memory_session_factory, set_url)
        await transport.resolve(
            handle,
            QueuedRequest(request=_request_for(set_url), request_id=rid),
        )

        # Force a rebuild, then navigate again on the fresh page.
        recycled = await transport.acquire(0)
        recycled = await transport.acquire(0)
        assert recycled is not handle

        echo_url = f"{server.base_url}/echo"
        rid2 = await _insert_request_row(memory_session_factory, echo_url)
        await transport.resolve(
            recycled,
            QueuedRequest(request=_request_for(echo_url), request_id=rid2),
        )

        assert any("SESSIONID=abc123" in c for c in seen_cookies), seen_cookies
    finally:
        await transport.aclose()
        await server.runner.cleanup()
