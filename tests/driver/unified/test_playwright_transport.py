"""Tests for ``PlaywrightTransport`` (B1 lifecycle + B2 navigation).

B1 covers engine + page lifecycle (``open``/``aclose``, ``acquire``/
``release``). B2 adds the navigation path (``resolve``): navigate, apply
``await_conditions``, snapshot the DOM, persist incidental sub-requests
against the request's row id, and stage a forked tab from a parent's cached
response. ``resolve_archive``/``finish_archiving`` remain stubbed (B3/B4).

Browser-dependent tests are skipped cleanly when no browser engine can launch
(camoufox/chromium may be absent here); the structural-conformance checks run
without a browser. The navigation tests stand up a local aiohttp server and a
real in-memory DB.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from aiohttp import web
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from pyrate_limiter import Duration, Limiter, Rate

from jkent.common.decorators import step
from jkent.common.exceptions import TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    Selector,
    WaitForLoadState,
    WaitForSelector,
)
from jkent.driver.browser_engine.worker_page import WorkerPage
from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle, Recoverable
from jkent.driver.unified_driver.rate_limiter import (
    PyrateRateLimiter,
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
from jkent.driver.unified_driver.worker import PoolWorker
from tests.db_staging import (
    insert_request_row as _insert_request_row,
)
from tests.db_staging import (
    insert_staged_parent as _insert_staged_parent,
)
from tests.driver.unified.conftest import (
    StartedServer,
    serve_archive_download,
    single_page_app,
    start_app,
    status_app,
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator, Sequence

    from sqlalchemy.ext.asyncio import async_sessionmaker


class _Scraper(BaseScraper[None]):
    """Minimal scraper (no CFCAP requirement -> standard playwright engine)."""

    def get_entry(self):  # type: ignore[no-untyped-def]
        return iter(())

    @step
    def parse(self, response: Response) -> Generator[Request, None, None]:
        """No-op continuation so requests can reference ``continuation='parse'``.

        Worker tests drive a mocked continuation executor, but the worker still
        resolves this step's (empty) await_list before fetching, so it must be a
        discoverable continuation.
        """
        yield from ()


@pytest.fixture
async def transport(has_browser: bool):  # type: ignore[no-untyped-def]
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")
    subject = PlaywrightTransport(_Scraper(), headless=True)
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


def _sql_manager(sf: async_sessionmaker) -> SQLManager:
    """Build an ``SQLManager`` over a test ``async_sessionmaker``.

    The parent-response read + incidental write touch only the session
    factory and lock; the engine is taken off the factory's bind.
    """
    engine = sf.kw["bind"]
    return SQLManager(engine, sf)


# --- local server --------------------------------------------------------


def _request_for(url: str) -> Request:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse",
    )


# --- Always-runnable structural conformance (no browser) ------------------


def test_exposes_full_transport_surface() -> None:
    """``PlaywrightTransport`` has the whole ``Transport`` method surface."""
    for name in (
        "open",
        "aclose",
        "acquire",
        "release",
        "resolve",
        "resolve_archive",
        "finish_archiving",
    ):
        assert callable(getattr(PlaywrightTransport, name))


async def test_export_cookies_none_without_context() -> None:
    """``export_cookies`` returns None before a context exists (no browser)."""
    subject = PlaywrightTransport(_Scraper())
    assert await subject.export_cookies() is None


async def test_cookie_round_trip_through_context(
    transport: PlaywrightTransport,
) -> None:
    """Cookies set on one context export and re-import onto a fresh one."""
    await transport._require_context().add_cookies(
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
    assert "round-trip" in exported

    fresh = PlaywrightTransport(_Scraper(), headless=True)
    await fresh.open()
    try:
        assert await fresh.export_cookies() == "[]"  # empty to start
        await fresh.import_cookies(exported)
        reexported = await fresh.export_cookies()
        assert reexported is not None
        assert "round-trip" in reexported
    finally:
        await fresh.aclose()


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


# --- Browser-dependent lifecycle (skipped cleanly w/o a browser) ----------


async def test_open_then_aclose_clears_refs(has_browser: bool) -> None:
    """``aclose`` releases everything ``open`` acquired (no leak)."""
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")
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

    @pytest.fixture
    async def subject(self, has_browser: bool):  # type: ignore[no-untyped-def]
        # Yield + aclose in teardown: the base suite's
        # ``test_open_awaits_to_none`` opens without closing, which would leak
        # a live browser (and, for camoufox, deadlock the profile lock for the
        # next test). The teardown guarantees cleanup after every case.
        if not has_browser:
            pytest.skip("no launchable browser engine in this environment")
        transport = PlaywrightTransport(_Scraper(), headless=True)
        try:
            yield transport
        finally:
            await transport.aclose()

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


# --- B2 navigation (skipped cleanly w/o a browser) ------------------------


@pytest.fixture
async def nav_transport(  # type: ignore[no-untyped-def]
    has_browser: bool,
    memory_session_factory: async_sessionmaker,
):
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")
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
    memory_session_factory: async_sessionmaker,
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
        rid = await _insert_request_row(memory_session_factory, url, qc=1)
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
    memory_session_factory: async_sessionmaker,
) -> None:
    """``await_conditions`` are applied before snapshotting (load + selector)."""
    html = "<html><body><div class='ready'>here</div></body></html>"

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/page", handler)
    server = await start_app(app)
    try:
        handle = await nav_transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url, qc=1)
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
        assert "here" in resp.text
        assert resp.status_code == 200
    finally:
        await server.runner.cleanup()


async def test_resolve_persists_incidentals(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
) -> None:
    """A sub-request (fetch) is captured and persisted against request_id."""
    page_html = (
        "<html><body><script>"
        "fetch('/sub.json').then(r => r.text());"
        "</script></body></html>"
    )

    async def page_handler(_request: web.Request) -> web.Response:
        return web.Response(
            status=200, body=page_html, content_type="text/html"
        )

    async def sub_handler(_request: web.Request) -> web.Response:
        return web.Response(
            status=200, body='{"k": "v"}', content_type="application/json"
        )

    app = web.Application()
    app.router.add_get("/page", page_handler)
    app.router.add_get("/sub.json", sub_handler)
    server = await start_app(app)
    try:
        handle = await nav_transport.acquire(0)
        url = f"{server.base_url}/page"
        rid = await _insert_request_row(memory_session_factory, url, qc=1)
        request = _request_for(url)
        queued = QueuedRequest(request=request, request_id=rid)
        await nav_transport.resolve(
            handle,
            queued,
            # Give the fetch time to fire + complete before snapshotting.
            await_conditions=(
                WaitForLoadState(state="networkidle", timeout=5000),
            ),
        )

        sf = memory_session_factory
        async with sf() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT parent_request_id, url FROM "
                        "incidental_requests"
                    )
                )
            ).all()
        # Every captured incidental is tagged to the navigating request id.
        assert rows, "no incidental requests were persisted"
        assert all(row[0] == rid for row in rows)
        assert any("sub.json" in row[1] for row in rows)
    finally:
        await server.runner.cleanup()


async def test_resolve_stages_parent_then_via_navigates(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
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
                        status, priority, queue_counter, method, url,
                        continuation, current_location, response_status_code,
                        response_url, response_headers_json, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES ('completed', 9, 1, 'GET', :url, 'parse', '', 200,
                        :url, NULL, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
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
        child_id = await _insert_request_row(sf, child_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=child_url),
            continuation="parse",
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
    memory_session_factory: async_sessionmaker,
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
            sf, "https://staged.example/parent", qc=1
        )
        child_url = f"{server.base_url}/own"
        child_id = await _insert_request_row(sf, child_url, qc=2)

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


# --- T2.2: ViaFormSubmit navigation ---------------------------------------


async def test_resolve_via_form_submit_navigates_with_submitted_data(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            continuation="parse",
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
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            continuation="parse",
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
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            continuation="parse",
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


async def test_resolve_via_form_submit_event_target_postback(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            continuation="parse",
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
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        result_url = f"{server.base_url}/results"
        child_id = await _insert_request_row(sf, result_url, qc=2)
        request = Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=result_url),
            continuation="parse",
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
    memory_session_factory: async_sessionmaker,
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
            sf, url=form_url, body=form_body, qc=1
        )

        file_url = f"{server.base_url}/dl"
        child_id = await _insert_request_row(sf, file_url, qc=2)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST, url=file_url, timeout=30
            ),
            continuation="collect",
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


# --- B3 crash recovery (browser-free) -------------------------------------


class _RecoveryTransport(PlaywrightTransport):
    """Transport with a browser-free rebuild step for recovery tests.

    Overrides the single browser-touching method ``_rebuild_context`` with a
    no-op counter, so the generation / single-flight logic is exercisable
    without launching a browser.
    """

    def __init__(self) -> None:
        super().__init__(_Scraper())
        self.rebuild_count = 0
        # A non-None sentinel so _require_context() / acquire don't trip the
        # "used before open()" guard during recovery tests.
        self._context = object()  # type: ignore[assignment]

    async def _rebuild_context(self) -> None:
        self.rebuild_count += 1
        self._context = object()  # type: ignore[assignment]


class TestPlaywrightTransportRecoverable(RecoverableConformance):
    """Run the Recoverable conformance suite against a browser-free subject."""

    @pytest.fixture
    def subject(self) -> Recoverable:
        return _RecoveryTransport()

    def make_subject(self) -> Recoverable:
        return _RecoveryTransport()

    def dead_exc(self) -> BaseException:
        return Exception("Connection closed")

    # The two ``@given`` property methods are inherited from the shared base;
    # collecting a second conformance subclass in the same session makes
    # hypothesis see them called from "differing executors". The property is
    # still exercised correctly per-class — re-declare with the suppression.
    @pytest.mark.generative
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(k=st.integers(min_value=1, max_value=50))
    def test_k_sequential_restarts_advance_by_k(self, k: int) -> None:
        """K sequential in-band restarts advance generation by exactly K."""

        async def drive() -> int:
            subject = self.make_subject()
            for _ in range(k):
                await subject.restart(subject.generation)
            return subject.generation

        assert asyncio.run(drive()) == k

    @pytest.mark.generative
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(n=st.integers(min_value=1, max_value=50))
    def test_concurrent_restarts_rebuild_once(self, n: int) -> None:
        """N concurrent restarts at the same seen generation rebuild once."""

        async def drive() -> int:
            subject = self.make_subject()
            seen = subject.generation
            await asyncio.gather(*(subject.restart(seen) for _ in range(n)))
            return subject.generation

        assert asyncio.run(drive()) == 1


def test_should_restart_recognizes_each_dead_message() -> None:
    """All three ported dead-connection messages are recognized as death."""
    t = _RecoveryTransport()
    for msg in (
        "Connection closed",
        "Browser has been closed",
        "Target page, context or browser has been closed",
    ):
        assert t.should_restart(Exception(f"... {msg} ...")) is True
    assert t.should_restart(Exception("some other error")) is False


# ``test_concurrent_restart_rebuilds_once`` lived here but duplicated both the
# RecoverableConformance binding above and the property-based concurrent-restart
# test in test_playwright_concurrency.py (which asserts rebuild_count AND
# generation); it was removed to keep a single owner of that invariant.


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
        await t.resolve(handle, object())  # type: ignore[arg-type]

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
        await t.resolve(sentinel, object())  # type: ignore[arg-type]
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
            self.incidental_requests: list[dict] = []
            self.current_parent_request_id: int | None = None

        def clear_request_state(self) -> None:
            self.incidental_requests = []

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        async def insert_incidental_request(self, **_kw: object) -> None:
            return None

    t = PlaywrightTransport(_Scraper(), db=_FakeDB())  # type: ignore[arg-type]
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
    assert snapshot.text == "<html>partial</html>"
    assert snapshot.url == "https://example.com/x"
    # The runaway load is stopped before the snapshot: Playwright's timeout
    # only ends the wait, so window.stop() must abort the browser-side load.
    assert handle.page.calls == ["window.stop()", "content"]


async def test_resolve_goto_honors_request_timeout() -> None:
    """The request's timeout reaches page.goto as milliseconds.

    A ``(connect, read)`` tuple uses the read element; an explicit ``None``
    passes no ``timeout`` kwarg so Playwright's default applies. An unset
    timeout gets the ``HTTPRequestParams`` default (60 s).
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
            self.incidental_requests: list[dict] = []

        def clear_request_state(self) -> None:
            self.incidental_requests = []

        async def drain_captures(self, timeout: float = 10.0) -> None:
            return None

    class _FakeDB:
        async def insert_incidental_request(self, **_kw: object) -> None:
            return None

    t = PlaywrightTransport(_Scraper(), db=_FakeDB())  # type: ignore[arg-type]

    cases: tuple[tuple[dict[str, Any], float | None], ...] = (
        ({"timeout": 45}, 45000.0),
        ({"timeout": (5, 90)}, 90000.0),
        ({"timeout": None}, None),
        ({}, 60000.0),  # unset -> HTTPRequestParams default
    )
    for extra, expected in cases:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com/x",
                **extra,
            ),
            continuation="parse",
        )
        handle = _FakeHandle()
        await t._resolve(
            handle,  # type: ignore[arg-type]
            QueuedRequest(request=request, request_id=7),
            [],
        )
        assert handle.page.goto_kwargs.get("timeout") == expected


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
        await t.resolve(sentinel, object())  # type: ignore[arg-type]
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
    assert isinstance(handle.page, _FreshPage)
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
            self._context = _DeadOnceContext()  # type: ignore[assignment]
            self.rebuild_count = 0

        async def _rebuild_context(self) -> None:
            self.rebuild_count += 1
            self._context.alive = True  # type: ignore[union-attr, attr-defined]

    t = _EscalatingTransport()
    # new_page raises dead -> restart() rebuilds -> retry succeeds.
    page = await t._new_page()
    assert page is not None
    assert t.rebuild_count == 1
    assert t.generation == 1


async def test_acquire_transient_when_restart_cannot_rebuild() -> None:
    """If new_page stays dead after restart, surface a TransientException."""

    class _AlwaysDeadContext:
        async def new_page(self) -> object:
            raise Exception("Connection closed")

    class _NoEngineTransport(PlaywrightTransport):
        def __init__(self) -> None:
            super().__init__(_Scraper())
            self._context = _AlwaysDeadContext()  # type: ignore[assignment]

        async def _rebuild_context(self) -> None:
            # No-op rebuild: the context stays dead, mirroring an engine
            # that "restarted" but the connection is still gone.
            return None

    t = _NoEngineTransport()
    with pytest.raises(TransientException):
        await t._new_page()


# --- B3 crash recovery (browser-gated) ------------------------------------


async def test_real_restart_reassigns_context_and_clears_handles(
    transport: PlaywrightTransport,
) -> None:
    """A real engine restart rebuilds the context and clears all handles."""
    engine = transport._engine
    if engine is None or not engine.supports_restart:
        pytest.skip("engine does not support restart")
    await transport.acquire(0)
    old_context = transport._context
    assert transport._handles
    await transport.restart(transport.generation)
    assert transport._handles == {}
    assert transport._context is not old_context
    assert transport.generation == 1


# --- B5: full Transport conformance over a real browser -------------------


class TestPlaywrightTransportConformance(TransportConformance):
    """Run the shared ``Transport`` conformance suite over a real browser.

    Fidelity strategy (the B5 "decide the approach" deliverable):
    Playwright is tested against a REAL headless chromium browser, NOT a
    stubbed engine. The engine/page lifecycle and navigation ARE the
    behavior, so a stub would assert nothing meaningful; the cost is that
    these tests skip cleanly where no browser is installed. We deliberately
    do NOT build a heavy Playwright mock or a separate matching-server
    fidelity rig — that is out of scope.

    Response/archive *fidelity* (served HTML round-trips, await-conditions
    applied, incidentals captured, parent staging, downloaded bytes equal
    served bytes) is already covered by the B2 navigation tests and the B4
    archive tests above; crash recovery by the B3 ``RecoverableConformance``
    binding. This conformance binding pins the shared ``Transport`` protocol
    surface — lifecycle, ``acquire`` stability/freshness, ``resolve``, and
    archive streaming — over the real engine.

    ``subject`` serves a minimal no-subresource page (a plain ``resolve``
    produces no incidentals, so no FK surprises) and inserts the ``requests``
    row that ``make_queued`` references; the conformance tests drive
    ``open``/``aclose`` themselves. The two archive conformance methods are
    overridden because Playwright's ``resolve_archive`` needs a ``via``-driven
    download, which a plain GET cannot trigger.
    """

    @pytest.fixture
    async def subject(  # type: ignore[override]
        self,
        has_browser: bool,
        memory_session_factory: async_sessionmaker,
    ) -> AsyncIterator[PlaywrightTransport]:
        if not has_browser:
            pytest.skip("no launchable browser engine in this environment")

        html = "<html><body><h1 id='ok'>conformance</h1></body></html>"
        server = await start_app(single_page_app(html))
        self._base_url = server.base_url  # type: ignore
        # Insert the FK target row for incidentals; record its id so
        # make_queued references exactly this request row.
        self._request_id = await _insert_request_row(  # type: ignore
            memory_session_factory, f"{server.base_url}/page", qc=1
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

    def make_queued(self, *, request_id: int | None = None) -> QueuedRequest:
        rid = self._request_id if request_id is None else request_id  # type: ignore
        return QueuedRequest(
            request=_request_for(f"{self._base_url}/page"),  # type: ignore
            request_id=rid,
        )

    async def test_resolve_archive_metadata_and_chunks(  # type: ignore[override]
        self, subject: PlaywrightTransport
    ) -> None:
        """``resolve_archive`` yields valid metadata then ``bytes`` chunks.

        Overridden: Playwright needs a ``via``-driven download, so this stands
        up an attachment endpoint + parent page, lands the worker on the
        parent, and drives the download via a ``ViaLink``.
        """
        server, queued = await self._stage_archive(subject)
        try:
            await subject.open()
            handle = await subject.acquire(0)
            await handle.page.goto(
                f"{server.base_url}/parent", wait_until="domcontentloaded"
            )
            stream = await subject.resolve_archive(handle, queued)
            assert isinstance(stream, ArchiveStream)
            assert isinstance(stream.status_code, int)
            assert isinstance(stream.headers, dict)
            assert isinstance(stream.url, str)

            chunks = [chunk async for chunk in stream]
            assert chunks
            assert all(isinstance(chunk, bytes) for chunk in chunks)
            await subject.aclose()
        finally:
            await server.runner.cleanup()

    async def test_finish_archiving_is_no_throw(  # type: ignore[override]
        self, subject: PlaywrightTransport
    ) -> None:
        """``finish_archiving`` is no-throw and removes the staged temp file."""
        server, queued = await self._stage_archive(subject)
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
    async def _stage_archive(
        _subject: PlaywrightTransport,
    ) -> tuple[StartedServer, QueuedRequest]:
        """Serve an attachment endpoint + parent page; build the archive request."""
        body = b"%PDF-1.7\nconformance-pdf\n" + bytes(range(256)) * 4
        server = await serve_archive_download(body)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}/file.bin",
                timeout=30,
            ),
            continuation="collect",
            archive=True,
            via=ViaLink(
                selector=Selector.CSS("#dl"),
                description="download link",
            ),
        )
        # No parent_request_id: the worker is landed on the parent page
        # directly, so resolve_archive does not touch the DB to stage.
        return server, QueuedRequest(request=request, request_id=1)


# --- T2.4: Playwright + rate-limiter gate ---------------------------------


@dataclass
class _GateSpyLimiter:
    """A ``RateLimiter`` recording every ``gate`` call (consult count)."""

    gate_calls: list[bool] = field(default_factory=list)

    async def gate(self, request: Request) -> None:
        self.gate_calls.append(getattr(request, "bypass_rate_limit", False))

    @property
    def max_rate_per_second(self) -> float | None:
        return None


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
class _RecordingContinuation:
    """Continuation that records completed request ids."""

    completed: list[int] = field(default_factory=list)

    async def complete_request(
        self, request_id, response, request, continuation_name, **_
    ) -> None:  # type: ignore[no-untyped-def]
        self.completed.append(request_id)


@dataclass
class _RecordingStorage:
    """Storage that records failures (none expected on the happy path)."""

    failed: list[tuple[int, str]] = field(default_factory=list)

    async def handle_retry(self, request_id, error):  # type: ignore[no-untyped-def]
        return None

    async def mark_request_failed(self, request_id, error_message) -> None:  # type: ignore[no-untyped-def]
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id) -> None:  # type: ignore[no-untyped-def]
        return None


def _pool_worker(  # type: ignore[no-untyped-def]
    *, queue, transport, rate_limiter, continuation, storage
):
    return PoolWorker(
        worker_id=0,
        queue=queue,
        transport=transport,
        rate_limiter=rate_limiter,
        continuation=continuation,
        storage=storage,
        stop_event=asyncio.Event(),
        scraper=_Scraper(),
        archive_handler=None,
    )


async def test_pool_worker_gates_each_playwright_resolve(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
) -> None:
    """The worker consults the limiter once per non-bypass Playwright resolve."""
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
            rid = await _insert_request_row(sf, url, qc=i + 1)
            items.append((rid, _request_for(url), None, False))

        limiter = _GateSpyLimiter()
        continuation = _RecordingContinuation()
        storage = _RecordingStorage()
        worker = _pool_worker(
            queue=_RecordingQueue(items=items),
            transport=nav_transport,
            rate_limiter=limiter,
            continuation=continuation,
            storage=storage,
        )

        await worker.run()

        assert storage.failed == []
        assert len(continuation.completed) == n  # all resolved + completed
        # Gate consulted exactly once per request, none of them bypassing.
        assert limiter.gate_calls == [False] * n
    finally:
        await server.runner.cleanup()


async def test_pool_worker_bypass_skips_token_acquire(
    nav_transport: PlaywrightTransport,
    memory_session_factory: async_sessionmaker,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bypass_rate_limit=True`` requests skip the real limiter's token acquire.

    Drives the real ``PyrateRateLimiter.gate`` (its bypass branch returns
    before acquiring) and spies on ``Limiter.try_acquire_async`` to prove no
    token was consumed, while the Playwright resolve still runs to completion.
    """
    acquires: list[str] = []

    async def fake_acquire(self, name="pyrate", weight=1, **_):  # type: ignore[no-untyped-def]
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
            rid = await _insert_request_row(sf, url, qc=i + 1)
            request = Request(
                request=HTTPRequestParams(method=HttpMethod.GET, url=url),
                continuation="parse",
                bypass_rate_limit=True,
            )
            items.append((rid, request, None, False))

        continuation = _RecordingContinuation()
        storage = _RecordingStorage()
        worker = _pool_worker(
            queue=_RecordingQueue(items=items),
            transport=nav_transport,
            rate_limiter=PyrateRateLimiter([Rate(1, Duration.SECOND)]),
            continuation=continuation,
            storage=storage,
        )

        await worker.run()

        assert storage.failed == []
        assert len(continuation.completed) == n
        # Bypass requests never consumed a rate-limiter token.
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

    # pyre-ignore[13]: bound by the autouse ``_bind_env`` fixture before
    # every test, which pyre can't see.
    _sf: async_sessionmaker

    @pytest.fixture(autouse=True)
    async def _bind_env(
        self,
        has_browser: bool,
        memory_session_factory: async_sessionmaker,
    ) -> None:
        if not has_browser:
            pytest.skip("no launchable browser engine in this environment")
        self._sf = memory_session_factory

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
            self._sf, f"{server.base_url}/page", qc=1
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
                continuation="parse",
                is_speculative=speculative,
            )
            return await transport.resolve(
                handle, QueuedRequest(request=request, request_id=request_id)
            )
        finally:
            await transport.aclose()
            await server.runner.cleanup()
