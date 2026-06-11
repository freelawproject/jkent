"""Concurrency tests for ``PlaywrightTransport`` (T1.1 + T1.2).

T1.1 — worker-page registry invariants: ``acquire``/``release`` keep a 1:1,
get-or-create mapping between worker ids and pages, stable under concurrent
interleaving, fresh after release, rebuilt after a crash. The page-registry
half runs against a real browser; the crash/generation half uses the
browser-free recovery override (single-flight rebuild under the generation).

T1.2 — concurrent DOM-snapshot / incidental isolation: under forced
interleaving, each ``resolve`` snapshots ITS page (no content/url bleed) and
each captured incidental is tagged to ITS ``request_id`` (no cross-contamination
across worker pages).

Browser-gated: skipped cleanly when no engine launches. Every transport opened
here is torn down (yield+finally / try+finally) so no browser process leaks.
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from aiohttp import web
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jkent.data_types import WaitForLoadState
from jkent.driver.unified_driver.transport import QueuedRequest
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.test_playwright_transport import (
    _insert_request_row,
    _RecoveryTransport,
    _request_for,
    _Scraper,
    _sql_manager,
    _start_server,
)

if TYPE_CHECKING:
    from jkent.driver.database_engine.scoped_session import (
        ScopedSessionFactory,
    )


async def _browser_launches() -> bool:
    """Probe whether a real browser context can be brought up + torn down."""
    transport = PlaywrightTransport(_Scraper(), headless=True)
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
    return asyncio.run(_browser_launches())


# --- T1.1: worker-page registry invariants -------------------------------


@pytest.fixture
async def registry_transport(has_browser: bool):  # type: ignore[no-untyped-def]
    """An opened browser transport; torn down so no page/process leaks."""
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")
    subject = PlaywrightTransport(_Scraper(), headless=True)
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


@pytest.mark.generative
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(data=st.data())
async def test_acquire_release_registry_invariants(
    registry_transport: PlaywrightTransport, data: st.DataObject
) -> None:
    """Random acquire/release/close sequences never violate registry invariants.

    P1 get-or-create: acquire(wid) is the same handle until released/crashed.
    P2 1:1: two live workers never share a page.
    P3 fresh-after-release: re-acquiring a released worker builds a new page.
    P4 rebuilt-after-crash: a closed page is detected and replaced.
    """
    transport = registry_transport
    model: dict[int, Any] = {}  # worker_id -> expected page object
    # Hold references to retired pages so their id() can't be recycled by GC
    # onto a fresh page (which would make a clean allocation look reused).
    retired: list[Any] = []

    n_ops = data.draw(st.integers(min_value=10, max_value=40), label="n_ops")
    for _ in range(n_ops):
        op = data.draw(
            st.sampled_from(["acquire", "release", "close_page"]), label="op"
        )
        wid = data.draw(st.integers(min_value=0, max_value=4), label="wid")
        # Force event-loop interleaving between operations.
        await asyncio.sleep(0)

        if op == "acquire":
            handle = await transport.acquire(wid)
            if wid in model:
                assert handle.page is model[wid], (
                    f"P1: worker {wid} got a different handle while held"
                )
            else:
                model[wid] = handle.page
            # P2: no two live workers share the same page object.
            live = list(model.values())
            assert len({id(p) for p in live}) == len(live), "P2: shared page"
            # A fresh allocation is never a previously-retired page (identity).
            assert all(handle.page is not p for p in retired), (
                f"reused a retired page for worker {wid}"
            )
        elif op == "release":
            if wid in model:
                retired.append(model.pop(wid))
            await transport.release(wid)
            assert wid not in transport._handles, (
                f"P3: worker {wid} still registered after release"
            )
        else:  # close_page — simulate a browser-side crash of the page.
            held = transport._handles.get(wid)
            if held is not None:
                old = held.page
                await old.close()
                fresh = await transport.acquire(wid)
                assert fresh.page is not old, (
                    f"P4: worker {wid} kept its closed page"
                )
                assert not fresh.page.is_closed(), "P4: got a closed page"
                retired.append(old)
                model[wid] = fresh.page

    # Final: every page still registered is open.
    for wid, handle in transport._handles.items():
        assert not handle.page.is_closed(), f"worker {wid} left a closed page"


@pytest.mark.generative
@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(worker_ids=st.lists(st.integers(0, 6), min_size=2, max_size=6))
async def test_concurrent_acquires_never_share_pages(
    registry_transport: PlaywrightTransport, worker_ids: list[int]
) -> None:
    """Concurrent acquires across distinct workers each get a distinct page."""
    transport = registry_transport
    unique = sorted(set(worker_ids))

    async def acquire(wid: int) -> Any:
        await asyncio.sleep(0)  # interleave the racing acquires
        return await transport.acquire(wid)

    handles = await asyncio.gather(*(acquire(wid) for wid in unique))
    pages = [h.page for h in handles]
    assert len({id(p) for p in pages}) == len(pages), "concurrent share"
    for wid in unique:
        await transport.release(wid)


async def test_release_then_reacquire_is_fresh(
    registry_transport: PlaywrightTransport,
) -> None:
    """Releasing then re-acquiring a worker yields a brand-new page."""
    transport = registry_transport
    first = await transport.acquire(0)
    await transport.release(0)
    second = await transport.acquire(0)
    assert second.page is not first.page
    await transport.release(0)


# --- T1.1: crash → rebuild (browser-free, via the recovery override) -----


@pytest.mark.generative
@settings(suppress_health_check=[HealthCheck.differing_executors])
@given(n=st.integers(min_value=2, max_value=32))
async def test_concurrent_restarts_rebuild_once(n: int) -> None:
    """N concurrent restarts at one seen generation rebuild exactly once.

    The poison→rebuild path tied to ``Recoverable``: a crashed/poisoned engine
    is rebuilt a single time across racing callers, advancing the generation
    by one (interleaved with ``asyncio.sleep(0)``).
    """
    subject = _RecoveryTransport()
    seen = subject.generation

    async def restart() -> None:
        await asyncio.sleep(0)
        await subject.restart(seen)

    await asyncio.gather(*(restart() for _ in range(n)))
    assert subject.rebuild_count == 1
    assert subject.generation == 1


# --- T1.2: concurrent snapshot / incidental isolation --------------------


def _fingerprint_app(n_pages: int) -> tuple[web.Application, list[str]]:
    """Serve N distinct fingerprinted pages, each with one sub-resource fetch.

    Slow every other page (and stagger the sub-resource) to widen the race
    window so a snapshot/incidental bleed across worker pages would surface.
    """
    page_ids = [f"page-{i}" for i in range(n_pages)]

    async def page_handler(request: web.Request) -> web.Response:
        pid = request.match_info["pid"]
        idx = int(pid.rsplit("-", 1)[-1])
        if idx % 2 == 0:
            await asyncio.sleep(0.05)
        html = (
            f"<html><head><title>{pid}</title></head><body>"
            f"<h1 id='fp'>{pid}</h1>"
            f"<script>fetch('/sub/{pid}').then(r => r.text());</script>"
            "</body></html>"
        )
        return web.Response(text=html, content_type="text/html")

    async def sub_handler(request: web.Request) -> web.Response:
        pid = request.match_info["pid"]
        await asyncio.sleep(0.02)
        return web.Response(
            body=f"sub-{pid}".encode(), content_type="application/octet-stream"
        )

    app = web.Application()
    app.router.add_get("/fp/{pid}", page_handler)
    app.router.add_get("/sub/{pid}", sub_handler)
    return app, page_ids


@pytest.fixture
async def iso_transport(  # type: ignore[no-untyped-def]
    has_browser: bool,
    memory_session_factory: ScopedSessionFactory,
):
    """A DB-backed browser transport for the isolation test; torn down after."""
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


def _patch_resolve_yields(transport: PlaywrightTransport) -> None:
    """Force interleaving at the snapshot→persist boundary inside ``_resolve``."""
    original = transport._resolve

    @functools.wraps(original)
    async def patched(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        result = await original(*args, **kwargs)
        await asyncio.sleep(0)
        return result

    transport._resolve = patched  # type: ignore[method-assign]


async def test_concurrent_resolves_isolate_snapshot_and_incidentals(
    iso_transport: PlaywrightTransport,
    memory_session_factory: ScopedSessionFactory,
) -> None:
    """Concurrent resolves on distinct worker pages don't cross-contaminate.

    Each response's content/url must be ITS served page (no snapshot bleed),
    and each captured incidental must be tagged to ITS request_id (no
    cross-attribution). Drives one worker page per request and forces
    interleaving with ``asyncio.sleep(0)`` at the snapshot→persist boundary.
    """
    transport = iso_transport
    _patch_resolve_yields(transport)
    sf = memory_session_factory
    n_pages = 6
    app, page_ids = _fingerprint_app(n_pages)
    server = await _start_server(app)
    try:
        # Distinct worker page per request -> isolation is by WorkerPage.
        plan: list[tuple[int, int, str, str]] = []  # worker_id, rid, pid, url
        for i, pid in enumerate(page_ids):
            url = f"{server.base_url}/fp/{pid}"
            rid = await _insert_request_row(sf, url, qc=i + 1)
            plan.append((i, rid, pid, url))

        async def resolve_one(worker_id: int, rid: int, url: str):  # type: ignore[no-untyped-def]
            handle = await transport.acquire(worker_id)
            queued = QueuedRequest(request=_request_for(url), request_id=rid)
            return await transport.resolve(
                handle,
                queued,
                await_conditions=(
                    WaitForLoadState(state="networkidle", timeout=5000),
                ),
            )

        responses = await asyncio.gather(
            *(resolve_one(w, rid, url) for (w, rid, pid, url) in plan)
        )

        # No snapshot bleed: each response is its own page's fingerprint + url
        # (and no OTHER page's fingerprint leaked in).
        for (_w, _rid, pid, url), resp in zip(plan, responses, strict=True):
            assert f">{pid}</h1>" in resp.text, (
                f"snapshot bleed: expected {pid}, got {resp.text[:120]!r}"
            )
            others = [other for other in page_ids if other != pid]
            assert not any(
                f">{other}</h1>" in resp.text for other in others
            ), f"foreign fingerprint bled into {pid}"
            assert resp.url == url

        # No incidental cross-contamination: every incidental's url references
        # its own parent_request_id's page.
        rid_to_pid = {rid: pid for (_w, rid, pid, _url) in plan}
        async with sf() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT parent_request_id, url FROM "
                        "incidental_requests"
                    )
                )
            ).all()
        assert rows, "no incidentals captured"
        misattributed = [
            (parent_id, url)
            for parent_id, url in rows
            if "about:blank" not in url
            and f"/sub/{rid_to_pid.get(parent_id, '<none>')}" not in url
            and f"/fp/{rid_to_pid.get(parent_id, '<none>')}" not in url
        ]
        assert misattributed == [], (
            f"incidentals attributed to wrong parent: {misattributed[:10]}"
        )
    finally:
        await server.runner.cleanup()
