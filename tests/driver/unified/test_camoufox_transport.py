"""Tests for ``CamoufoxTransport`` (B6).

``CamoufoxTransport`` is a one-method subclass of ``PlaywrightTransport``: it
forces the camoufox engine. Everything else (lifecycle, resolve, crash
recovery, archive) is inherited and already proven for the parent in B1–B5, so
this module covers:

  - the engine-selection delta (browser-free): the subclass always builds a
    camoufox engine, where the parent defaults to playwright;
  - that it remains structurally a ``Transport`` and a ``PlaywrightTransport``;
  - that the inherited crash predicate recognizes camoufox's Firefox page-error
    crash (a ``Connection closed`` channel error);
  - the full ``TransportConformance`` over a REAL camoufox, gated on a launch
    probe so it skips cleanly where the camoufox/Firefox binary is absent.

Fidelity strategy mirrors B5: a real headless engine, not a stub. The archive
``resolve_archive`` path is inherited byte-for-byte from ``PlaywrightTransport``
(exercised by B4 + B5), so the two archive conformance methods are skipped here
rather than re-driving a camoufox download.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.browser_engine.engines import (
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.unified_driver import CamoufoxTransport, QueuedRequest
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.transport import Transport, WorkerHandle
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.test_async_lifecycle_conformance import (
    AsyncLifecycleConformance,
)
from tests.driver.unified.test_playwright_transport import (
    _insert_request_row,
    _Scraper,
    _sql_manager,
    _start_server,
)
from tests.driver.unified.test_transport_conformance import (
    TransportConformance,
)

if TYPE_CHECKING:
    from jkent.driver.database_engine.scoped_session import (
        ScopedSessionFactory,
    )

# Camoufox can only launch one Firefox at a time, so co-locate every test in
# this module on a single xdist worker (honored under --dist loadgroup) — they
# never run concurrently with each other.
pytestmark = pytest.mark.xdist_group("camoufox")


# --- Browser-free unit checks --------------------------------------------


def test_build_engine_is_camoufox() -> None:
    """The subclass always builds a camoufox engine (no CFCAP_HANDLER needed)."""
    engine = CamoufoxTransport(_Scraper())._build_engine()
    assert isinstance(engine, CamoufoxEngine)


def test_parent_defaults_to_playwright_engine() -> None:
    """Contrast: a plain ``PlaywrightTransport`` builds a playwright engine."""
    engine = PlaywrightTransport(_Scraper())._build_engine()
    assert isinstance(engine, PlaywrightEngine)


def test_is_a_playwright_transport() -> None:
    """It inherits the whole Playwright transport surface."""
    transport = CamoufoxTransport(_Scraper())
    assert isinstance(transport, PlaywrightTransport)
    for method in (
        "open",
        "aclose",
        "acquire",
        "release",
        "resolve",
        "resolve_archive",
        "finish_archiving",
    ):
        assert callable(getattr(transport, method))


def test_should_restart_recognizes_camoufox_crash() -> None:
    """The inherited predicate flags camoufox's ``Connection closed`` crash."""
    transport = CamoufoxTransport(_Scraper())
    assert transport.should_restart(Exception("Connection closed")) is True
    assert transport.should_restart(ValueError("unrelated")) is False


# --- Real-camoufox conformance (skipped cleanly without the binary) -------


@pytest.fixture(scope="session")
def has_camoufox() -> bool:
    """Whether a camoufox engine can actually launch in this environment."""

    async def _launches() -> bool:
        transport = CamoufoxTransport(_Scraper(), headless=True)
        try:
            await transport.open()
            await transport.aclose()
        except Exception:
            return False
        return True

    return asyncio.run(_launches())


class TestCamoufoxTransportLifecycle(AsyncLifecycleConformance):
    """``CamoufoxTransport`` honors the open -> use -> aclose lifecycle."""

    @pytest.fixture
    async def subject(self, has_camoufox: bool):  # type: ignore[no-untyped-def]
        # Yield + aclose in teardown: the base suite's
        # ``test_open_awaits_to_none`` opens without closing, which for camoufox
        # would leave a Firefox holding the single-instance profile lock and
        # deadlock the next camoufox test. The teardown guarantees cleanup.
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        transport = CamoufoxTransport(_Scraper(), headless=True)
        try:
            yield transport
        finally:
            await transport.aclose()

    async def test_open_then_aclose_releases_resources(
        self, subject: AsyncLifecycle
    ) -> None:
        """``aclose`` drops the engine + context acquired in ``open``."""
        assert isinstance(subject, CamoufoxTransport)
        await subject.open()
        assert subject._engine is not None
        assert subject._context is not None
        await subject.aclose()
        assert subject._engine is None
        # type-checkers keep the narrowing from the pre-close asserts; aclose() really does reset
        assert subject._engine_cm is None  # type: ignore[unreachable]
        assert subject._context is None
        assert subject._handles == {}


def _no_subresource_app(server_holder: dict[str, str]) -> web.Application:
    async def page(_request: web.Request) -> web.Response:
        return web.Response(
            status=200,
            content_type="text/html",
            text="<html><body><p>camoufox conformance</p></body></html>",
        )

    app = web.Application()
    app.router.add_get("/page", page)
    return app


class TestCamoufoxTransportConformance(TransportConformance):
    """Run the shared ``Transport`` contract against a real camoufox engine."""

    @pytest.fixture
    async def subject(
        self,
        has_camoufox: bool,
        memory_session_factory: ScopedSessionFactory,
    ) -> AsyncIterator[CamoufoxTransport]:
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        server = await _start_server(_no_subresource_app({}))
        self._url = f"{server.base_url}/page"  # type: ignore
        self._request_id = await _insert_request_row(  # type: ignore
            memory_session_factory, self._url, qc=1
        )
        transport = CamoufoxTransport(
            _Scraper(),
            headless=True,
            db=_sql_manager(memory_session_factory),
        )
        try:
            yield transport
        finally:
            await server.runner.cleanup()

    def make_queued(self, *, request_id: int | None = None) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=self._url,  # type: ignore
                ),
                continuation="parse",
            ),
            request_id=request_id
            if request_id is not None
            else self._request_id,  # type: ignore
        )

    async def test_resolve_archive_metadata_and_chunks(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """Archive path is inherited unchanged — covered by B4 + B5."""
        pytest.skip("resolve_archive identical to PlaywrightTransport (B4/B5)")

    async def test_finish_archiving_is_no_throw(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """Archive path is inherited unchanged — covered by B4 + B5."""
        pytest.skip(
            "finish_archiving identical to PlaywrightTransport (B4/B5)"
        )
