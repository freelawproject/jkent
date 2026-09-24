"""Tests for the camoufox engine behind ``PlaywrightTransport``.

Camoufox is selected by a camoufox requirement or an explicit
``browser_type="camoufox"``. Everything else (lifecycle, resolve, crash
recovery, archive) is the Playwright transport tested in
``test_playwright_transport.py``, so this module covers:

  - the engine-selection delta (browser-free): ``browser_type="camoufox"``
    builds a camoufox engine, where the default is playwright;
  - that the crash predicate recognizes camoufox's Firefox page-error crash
    (a ``Connection closed`` channel error);
  - the full ``TransportConformance`` over a REAL camoufox, gated on a launch
    probe so it skips cleanly where the camoufox/Firefox binary is absent.

As there, a real headless engine, not a stub. The
archive conformance methods drive a real Firefox download; the pdf.js inline
render, the other Firefox archive shape, is pinned by the via-less PDF test.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.browser_engine.engines import (
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.unified_driver import QueuedRequest
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.test_async_lifecycle_conformance import (
    AsyncLifecycleConformance,
)
from tests.driver.unified.test_playwright_archive import fetch_via_less_pdf
from tests.driver.unified.test_playwright_transport import (
    PlaywrightArchiveConformance,
    _insert_request_row,
    _Scraper,
    _sql_manager,
)
from tests.driver.unified.test_transport_conformance import (
    ClassificationConformance,
)
from tests.servers import (
    single_page_app,
    start_app,
    status_app,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from jkent.data_types import BaseScraper, Response

# Camoufox can only launch one Firefox at a time, so co-locate every test in
# this module on a single xdist worker (honored under --dist loadgroup) — they
# never run concurrently with each other.
pytestmark = pytest.mark.xdist_group("camoufox")


# --- Browser-free unit checks --------------------------------------------


def _camoufox_transport(
    scraper: BaseScraper[Any], **kwargs: Any
) -> PlaywrightTransport:
    return PlaywrightTransport(scraper, browser_type="camoufox", **kwargs)


def test_build_engine_is_camoufox() -> None:
    """``browser_type="camoufox"`` builds a camoufox engine (no CFCAP needed)."""
    engine = _camoufox_transport(_Scraper())._build_engine()
    assert isinstance(engine, CamoufoxEngine)


def test_default_is_playwright_engine() -> None:
    """Contrast: a plain ``PlaywrightTransport`` builds a playwright engine."""
    engine = PlaywrightTransport(_Scraper())._build_engine()
    assert isinstance(engine, PlaywrightEngine)


def test_should_restart_recognizes_camoufox_crash() -> None:
    """Pre-``open``, the base engine predicate flags ``Connection closed``."""
    transport = _camoufox_transport(_Scraper())
    assert transport.should_restart(Exception("Connection closed")) is True
    assert transport.should_restart(ValueError("unrelated")) is False


# --- Real-camoufox conformance (skipped cleanly without the binary) -------


@pytest.fixture(scope="session")
def has_camoufox() -> bool:
    """Whether a camoufox engine can actually launch in this environment."""

    async def _launches() -> bool:
        transport = _camoufox_transport(_Scraper(), headless=True)
        try:
            await transport.open()
            await transport.aclose()
        except Exception:
            return False
        return True

    return asyncio.run(_launches())


class TestCamoufoxLifecycle(AsyncLifecycleConformance):
    """A camoufox transport honors the open -> use -> aclose lifecycle."""

    @override
    @pytest.fixture
    async def subject(self, has_camoufox: bool):
        # Yield + aclose in teardown: the base suite's
        # ``test_open_awaits_to_none`` opens without closing, which for camoufox
        # would leave a Firefox holding the single-instance profile lock and
        # deadlock the next camoufox test. The teardown guarantees cleanup.
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        transport = _camoufox_transport(_Scraper(), headless=True)
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


class TestCamoufoxConformance(PlaywrightArchiveConformance):
    """Run the shared ``Transport`` contract against a real camoufox engine."""

    @override
    @pytest.fixture
    async def subject(
        self,
        has_camoufox: bool,
        memory_session_factory: async_sessionmaker[AsyncSession],
    ) -> AsyncIterator[PlaywrightTransport]:
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        html = "<html><body><p>camoufox conformance</p></body></html>"
        server = await start_app(single_page_app(html))
        self._url = f"{server.base_url}/page"
        self._request_id = await _insert_request_row(
            memory_session_factory, self._url
        )
        transport = _camoufox_transport(
            _Scraper(),
            headless=True,
            db=_sql_manager(memory_session_factory),
        )
        try:
            yield transport
        finally:
            # The conformance tests drive open()/aclose() themselves, but a
            # failure between them would skip their aclose() and leak the live
            # browser, deadlocking the profile lock for the next camoufox test.
            # aclose() is idempotent, so guarantee teardown here.
            await transport.aclose()
            await server.runner.cleanup()

    @override
    def make_queued(self, *, request_id: int | None = None) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=self._url,
                ),
                step="parse",
            ),
            request_id=request_id
            if request_id is not None
            else self._request_id,
        )


class TestCamoufoxClassification(ClassificationConformance):
    """Status classification over a real camoufox/firefox navigation.

    The classification logic is the Playwright transport's; what this binding pins is the engine half —
    firefox surfacing non-200 navigation statuses into ``resolve`` the same
    way chromium does.
    """

    # Bound by the autouse ``_bind_env`` fixture before every test.
    _sf: async_sessionmaker[AsyncSession]

    @pytest.fixture(autouse=True)
    async def _bind_env(
        self,
        has_camoufox: bool,
        memory_session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
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
        transport = _camoufox_transport(
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


async def test_via_less_pdf_is_archived_not_displayed(
    has_camoufox: bool,
) -> None:
    """Firefox opens an ``application/pdf`` in pdf.js rather than downloading
    it; the archive must still be the file's bytes (the inline branch)."""
    if not has_camoufox:
        pytest.skip("no launchable camoufox engine in this environment")
    transport = _camoufox_transport(_Scraper(), headless=True)
    await transport.open()
    try:
        await fetch_via_less_pdf(transport)
    finally:
        await transport.aclose()
