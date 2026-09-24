"""One "no timeout" default across the transport layer.

A request that sets no ``HTTPRequestParams.timeout`` (and whose step sets
none) falls back to the transport's ``timeout``, and a transport built
without one falls back to ``DEFAULT_TIMEOUT_S`` — on both transports, on
every wait.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar

import pytest
from aiohttp import web

from jkent.common.exceptions import RequestTimeoutException
from jkent.common.request import DEFAULT_TIMEOUT_S
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from jkent.driver.unified_driver.bootstrap import build_transport
from jkent.driver.unified_driver.transport import httpx_transport
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
    ResolveTimeout,
)
from tests.db_staging import insert_request_row
from tests.driver.unified.test_playwright_transport import (
    _Scraper,
    _sql_manager,
)
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _BrowserScraper(_Scraper):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.JS_EVAL
    ]


def _untimed(url: str) -> Request:
    """A request spelled the way a scraper spells it: no ``timeout``."""
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        step="parse",
    )


@pytest.fixture
async def slow_server() -> AsyncIterator[StartedServer]:
    """``/slow`` answers after two seconds — longer than any timeout here."""

    async def slow(_request: web.Request) -> web.Response:
        await asyncio.sleep(2.0)
        return web.Response(
            text="<html><body>late</body></html>", content_type="text/html"
        )

    app = web.Application()
    app.router.add_get("/slow", slow)
    started = await start_app(app)
    try:
        yield started
    finally:
        await started.aclose()


class TestBootstrapForwardsTimeout:
    def test_browser_transport_receives_the_timeout(self) -> None:
        transport = build_transport(_BrowserScraper(), timeout=5.0)
        assert isinstance(transport, PlaywrightTransport)
        assert transport.timeout == 5.0

    def test_both_transports_share_one_default(self) -> None:
        assert build_transport(_BrowserScraper()).timeout == DEFAULT_TIMEOUT_S
        assert build_transport(BaseScraper()).timeout == DEFAULT_TIMEOUT_S

    def test_an_unset_request_timeout_defers_to_the_transport(self) -> None:
        assert (
            HTTPRequestParams(method=HttpMethod.GET, url="x").timeout is None
        )


class TestPlaywrightFallback:
    @pytest.mark.parametrize(
        ("transport_timeout", "request_timeout", "expected_ms"),
        [
            pytest.param(2.5, None, 2500.0, id="unset-uses-transport"),
            pytest.param(
                None, None, DEFAULT_TIMEOUT_S * 1000.0, id="unset-default"
            ),
            pytest.param(2.5, 7, 7000.0, id="request-wins"),
            pytest.param(2.5, (1.0, 4.0), 4000.0, id="tuple-uses-read"),
        ],
    )
    def test_request_timeout_falls_back_to_transport_then_default(
        self,
        transport_timeout: float | None,
        request_timeout: float | tuple[float, float] | None,
        expected_ms: float,
    ) -> None:
        transport = (
            PlaywrightTransport(_Scraper())
            if transport_timeout is None
            else PlaywrightTransport(_Scraper(), timeout=transport_timeout)
        )
        assert transport._timeout_ms(request_timeout) == expected_ms

    async def test_untimed_request_times_out_at_the_transport_timeout(
        self,
        require_browser: None,
        memory_session_factory: async_sessionmaker[AsyncSession],
        slow_server: StartedServer,
    ) -> None:
        transport = PlaywrightTransport(
            _Scraper(),
            headless=True,
            timeout=0.3,
            db=_sql_manager(memory_session_factory),
        )
        await transport.open()
        try:
            handle = await transport.acquire(0)
            url = f"{slow_server.base_url}/slow"
            rid = await insert_request_row(memory_session_factory, url)
            with pytest.raises(ResolveTimeout) as excinfo:
                await transport.resolve(
                    handle,
                    QueuedRequest(request=_untimed(url), request_id=rid),
                )
            assert excinfo.value.timeout_seconds == pytest.approx(0.3)
        finally:
            await transport.aclose()


class TestHttpxFallback:
    @pytest.mark.parametrize(
        "archive", [False, True], ids=["resolve", "archive"]
    )
    async def test_untimed_request_times_out_at_the_transport_timeout(
        self, slow_server: StartedServer, archive: bool
    ) -> None:
        transport = HttpxTransport(scraper=_Scraper, timeout=0.3)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            url = f"{slow_server.base_url}/slow"
            queued = QueuedRequest(request=_untimed(url), request_id=1)
            with pytest.raises(RequestTimeoutException) as excinfo:
                if archive:
                    await transport.resolve_archive(handle, queued)
                else:
                    await transport.resolve(handle, queued)
            assert excinfo.value.timeout_seconds == pytest.approx(0.3)
        finally:
            await transport.aclose()

    async def test_untimed_request_on_untimed_transport_uses_the_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        slow_server: StartedServer,
    ) -> None:
        # Shrink the shared default so the test is quick; the point is that
        # a transport built without a timeout no longer means "no timeout".
        monkeypatch.setattr(httpx_transport, "DEFAULT_TIMEOUT_S", 0.3)
        transport = HttpxTransport(scraper=_Scraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            url = f"{slow_server.base_url}/slow"
            with pytest.raises(RequestTimeoutException) as excinfo:
                await transport.resolve(
                    handle, QueuedRequest(request=_untimed(url), request_id=1)
                )
            assert excinfo.value.timeout_seconds == pytest.approx(0.3)
        finally:
            await transport.aclose()
