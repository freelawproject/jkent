"""FOLLOW_REDIRECTS through ``HttpxTransport``.

The unified port of the old request-manager behavior
(tests/unorganized/test_follow_redirects.py): a scraper opts into httpx
redirect-following by declaring ``DriverRequirement.FOLLOW_REDIRECTS``;
without it, redirect responses come back unfollowed. Covers both the
resolve and the streaming (archive) paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import pytest
from aiohttp import web

from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from tests.httpx_mock import mocked_httpx_transport
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _RedirectingScraper(BaseScraper[dict[str, Any]]):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.FOLLOW_REDIRECTS,
    ]


class _PlainScraper(BaseScraper[dict[str, Any]]):
    pass


class _ManualRedirectScraper(BaseScraper[dict[str, Any]]):
    """No FOLLOW_REDIRECTS: consumes redirect responses itself.

    302 is in no default bucket, so the classifier's unlisted-is-persistent
    fallback would fail it fast; a scraper that reads redirects manually
    must claim the code as successful to receive the response.
    """

    HTTP_CODE_TYPES: ClassVar = {302: HTTPCodeType.SUCCESSFUL}


def _redirect_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/start":
        return httpx.Response(302, headers={"Location": "/dest"})
    if request.url.path == "/dest":
        return httpx.Response(200, text="final")
    return httpx.Response(404)


def _make_request(url: str) -> Request:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        step="parse_page",
    )


@pytest.fixture
async def redirect_server() -> AsyncIterator[StartedServer]:
    """``/a`` redirects to ``/b``, which answers 200."""

    async def a(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/b")

    async def b(_request: web.Request) -> web.Response:
        return web.Response(text="final")

    app = web.Application()
    app.router.add_get("/a", a)
    app.router.add_get("/b", b)
    started = await start_app(app)
    try:
        yield started
    finally:
        await started.aclose()


class TestHttpxTransportFollowRedirects:
    async def test_resolve_follows_redirect(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _RedirectingScraper
        )
        try:
            handle = await transport.acquire(0)
            response = await transport.resolve(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            assert response.status_code == 200
            assert response.text == "final"
        finally:
            await transport.aclose()

    async def test_resolve_does_not_follow_when_off(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _ManualRedirectScraper
        )
        try:
            handle = await transport.acquire(0)
            response = await transport.resolve(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            assert response.status_code == 302
        finally:
            await transport.aclose()

    async def test_archive_stream_follows_redirect(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _RedirectingScraper
        )
        try:
            handle = await transport.acquire(0)
            stream = await transport.resolve_archive(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            try:
                body = b"".join([chunk async for chunk in stream])
                assert stream.status_code == 200
                assert body == b"final"
            finally:
                await transport.finish_archiving(stream)
        finally:
            await transport.aclose()

    async def test_archive_stream_does_not_follow_when_off(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _ManualRedirectScraper
        )
        try:
            handle = await transport.acquire(0)
            stream = await transport.resolve_archive(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            try:
                assert stream.status_code == 302
            finally:
                await transport.finish_archiving(stream)
        finally:
            await transport.aclose()

    @pytest.mark.parametrize(
        "archive", [False, True], ids=["resolve", "archive"]
    )
    async def test_url_is_the_final_url_after_redirects(
        self, redirect_server: StartedServer, archive: bool
    ) -> None:
        transport = HttpxTransport(scraper=_RedirectingScraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            queued = QueuedRequest(
                request=_make_request(f"{redirect_server.base_url}/a"),
                request_id=1,
            )
            if archive:
                stream = await transport.resolve_archive(handle, queued)
                try:
                    assert stream.status_code == 200
                    assert stream.url == f"{redirect_server.base_url}/b"
                finally:
                    await transport.finish_archiving(stream)
            else:
                response = await transport.resolve(handle, queued)
                assert response.status_code == 200
                assert response.url == f"{redirect_server.base_url}/b"
        finally:
            await transport.aclose()
