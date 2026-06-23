"""FOLLOW_REDIRECTS through ``HttpxTransport``.

The unified port of the old request-manager behavior
(tests/unorganized/test_follow_redirects.py): a scraper opts into httpx
redirect-following by declaring ``DriverRequirement.FOLLOW_REDIRECTS``;
without it, redirect responses come back unfollowed. Covers both the
resolve and the streaming (archive) paths.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from jkent.driver.unified_driver.transport.httpx_transport import (
    _wants_follow_redirects,
)


class _RedirectingScraper(BaseScraper[dict]):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.FOLLOW_REDIRECTS,
    ]


class _PlainScraper(BaseScraper[dict]):
    pass


class _ManualRedirectScraper(BaseScraper[dict]):
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
        continuation="parse_page",
    )


def _mocked_transport(
    scraper: type[BaseScraper] | BaseScraper,
) -> HttpxTransport:
    transport = HttpxTransport(scraper=scraper)
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_redirect_handler)
    )
    return transport


class TestWantsFollowRedirectsHelper:
    def test_default_false(self) -> None:
        assert _wants_follow_redirects(BaseScraper) is False

    def test_opt_in_true(self) -> None:
        assert _wants_follow_redirects(_RedirectingScraper) is True


class TestHttpxTransportFollowRedirects:
    def test_caches_flag_from_scraper(self) -> None:
        transport = HttpxTransport(scraper=_RedirectingScraper)
        assert transport._follow_redirects is True

    def test_default_scraper_does_not_follow(self) -> None:
        transport = HttpxTransport(scraper=_PlainScraper)
        assert transport._follow_redirects is False

    async def test_resolve_follows_redirect(self) -> None:
        transport = _mocked_transport(_RedirectingScraper)
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
        transport = _mocked_transport(_ManualRedirectScraper)
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
        transport = _mocked_transport(_RedirectingScraper)
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
        transport = _mocked_transport(_ManualRedirectScraper)
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
