"""Tests for the FOLLOW_REDIRECTS DriverRequirement.

Verifies the request manager respects the per-scraper opt-in for httpx
redirect-following, in both sync and async / request and stream paths.
"""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from kent.common.request_manager import (
    AsyncRequestManager,
    SyncRequestManager,
    _wants_follow_redirects,
)
from kent.data_types import (
    BaseRequest,
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
)


class _RedirectingScraper(BaseScraper[dict]):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.FOLLOW_REDIRECTS,
    ]


class _PlainScraper(BaseScraper[dict]):
    pass


def _redirect_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/start":
        return httpx.Response(302, headers={"Location": "/dest"})
    if request.url.path == "/dest":
        return httpx.Response(200, text="final")
    return httpx.Response(404)


def _make_request(url: str) -> BaseRequest:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse_page",
    )


class TestWantsFollowRedirectsHelper:
    def test_default_false(self) -> None:
        assert _wants_follow_redirects(BaseScraper) is False

    def test_opt_in_true(self) -> None:
        assert _wants_follow_redirects(_RedirectingScraper) is True


class TestSyncRequestManagerFollowRedirects:
    def test_caches_flag_from_scraper(self) -> None:
        rm = SyncRequestManager(scraper=_RedirectingScraper)
        assert rm._follow_redirects is True

    def test_default_scraper_does_not_follow(self) -> None:
        rm = SyncRequestManager(scraper=_PlainScraper)
        assert rm._follow_redirects is False

    def test_resolve_request_follows_redirect(self) -> None:
        rm = SyncRequestManager(scraper=_RedirectingScraper)
        rm._client = httpx.Client(
            transport=httpx.MockTransport(_redirect_handler)
        )
        response = rm.resolve_request(_make_request("http://test/start"))
        assert response.status_code == 200
        assert response.text == "final"

    def test_resolve_request_does_not_follow_when_off(self) -> None:
        rm = SyncRequestManager(scraper=_PlainScraper)
        rm._client = httpx.Client(
            transport=httpx.MockTransport(_redirect_handler)
        )
        response = rm.resolve_request(_make_request("http://test/start"))
        assert response.status_code == 302


class TestAsyncRequestManagerFollowRedirects:
    def test_caches_flag_from_scraper(self) -> None:
        rm = AsyncRequestManager(scraper=_RedirectingScraper)
        assert rm._follow_redirects is True

    def test_default_scraper_does_not_follow(self) -> None:
        rm = AsyncRequestManager(scraper=_PlainScraper)
        assert rm._follow_redirects is False

    @pytest.mark.asyncio
    async def test_resolve_request_follows_redirect(self) -> None:
        rm = AsyncRequestManager(scraper=_RedirectingScraper)
        rm._client = httpx.AsyncClient(
            transport=httpx.MockTransport(_redirect_handler)
        )
        try:
            response = await rm.resolve_request(
                _make_request("http://test/start")
            )
            assert response.status_code == 200
            assert response.text == "final"
        finally:
            await rm.close()

    @pytest.mark.asyncio
    async def test_resolve_request_does_not_follow_when_off(self) -> None:
        rm = AsyncRequestManager(scraper=_PlainScraper)
        rm._client = httpx.AsyncClient(
            transport=httpx.MockTransport(_redirect_handler)
        )
        try:
            response = await rm.resolve_request(
                _make_request("http://test/start")
            )
            assert response.status_code == 302
        finally:
            await rm.close()
