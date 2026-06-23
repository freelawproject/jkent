"""Scraper ``default_headers`` through ``HttpxTransport``.

A scraper's ``default_headers`` ClassVar is the baseline for every httpx
request; a per-request header with the same name (any case) overrides the
default. Covers both the resolve and the streaming (archive) paths. The
Playwright/Camoufox transports ignore ``default_headers`` entirely.
"""

from __future__ import annotations

from typing import ClassVar

import httpx

from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest


class _HeaderedScraper(BaseScraper[dict]):
    default_headers: ClassVar[dict[str, str]] = {
        "User-Agent": "jkent-default/1.0",
        "X-Court-Client": "jkent",
    }


class _PlainScraper(BaseScraper[dict]):
    pass


_seen: list[httpx.Headers] = []


def _echo_handler(request: httpx.Request) -> httpx.Response:
    _seen.append(request.headers)
    return httpx.Response(200, text="ok")


def _make_request(headers: dict[str, str] | None = None) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="http://test/page", headers=headers
        ),
        continuation="parse_page",
    )


def _mocked_transport(
    scraper: type[BaseScraper] | BaseScraper,
) -> HttpxTransport:
    transport = HttpxTransport(scraper=scraper)
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(_echo_handler)
    )
    return transport


async def _wire_headers(
    scraper: type[BaseScraper] | BaseScraper,
    request: Request,
) -> httpx.Headers:
    transport = _mocked_transport(scraper)
    try:
        handle = await transport.acquire(0)
        await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
        return _seen[-1]
    finally:
        await transport.aclose()


class TestDefaultHeaders:
    async def test_defaults_sent_when_request_has_no_headers(self) -> None:
        headers = await _wire_headers(_HeaderedScraper, _make_request())
        assert headers["user-agent"] == "jkent-default/1.0"
        assert headers["x-court-client"] == "jkent"

    async def test_per_request_header_overrides_default(self) -> None:
        headers = await _wire_headers(
            _HeaderedScraper, _make_request({"X-Court-Client": "special"})
        )
        assert headers["x-court-client"] == "special"
        # Untouched defaults still ride along.
        assert headers["user-agent"] == "jkent-default/1.0"

    async def test_override_matches_case_insensitively(self) -> None:
        headers = await _wire_headers(
            _HeaderedScraper, _make_request({"user-agent": "custom/2.0"})
        )
        # Exactly one value on the wire — the default must not be sent
        # alongside a differently-cased override.
        assert headers.get_list("user-agent") == ["custom/2.0"]

    async def test_permanent_headers_override_defaults(self) -> None:
        # Permanent headers are merged into the request's headers before
        # the transport sees them, so they win over defaults too.
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="http://test/page"
            ),
            continuation="parse_page",
            permanent={"headers": {"X-Court-Client": "from-permanent"}},
        )
        headers = await _wire_headers(_HeaderedScraper, request)
        assert headers["x-court-client"] == "from-permanent"

    async def test_plain_scraper_sends_no_extras(self) -> None:
        headers = await _wire_headers(_PlainScraper, _make_request())
        assert "x-court-client" not in headers

    async def test_archive_stream_gets_defaults(self) -> None:
        transport = _mocked_transport(_HeaderedScraper)
        try:
            handle = await transport.acquire(0)
            stream = await transport.resolve_archive(
                handle,
                QueuedRequest(request=_make_request(), request_id=1),
            )
            try:
                assert stream.status_code == 200
            finally:
                await transport.finish_archiving(stream)
            assert _seen[-1]["x-court-client"] == "jkent"
        finally:
            await transport.aclose()
