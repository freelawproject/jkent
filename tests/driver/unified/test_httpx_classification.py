"""Generative status-code classification sweep for ``HttpxTransport``.

``resolve`` consults the scraper's classifier — ``classify``, which merges the
framework defaults with the scraper's ``HTTP_CODE_TYPES`` override mapping —
via the shared
``Transport.classify_and_raise``, and:

  - returns a ``Response`` for successful codes,
  - raises ``HTTPResponseAssumptionException`` for transient codes,
  - raises ``PersistentHTTPResponseException`` for persistent codes — which
    includes any code absent from the active map (the classifier's
    unlisted-is-persistent fallback).

The cross-transport contract (fixed representative codes, dynamic
content-based overrides, speculative narrowing) lives in
``ClassificationConformance`` (``test_transport_conformance``), bound to every
transport. What stays here is the *generative* half — a sweep over codes from
every bucket (plus some in none) and
arbitrary set-overrides, hypothesis-driven with the scraper's own classifier
as the oracle — which only the cheap HTTP transport can afford to sweep (a
fresh server + transport per example is too slow for a browser).

Out of scope: archive download. Header-based dynamic overrides aren't
exercised (server/client add their own headers, so the oracle stays
header-independent).
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, cast

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.data_types import (
    BaseScraper,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from tests.httpx_mock import mocked_httpx_transport
from tests.servers import start_app, status_app

_SUCCESS = [200, 201, 202, 203]
_TRANSIENT = [408, 425, 429, 502, 503, 504]
_PERSISTENT = [400, 401, 403, 404, 409, 422, 500, 501]
# In no default bucket (redirects, nginx/Cloudflare nonstandards): persistent
# via the classifier's fallback unless an override map rescues them.
_UNLISTED = [302, 499, 520, 522]
_ALL = _SUCCESS + _TRANSIENT + _PERSISTENT + _UNLISTED

Outcome = Literal["response", "transient", "persistent"]


def _expected(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> Outcome:
    """The scraper's own verdict — the oracle ``_classify_and_raise`` follows."""
    verdict = scraper.classify(code, None, body)
    if verdict is HTTPCodeType.TRANSIENT:
        return "transient"
    if verdict is HTTPCodeType.PERSISTENT:
        return "persistent"
    return "response"


def _make_scraper(
    code_types: Mapping[int, HTTPCodeType],
) -> type[BaseScraper[Any]]:
    """A BaseScraper subclass with the given status-code override map."""
    cls = type(
        "_OverrideScraper",
        (BaseScraper,),
        {"HTTP_CODE_TYPES": dict(code_types)},
    )
    return cast("type[BaseScraper[Any]]", cls)


async def _resolve_status(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> Any:
    """Serve ``code``+``body`` and resolve one GET through HttpxTransport(scraper)."""
    server = await start_app(status_app(code, body))
    transport = HttpxTransport(scraper=scraper)
    await transport.open()
    handle = await transport.acquire(0)
    try:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{server.base_url}/r"
            ),
            step="parse",
        )
        return await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
    finally:
        await transport.aclose()
        await server.runner.cleanup()


def _assert_outcome(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> None:
    outcome = _expected(scraper, code, body)

    async def run() -> None:
        if outcome == "response":
            resp = await _resolve_status(scraper, code, body)
            assert resp.status_code == code
            assert resp.content == body
        elif outcome == "transient":
            with pytest.raises(HTTPResponseAssumptionException):
                await _resolve_status(scraper, code, body)
        else:
            with pytest.raises(PersistentHTTPResponseException):
                await _resolve_status(scraper, code, body)

    asyncio.run(run())


# --- default classification ----------------------------------------------


@pytest.mark.parametrize("code", _ALL)
def test_default_classification(code: int) -> None:
    _assert_outcome(BaseScraper, code, b"<html>body</html>")


# --- arbitrary set overrides ----------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(
    code=st.sampled_from(_ALL),
    # A mapping holds one type per code, so the override is disjoint by
    # construction — a code can never land in two buckets.
    code_types=st.dictionaries(
        keys=st.sampled_from(_ALL),
        values=st.sampled_from(list(HTTPCodeType)),
        max_size=5,
    ),
)
def test_override_classification(
    code: int, code_types: dict[int, HTTPCodeType]
) -> None:
    scraper = _make_scraper(code_types)
    _assert_outcome(scraper, code, b"x")


@pytest.mark.parametrize(
    ("headers", "body", "stored"),
    [
        (None, None, None),
        # Status and headers read, body not: nothing worth writing over a
        # previous attempt's stored block page.
        ({"Retry-After": "5"}, None, None),
        # An empty body the server did send is an observation.
        ({}, b"", b""),
        ({}, b"blocked", b"blocked"),
    ],
    ids=["nothing", "headers only", "empty body", "body"],
)
def test_debug_response_needs_an_observed_body(
    headers: dict[str, str] | None, body: bytes | None, stored: bytes | None
) -> None:
    request = Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url="https://e/x"),
        step="parse",
    )
    with pytest.raises(HTTPResponseAssumptionException) as excinfo:
        HttpxTransport().classify_and_raise(
            BaseScraper,
            request,
            status_code=503,
            headers=headers,
            body=body,
            url="https://e/x",
        )
    debug = excinfo.value.debug_response
    assert (None if debug is None else debug.content) == stored


# --- network failures ------------------------------------------------------


def _closed_port_url() -> str:
    """A URL on a 127.0.0.1 port nothing listens on."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/r"


@contextlib.asynccontextmanager
async def _hang_up_url() -> AsyncIterator[str]:
    """A URL whose server accepts the connection and closes it unanswered."""

    async def hang_up(
        _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        writer.close()

    server = await asyncio.start_server(hang_up, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/r"
    finally:
        server.close()
        await server.wait_closed()


def _get(url: str, timeout: Any = None) -> QueuedRequest:
    return QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=url, timeout=timeout
            ),
            step="parse",
        ),
        request_id=1,
    )


@pytest.mark.parametrize("failure", ["connection refused", "hang-up"])
async def test_a_network_failure_is_transient(failure: str) -> None:
    """An httpx ``TransportError`` becomes a NETWORK ``TransientException``."""
    async with contextlib.AsyncExitStack() as stack:
        if failure == "hang-up":
            url = await stack.enter_async_context(_hang_up_url())
        else:
            url = _closed_port_url()
        transport = HttpxTransport(scraper=BaseScraper)
        await transport.open()
        stack.push_async_callback(transport.aclose)
        handle = await transport.acquire(0)

        with pytest.raises(TransientException) as excinfo:
            await transport.resolve(handle, _get(url))

    assert type(excinfo.value) is TransientException
    assert excinfo.value.kind is TransientKind.NETWORK
    assert excinfo.value.url == url
    assert isinstance(excinfo.value.__cause__, httpx.TransportError)


@pytest.mark.parametrize(
    ("timeout", "sent", "reported"),
    [
        (None, (7.0, 7.0), 7.0),
        (3.0, (3.0, 3.0), 3.0),
        ((1.0, 4.0), (1.0, 4.0), 4.0),
    ],
    ids=["unset", "float", "connect-read tuple"],
)
async def test_request_timeout_reaches_httpx_and_the_error(
    timeout: Any, sent: tuple[float, float], reported: float
) -> None:
    """The request's timeout goes to httpx as (connect, read); a timeout
    error reports the read value, else the transport's own."""
    seen: list[dict[str, float]] = []

    def time_out(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        raise httpx.ReadTimeout("slow", request=request)

    transport = mocked_httpx_transport(time_out, timeout=7.0)
    try:
        handle = await transport.acquire(0)
        with pytest.raises(RequestTimeoutException) as excinfo:
            await transport.resolve(handle, _get("http://test/r", timeout))
    finally:
        await transport.aclose()

    assert (seen[0]["connect"], seen[0]["read"]) == sent
    assert excinfo.value.timeout_seconds == reported


@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_an_unsupported_scheme_is_a_config_error(path: str) -> None:
    """A URL httpx cannot send (bad scheme) is the scraper's bug: persistent,
    not a NETWORK transient that retries to budget."""
    url = "ftp://127.0.0.1/r"
    transport = HttpxTransport(scraper=BaseScraper)
    await transport.open()
    try:
        handle = await transport.acquire(0)
        with pytest.raises(ScraperConfigError) as excinfo:
            if path == "resolve":
                await transport.resolve(handle, _get(url))
            else:
                await transport.resolve_archive(handle, _get(url))
    finally:
        await transport.aclose()

    assert url in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.UnsupportedProtocol)
