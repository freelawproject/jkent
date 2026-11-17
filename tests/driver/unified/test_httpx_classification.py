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
transport — including this one, via ``test_transport_impl_conformance``.

What stays here is the *generative* half — a sweep over codes from every
bucket (plus some in none) and arbitrary set-overrides, hypothesis-driven with
the scraper's own classifier as the oracle — which only the cheap HTTP
transport can afford to sweep (a fresh server + transport per example is too
slow for a browser).

Out of scope: archive download. Header-based dynamic overrides aren't
exercised (server/client add their own headers, so the oracle stays
header-independent).
"""

from __future__ import annotations

import asyncio
import contextlib
import pathlib
import socket
from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal, cast

import httpx2
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typing_extensions import override

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.common.incidental import Singular
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    TimeoutType,
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
) -> Response:
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
def test_default_classification_over_every_code(code: int) -> None:
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
def test_override_classification_over_arbitrary_maps(
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


def _get(url: str, timeout: TimeoutType = None) -> QueuedRequest:
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
    """An httpx2 ``TransportError`` becomes a NETWORK ``TransientException``."""
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
    assert isinstance(excinfo.value.__cause__, httpx2.TransportError)


@contextlib.asynccontextmanager
async def _raw_answer_url(answer: bytes) -> AsyncIterator[str]:
    """A URL whose server writes ``answer`` raw, then closes."""

    async def respond(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(answer)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/r"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.parametrize(
    "answer",
    [
        b"HTTP/1.1 200 OK\r\nContent-Le",
        b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabcd",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nabcd",
        b"NOT HTTP AT ALL\r\n\r\n",
    ],
    ids=["mid-headers", "short of length", "short of chunk", "malformed"],
)
@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_an_h11_protocol_error_is_network(
    answer: bytes, path: str
) -> None:
    """Every response h11 refuses is NETWORK: a peer that hung up partway
    and a malformed response raise the same type, told apart only by the
    message, so the transport does not try."""
    async with contextlib.AsyncExitStack() as stack:
        url = await stack.enter_async_context(_raw_answer_url(answer))
        transport = HttpxTransport(scraper=BaseScraper)
        await transport.open()
        stack.push_async_callback(transport.aclose)
        handle = await transport.acquire(0)

        with pytest.raises(TransientException) as excinfo:
            if path == "resolve":
                await transport.resolve(handle, _get(url))
            else:
                stream = await transport.resolve_archive(handle, _get(url))
                try:
                    async for _ in stream:
                        pass
                finally:
                    await transport.finish_archiving(stream)

    assert excinfo.value.kind is TransientKind.NETWORK
    assert isinstance(excinfo.value.__cause__, httpx2.RemoteProtocolError)


async def test_a_request_after_aclose_is_a_network_transient() -> None:
    """A request that reaches the transport after ``aclose`` — a worker
    racing shutdown — is retried like one whose pool closed under it, not
    failed as a transport that was never opened."""
    transport = HttpxTransport(scraper=BaseScraper)
    await transport.open()
    handle = await transport.acquire(0)
    await transport.aclose()

    with pytest.raises(TransientException) as excinfo:
        await transport.resolve(handle, _get("http://test/r"))

    assert excinfo.value.kind is TransientKind.NETWORK


async def test_a_request_before_open_is_a_runtime_error() -> None:
    """A transport never opened is a programming error, not a transient."""
    transport = HttpxTransport(scraper=BaseScraper)
    handle = await transport.acquire(0)

    with pytest.raises(RuntimeError, match="before open"):
        await transport.resolve(handle, _get("http://test/r"))


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
    timeout: TimeoutType, sent: tuple[float, float], reported: float
) -> None:
    """The request's timeout goes to httpx2 as (connect, read); a timeout
    error reports the read value, else the transport's own."""
    seen: list[dict[str, float]] = []

    def time_out(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.extensions["timeout"])
        raise httpx2.ReadTimeout("slow", request=request)

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
    """A URL httpx2 cannot send (bad scheme) is the scraper's bug: persistent,
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
    assert isinstance(excinfo.value.__cause__, httpx2.UnsupportedProtocol)


@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_an_incidental_request_is_a_config_error(path: str) -> None:
    """An ``incidental=`` request needs a transport that captures sub-requests;
    over HTTP it raises rather than fetching its URL as if it were the
    capture."""
    fetched: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        fetched.append(str(request.url))
        return httpx2.Response(200)

    queued = QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="http://test/api"
            ),
            step="parse",
            incidental=Singular(url="*/api*"),
        ),
        request_id=1,
    )
    transport = mocked_httpx_transport(handler)
    try:
        handle = await transport.acquire(0)
        with pytest.raises(ScraperConfigError, match="incidental"):
            if path == "resolve":
                await transport.resolve(handle, queued)
            else:
                await transport.resolve_archive(handle, queued)
    finally:
        await transport.aclose()

    assert fetched == []


@pytest.mark.parametrize(
    ("error", "reported"),
    [
        (httpx2.ConnectTimeout, 1.0),
        (httpx2.ReadTimeout, 4.0),
        (httpx2.PoolTimeout, 4.0),
    ],
    ids=["connect", "read", "pool"],
)
async def test_a_tuple_timeout_reports_the_phase_that_timed_out(
    error: type[httpx2.TimeoutException], reported: float
) -> None:
    """Of a ``(connect, read)`` timeout, a connect timeout reports connect;
    every other phase runs under (and reports) the read value."""

    def time_out(request: httpx2.Request) -> httpx2.Response:
        raise error("slow", request=request)

    transport = mocked_httpx_transport(time_out, timeout=7.0)
    try:
        handle = await transport.acquire(0)
        with pytest.raises(RequestTimeoutException) as excinfo:
            await transport.resolve(handle, _get("http://test/r", (1.0, 4.0)))
    finally:
        await transport.aclose()

    assert excinfo.value.timeout_seconds == reported


async def test_an_invalid_url_is_a_config_error() -> None:
    url = "http://[::1/r"
    transport = HttpxTransport(scraper=BaseScraper)
    await transport.open()
    try:
        handle = await transport.acquire(0)
        with pytest.raises(ScraperConfigError) as excinfo:
            await transport.resolve(handle, _get(url))
    finally:
        await transport.aclose()

    assert isinstance(excinfo.value.__cause__, httpx2.InvalidURL)


async def test_a_header_h11_refuses_to_send_is_a_config_error() -> None:
    """A CR/LF in a header value is our request's fault, not the network's:
    persistent rather than a NETWORK transient retried to budget."""
    async with _hang_up_url() as url:
        transport = HttpxTransport(scraper=BaseScraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            request = Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=url, headers={"X": "a\r\nb"}
                ),
                step="parse",
            )
            with pytest.raises(ScraperConfigError) as excinfo:
                await transport.resolve(
                    handle, QueuedRequest(request=request, request_id=1)
                )
        finally:
            await transport.aclose()

    assert isinstance(excinfo.value.__cause__, httpx2.LocalProtocolError)


class _FollowRedirects(BaseScraper[Any]):
    driver_requirements = [DriverRequirement.FOLLOW_REDIRECTS]


def _redirect_loop(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(302, headers={"Location": str(request.url)})


def _corrupt_gzip(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"Content-Encoding": "gzip"}, content=b"not gzip"
    )


@pytest.mark.parametrize(
    ("handler", "scraper", "cause"),
    [
        (_redirect_loop, _FollowRedirects, httpx2.TooManyRedirects),
        (_corrupt_gzip, BaseScraper, httpx2.DecodingError),
    ],
    ids=["redirect loop", "corrupt gzip"],
)
async def test_a_malformed_response_is_a_site_degraded_transient(
    handler: Any, scraper: type[BaseScraper[Any]], cause: type[Exception]
) -> None:
    """A response the site got wrong is a ``SITE_DEGRADED`` transient, not
    a raw httpx2 error the worker cannot classify."""
    transport = mocked_httpx_transport(handler, scraper)
    try:
        handle = await transport.acquire(0)
        with pytest.raises(TransientException) as excinfo:
            await transport.resolve(handle, _get("http://test/r"))
    finally:
        await transport.aclose()

    assert excinfo.value.kind is TransientKind.SITE_DEGRADED
    assert isinstance(excinfo.value.__cause__, cause)


class _Body(httpx2.AsyncByteStream):
    """A body httpx2 reads lazily, so decoding happens mid-download."""

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"not gzip"


def _corrupt_gzip_stream(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(
        200, headers={"Content-Encoding": "gzip"}, stream=_Body()
    )


async def test_a_corrupt_archive_body_is_a_site_degraded_transient() -> None:
    """Mid-download, a body that will not decode is translated too."""
    transport = mocked_httpx_transport(_corrupt_gzip_stream)
    try:
        handle = await transport.acquire(0)
        stream = await transport.resolve_archive(handle, _get("http://test/r"))
        try:
            with pytest.raises(TransientException) as excinfo:
                async for _ in stream:
                    pass
        finally:
            await transport.finish_archiving(stream)
    finally:
        await transport.aclose()

    assert excinfo.value.kind is TransientKind.SITE_DEGRADED
    assert isinstance(excinfo.value.__cause__, httpx2.DecodingError)


async def test_an_unloadable_verify_path_is_a_config_error(
    tmp_path: pathlib.Path,
) -> None:
    """A CA path that is missing or not a certificate is the scraper's
    mistake, not the network's: a ``ScraperConfigError``, not retried."""
    garbage = tmp_path / "ca.pem"
    garbage.write_text("not a certificate")
    transport = HttpxTransport(scraper=BaseScraper)
    await transport.open()
    try:
        handle = await transport.acquire(0)
        for verify in (str(tmp_path / "missing.pem"), str(garbage)):
            queued = QueuedRequest(
                request=Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/",
                        verify=verify,
                    ),
                    step="parse",
                ),
                request_id=1,
            )
            with pytest.raises(ScraperConfigError, match="verify="):
                await transport.resolve(handle, queued)
    finally:
        await transport.aclose()


@pytest.mark.parametrize(
    ("message", "refused"),
    [
        # httpcore2's HTTP CONNECT failures: "<status> <reason>".
        ("407 Proxy Authentication Required", True),
        ("403 Forbidden", True),
        ("502 Bad Gateway", False),
        ("503 Service Unavailable", False),
        ("504 Gateway Timeout", False),
        # httpcore2's SOCKS5 failures.
        ("Invalid username/password", True),
        ("Requested NO_AUTH_REQUIRED from proxy server, but got X.", True),
        (
            "Proxy Server could not connect: Connection not allowed by ruleset.",
            True,
        ),
        ("Proxy Server could not connect: Host unreachable.", False),
        ("Proxy Server could not connect: Connection refused.", False),
        (
            "Proxy Server could not connect: General SOCKS server failure.",
            False,
        ),
    ],
)
async def test_a_proxy_refusal_is_a_config_error(
    message: str, refused: bool
) -> None:
    """A proxy that refuses us is the run's configuration; one that cannot
    reach the site is a NETWORK transient, retried like any other."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ProxyError(message, request=request)

    transport = mocked_httpx_transport(handler)
    try:
        handle = await transport.acquire(0)
        expected = ScraperConfigError if refused else TransientException
        with pytest.raises(expected) as excinfo:
            await transport.resolve(handle, _get("http://test/r"))
    finally:
        await transport.aclose()

    if not refused:
        assert isinstance(excinfo.value, TransientException)
        assert excinfo.value.kind is TransientKind.NETWORK
