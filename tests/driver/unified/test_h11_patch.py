"""Tests for the h11 leniency patch and ``H11_HEADER_FIXES``."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import h11
import h11._events as _h11_events
import h11._headers as _h11_headers
import pytest
from h11._util import LocalProtocolError, RemoteProtocolError

from jkent.common.exceptions import TransientException, TransientKind
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from jkent.driver.unified_driver.transport.httpx_transport import (
    _lenient_te,
    _patched_normalize_and_validate,
    lenient_te,
)


def _dup_te_headers() -> list[tuple[bytes, bytes]]:
    return [
        (b"content-type", b"text/html"),
        (b"transfer-encoding", b"chunked"),
        (b"transfer-encoding", b"chunked"),
    ]


def test_patch_is_installed_globally() -> None:
    assert (
        _h11_headers.normalize_and_validate is _patched_normalize_and_validate
    )
    # Response-parsing path resolves the symbol through _events' module
    # globals; both bindings must be patched or the response path bypasses us.
    assert (
        _h11_events.normalize_and_validate is _patched_normalize_and_validate
    )


def _feed_response_with_dup_te() -> None:
    """Drive h11 through the real response-parse path with duplicate TE.

    Uses the client-side state machine: send a request, then feed back a
    server response that contains two ``Transfer-Encoding: chunked``
    headers. h11 raises ``RemoteProtocolError`` on the bad header unless
    our patch dedupes them.
    """
    conn = h11.Connection(our_role=h11.CLIENT)
    conn.send(
        h11.Request(
            method="GET",
            target="/",
            headers=[("host", "example.com")],
        )
    )
    conn.send(h11.EndOfMessage())
    conn.receive_data(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    while True:
        event = conn.next_event()
        if isinstance(event, h11.Response):
            return
        if event in (h11.NEED_DATA, h11.PAUSED):
            raise AssertionError("h11 did not produce a Response event")


def test_response_parser_strict_by_default() -> None:
    with pytest.raises(
        RemoteProtocolError, match="multiple Transfer-Encoding"
    ):
        _feed_response_with_dup_te()


def test_response_parser_lenient_under_scope() -> None:
    with lenient_te():
        _feed_response_with_dup_te()


def test_lenient_scope_accepts_duplicate_transfer_encoding() -> None:
    with lenient_te():
        result = _h11_headers.normalize_and_validate(_dup_te_headers(), True)
    # Only one TE entry survives after dedupe.
    te_entries = [
        (n, v)
        for n, v in [(item[0], item[1]) for item in result]
        if n == b"transfer-encoding"
    ]
    assert len(te_entries) == 1


@pytest.mark.parametrize(
    "values",
    [[b"chunked, chunked"], [b"chunked", b"Chunked, chunked"]],
    ids=["one line", "across lines"],
)
def test_lenient_scope_accepts_comma_joined_duplicate_transfer_encoding(
    values: list[bytes],
) -> None:
    headers = [(b"transfer-encoding", value) for value in values]
    with lenient_te():
        result = _h11_headers.normalize_and_validate(headers, True)
    assert [(item[0], item[1]) for item in result] == [
        (b"transfer-encoding", b"chunked")
    ]


def test_outbound_request_headers_stay_strict_in_lenient_scope() -> None:
    """Lenient scope must not loosen validation for headers we generate.

    Request smuggling protections rely on h11 rejecting our own
    duplicate Transfer-Encoding emissions.
    """
    with (
        lenient_te(),
        pytest.raises(LocalProtocolError, match="multiple Transfer-Encoding"),
    ):
        _h11_headers.normalize_and_validate(_dup_te_headers(), False)


def test_lenient_scope_passes_distinct_transfer_encodings_through() -> None:
    """Only identical repeats are dropped.

    ``chunked`` + ``gzip`` is real framing, so h11 reports it rather than
    reading the body as plain chunked.
    """
    headers = [
        (b"transfer-encoding", b"chunked"),
        (b"transfer-encoding", b"gzip"),
    ]
    with (
        lenient_te(),
        pytest.raises(LocalProtocolError, match="Transfer-Encoding"),
    ):
        _h11_headers.normalize_and_validate(headers, True)


@contextlib.asynccontextmanager
async def _dup_te_url() -> AsyncIterator[str]:
    """A URL whose server answers with ``Transfer-Encoding: chunked`` twice."""

    async def answer(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"2\r\nok\r\n0\r\n\r\n"
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/r"
    finally:
        server.close()
        await server.wait_closed()


class _LenientScraper(BaseScraper[Any]):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.H11_HEADER_FIXES
    ]


def _get(url: str) -> QueuedRequest:
    return QueuedRequest(
        request=Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=url),
            step="parse",
        ),
        request_id=1,
    )


@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_the_transport_is_lenient_for_a_scraper_that_opts_in(
    path: str,
) -> None:
    """``H11_HEADER_FIXES`` takes effect inside the transport itself: no
    run-level wrapper is entered, and the flag does not outlive the send."""
    async with _dup_te_url() as url:
        transport = HttpxTransport(scraper=_LenientScraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            if path == "resolve":
                response = await transport.resolve(handle, _get(url))
                assert response.content == b"ok"
            else:
                stream = await transport.resolve_archive(handle, _get(url))
                try:
                    assert b"".join([c async for c in stream]) == b"ok"
                finally:
                    await transport.finish_archiving(stream)
            assert _lenient_te.get() is False
        finally:
            await transport.aclose()


async def test_the_transport_stays_strict_for_other_scrapers() -> None:
    async with _dup_te_url() as url:
        transport = HttpxTransport(scraper=BaseScraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            with pytest.raises(TransientException) as excinfo:
                await transport.resolve(handle, _get(url))
        finally:
            await transport.aclose()

    # h11 rejected the response as malformed. That is the site's fault,
    # but h11 raises the same type for a peer that hangs up partway, so
    # the transport classes both as NETWORK.
    assert excinfo.value.kind is TransientKind.NETWORK
