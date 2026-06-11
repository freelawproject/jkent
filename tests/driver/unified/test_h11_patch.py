"""Tests for h11_patch."""

from __future__ import annotations

import asyncio

import h11
import h11._events as _h11_events
import h11._headers as _h11_headers
import pytest
from h11._util import LocalProtocolError, RemoteProtocolError

from jkent.driver.unified_driver.transport.httpx_transport import (
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


def test_strict_by_default_rejects_duplicate_transfer_encoding() -> None:
    with pytest.raises(LocalProtocolError, match="multiple Transfer-Encoding"):
        _h11_headers.normalize_and_validate(_dup_te_headers(), True)


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


def test_scope_resets_on_exit() -> None:
    with lenient_te():
        pass
    with pytest.raises(LocalProtocolError, match="multiple Transfer-Encoding"):
        _h11_headers.normalize_and_validate(_dup_te_headers(), True)


def test_scope_does_not_leak_between_concurrent_tasks() -> None:
    """ContextVar must isolate the lenient scope per asyncio Task."""

    results: dict[str, bool] = {}

    async def lenient_task() -> None:
        with lenient_te():
            await asyncio.sleep(0.01)
            try:
                _h11_headers.normalize_and_validate(_dup_te_headers(), True)
                results["lenient"] = True
            except LocalProtocolError:
                results["lenient"] = False

    async def strict_task() -> None:
        await asyncio.sleep(0)
        try:
            _h11_headers.normalize_and_validate(_dup_te_headers(), True)
            results["strict"] = True
        except LocalProtocolError:
            results["strict"] = False

    async def main() -> None:
        await asyncio.gather(lenient_task(), strict_task())

    asyncio.run(main())
    assert results == {"lenient": True, "strict": False}
