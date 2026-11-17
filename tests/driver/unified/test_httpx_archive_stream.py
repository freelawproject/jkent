"""``HttpxTransport``'s archive stream: single consumption and cleanup."""

from __future__ import annotations

import gzip
import hashlib
import inspect
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest
from typing_extensions import override

from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    TransientException,
    TransientKind,
)
from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.unified_driver import QueuedRequest
from tests.httpx_mock import mocked_httpx_transport


def _archive(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(200, content=b"%PDF" * 1024)


def _queued() -> QueuedRequest:
    return QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="http://test/doc.pdf"
            ),
            step="parse",
        ),
        request_id=1,
    )


async def test_a_second_iteration_raises_up_front() -> None:
    transport = mocked_httpx_transport(_archive)
    try:
        handle = await transport.acquire(0)
        stream = await transport.resolve_archive(handle, _queued())
        try:
            body = b"".join([chunk async for chunk in stream])
            assert body == b"%PDF" * 1024
            with pytest.raises(RuntimeError, match="already consumed"):
                stream.__aiter__()
        finally:
            await transport.finish_archiving(stream)
    finally:
        await transport.aclose()


async def test_finish_archiving_closes_an_abandoned_read() -> None:
    """A body read stopped partway is closed by ``finish_archiving``, not
    left suspended for asyncio's async-generator finalizer."""
    transport = mocked_httpx_transport(_archive)
    try:
        handle = await transport.acquire(0)
        stream = await transport.resolve_archive(handle, _queued())
        iterator = stream.__aiter__()
        await iterator.__anext__()
        await transport.finish_archiving(stream)
        assert inspect.isasyncgen(iterator)
        # A closed async generator has no frame; a suspended one does.
        assert iterator.ag_frame is None
    finally:
        await transport.aclose()


def _missing(_request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(404)


async def test_a_classified_failure_survives_a_failing_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classifier's error is raised, not the error closing the
    rejected stream raised after it."""

    async def failing_aclose(_self: httpx2.Response) -> None:
        raise httpx2.ReadError("reset while closing")

    monkeypatch.setattr(httpx2.Response, "aclose", failing_aclose)
    transport = mocked_httpx_transport(_missing)
    try:
        handle = await transport.acquire(0)
        with pytest.raises(PersistentHTTPResponseException):
            await transport.resolve_archive(handle, _queued())
    finally:
        monkeypatch.undo()
        await transport.aclose()


async def test_the_response_closes_when_closing_the_read_fails() -> None:
    """An error closing the body's iterator still closes the response, and
    is the error raised."""

    class _FailingIterator:
        async def aclose(self) -> None:
            raise RuntimeError("iterator close failed")

    transport = mocked_httpx_transport(_archive)
    try:
        handle = await transport.acquire(0)
        stream = await transport.resolve_archive(handle, _queued())
        stream.__aiter__()
        stream._iterator = _FailingIterator()  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="iterator close failed"):
            await transport.finish_archiving(stream)
        assert stream._response.is_closed  # type: ignore[attr-defined]
    finally:
        await transport.aclose()


_DECODED = b"[" + b'{"opinion": 1},' * 4096 + b"{}]"
_ENCODED = gzip.compress(_DECODED)


class _Lazy(httpx2.AsyncByteStream):
    """``_ENCODED``, read off the wire as a network body is, in chunks."""

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for start in range(0, len(_ENCODED), 64):
            yield _ENCODED[start : start + 64]


def _stored_gzipped(
    etag_for: bytes,
) -> Callable[[httpx2.Request], httpx2.Response]:
    """An S3 object stored gzipped: its ETag is the MD5 of ``etag_for``, and
    its Content-Length counts the encoded bytes."""
    etag = hashlib.md5(etag_for, usedforsecurity=False).hexdigest()

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={
                "Content-Encoding": "gzip",
                "Content-Length": str(len(_ENCODED)),
                "ETag": f'"{etag}"',
            },
            stream=_Lazy(),
        )

    return handler


@pytest.mark.parametrize(
    ("etag_for", "verified"),
    [(_ENCODED, True), (_DECODED, False)],
    ids=["etag of the encoded bytes", "etag of anything else"],
)
async def test_an_encoded_body_is_checked_against_its_encoded_bytes(
    etag_for: bytes, verified: bool
) -> None:
    """The handler gets decoded bytes, so the transport checks the ETag
    against the encoded ones it read them from."""
    transport = mocked_httpx_transport(_stored_gzipped(etag_for))
    try:
        handle = await transport.acquire(0)
        stream = await transport.resolve_archive(handle, _queued())
        try:
            if verified:
                assert b"".join([c async for c in stream]) == _DECODED
            else:
                with pytest.raises(TransientException, match="ETag MD5") as e:
                    async for _ in stream:
                        pass
                assert e.value.kind is TransientKind.ARCHIVE
        finally:
            await transport.finish_archiving(stream)
    finally:
        await transport.aclose()
