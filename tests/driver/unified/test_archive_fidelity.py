"""Archive-download fidelity for ``HttpxTransport.resolve_archive``.

The *worker* owns the archive handler (``should_download``
for dedup, ``save_stream`` to persist) and the *transport* just streams bytes:
``resolve_archive`` returns an ``ArchiveStream``; the worker feeds it to
``handler.save_stream``; ``finish_archiving`` releases the backing (the open
httpx connection for HTTP).

This rig mocks a capturing streaming handler and drives that worker-style flow:

- **HttpxTransport** streams a file served by a live aiohttp server; the
  captured bytes must equal what was served, and the stream reports the right
  status/URL.

Scope: the download path (``decision.download is True``). The skip path
(handler returns ``download=False`` for an already-present file) is an
orchestration decision the worker makes by *not* calling ``resolve_archive``;
``test_worker.py`` covers it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import web
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.data_types import (
    ArchiveDecision,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import (
    ArchiveStream,
    HttpxTransport,
    QueuedRequest,
)
from tests.servers import (
    start_app,
)


@dataclass
class _CapturingHandler:
    """A mock streaming archive handler that captures the streamed bytes."""

    saved: bytes | None = None

    async def should_download(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
    ) -> ArchiveDecision:
        return ArchiveDecision(download=True, file_url="")

    async def save_stream(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
        chunks: AsyncIterable[bytes],
    ) -> str:
        buffer = bytearray()
        async for chunk in chunks:
            buffer.extend(chunk)
        self.saved = bytes(buffer)
        return f"saved://{deduplication_key}"


async def _download(
    transport: Any,
    handle: Any,
    handler: _CapturingHandler,
    request: Request,
) -> ArchiveStream:
    """The worker-style flow: decide, stream, persist, release."""
    dedup = request.effective_deduplication_key
    decision = await handler.should_download(
        request.request.url, dedup, None, None
    )
    assert decision.download
    queued = QueuedRequest(request=request, request_id=1)
    stream = await transport.resolve_archive(handle, queued)
    try:
        await handler.save_stream(
            request.request.url, dedup, None, None, stream
        )
    finally:
        await transport.finish_archiving(stream)
    return stream


# --- HttpxTransport: stream a served file --------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(files=st.lists(st.binary(max_size=400), min_size=1, max_size=4))
def test_httpx_archive_streams_served_file(files: list[bytes]) -> None:
    async def run() -> None:
        app = web.Application()

        async def handler(request: web.Request) -> web.Response:
            return web.Response(
                status=200, body=files[int(request.match_info["idx"])]
            )

        app.router.add_route("GET", "/a{idx}", handler)
        transport = HttpxTransport()
        # start_app is before the try so a setup failure here can't leak a
        # half-started server; everything that can raise after it (open,
        # acquire, the loop) runs inside the try whose finally tears both down.
        server = await start_app(app)
        try:
            base = server.base_url
            await transport.open()
            handle = await transport.acquire(0)
            for i, content in enumerate(files):
                request = Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET, url=f"{base}/a{i}"
                    ),
                    step="parse",
                )
                capturing = _CapturingHandler()
                stream = await _download(transport, handle, capturing, request)
                assert capturing.saved == content
                assert stream.status_code == 200
                assert stream.url == request.request.url
        finally:
            await transport.aclose()
            await server.runner.cleanup()

    asyncio.run(run())
