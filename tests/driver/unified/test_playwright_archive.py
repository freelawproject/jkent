"""Tests for ``PlaywrightTransport`` archive download (B4).

``resolve_archive`` triggers a browser download via the request's ``via``,
stages it to a temp file, and returns a ``FileArchiveStream`` that
streams the staged file in chunks. ``finish_archiving`` deletes that temp file.

The browser-gated test stands up a local aiohttp server: a parent HTML page
with a download link, plus an endpoint serving a binary body with
``Content-Disposition: attachment`` so the browser downloads it. Reuses B1's
``has_browser`` gate so everything skips cleanly with no browser. The
browser-free tests prove streaming + temp-file deletion + protocol conformance
by constructing the stream over a real temp file directly.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import pytest

from jkent.common.page_element import ViaLink
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
)
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    FileArchiveStream,
    QueuedRequest,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.conftest import (
    serve_archive_download,
    serve_inline_render,
)
from tests.driver.unified.test_playwright_transport import (
    _Scraper,
    _sql_manager,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


# --- Browser-free streaming + lifecycle (always runs) ---------------------


async def _drain(stream: ArchiveStream) -> bytes:
    return b"".join([chunk async for chunk in stream])


async def test_archive_stream_satisfies_protocol() -> None:
    """``FileArchiveStream`` is a structural ``ArchiveStream``."""
    stream = FileArchiveStream(
        status_code=200, headers={}, url="http://x/y", file_path="/dev/null"
    )
    assert isinstance(stream, ArchiveStream)


async def test_archive_stream_streams_the_staged_file() -> None:
    """Iterating the stream yields the staged file's bytes (chunked)."""
    body = b"%PDF-1.7\n" + (b"binary-payload" * 10000)
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(body)
        path = fh.name
    try:
        stream = FileArchiveStream(
            status_code=200,
            headers={},
            url="http://x/y.pdf",
            file_path=path,
            chunk_size=4096,
        )
        assert await _drain(stream) == body
    finally:
        if os.path.exists(path):
            os.unlink(path)


async def test_finish_archiving_deletes_temp_file_idempotently() -> None:
    """``finish_archiving`` deletes the staged file; a second call is a no-op."""
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"data")
        path = fh.name
    stream = FileArchiveStream(
        status_code=200, headers={}, url="http://x/y", file_path=path
    )
    transport = PlaywrightTransport(_Scraper())

    assert os.path.exists(path)
    await transport.finish_archiving(stream)
    assert not os.path.exists(path)
    # Idempotent: deleting an already-gone file is fine.
    await transport.finish_archiving(stream)


async def test_finish_archiving_ignores_foreign_stream() -> None:
    """``finish_archiving`` is a no-op for non-Playwright streams."""

    class _Foreign:
        status_code = 200
        headers: dict[str, str] = {}
        url = "http://x/y"

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def _gen():  # type: ignore[no-untyped-def]
                if False:
                    # Unreachable on purpose: marks _gen as an (empty)
                    # async generator.
                    yield b""  # type: ignore[unreachable]

            return _gen()

    transport = PlaywrightTransport(_Scraper())
    await transport.finish_archiving(_Foreign())  # type: ignore[arg-type]


# --- Browser-gated end-to-end download (skips cleanly w/o a browser) -------


@pytest.fixture
async def archive_transport(  # type: ignore[no-untyped-def]
    has_browser: bool,
    memory_session_factory: async_sessionmaker,
):
    if not has_browser:
        pytest.skip("no launchable browser engine in this environment")
    subject = PlaywrightTransport(
        _Scraper(),
        headless=True,
        # The download path doesn't itself touch the DB when there is no
        # parent to stage, but construct with one for parity.
        db=_sql_manager(memory_session_factory),
    )
    await subject.open()
    try:
        yield subject
    finally:
        await subject.aclose()


async def test_resolve_archive_downloads_and_streams(
    archive_transport: PlaywrightTransport,
) -> None:
    """End-to-end: navigate parent, click link, stream the downloaded bytes."""
    body = b"%PDF-1.7\nfake-pdf-body\n" + bytes(range(256)) * 8
    server = await serve_archive_download(body)
    try:
        handle = await archive_transport.acquire(0)
        # Land the worker page on the parent so the link is clickable. The
        # download request has no parent_request_id (nothing to stage from
        # the DB); we put the worker on the parent page directly.
        await handle.page.goto(
            f"{server.base_url}/parent", wait_until="domcontentloaded"
        )

        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}/file.bin",
                timeout=30,
            ),
            continuation="collect",
            archive=True,
            via=ViaLink(
                selector=Selector.CSS("#dl"),
                description="download link",
            ),
        )
        queued = QueuedRequest(request=request, request_id=1)
        stream = await archive_transport.resolve_archive(handle, queued)

        assert stream.status_code == 200
        assert stream.url
        staged_path = stream.file_path  # type: ignore[attr-defined]
        data = await _drain(stream)
        assert data == body

        # The staged temp file exists until finish_archiving releases it.
        assert os.path.exists(staged_path)
        await archive_transport.finish_archiving(stream)
        assert not os.path.exists(staged_path)
    finally:
        await server.runner.cleanup()


async def test_resolve_archive_captures_inline_render(
    archive_transport: PlaywrightTransport,
) -> None:
    """A click that navigates instead of downloading still yields the file.

    The server renders the linked ``.pdf`` path inline (no download event
    fires — the Firefox pdf.js shape), so the transport must reconstruct the
    archive from the navigation's own response bytes and stage it to a temp
    file exactly like a real download.
    """
    body = b"<html><body>inline-rendered-archive-body</body></html>"
    server = await serve_inline_render(body)
    try:
        handle = await archive_transport.acquire(0)
        await handle.page.goto(
            f"{server.base_url}/parent", wait_until="domcontentloaded"
        )

        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}/doc.pdf",
                timeout=30,
            ),
            continuation="collect",
            archive=True,
            via=ViaLink(
                selector=Selector.CSS("#dl"),
                description="inline-rendered link",
            ),
        )
        queued = QueuedRequest(request=request, request_id=1)
        stream = await archive_transport.resolve_archive(handle, queued)

        assert stream.status_code == 200
        assert stream.url.endswith("/doc.pdf")
        staged_path = stream.file_path  # type: ignore[attr-defined]
        assert await _drain(stream) == body

        # Inline staging mirrors a real download: the temp file lives until
        # finish_archiving unlinks it.
        assert os.path.exists(staged_path)
        await archive_transport.finish_archiving(stream)
        assert not os.path.exists(staged_path)
    finally:
        await server.runner.cleanup()
