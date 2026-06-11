"""Tests for ``PlaywrightTransport`` archive download (B4).

``resolve_archive`` triggers a browser download via the request's ``via``,
stages it to a temp file, and returns a ``_PlaywrightArchiveStream`` that
streams the staged file in chunks. ``finish_archiving`` deletes that temp file.

The browser-gated test stands up a local aiohttp server: a parent HTML page
with a download link, plus an endpoint serving a binary body with
``Content-Disposition: attachment`` so the browser downloads it. Reuses B1's
``has_browser`` gate so everything skips cleanly with no browser. The
browser-free tests prove streaming + temp-file deletion + protocol conformance
by constructing the stream over a real temp file directly.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from jkent.common.page_element import ViaLink
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.transport import ArchiveStream, QueuedRequest
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
    _PlaywrightArchiveStream,
)
from tests.driver.unified.test_playwright_transport import (
    _Scraper,
    _start_server,
)

if TYPE_CHECKING:
    from jkent.driver.database_engine.scoped_session import (
        ScopedSessionFactory,
    )


async def _browser_launches() -> bool:
    """Probe whether a real browser context can be brought up + torn down."""
    transport = PlaywrightTransport(_Scraper(), headless=True)
    try:
        await transport.open()
    except Exception:
        try:
            await transport.aclose()
        except Exception:
            pass
        return False
    await transport.aclose()
    return True


@pytest.fixture(scope="session")
def has_browser() -> bool:
    return asyncio.run(_browser_launches())


def _sql_manager(sf: ScopedSessionFactory) -> SQLManager:
    engine = sf._factory.kw["bind"]
    return SQLManager(engine, sf)


# --- Browser-free streaming + lifecycle (always runs) ---------------------


async def _drain(stream: ArchiveStream) -> bytes:
    return b"".join([chunk async for chunk in stream])


async def test_archive_stream_satisfies_protocol() -> None:
    """``_PlaywrightArchiveStream`` is a structural ``ArchiveStream``."""
    stream = _PlaywrightArchiveStream(
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
        stream = _PlaywrightArchiveStream(
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
    stream = _PlaywrightArchiveStream(
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


@dataclass
class _ArchiveServer:
    base_url: str
    body: bytes
    runner: web.AppRunner


async def _start_archive_server() -> _ArchiveServer:
    """A parent page with a download link + an attachment endpoint."""
    body = b"%PDF-1.7\nfake-pdf-body\n" + bytes(range(256)) * 8

    async def parent(_req: web.Request) -> web.Response:
        html = (
            "<html><body>"
            "<a id='dl' href='/file.bin'>download</a>"
            "</body></html>"
        )
        return web.Response(status=200, body=html, content_type="text/html")

    async def file_(_req: web.Request) -> web.Response:
        return web.Response(
            status=200,
            body=body,
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="file.bin"',
            },
        )

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_get("/file.bin", file_)
    server = await _start_server(app)
    return _ArchiveServer(
        base_url=server.base_url, body=body, runner=server.runner
    )


@pytest.fixture
async def archive_transport(  # type: ignore[no-untyped-def]
    has_browser: bool,
    memory_session_factory: ScopedSessionFactory,
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
    server = await _start_archive_server()
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
        assert data == server.body

        # The staged temp file exists until finish_archiving releases it.
        assert os.path.exists(staged_path)
        await archive_transport.finish_archiving(stream)
        assert not os.path.exists(staged_path)
    finally:
        await server.runner.cleanup()
