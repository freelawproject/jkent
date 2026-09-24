"""Tests for ``PlaywrightTransport`` archive download.

``resolve_archive`` triggers a browser download via the request's ``via``,
stages it to a temp file, and returns a ``FileArchiveStream`` that
streams the staged file in chunks. ``finish_archiving`` deletes that temp file.

The browser-gated test stands up a local aiohttp server: a parent HTML page
with a download link, plus an endpoint serving a binary body with
``Content-Disposition: attachment`` so the browser downloads it. The
``require_browser`` gate skips it cleanly with no browser. The
browser-free tests prove streaming + temp-file deletion + protocol conformance
by constructing the stream over a real temp file directly.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from typing_extensions import override

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    TransientException,
    TransientKind,
)
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.data_types import (
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
)
from jkent.driver.browser_engine import worker_page
from jkent.driver.unified_driver import transport as transport_module
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
    serve_pdf,
)
from tests.driver.unified.test_playwright_transport import (
    _Scraper,
    _sql_manager,
)
from tests.servers import start_app

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


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


async def test_archive_stream_opens_its_file_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Like the reads, the ``open()`` runs in a worker thread."""
    path = tmp_path / "body.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    loop_thread = threading.current_thread()
    opened_on: list[threading.Thread] = []

    def spy_open(*args: Any, **kwargs: Any) -> Any:
        opened_on.append(threading.current_thread())
        return open(*args, **kwargs)

    monkeypatch.setattr(transport_module, "open", spy_open, raising=False)
    stream = FileArchiveStream(
        status_code=200, headers={}, url="http://x/y.pdf", file_path=str(path)
    )
    assert await _drain(stream) == b"%PDF-1.7\n"
    assert opened_on and opened_on[0] is not loop_thread


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


def _open_fds() -> int:
    return len(os.listdir("/dev/fd"))


async def test_finishing_a_partly_read_stream_closes_its_file() -> None:
    """A download abandoned mid-body releases its fd at finish, not at GC."""
    with tempfile.NamedTemporaryFile(delete=False) as fh:
        fh.write(b"x" * 20000)
        path = fh.name
    stream = FileArchiveStream(
        status_code=200,
        headers={},
        url="http://x/y",
        file_path=path,
        chunk_size=4096,
    )
    before = _open_fds()
    chunks = aiter(stream)
    await anext(chunks)  # the consumer stops after one chunk
    assert _open_fds() == before + 1

    await PlaywrightTransport(_Scraper()).finish_archiving(stream)

    assert _open_fds() == before
    assert not os.path.exists(path)


async def test_finish_archiving_ignores_foreign_stream() -> None:
    """A stream Playwright did not stage is closed, and nothing deleted."""

    class _Foreign(ArchiveStream):
        closed = False

        def __aiter__(self) -> AsyncIterator[bytes]:
            async def _gen() -> AsyncIterator[bytes]:
                if False:
                    # Unreachable on purpose: marks _gen as an (empty)
                    # async generator.
                    yield b""  # type: ignore[unreachable]

            return _gen()

        @override
        async def aclose(self) -> None:
            self.closed = True

    stream = _Foreign(status_code=200, headers={}, url="http://x/y")
    await PlaywrightTransport(_Scraper()).finish_archiving(stream)
    assert stream.closed


# --- Browser-gated end-to-end download (skips cleanly w/o a browser) -------


@pytest.fixture
async def archive_transport(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
):
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
            step="collect",
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


_PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << >>\n%%EOF\n"


async def fetch_via_less_pdf(transport: PlaywrightTransport) -> None:
    """An archive request with no ``via`` downloads its URL directly."""
    server = await serve_pdf(_PDF)
    try:
        handle = await transport.acquire(0)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}/doc.pdf",
                timeout=30,
            ),
            step="collect",
            archive=True,
            expected_type="pdf",
        )
        stream = await transport.resolve_archive(
            handle, QueuedRequest(request=request, request_id=1)
        )
        try:
            assert stream.status_code == 200
            assert await _drain(stream) == _PDF
        finally:
            await transport.finish_archiving(stream)
    finally:
        await server.runner.cleanup()


async def test_resolve_archive_without_a_via_fetches_the_url(
    archive_transport: PlaywrightTransport,
) -> None:
    await fetch_via_less_pdf(archive_transport)


def _postback_app() -> web.Application:
    """An ASP.NET-style page whose download is a ``__doPostBack`` link.

    The form has no submit button; the server answers the POST with the file
    only when ``__EVENTTARGET`` names the download link.
    """

    async def parent(_request: web.Request) -> web.Response:
        return web.Response(
            text=(
                "<html><body><form id='f' method='post' action='/postback'>"
                "<input type='hidden' name='__EVENTTARGET' value=''>"
                "<input type='hidden' name='__VIEWSTATE' value='vs'>"
                "<a href=\"javascript:__doPostBack('lnkPdf','')\">pdf</a>"
                "</form></body></html>"
            ),
            content_type="text/html",
        )

    async def postback(request: web.Request) -> web.Response:
        form = await request.post()
        if form.get("__EVENTTARGET") != "lnkPdf":
            return web.Response(
                text="<html>same page</html>", content_type="text/html"
            )
        return web.Response(
            body=_PDF,
            headers={
                "Content-Type": "application/pdf",
                "Content-Disposition": 'attachment; filename="doc.pdf"',
            },
        )

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_post("/postback", postback)
    return app


async def test_resolve_archive_follows_an_eventtarget_postback(
    archive_transport: PlaywrightTransport,
) -> None:
    server = await start_app(_postback_app())
    try:
        handle = await archive_transport.acquire(0)
        await handle.page.goto(
            f"{server.base_url}/parent", wait_until="domcontentloaded"
        )
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url=f"{server.base_url}/postback",
                timeout=30,
            ),
            step="collect",
            archive=True,
            via=ViaFormSubmit(
                form_selector=Selector.CSS("#f"),
                submit_selector=None,
                field_data={"__EVENTTARGET": "lnkPdf", "__VIEWSTATE": "vs"},
                description="postback download",
            ),
        )
        stream = await archive_transport.resolve_archive(
            handle, QueuedRequest(request=request, request_id=1)
        )
        try:
            assert await _drain(stream) == _PDF
        finally:
            await archive_transport.finish_archiving(stream)
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
            step="collect",
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


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/opinion.txt", "text/plain"),
        ("/docket?id=7", "application/json"),
    ],
    ids=["text", "json-extensionless"],
)
async def test_resolve_archive_captures_any_non_html_inline_render(
    archive_transport: PlaywrightTransport, path: str, content_type: str
) -> None:
    """Whatever file the page navigated to is the archive, by its response.

    Not only a ``.pdf``/``.doc``/``.docx`` path or a PDF content type: a
    browser renders text and JSON inline too, at any URL.
    """
    body = b'{"opinion": "inline-rendered"}'

    async def parent(_request: web.Request) -> web.Response:
        return web.Response(
            text=f"<html><body><a id='dl' href='{path}'>view</a></body></html>",
            content_type="text/html",
        )

    async def file_(_request: web.Request) -> web.Response:
        return web.Response(body=body, headers={"Content-Type": content_type})

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_get(path.split("?", 1)[0], file_)
    server = await start_app(app)
    try:
        handle = await archive_transport.acquire(0)
        await handle.page.goto(
            f"{server.base_url}/parent", wait_until="domcontentloaded"
        )
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server.base_url}{path}",
                timeout=30,
            ),
            step="collect",
            archive=True,
            via=ViaLink(selector=Selector.CSS("#dl"), description="link"),
        )
        stream = await archive_transport.resolve_archive(
            handle, QueuedRequest(request=request, request_id=1)
        )
        try:
            assert stream.url.endswith(path)
            assert await _drain(stream) == body
        finally:
            await archive_transport.finish_archiving(stream)
    finally:
        await server.aclose()


# --- Archive rows record what the server actually sent ---------------------
#
# The httpx archive path streams the response's real status + headers and
# runs them through the scraper's classifier before the first byte is saved.
# A browser download used to hard-code ``status_code=200, headers={}`` and
# skip classification, so a run db recorded every download — including one
# the scraper would have retried or failed — as a clean 200 with no headers.

_TAG = "X-Archive-Tag"


def _tagged_app(*, status: int, inline: bool, tag: str) -> web.Application:
    """``/parent`` links to ``/file.pdf``, served with ``X-Archive-Tag``.

    ``inline=False`` serves an attachment (a real download event);
    ``inline=True`` serves text/html so the browser navigates instead — the
    inline-render branch.
    """

    async def parent(_request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body><a id='dl' href='/file.pdf'>dl</a></body></html>",
            content_type="text/html",
        )

    async def file_(_request: web.Request) -> web.Response:
        headers = {_TAG: tag}
        if inline:
            headers["Content-Type"] = "text/html"
        else:
            headers["Content-Type"] = "application/pdf"
            headers["Content-Disposition"] = 'attachment; filename="file.pdf"'
        return web.Response(
            status=status, body=b"%PDF-1.7 fake", headers=headers
        )

    app = web.Application()
    app.router.add_get("/parent", parent)
    app.router.add_get("/file.pdf", file_)
    return app


class _TagClassifyingScraper(_Scraper):
    """Retries any archive whose response is tagged ``retry``."""

    @classmethod
    @override
    def classify(
        cls,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> HTTPCodeType:
        tag = {k.lower(): v for k, v in (headers or {}).items()}.get(
            _TAG.lower()
        )
        if tag == "retry":
            return HTTPCodeType.TRANSIENT
        return super().classify(status_code, headers, content)


async def _trigger(
    transport: PlaywrightTransport, base_url: str
) -> ArchiveStream:
    handle = await transport.acquire(0)
    await handle.page.goto(f"{base_url}/parent", wait_until="domcontentloaded")
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url=f"{base_url}/file.pdf", timeout=30
        ),
        step="collect",
        archive=True,
        via=ViaLink(selector=Selector.CSS("#dl"), description="link"),
    )
    return await transport.resolve_archive(
        handle, QueuedRequest(request=request, request_id=1)
    )


def _lower(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


@pytest.fixture
def staged_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every file a ``FileArchiveStream`` was built over during the test.

    A raising verdict never hands the stream back, so this is the only way
    to find the download (or inline-render) file it staged.
    """
    paths: list[str] = []
    init = FileArchiveStream.__init__

    def recording_init(self: FileArchiveStream, **kwargs: Any) -> None:
        init(self, **kwargs)
        paths.append(self.file_path)

    monkeypatch.setattr(FileArchiveStream, "__init__", recording_init)
    return paths


@pytest.mark.parametrize("inline", [False, True], ids=["download", "inline"])
async def test_archive_stream_carries_the_responses_headers(
    archive_transport: PlaywrightTransport, inline: bool
) -> None:
    server = await start_app(
        _tagged_app(status=200, inline=inline, tag="tagged")
    )
    try:
        stream = await _trigger(archive_transport, server.base_url)
        try:
            assert stream.status_code == 200
            assert _lower(stream.headers)[_TAG.lower()] == "tagged"
        finally:
            await archive_transport.finish_archiving(stream)
    finally:
        await server.aclose()


@pytest.mark.parametrize("inline", [False, True], ids=["download", "inline"])
async def test_archive_file_is_not_captured_as_an_incidental(
    archive_transport: PlaywrightTransport,
    monkeypatch: pytest.MonkeyPatch,
    inline: bool,
) -> None:
    """The archive path records no incidentals, so it reads no file body.

    Nothing persists them there, so a capture only reads and compresses the
    file a second time and leaves it on the handle until the next request.
    """
    compressed: list[bytes] = []
    real_compress = worker_page.compress

    def recording_compress(data: bytes, *args: Any, **kwargs: Any) -> bytes:
        compressed.append(data)
        return real_compress(data, *args, **kwargs)

    monkeypatch.setattr(worker_page, "compress", recording_compress)
    server = await start_app(
        _tagged_app(status=200, inline=inline, tag="tagged")
    )
    try:
        stream = await _trigger(archive_transport, server.base_url)
        await archive_transport.finish_archiving(stream)
        # The handle as the archive left it — a fresh acquire would reset it.
        handle = archive_transport._handles[0]
        await handle.drain_captures()
        assert handle.incidental_requests == []
        assert b"%PDF-1.7 fake" not in compressed
    finally:
        await server.aclose()


async def test_download_is_classified_through_the_scraper(
    require_browser: None,
    memory_session_factory: async_sessionmaker[AsyncSession],
    staged_files: list[str],
) -> None:
    """A header-based classifier sees the download's headers, like httpx.

    The raising verdict releases the staged download.
    """
    transport = PlaywrightTransport(
        _TagClassifyingScraper(),
        headless=True,
        db=_sql_manager(memory_session_factory),
    )
    await transport.open()
    server = await start_app(
        _tagged_app(status=200, inline=False, tag="retry")
    )
    try:
        with pytest.raises(HTTPResponseAssumptionException) as excinfo:
            await _trigger(transport, server.base_url)
        assert excinfo.value.status_code == 200
        [staged] = staged_files
        assert not os.path.exists(staged)
    finally:
        await server.aclose()
        await transport.aclose()


async def test_inline_status_is_classified_through_the_scraper(
    archive_transport: PlaywrightTransport, staged_files: list[str]
) -> None:
    """An inline render's real status goes through the default code map.

    The raising verdict deletes the ``jkent-inline-archive-*`` temp file.
    """
    server = await start_app(_tagged_app(status=503, inline=True, tag="x"))
    try:
        with pytest.raises(HTTPResponseAssumptionException) as excinfo:
            await _trigger(archive_transport, server.base_url)
        assert excinfo.value.status_code == 503
        [staged] = staged_files
        assert os.path.basename(staged).startswith("jkent-inline-archive-")
        assert not os.path.exists(staged)
    finally:
        await server.aclose()


async def test_stalled_download_times_out_as_archive_transient(
    archive_transport: PlaywrightTransport,
) -> None:
    """A download that stalls past the request's timeout is a transient."""
    release = asyncio.Event()

    async def stalled(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": 'attachment; filename="slow.bin"',
                "Content-Length": "1000000",
            }
        )
        await response.prepare(request)
        await response.write(b"x" * 1024)
        await release.wait()
        return response

    app = web.Application()
    app.router.add_get("/slow.bin", stalled)
    server = await start_app(app)
    base = server.base_url
    handle = await archive_transport.acquire(0)
    request = Request(
        request=HTTPRequestParams(
            # Long enough that the download event lands inside it even on
            # a loaded machine; the body never completes, so only the
            # download wait can use it up.
            method=HttpMethod.GET,
            url=f"{base}/slow.bin",
            timeout=5,
        ),
        step="collect",
        archive=True,
    )
    try:
        with pytest.raises(
            TransientException, match="exceeded timeout"
        ) as excinfo:
            await asyncio.wait_for(
                archive_transport.resolve_archive(
                    handle, QueuedRequest(request=request, request_id=1)
                ),
                timeout=120,
            )
        assert excinfo.value.kind is TransientKind.ARCHIVE
    finally:
        release.set()
        await server.aclose()
