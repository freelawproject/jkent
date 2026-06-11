"""Archive-download fidelity for both transports' ``resolve_archive``.

In the unified split the *worker* owns the archive handler (``should_download``
for dedup, ``save_stream`` to persist) and the *transport* just streams bytes:
``resolve_archive`` returns an ``ArchiveStream``; the worker feeds it to
``handler.save_stream``; ``finish_archiving`` releases the backing (the open
httpx connection for HTTP; nothing for replay, whose file is the source DB's).

This rig mocks a capturing streaming handler and drives that worker-style flow:

- **HttpxTransport** streams a file served by a live aiohttp server; the
  captured bytes must equal what was served, and the stream reports the right
  status/URL.
- **ReplayTransport** streams a file recorded in a run DB (an ``archived_files``
  row pointing at an on-disk file); the captured bytes must equal that file. A
  request with no stored archive raises ``ReplayMiss``.

Scope: the download path (``decision.download is True``). The skip path
(handler returns ``download=False`` for an already-present file) is an
orchestration decision the worker makes by *not* calling ``resolve_archive``,
so it isn't exercised here.
"""

from __future__ import annotations

import asyncio
import shutil
import sqlite3
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path
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
from jkent.driver.replay.source_index import serialize_url_and_body
from jkent.driver.unified_driver import (
    ArchiveStream,
    HttpxTransport,
    QueuedRequest,
    ReplayMiss,
    ReplayTransport,
)


@dataclass
class _CapturingHandler:
    """A mock streaming archive handler that captures the streamed bytes."""

    download: bool = True
    saved: bytes | None = None

    async def should_download(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
    ) -> ArchiveDecision:
        return ArchiveDecision(
            download=self.download,
            file_url="" if self.download else "cached",
        )

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


def _dedup_of(request: Request) -> str | None:
    key = request.deduplication_key
    return key if isinstance(key, str) else None


async def _download(
    transport: Any,
    handle: Any,
    handler: _CapturingHandler,
    request: Request,
) -> ArchiveStream:
    """The worker-style flow: decide, stream, persist, release."""
    dedup = _dedup_of(request)
    decision = await handler.should_download(
        request.request.url, dedup, None, None
    )
    assert decision.download
    queued = QueuedRequest(request=request, request_id=1)
    stream = await transport.resolve_archive(handle, queued, decision)
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
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][0], runner.addresses[0][1]
        base = f"http://{host}:{port}"

        transport = HttpxTransport()
        await transport.open()
        handle = await transport.acquire(0)
        try:
            for i, content in enumerate(files):
                request = Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET, url=f"{base}/a{i}"
                    ),
                    continuation="parse",
                )
                capturing = _CapturingHandler()
                stream = await _download(transport, handle, capturing, request)
                assert capturing.saved == content
                assert stream.status_code == 200
                assert stream.url == request.request.url
        finally:
            await transport.aclose()
            await runner.cleanup()

    asyncio.run(run())


# --- ReplayTransport: stream a stored file -------------------------------


def _materialize_archives(
    template: Path, dest: Path, workdir: Path, files: list[bytes]
) -> list[Request]:
    """Write each file to disk and record an archive row + archived_files row."""
    shutil.copy(template, dest)
    conn = sqlite3.connect(str(dest))
    requests: list[Request] = []
    try:
        for i, content in enumerate(files):
            url = f"https://archive.test/a{i}"
            request = Request(
                request=HTTPRequestParams(method=HttpMethod.GET, url=url),
                continuation="parse",
            )
            requests.append(request)
            assert isinstance(request.deduplication_key, str)
            file_path = workdir / f"file{i}.bin"
            file_path.write_bytes(content)
            ser_url, body = serialize_url_and_body(request.request)
            cur = conn.execute(
                """
                INSERT INTO requests (
                    status, priority, queue_counter, method, url, body,
                    continuation, current_location, deduplication_key,
                    request_type, response_status_code, response_url,
                    completed_at_ns, created_at_ns)
                VALUES ('completed', 9, ?, 'GET', ?, ?, 'parse', '', ?,
                    'archive', 200, ?, ?, ?)
                """,
                (
                    i + 1,
                    ser_url,
                    body,
                    request.deduplication_key,
                    url,
                    i + 1,
                    i + 1,
                ),
            )
            conn.execute(
                "INSERT INTO archived_files (request_id, file_path, original_url) "
                "VALUES (?, ?, ?)",
                (cur.lastrowid, str(file_path), url),
            )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return requests


@pytest.mark.generative
@settings(deadline=None)
@given(files=st.lists(st.binary(max_size=400), min_size=1, max_size=4))
def test_replay_archive_streams_stored_file(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    files: list[bytes],
) -> None:
    workdir = tmp_path_factory.mktemp("archive")
    dest = workdir / "run.db"
    requests = _materialize_archives(schema_template, dest, workdir, files)

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            for content, request in zip(files, requests, strict=True):
                capturing = _CapturingHandler()
                stream = await _download(transport, handle, capturing, request)
                assert capturing.saved == content
                assert stream.status_code == 200
        finally:
            await transport.aclose()

    asyncio.run(run())


async def test_replay_archive_miss(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    dest = tmp_path_factory.mktemp("archive") / "empty.db"
    shutil.copy(schema_template, dest)

    transport = ReplayTransport([dest])
    await transport.open()
    handle = await transport.acquire(0)
    try:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://archive.test/missing"
            ),
            continuation="parse",
        )
        with pytest.raises(ReplayMiss):
            await transport.resolve_archive(
                handle, QueuedRequest(request=request, request_id=1)
            )
    finally:
        await transport.aclose()
