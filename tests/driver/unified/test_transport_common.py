"""Tests for the concrete shared bases in ``transport.common``.

``StatelessTransport``, ``NoopHandle`` and ``FileArchiveStream`` are the
pieces HTTP, replay and Playwright reuse rather than repeat. They are
abstract enough that their real coverage comes from the concrete
transports, so this file does two things:

- Binds a minimal ``StatelessTransport`` subclass — canned responses,
  archives served as a ``FileArchiveStream`` over a real file — into
  ``TransportConformance``. That runs the inherited ``acquire``/``release``
  cache, ``NoopHandle``, ``FileArchiveStream`` and the default
  ``finish_archiving`` through the same contract every transport must meet.
- Pins what the suite can't see: ``NoopHandle`` is pageless, the handle
  cache is per worker, and ``FileArchiveStream`` chunks, closes its file
  when abandoned, and does its file I/O off the event loop.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typing_extensions import override

from jkent.data_types import Response
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    FileArchiveStream,
    NoopHandle,
    QueuedRequest,
    StatelessTransport,
)
from jkent.driver.unified_driver.transport import common as transport_module
from tests.driver.unified.test_transport_conformance import (
    TransportConformance,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jkent.driver.unified_driver.transport import AwaitCondition


class _FileBackedStatelessTransport(StatelessTransport):
    """Only what ``StatelessTransport`` leaves abstract, served from disk."""

    def __init__(self, archive: Path) -> None:
        super().__init__()
        self._archive = archive

    async def open(self) -> None:
        return None

    async def aclose(self) -> None:
        return None

    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        return Response(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"<html></html>",
            text="<html></html>",
            url=queued.request.request.url,
            request=queued.request,
        )

    async def resolve_archive(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
    ) -> FileArchiveStream:
        return FileArchiveStream(
            status_code=200,
            headers={"content-type": "application/pdf"},
            url=queued.request.request.url,
            file_path=str(self._archive),
            chunk_size=4,
        )


class TestStatelessTransportConformance(
    TransportConformance[_FileBackedStatelessTransport]
):
    """The shared bases, run through the transport contract."""

    @pytest.fixture
    @override
    def subject(self, tmp_path: Path) -> _FileBackedStatelessTransport:
        archive = tmp_path / "archive.pdf"
        archive.write_bytes(b"%PDF-1.7\nbody spanning several chunks")
        return _FileBackedStatelessTransport(archive)


# --- StatelessTransport / NoopHandle ----------------------------------------


async def test_noop_handle_is_pageless() -> None:
    """The worker passes ``handle.page`` to steps; pageless means ``None``."""
    assert NoopHandle().page is None


async def test_stateless_handles_are_per_worker(tmp_path: Path) -> None:
    """Each worker gets its own handle; releasing one leaves the others."""
    transport = _FileBackedStatelessTransport(tmp_path / "unused")
    first = await transport.acquire(1)
    second = await transport.acquire(2)
    assert first is not second
    await transport.release(1)
    assert await transport.acquire(2) is second


async def test_stateless_release_of_unknown_worker_is_a_noop(
    tmp_path: Path,
) -> None:
    """A worker that never acquired (or already released) can release."""
    transport = _FileBackedStatelessTransport(tmp_path / "unused")
    await transport.release(42)
    await transport.acquire(1)
    await transport.release(1)
    await transport.release(1)


# --- FileArchiveStream -------------------------------------------------------


async def _drain(stream: ArchiveStream) -> list[bytes]:
    return [chunk async for chunk in stream]


def _stream(path: Path, *, chunk_size: int = 65536) -> FileArchiveStream:
    return FileArchiveStream(
        status_code=200,
        headers={},
        url="http://x/y.pdf",
        file_path=str(path),
        chunk_size=chunk_size,
    )


async def test_file_stream_yields_the_file_in_bounded_chunks(
    tmp_path: Path,
) -> None:
    """The chunks reassemble to the file, none larger than ``chunk_size``."""
    body = b"%PDF-1.7\n" + b"binary-payload" * 1000
    path = tmp_path / "body.pdf"
    path.write_bytes(body)
    chunks = await _drain(_stream(path, chunk_size=4096))
    assert b"".join(chunks) == body
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 4096 for chunk in chunks)


async def test_file_stream_of_an_empty_file_yields_nothing(
    tmp_path: Path,
) -> None:
    """An empty body is zero chunks, not one empty chunk."""
    path = tmp_path / "empty"
    path.write_bytes(b"")
    assert await _drain(_stream(path)) == []


async def test_file_stream_aclose_before_iterating_is_a_noop(
    tmp_path: Path,
) -> None:
    """``aclose`` on an untouched stream is safe, and idempotent."""
    stream = _stream(tmp_path / "never-opened")
    await stream.aclose()
    await stream.aclose()


def _open_fds() -> int:
    return len(list(Path("/dev/fd").iterdir()))


async def test_file_stream_aclose_releases_a_partly_read_file(
    tmp_path: Path,
) -> None:
    """A body abandoned mid-read releases its fd at ``aclose``, not at GC."""
    path = tmp_path / "body"
    path.write_bytes(b"x" * 20000)
    stream = _stream(path, chunk_size=4096)
    before = _open_fds()
    chunks = aiter(stream)
    await anext(chunks)
    assert _open_fds() == before + 1

    await stream.aclose()

    assert _open_fds() == before


async def test_file_stream_missing_file_raises_on_read_not_on_close(
    tmp_path: Path,
) -> None:
    """A vanished file surfaces when read; the cleanup ``aclose`` still runs."""
    stream = _stream(tmp_path / "gone")
    with pytest.raises(FileNotFoundError):
        await _drain(stream)
    await stream.aclose()


async def test_file_stream_does_its_file_io_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``open()`` runs in a worker thread, like the reads."""
    path = tmp_path / "body.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    loop_thread = threading.current_thread()
    opened_on: list[threading.Thread] = []

    def spy_open(*args: Any, **kwargs: Any) -> Any:
        opened_on.append(threading.current_thread())
        # Spies on the builtin the stream calls; Path.open would not be
        # the same callable.
        return open(*args, **kwargs)  # noqa: PTH123

    monkeypatch.setattr(transport_module, "open", spy_open, raising=False)
    assert b"".join(await _drain(_stream(path))) == b"%PDF-1.7\n"
    assert opened_on and opened_on[0] is not loop_thread
