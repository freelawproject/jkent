"""Tests for the default streaming archive handler's filename scheme.

:class:`LocalAsyncStreamingArchiveHandler` writes files to
``{storage_dir}/{xx}/{yy}/{deduplication_key}/{sha256}.{expected_type}`` by
default, where ``xx`` and ``yy`` are the first two pairs of hex digits of the
SHA-256 of the deduplication key. These tests lock in that layout and confirm
the hashing + atomic-rename plumbing, plus the skip/cleanup of partial temp
files left by an interrupted download.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jkent.common.exceptions import ScraperConfigError
from jkent.driver.archive_handler import (
    STALE_STREAM_TMP_AGE_S,
    LocalAsyncStreamingArchiveHandler,
    NoArchiveHandler,
)


def _iter_chunks(data: bytes, chunk_size: int = 4) -> list[bytes]:
    """Split ``data`` into a list of byte chunks for streaming."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


def _dedup_path(storage_dir: Path, key: str) -> Path:
    """Mirror of the handler's nested dedup-dir layout."""
    sha = hashlib.sha256(key.encode()).hexdigest()
    return storage_dir / sha[:2] / sha[2:4] / key


class TestAsyncStreamingLayout:
    async def test_dedup_key_and_expected_type(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"async streamed bytes"
        sha = hashlib.sha256(payload).hexdigest()

        path = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key="case-99",
            expected_type="audio",
            hash_header_value=None,
            chunks=_aiter(_iter_chunks(payload)),
        )

        expected = _dedup_path(tmp_path, "case-99") / f"{sha}.audio"
        assert Path(path) == expected
        assert expected.read_bytes() == payload

    async def test_no_dedup_key(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"flat file"
        sha = hashlib.sha256(payload).hexdigest()

        path = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="pdf",
            hash_header_value=None,
            chunks=_aiter([payload]),
        )

        assert Path(path) == tmp_path / f"{sha}.pdf"

    async def test_no_expected_type(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"plain content"
        sha = hashlib.sha256(payload).hexdigest()

        path = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key="dedup",
            expected_type=None,
            hash_header_value=None,
            chunks=_aiter([payload]),
        )

        dedup_dir = _dedup_path(tmp_path, "dedup")
        assert Path(path) == dedup_dir / sha
        # No leftover ``.tmp`` files.
        assert [p.name for p in dedup_dir.iterdir()] == [sha]

    async def test_sha_computed_across_chunks(self, tmp_path: Path) -> None:
        """The SHA is built incrementally from every chunk, not just one."""
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        chunks = [b"alpha-", b"beta-", b"gamma"]
        payload = b"".join(chunks)
        sha = hashlib.sha256(payload).hexdigest()

        path = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="txt",
            hash_header_value=None,
            chunks=_aiter(chunks),
        )
        assert Path(path).name == f"{sha}.txt"
        assert Path(path).read_bytes() == payload

    async def test_tempfile_cleaned_up_on_stream_error(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        async def _boom() -> AsyncIterator[bytes]:
            yield b"first"
            raise RuntimeError("stream exploded")

        with pytest.raises(RuntimeError, match="stream exploded"):
            await handler.save_stream(
                url="u",
                deduplication_key="k",
                expected_type="pdf",
                hash_header_value=None,
                chunks=_boom(),
            )

        # save_stream creates the dedup dir before streaming; neither a final
        # file nor a stray .tmp file may be left in it.
        assert list(_dedup_path(tmp_path, "k").iterdir()) == []


class TestPartialTempFileHandling:
    """A partial ``.stream-*.tmp`` left by an interrupted download must never
    be served as a finished archive, and is swept on the next download."""

    async def test_should_download_ignores_partial_tmp(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_path(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        # Simulate a download killed mid-write: a leftover temp file.
        (dedup_dir / ".stream-abc.tmp").write_bytes(b"partial bytes")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            hash_header_value=None,
        )

        # Must not treat the partial temp file as an existing archive.
        assert decision.download is True

    async def test_should_download_ignores_empty_file(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_path(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        (dedup_dir / "zero.pdf").write_bytes(b"")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            hash_header_value=None,
        )
        assert decision.download is True

    async def test_save_stream_sweeps_stale_tmp(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_path(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        stale = dedup_dir / ".stream-stale.tmp"
        stale.write_bytes(b"leftover")
        an_hour_and_more_ago = time.time() - STALE_STREAM_TMP_AGE_S - 1
        os.utime(stale, (an_hour_and_more_ago, an_hour_and_more_ago))

        payload = b"fresh content"
        sha = hashlib.sha256(payload).hexdigest()
        path = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            hash_header_value=None,
            chunks=_aiter([payload]),
        )

        # The stale temp file is gone and only the finished file remains.
        assert not stale.exists()
        assert Path(path) == dedup_dir / f"{sha}.pdf"
        assert [p.name for p in dedup_dir.iterdir()] == [f"{sha}.pdf"]


async def test_a_sibling_download_in_flight_survives(tmp_path: Path) -> None:
    """A second download into the same directory leaves the first's temp."""
    handler = LocalAsyncStreamingArchiveHandler(tmp_path)
    first_started = asyncio.Event()
    second_done = asyncio.Event()

    async def slow_chunks() -> AsyncIterator[bytes]:
        yield b"first half, "
        first_started.set()
        await second_done.wait()
        yield b"second half"

    async def second() -> str:
        await first_started.wait()
        try:
            return await handler.save_stream(
                url="https://example.com/b",
                deduplication_key=None,
                expected_type="pdf",
                hash_header_value=None,
                chunks=_aiter([b"other file"]),
            )
        finally:
            second_done.set()

    first_path, second_path = await asyncio.gather(
        handler.save_stream(
            url="https://example.com/a",
            deduplication_key=None,
            expected_type="pdf",
            hash_header_value=None,
            chunks=slow_chunks(),
        ),
        second(),
    )

    assert Path(first_path).read_bytes() == b"first half, second half"
    assert Path(second_path).read_bytes() == b"other file"


class TestDedupReuse:
    """``should_download`` reuses a finished file already in the dedup dir."""

    async def test_reuses_a_finished_file(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_path(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        # A partial temp and an empty file sit beside the finished one.
        (dedup_dir / ".stream-abc.tmp").write_bytes(b"partial bytes")
        (dedup_dir / "zero.pdf").write_bytes(b"")
        finished = dedup_dir / "done.pdf"
        finished.write_bytes(b"%PDF")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            hash_header_value=None,
        )

        assert decision.download is False
        assert decision.file_url == str(finished)

    @pytest.mark.parametrize("key", [None, "never-seen"])
    async def test_downloads_without_a_stored_file(
        self, tmp_path: Path, key: str | None
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key=key,
            expected_type="pdf",
            hash_header_value=None,
        )

        assert decision.download is True


@pytest.mark.parametrize(
    ("name", "age_s", "survives"),
    [
        (".stream-old.tmp", STALE_STREAM_TMP_AGE_S + 60, False),
        (".stream-fresh.tmp", STALE_STREAM_TMP_AGE_S - 60, True),
        ("old-finished.pdf", STALE_STREAM_TMP_AGE_S + 60, True),
    ],
    ids=["stale temp", "fresh temp", "finished file"],
)
async def test_save_stream_sweeps_only_stale_temps(
    tmp_path: Path, name: str, age_s: float, survives: bool
) -> None:
    """Only a ``.stream-*`` file older than the stale age is swept."""
    handler = LocalAsyncStreamingArchiveHandler(tmp_path)
    dedup_dir = _dedup_path(tmp_path, "k")
    dedup_dir.mkdir(parents=True)
    leftover = dedup_dir / name
    leftover.write_bytes(b"leftover")
    then = time.time() - age_s
    os.utime(leftover, (then, then))

    await handler.save_stream(
        url="https://example.com/file",
        deduplication_key="k",
        expected_type="pdf",
        hash_header_value=None,
        chunks=_aiter([b"fresh content"]),
    )

    assert leftover.exists() is survives


@pytest.mark.parametrize("method", ["should_download", "save_stream"])
async def test_no_archive_handler_refuses(method: str) -> None:
    """Both methods raise ``ScraperConfigError`` naming the configuration fix."""
    handler = NoArchiveHandler()
    args: dict[str, object] = {
        "url": "https://example.com/file",
        "deduplication_key": "k",
        "expected_type": "pdf",
        "hash_header_value": None,
    }
    if method == "save_stream":
        args["chunks"] = _aiter([b"bytes"])

    with pytest.raises(ScraperConfigError) as excinfo:
        await getattr(handler, method)(**args)

    message = str(excinfo.value)
    assert "https://example.com/file" in message
    assert "storage_dir=" in message
    assert "archive_handler=" in message
