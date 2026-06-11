"""Tests for the default streaming archive handlers' filename scheme.

The sync and async ``LocalStreamingArchiveHandler`` variants write files
to ``{storage_dir}/{xx}/{yy}/{deduplication_key}/{sha256}.{expected_type}``
by default, where ``xx`` and ``yy`` are the first two pairs of hex digits
of the SHA-256 of the deduplication key. These tests lock in that layout
and confirm the hashing + atomic-rename plumbing.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from jkent.driver.archive_handler import (
    LocalAsyncStreamingArchiveHandler,
    LocalSyncStreamingArchiveHandler,
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


class TestSyncStreamingLayout:
    def test_dedup_key_and_expected_type(self, tmp_path: Path) -> None:
        handler = LocalSyncStreamingArchiveHandler(tmp_path)
        payload = b"some pdf bytes would go here"
        sha = hashlib.sha256(payload).hexdigest()

        path = handler.save_stream(
            url="https://example.com/ignored",
            deduplication_key="case-123",
            expected_type="pdf",
            hash_header_value=None,
            chunks=iter(_iter_chunks(payload)),
        )

        expected = _dedup_path(tmp_path, "case-123") / f"{sha}.pdf"
        assert Path(path) == expected
        assert expected.read_bytes() == payload

    def test_no_dedup_key(self, tmp_path: Path) -> None:
        handler = LocalSyncStreamingArchiveHandler(tmp_path)
        payload = b"anonymous content"
        sha = hashlib.sha256(payload).hexdigest()

        path = handler.save_stream(
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="bin",
            hash_header_value=None,
            chunks=iter([payload]),
        )

        assert Path(path) == tmp_path / f"{sha}.bin"

    def test_no_expected_type(self, tmp_path: Path) -> None:
        handler = LocalSyncStreamingArchiveHandler(tmp_path)
        payload = b"plain content"
        sha = hashlib.sha256(payload).hexdigest()

        path = handler.save_stream(
            url="https://example.com/file",
            deduplication_key="dedup",
            expected_type=None,
            hash_header_value=None,
            chunks=iter([payload]),
        )

        dedup_dir = _dedup_path(tmp_path, "dedup")
        assert Path(path) == dedup_dir / sha
        # No leftover ``.tmp`` files.
        assert [p.name for p in dedup_dir.iterdir()] == [sha]

    def test_sha_computed_across_chunks(self, tmp_path: Path) -> None:
        """The SHA is built incrementally from every chunk, not just one."""
        handler = LocalSyncStreamingArchiveHandler(tmp_path)
        chunks = [b"alpha-", b"beta-", b"gamma"]
        payload = b"".join(chunks)
        sha = hashlib.sha256(payload).hexdigest()

        path = handler.save_stream(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="txt",
            hash_header_value=None,
            chunks=iter(chunks),
        )
        assert Path(path).name == f"{sha}.txt"
        assert Path(path).read_bytes() == payload

    def test_tempfile_cleaned_up_on_stream_error(self, tmp_path: Path) -> None:
        handler = LocalSyncStreamingArchiveHandler(tmp_path)

        def _boom() -> Iterator[bytes]:
            yield b"first"
            raise RuntimeError("stream exploded")

        with pytest.raises(RuntimeError, match="stream exploded"):
            handler.save_stream(
                url="u",
                deduplication_key="k",
                expected_type="pdf",
                hash_header_value=None,
                chunks=_boom(),
            )

        dedup_dir = _dedup_path(tmp_path, "k")
        # Neither a final file nor a stray .tmp file should remain.
        assert not dedup_dir.exists() or list(dedup_dir.iterdir()) == []


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

        dedup_dir = _dedup_path(tmp_path, "k")
        # Neither a final file nor a stray .tmp file should remain.
        assert not dedup_dir.exists() or list(dedup_dir.iterdir()) == []
