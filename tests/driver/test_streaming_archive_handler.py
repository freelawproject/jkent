"""Tests for the default streaming archive handler's filename scheme.

:class:`LocalAsyncStreamingArchiveHandler` stores a download with a
deduplication key at ``{storage_dir}/{xx}/{yy}/{sha}.{expected_type}``, where
``sha`` is the SHA-256 of the key and ``xx``/``yy`` are its first two pairs of
hex digits; a download that skipped the dedup check is content-addressed
under the same fanout at
``{storage_dir}/{xx}/{yy}/{sha256-of-payload}.{expected_type}``, with a
``_1``, ``_2`` ... suffix rather than clobbering an earlier copy. These
tests lock in
that layout -- including that no part of it comes from the key's own text --
and confirm the hashing + atomic-rename plumbing, plus the skip/cleanup of
partial temp files left by an interrupted download.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jkent.common.exceptions import (
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.driver.archive_handler import (
    ARCHIVE_FILE_MODE,
    STALE_STREAM_TMP_AGE_S,
    LocalAsyncStreamingArchiveHandler,
    NoArchiveHandler,
    plan_digests,
)


def _iter_chunks(data: bytes, chunk_size: int = 4) -> list[bytes]:
    """Split ``data`` into a list of byte chunks for streaming."""
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


async def _aiter(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


async def _save(
    handler: LocalAsyncStreamingArchiveHandler, **kwargs: object
) -> Path:
    """``save_stream``, keeping just the stored file's path."""
    result = await handler.save_stream(**kwargs)  # type: ignore[arg-type]
    return Path(result.file_url)


def _dedup_path(
    storage_dir: Path, key: str, expected_type: str | None = "pdf"
) -> Path:
    """Mirror of the handler's stored-file path for a deduplication key."""
    sha = hashlib.sha256(key.encode()).hexdigest()
    filename = f"{sha}.{expected_type}" if expected_type else sha
    return storage_dir / sha[:2] / sha[2:4] / filename


def _content_path(
    storage_dir: Path,
    payload: bytes,
    expected_type: str | None = "pdf",
    index: int = 0,
) -> Path:
    """Mirror of the handler's content-addressed path for a payload."""
    sha = hashlib.sha256(payload).hexdigest()
    stem = sha if index == 0 else f"{sha}_{index}"
    name = f"{stem}.{expected_type}" if expected_type else stem
    return storage_dir / sha[:2] / sha[2:4] / name


def _dedup_dir(storage_dir: Path, key: str) -> Path:
    """The ``{xx}/{yy}`` bucket a key hashes into."""
    return _dedup_path(storage_dir, key).parent


class TestAsyncStreamingLayout:
    async def test_dedup_key_and_expected_type(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"async streamed bytes"
        sha = hashlib.sha256(payload).hexdigest()

        path = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key="case-99",
            expected_type="audio",
            chunks=_aiter(_iter_chunks(payload)),
        )

        expected = _dedup_path(tmp_path, "case-99", "audio")
        assert Path(path) == expected
        assert expected.read_bytes() == payload
        # The payload's own hash is nowhere in the path: the key fixes it.
        assert sha not in str(expected.relative_to(tmp_path))

    async def test_no_dedup_key(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"flat file"
        sha = hashlib.sha256(payload).hexdigest()

        path = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="pdf",
            chunks=_aiter([payload]),
        )

        assert path == _content_path(tmp_path, payload)
        assert path.read_bytes() == payload
        assert sha in path.name

    async def test_no_expected_type(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"plain content"

        path = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key="dedup",
            expected_type=None,
            chunks=_aiter([payload]),
        )

        key_sha = hashlib.sha256(b"dedup").hexdigest()
        assert Path(path) == _dedup_path(tmp_path, "dedup", None)
        assert Path(path).name == key_sha
        assert Path(path).read_bytes() == payload
        # No leftover ``.tmp`` files.
        assert [p.name for p in _dedup_dir(tmp_path, "dedup").iterdir()] == [
            key_sha
        ]

    async def test_sha_computed_across_chunks(self, tmp_path: Path) -> None:
        """The SHA is built incrementally from every chunk, not just one.

        Keyless, because that is the path the payload hash still names.
        """
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        chunks = [b"alpha-", b"beta-", b"gamma"]
        payload = b"".join(chunks)
        sha = hashlib.sha256(payload).hexdigest()

        path = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="txt",
            chunks=_aiter(chunks),
        )
        assert path == _content_path(tmp_path, payload, "txt")
        assert path.read_bytes() == payload
        assert sha in path.name

    @pytest.mark.parametrize(
        "key",
        ["/etc/passwd", "../../escape", "a/b/c", ".", ".."],
        ids=["absolute", "traversal", "separators", "dot", "dotdot"],
    )
    async def test_key_never_steers_the_path(
        self, tmp_path: Path, key: str
    ) -> None:
        """A key is scraped content; no part of it reaches the filesystem.

        Interpolating one as a path segment would let a site write outside
        ``storage_dir`` -- ``Path(storage) / "/etc/passwd"`` is ``/etc/passwd``.
        """
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        path = Path(
            await _save(
                handler,
                url="https://example.com/file",
                deduplication_key=key,
                expected_type="pdf",
                chunks=_aiter([b"payload"]),
            )
        )

        assert path.resolve().is_relative_to(tmp_path.resolve())
        assert path.read_bytes() == b"payload"

    async def test_tempfile_cleaned_up_on_stream_error(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        async def _boom() -> AsyncIterator[bytes]:
            yield b"first"
            raise RuntimeError("stream exploded")

        with pytest.raises(RuntimeError, match="stream exploded"):
            await _save(
                handler,
                url="u",
                deduplication_key="k",
                expected_type="pdf",
                chunks=_boom(),
            )

        # save_stream creates the dedup dir before streaming; neither a final
        # file nor a stray .tmp file may be left in it.
        assert list(_dedup_dir(tmp_path, "k").iterdir()) == []


class TestPartialTempFileHandling:
    """A partial ``.stream-*.tmp`` left by an interrupted download must never
    be served as a finished archive, and is swept on the next download."""

    async def test_should_download_ignores_partial_tmp(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_dir(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        # Simulate a download killed mid-write: a leftover temp file.
        (dedup_dir / ".stream-abc.tmp").write_bytes(b"partial bytes")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
        )

        # Must not treat the partial temp file as an existing archive.
        assert decision.download is True

    async def test_should_download_ignores_empty_file(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        stored = _dedup_path(tmp_path, "k")
        stored.parent.mkdir(parents=True)
        stored.write_bytes(b"")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
        )
        assert decision.download is True

    async def test_save_stream_sweeps_stale_tmp(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        dedup_dir = _dedup_dir(tmp_path, "k")
        dedup_dir.mkdir(parents=True)
        stale = dedup_dir / ".stream-stale.tmp"
        stale.write_bytes(b"leftover")
        an_hour_and_more_ago = time.time() - STALE_STREAM_TMP_AGE_S - 1
        os.utime(stale, (an_hour_and_more_ago, an_hour_and_more_ago))

        payload = b"fresh content"
        path = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            chunks=_aiter([payload]),
        )

        # The stale temp file is gone and only the finished file remains.
        stored = _dedup_path(tmp_path, "k")
        assert not stale.exists()
        assert Path(path) == stored
        assert [p.name for p in dedup_dir.iterdir()] == [stored.name]


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

    async def second() -> Path:
        await first_started.wait()
        try:
            return await _save(
                handler,
                url="https://example.com/b",
                deduplication_key=None,
                expected_type="pdf",
                chunks=_aiter([b"other file"]),
            )
        finally:
            second_done.set()

    first_path, second_path = await asyncio.gather(
        _save(
            handler,
            url="https://example.com/a",
            deduplication_key=None,
            expected_type="pdf",
            chunks=slow_chunks(),
        ),
        second(),
    )

    assert first_path.read_bytes() == b"first half, second half"
    assert second_path.read_bytes() == b"other file"


class TestDedupReuse:
    """``should_download`` reuses the file a key already maps to."""

    async def test_reuses_a_finished_file(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        finished = _dedup_path(tmp_path, "k")
        finished.parent.mkdir(parents=True)
        # Another key's leftovers share the bucket; only the key's own path
        # counts, so neither may be served for it.
        (finished.parent / ".stream-abc.tmp").write_bytes(b"partial bytes")
        (finished.parent / "neighbor.pdf").write_bytes(b"%PDF other")
        finished.write_bytes(b"%PDF")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
        )

        assert decision.download is False
        assert decision.file_url == str(finished)

    async def test_a_different_expected_type_is_a_different_file(
        self, tmp_path: Path
    ) -> None:
        """The suffix is part of the path, so it is part of the identity."""
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        stored = _dedup_path(tmp_path, "k", "pdf")
        stored.parent.mkdir(parents=True)
        stored.write_bytes(b"%PDF")

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="wpd",
        )

        assert decision.download is True

    @pytest.mark.parametrize("key", [None, "never-seen"])
    async def test_downloads_without_a_stored_file(
        self, tmp_path: Path, key: str | None
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        decision = await handler.should_download(
            url="https://example.com/file",
            deduplication_key=key,
            expected_type="pdf",
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
    dedup_dir = _dedup_dir(tmp_path, "k")
    dedup_dir.mkdir(parents=True)
    leftover = dedup_dir / name
    leftover.write_bytes(b"leftover")
    then = time.time() - age_s
    os.utime(leftover, (then, then))

    await _save(
        handler,
        url="https://example.com/file",
        deduplication_key="k",
        expected_type="pdf",
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
    }
    if method == "save_stream":
        args["chunks"] = _aiter([b"bytes"])

    with pytest.raises(ScraperConfigError) as excinfo:
        await getattr(handler, method)(**args)

    message = str(excinfo.value)
    assert "https://example.com/file" in message
    assert "no archive handler" in message


class TestWriteResult:
    """``save_stream`` reports what it wrote, so nothing re-reads the file."""

    @pytest.mark.parametrize("key", ["k", None], ids=["keyed", "keyless"])
    async def test_reports_size_and_hash(
        self, tmp_path: Path, key: str | None
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"x" * 1234

        result = await handler.save_stream(
            url="https://example.com/file",
            deduplication_key=key,
            expected_type="pdf",
            chunks=_aiter(_iter_chunks(payload, 100)),
        )

        assert result.file_size == len(payload)
        assert result.content_hash == hashlib.sha256(payload).hexdigest()
        assert Path(result.file_url).read_bytes() == payload


class TestContentAddressedDuplicates:
    """A skipped dedup check means "store this fetch", so repeats are kept."""

    async def test_identical_bytes_get_a_suffix(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"the same bytes twice"

        paths = [
            await _save(
                handler,
                url="https://example.com/file",
                deduplication_key=None,
                expected_type="pdf",
                chunks=_aiter([payload]),
            )
            for _ in range(3)
        ]

        assert paths == [
            _content_path(tmp_path, payload, "pdf", index)
            for index in range(3)
        ]
        assert all(p.read_bytes() == payload for p in paths)

    async def test_concurrent_identical_bytes_both_survive(
        self, tmp_path: Path
    ) -> None:
        """Two workers finishing the same payload at once keep both copies."""
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"a racing payload"

        first, second = await asyncio.gather(
            *(
                _save(
                    handler,
                    url="https://example.com/file",
                    deduplication_key=None,
                    expected_type="pdf",
                    chunks=_aiter(_iter_chunks(payload)),
                )
                for _ in range(2)
            )
        )

        assert {first, second} == {
            _content_path(tmp_path, payload, "pdf", 0),
            _content_path(tmp_path, payload, "pdf", 1),
        }

    async def test_a_different_expected_type_is_a_different_stem(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        payload = b"same bytes, two labels"

        pdf = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="pdf",
            chunks=_aiter([payload]),
        )
        mp3 = await _save(
            handler,
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="mp3",
            chunks=_aiter([payload]),
        )

        assert pdf == _content_path(tmp_path, payload, "pdf")
        assert mp3 == _content_path(tmp_path, payload, "mp3")

    async def test_the_storage_root_holds_only_temp_files(
        self, tmp_path: Path
    ) -> None:
        """Finished keyless files live in a bucket, so the sweep stays cheap."""
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        await _save(
            handler,
            url="https://example.com/file",
            deduplication_key=None,
            expected_type="pdf",
            chunks=_aiter([b"content"]),
        )

        assert [p.name for p in tmp_path.iterdir() if p.is_file()] == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
async def test_archived_file_is_readable_by_other_users(
    tmp_path: Path,
) -> None:
    """Not 0600: another service reads this archive.

    ``NamedTemporaryFile`` creates at 0600 and ``os.replace`` preserves it,
    so the handler has to set the mode a plain ``open(..., "wb")`` would
    have produced.
    """
    handler = LocalAsyncStreamingArchiveHandler(tmp_path)

    path = await _save(
        handler,
        url="https://example.com/file",
        deduplication_key="k",
        expected_type="pdf",
        chunks=_aiter([b"readable"]),
    )

    assert ARCHIVE_FILE_MODE is not None
    assert stat.S_IMODE(path.stat().st_mode) == ARCHIVE_FILE_MODE
    assert ARCHIVE_FILE_MODE & stat.S_IRGRP


class TestIntegrityVerification:
    """What the response said the bytes would be is checked against them."""

    async def _save_with(
        self, handler: LocalAsyncStreamingArchiveHandler, **kwargs: object
    ) -> Path:
        return await _save(
            handler,
            url="https://example.com/file",
            deduplication_key="k",
            expected_type="pdf",
            **kwargs,
        )

    async def test_content_length_mismatch_is_transient(
        self, tmp_path: Path
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        with pytest.raises(TransientException) as excinfo:
            await self._save_with(
                handler,
                headers={"Content-Length": "999"},
                chunks=_aiter([b"short"]),
            )

        assert excinfo.value.kind is TransientKind.ARCHIVE
        assert "999" in str(excinfo.value)
        # Nothing was published, and the partial bytes are gone.
        assert list(_dedup_dir(tmp_path, "k").iterdir()) == []

    async def test_content_length_match_passes(self, tmp_path: Path) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        path = await self._save_with(
            handler,
            headers={"content-length": "5"},
            chunks=_aiter([b"ex", b"act"]),
        )

        assert path.read_bytes() == b"exact"

    async def test_content_length_ignored_when_encoded(
        self, tmp_path: Path
    ) -> None:
        """The header counts compressed bytes; the handler sees decoded ones."""
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        path = await self._save_with(
            handler,
            headers={"Content-Length": "12", "Content-Encoding": "gzip"},
            chunks=_aiter([b"decoded and much longer"]),
        )

        assert path.read_bytes() == b"decoded and much longer"

    @pytest.mark.parametrize(
        ("etag_for", "stored"),
        [(b"payload", True), (b"other bytes", False)],
        ids=["matching etag", "mismatched etag"],
    )
    async def test_md5_etag_is_checked(
        self, tmp_path: Path, etag_for: bytes, stored: bool
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)
        etag = hashlib.md5(etag_for, usedforsecurity=False).hexdigest()

        async def run() -> Path:
            return await self._save_with(
                handler,
                headers={"ETag": f'"{etag}"'},
                chunks=_aiter(_iter_chunks(b"payload")),
            )

        if stored:
            assert (await run()).read_bytes() == b"payload"
        else:
            with pytest.raises(TransientException, match="ETag MD5"):
                await run()

    @pytest.mark.parametrize(
        "etag",
        ['W/"0123456789abcdef0123456789abcdef"', '"deadbeef-3"', '"xyz"'],
        ids=["weak", "multipart", "opaque"],
    )
    async def test_an_etag_that_is_not_an_md5_is_ignored(
        self, tmp_path: Path, etag: str
    ) -> None:
        handler = LocalAsyncStreamingArchiveHandler(tmp_path)

        path = await self._save_with(
            handler,
            headers={"ETag": etag},
            chunks=_aiter([b"payload"]),
        )

        assert path.read_bytes() == b"payload"
        assert "md5" not in plan_digests({"ETag": etag})

    async def test_md5_runs_only_when_an_etag_can_check_it(self) -> None:
        """A second digest is a second pass over every byte, so none is free."""
        assert set(plan_digests(None)) == {"sha256"}
        assert set(plan_digests({"ETag": '"' + "a" * 32 + '"'})) == {
            "sha256",
            "md5",
        }
