"""Archive handler base class and the default local-filesystem target.

Defines :class:`AsyncStreamingArchiveHandler`, the abstract base that every
archive target implements, and :class:`LocalAsyncStreamingArchiveHandler`,
which streams downloaded bytes straight to a local directory without ever
buffering the whole file in memory.

Targets receive the download as an async iterator of chunks, so a new backend
(e.g. an S3-backed handler in the Django app) only has to subclass the ABC and
implement the two abstract methods.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from jkent.data_types import ArchiveDecision

logger = logging.getLogger(__name__)

# Prefix for the in-progress temp files the streaming handler writes into the
# destination directory. Used both to name them and to recognize (and skip /
# sweep) leftovers from an interrupted download.
_STREAM_TMP_PREFIX = ".stream-"
_STREAM_TMP_SUFFIX = ".tmp"


def _dedup_dir(storage_dir: Path, deduplication_key: str) -> Path:
    """Return the nested storage subdirectory for a deduplication key.

    Layout: ``{storage_dir}/{xx}/{yy}/{deduplication_key}``, where ``xx`` and
    ``yy`` are the first two pairs of hex digits of the SHA-256 of the key.
    The two-level fanout keeps any single directory from growing large
    enough to trigger filesystem link-count limits.
    """
    sha = hashlib.sha256(deduplication_key.encode()).hexdigest()
    return storage_dir / sha[:2] / sha[2:4] / deduplication_key


def _streaming_target_path(
    storage_dir: Path,
    deduplication_key: str | None,
    sha_hex: str,
    expected_type: str | None,
) -> Path:
    """Assemble the final target path for a streamed download.

    Layout: ``{storage_dir}/{xx}/{yy}/{deduplication_key}/{shasum}.{expected_type}``
    (the ``{xx}/{yy}/{deduplication_key}`` segment is omitted when
    ``deduplication_key`` is ``None`` and the ``.{expected_type}`` suffix
    is omitted when ``expected_type`` is ``None``). The parent directory is
    assumed to already exist — the streaming temp file is created inside it
    before this path is computed.
    """
    target_dir = (
        _dedup_dir(storage_dir, deduplication_key)
        if deduplication_key
        else storage_dir
    )
    filename = f"{sha_hex}.{expected_type}" if expected_type else sha_hex
    return target_dir / filename


def _existing_dedup_file(dedup_dir: Path) -> Path | None:
    """Return the first completed file in ``dedup_dir``, if any.

    Skips the streaming handler's in-progress ``.stream-*.tmp`` files and
    zero-byte files: an interrupted download (e.g. killed by SIGKILL before
    the cleanup handler runs) can leave a partial temp file behind, and we
    must never serve that as a finished archive.
    """
    if not dedup_dir.is_dir():
        return None
    for entry in dedup_dir.iterdir():
        if entry.name.startswith(_STREAM_TMP_PREFIX):
            continue
        try:
            if entry.is_file() and entry.stat().st_size > 0:
                return entry
        except OSError:
            continue
    return None


def _clear_stale_stream_tmps(target_dir: Path) -> None:
    """Remove leftover ``.stream-*.tmp`` files from an interrupted download.

    Called before a fresh download so a partial temp file from a process that
    died mid-write doesn't accumulate. Best-effort: ignores files that vanish
    or can't be removed.
    """
    try:
        entries = list(target_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith(_STREAM_TMP_PREFIX):
            try:
                entry.unlink()
            except OSError:
                pass


def _hash_and_write(sha: Any, tmp: Any, chunk: bytes) -> None:
    """Update a running hash and write ``chunk`` in a single thread dispatch."""
    sha.update(chunk)
    tmp.write(chunk)


class AsyncStreamingArchiveHandler(ABC):
    """Abstract base for streaming archive targets.

    The download is delivered as an async iterator of chunks so the target can
    persist it without buffering the whole file in memory. Concrete targets
    implement two methods:

    - :meth:`should_download` -- the dedup/skip policy. Return
      ``ArchiveDecision(download=False, file_url=...)`` to reuse an
      already-stored file, or ``ArchiveDecision(download=True)`` to fetch.
    - :meth:`save_stream` -- persist the streamed chunks and return a locator
      string (a path, URL, or key) for the stored file.
    """

    @abstractmethod
    async def should_download(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
    ) -> ArchiveDecision: ...

    @abstractmethod
    async def save_stream(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
        chunks: AsyncIterator[bytes],
    ) -> str: ...


class LocalAsyncStreamingArchiveHandler(AsyncStreamingArchiveHandler):
    """Streams downloaded bytes to a local directory.

    Writes chunks straight to disk instead of accepting a fully-buffered
    ``bytes`` payload. The default filename is content-addressed:
    ``{storage_dir}/{xx}/{yy}/{deduplication_key}/{sha256}.{expected_type}``.
    Bytes stream into a temp file alongside the final destination so the rename
    is atomic once the full SHA-256 is known.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir

    async def should_download(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
    ) -> ArchiveDecision:
        if deduplication_key:
            dedup_dir = _dedup_dir(self.storage_dir, deduplication_key)
            existing = await asyncio.to_thread(_existing_dedup_file, dedup_dir)
            if existing is not None:
                return ArchiveDecision(download=False, file_url=str(existing))
        return ArchiveDecision(download=True)

    async def save_stream(
        self,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        hash_header_value: str | None,
        chunks: AsyncIterator[bytes],
    ) -> str:
        target_dir = (
            _dedup_dir(self.storage_dir, deduplication_key)
            if deduplication_key
            else self.storage_dir
        )
        await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
        # Sweep any partial temp file left by a previously interrupted download
        # before starting a fresh one.
        await asyncio.to_thread(_clear_stale_stream_tmps, target_dir)

        sha = hashlib.sha256()
        tmp = await asyncio.to_thread(
            tempfile.NamedTemporaryFile,
            dir=target_dir,
            delete=False,
            prefix=_STREAM_TMP_PREFIX,
            suffix=_STREAM_TMP_SUFFIX,
        )
        logger.info(
            "save_stream: starting url=%s dedup_key=%s",
            url,
            deduplication_key,
        )
        bytes_total = 0
        chunk_count = 0
        start = time.monotonic()
        last_log = start
        last_chunk = start
        try:
            try:
                async for chunk in chunks:
                    now = time.monotonic()
                    gap = now - last_chunk
                    bytes_total += len(chunk)
                    chunk_count += 1
                    last_chunk = now
                    if now - last_log >= 30.0:
                        logger.info(
                            "save_stream: in flight url=%s elapsed=%.1fs "
                            "chunks=%d bytes=%d last_gap=%.2fs",
                            url,
                            now - start,
                            chunk_count,
                            bytes_total,
                            gap,
                        )
                        last_log = now
                    await asyncio.to_thread(_hash_and_write, sha, tmp, chunk)
            finally:
                await asyncio.to_thread(tmp.close)
            logger.info(
                "save_stream: chunks done url=%s elapsed=%.1fs chunks=%d "
                "bytes=%d",
                url,
                time.monotonic() - start,
                chunk_count,
                bytes_total,
            )
            final_path = await asyncio.to_thread(
                _streaming_target_path,
                self.storage_dir,
                deduplication_key,
                sha.hexdigest(),
                expected_type,
            )
            await asyncio.to_thread(os.replace, tmp.name, final_path)
        except BaseException:
            # Best-effort cleanup — use sync os.unlink so this stays safe
            # under cancellation (an await here could itself be cancelled).
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return str(final_path)
