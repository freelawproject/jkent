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
import re
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import IO, TYPE_CHECKING, NoReturn

from jkent.common.exceptions import (
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.data_types import ArchiveDecision, ArchiveWriteResult

if TYPE_CHECKING:
    # typeshed's hashlib object protocol; no runtime name to import.
    from hashlib import _Hash

logger = logging.getLogger(__name__)

# Prefix for the in-progress temp files the streaming handler writes into the
# destination directory. Used both to name them and to recognize (and skip /
# sweep) leftovers from an interrupted download.
_STREAM_TMP_PREFIX = ".stream-"
_STREAM_TMP_SUFFIX = ".tmp"
#: A ``.stream-*.tmp`` file untouched for this long is a leftover from a
#: killed download and may be swept. An in-flight download rewrites its temp
#: file on every chunk, so a sibling worker's (or another process's) file is
#: never this old.
#:
#: The window is deliberately generous because the sweep cannot tell a dead
#: writer from a stalled one: a download whose server has gone quiet for over
#: an hour will have its temp file unlinked out from under it by the next
#: worker into the same bucket, and its own :func:`os.replace` then fails
#: with ``FileNotFoundError``, discarding bytes that may have been complete.
#: An hour of no chunks at all is already a dead transfer in practice, so
#: that trade is accepted rather than tracked with in-flight bookkeeping.
STALE_STREAM_TMP_AGE_S = 3600.0

#: How many ``_1``, ``_2`` ... siblings a content-addressed name may grow
#: before the handler gives up. Only reached by a scraper that fetches the
#: same bytes this many times with the dedup check skipped.
_MAX_CONTENT_SUFFIX = 1000

#: A strong ETag that is exactly 32 hex digits is the body's MD5 (Apache and
#: S3 single-part uploads both emit one); anything else says nothing we can
#: check the bytes against.
_MD5_HEX = re.compile(r"[0-9a-f]{32}")


def _default_file_mode() -> int | None:
    """The mode archived files get, or ``None`` to leave the mode alone."""
    if os.name == "nt":
        return None
    previous = os.umask(0o077)
    os.umask(previous)
    return 0o666 & ~previous


ARCHIVE_FILE_MODE = _default_file_mode()


def _dedup_path(
    storage_dir: Path, deduplication_key: str, expected_type: str | None
) -> Path:
    """Return the stored file's path for a deduplication key.

    Layout: ``{storage_dir}/{xx}/{yy}/{sha}.{expected_type}``, where ``sha``
    is the SHA-256 of the key and ``xx``/``yy`` are its first two pairs of
    hex digits (the suffix is omitted when ``expected_type`` is ``None``).
    The two-level fanout keeps any single directory from growing large
    enough to trigger filesystem link-count limits.

    Every segment of the *key's* contribution comes from the hash, never
    from the key itself — keys are built from scraped values, and
    interpolating one as a path segment lets site content steer the write
    out of ``storage_dir`` (a key holding ``..``, or an absolute-looking
    one, which ``/`` resolves against the root). ``expected_type`` is
    interpolated as given: it is a scraper-written hint (``pdf``, ``mp3``),
    not a scraped value.

    A key maps to exactly one path, so the dedup check is a stat rather than
    a directory scan, at the cost of storing the same bytes once per key
    that reaches them and of a key fetched under two expected types
    landing twice.
    """
    sha = hashlib.sha256(deduplication_key.encode()).hexdigest()
    filename = f"{sha}.{expected_type}" if expected_type else sha
    return storage_dir / sha[:2] / sha[2:4] / filename


def _content_filename(
    sha_hex: str, expected_type: str | None, index: int
) -> str:
    """Name for the ``index``-th copy of a content-addressed payload."""
    stem = sha_hex if index == 0 else f"{sha_hex}_{index}"
    return f"{stem}.{expected_type}" if expected_type else stem


def _publish_content_addressed(
    tmp_name: str, storage_dir: Path, sha_hex: str, expected_type: str | None
) -> Path:
    """Give a finished temp file its content-addressed name.

    Layout mirrors :func:`_dedup_path` — ``{xx}/{yy}`` fanout from the
    hash's own hex digits — but the hash is of the payload, not of a key:
    ``{storage_dir}/{xx}/{yy}/{sha256-of-payload}.{expected_type}``.

    Only requests that skipped the dedup check land here, and skipping it
    means "store this fetch", so an existing file of the same bytes is a
    previous fetch to keep rather than a hit to collapse into: the name
    grows a ``_1``, ``_2`` ... suffix instead of being overwritten. The
    name is claimed with :func:`os.link`, which fails rather than clobbers,
    so two workers finishing identical bytes at once keep both copies.

    Raises:
        FileExistsError: :data:`_MAX_CONTENT_SUFFIX` copies of these bytes
            already exist under this expected type.
    """
    bucket = storage_dir / sha_hex[:2] / sha_hex[2:4]
    bucket.mkdir(parents=True, exist_ok=True)
    for index in range(_MAX_CONTENT_SUFFIX):
        candidate = bucket / _content_filename(sha_hex, expected_type, index)
        try:
            os.link(tmp_name, candidate)
        except FileExistsError:
            continue
        except OSError:
            os.replace(tmp_name, candidate)
            return candidate
        os.unlink(tmp_name)
        return candidate
    raise FileExistsError(
        f"{_MAX_CONTENT_SUFFIX} copies of {sha_hex} already stored in "
        f"{bucket}; refusing to add another"
    )


def _is_finished_file(path: Path) -> bool:
    """Whether ``path`` holds a completed download.

    Bytes only reach the final name through :func:`os.replace`, and the
    in-progress ``.stream-*.tmp`` files never carry it, so a partial write
    is not visible under this name. The size check still rejects a
    zero-byte file: re-fetching costs one request, serving nothing as a
    finished archive is silent data loss.
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _clear_stale_stream_tmps(target_dir: Path) -> None:
    """Remove leftover ``.stream-*.tmp`` files from an interrupted download.
    A best effort cleanup to remove any streaming file that hasn't
    managed to pull > 8kb(io.DEFAULT_BUFFER_SIZE) in the last
    hour (STALE_STREAM_TMP_AGE_S), which we view to be abandoned.
    """
    try:
        entries = list(target_dir.iterdir())
    except OSError:
        return
    cutoff = time.time() - STALE_STREAM_TMP_AGE_S
    for entry in entries:
        if not entry.name.startswith(_STREAM_TMP_PREFIX):
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
        except OSError:
            pass


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """Case-insensitive header lookup over a plain mapping."""
    if not headers:
        return None
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


def _etag_md5(headers: Mapping[str, str] | None) -> str | None:
    """The MD5 hex digest an ``ETag`` states, when it states one.

    An ETag is an opaque validator by spec, but the common servers make it
    the body's MD5, which is free integrity checking when they do. Ignored
    otherwise: a weak validator (``W/"..."``) is explicitly not
    byte-for-byte, an S3 multipart ETag (``"<hex>-<parts>"``) is a digest of
    digests, and an inode-and-mtime ETag is about the file, not its bytes.
    """
    etag = _header(headers, "etag")
    if etag is None:
        return None
    value = etag.strip()
    if value[:2].lower() == "w/":
        return None
    value = value.strip('"').lower()
    return value if _MD5_HEX.fullmatch(value) else None


def _stated_content_length(headers: Mapping[str, str] | None) -> int | None:
    """The body size the response promised, when it is comparable.

    ``None`` when the response didn't say (a chunked transfer has no
    ``Content-Length``), when the value isn't a number, or when a
    ``Content-Encoding`` is in play — that header then counts compressed
    bytes while the transport hands the handler decoded ones, and comparing
    the two would fail every gzipped download.
    """
    encoding = _header(headers, "content-encoding")
    if encoding and encoding.strip().lower() != "identity":
        return None
    raw = _header(headers, "content-length")
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def plan_digests(headers: Mapping[str, str] | None) -> dict[str, _Hash]:
    """The digests to run over this download's chunks.

    SHA-256 always: it names a content-addressed file and is what the
    ``archived_files`` row stores. MD5 only when this response carries an
    ETag that can be checked against it, since hashing a second time costs
    a pass over every byte. ``usedforsecurity=False`` says what the digest
    is for -- catching a truncated transfer, not authenticating one -- so
    it stays available on a FIPS-restricted build.
    """
    digests: dict[str, _Hash] = {"sha256": hashlib.sha256()}
    if _etag_md5(headers) is not None:
        digests["md5"] = hashlib.md5(usedforsecurity=False)
    return digests


def _integrity_failure(url: str, detail: str) -> TransientException:
    """The error a failed integrity check raises.

    Transient because a truncated or corrupted transfer is worth another
    attempt, and because the alternative is worse: the bytes would be
    renamed onto the dedup key's permanent path, where ``should_download``
    serves them to every later run without ever re-checking them.
    """
    return TransientException(
        f"archive download from {url} failed its integrity check: {detail}",
        url=url,
        kind=TransientKind.ARCHIVE,
    )


def verify_download(
    *,
    url: str,
    headers: Mapping[str, str] | None,
    digests: Mapping[str, _Hash],
    bytes_total: int,
) -> None:
    """Check the received bytes against what the response said they'd be.

    Two independent checks, each skipped when this response says nothing
    about it: the stated ``Content-Length``, and an ``ETag`` that is the
    body's MD5. A response carrying neither is accepted — the transport
    already raises on a connection that breaks mid-body, and these cover
    the quieter truncation it can't see.

    Raises:
        TransientException: a check failed; see :func:`_integrity_failure`.
    """
    stated_length = _stated_content_length(headers)
    if stated_length is not None and stated_length != bytes_total:
        raise _integrity_failure(
            url,
            f"Content-Length said {stated_length} bytes, received "
            f"{bytes_total}",
        )

    expected_md5 = _etag_md5(headers)
    if expected_md5 is not None and "md5" in digests:
        actual = digests["md5"].hexdigest()
        if actual != expected_md5:
            raise _integrity_failure(
                url, f"ETag MD5 {expected_md5} != received {actual}"
            )


def _hash_and_write(
    digests: Mapping[str, _Hash], tmp: IO[bytes], chunk: bytes
) -> None:
    """Update the running digests and write ``chunk`` in one dispatch."""
    for digest in digests.values():
        digest.update(chunk)
    tmp.write(chunk)


def _finalize_temp(tmp: IO[bytes]) -> None:
    """Flush, sync and chmod a temp file before it takes its final name."""
    tmp.flush()
    os.fsync(tmp.fileno())
    if ARCHIVE_FILE_MODE is not None:
        os.fchmod(tmp.fileno(), ARCHIVE_FILE_MODE)


class AsyncStreamingArchiveHandler(ABC):
    """Abstract base for streaming archive targets.

    The download is delivered as an async iterator of chunks so the target can
    persist it without buffering the whole file in memory. Concrete targets
    implement two methods:

    - :meth:`should_download` -- the dedup/skip policy. Return
      ``ArchiveDecision(download=False, file_url=...)`` to reuse an
      already-stored file, or ``ArchiveDecision(download=True)`` to fetch.
    - :meth:`save_stream` -- persist the streamed chunks and return an
      :class:`~jkent.common.response.ArchiveWriteResult`: a locator string
      (a path, URL, or key) plus the size and SHA-256 of what was written,
      which the driver stores on the ``archived_files`` row.
    """

    @abstractmethod
    async def should_download(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
    ) -> ArchiveDecision: ...

    @abstractmethod
    async def save_stream(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        headers: Mapping[str, str] | None = None,
        chunks: AsyncIterator[bytes],
    ) -> ArchiveWriteResult: ...


class NoArchiveHandler(AsyncStreamingArchiveHandler):
    """The handler for a run that was given nowhere to put files.

    An ``archive=True`` request reaching it is a configuration error — the
    scraper downloads files but the host configured no archive target — so
    it raises :class:`ScraperConfigError` naming the fix.
    """

    @staticmethod
    def _refuse(url: str) -> ScraperConfigError:
        return ScraperConfigError(
            f"archive request for {url} but the run has no archive handler."
        )

    async def should_download(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
    ) -> NoReturn:
        raise self._refuse(url)

    async def save_stream(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        headers: Mapping[str, str] | None = None,
        chunks: AsyncIterator[bytes],
    ) -> NoReturn:
        raise self._refuse(url)


class LocalAsyncStreamingArchiveHandler(AsyncStreamingArchiveHandler):
    """Streams downloaded bytes to a local directory.

    Writes chunks straight to disk instead of accepting a fully-buffered
    ``bytes`` payload. A request carrying a deduplication key is stored at
    ``{storage_dir}/{xx}/{yy}/{sha256-of-key}.{expected_type}``
    (:func:`_dedup_path`); one that skipped the dedup check is
    content-addressed under the same fanout as
    ``{storage_dir}/{xx}/{yy}/{sha256-of-payload}.{expected_type}``, with a
    ``_1``, ``_2`` ... suffix so a repeat fetch of identical bytes is kept
    rather than collapsed (:func:`_publish_content_addressed`). Bytes stream
    into a temp file on the same filesystem so the rename is atomic.
    """

    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir

    async def should_download(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
    ) -> ArchiveDecision:
        if deduplication_key:
            existing = _dedup_path(
                self.storage_dir, deduplication_key, expected_type
            )
            if await asyncio.to_thread(_is_finished_file, existing):
                return ArchiveDecision(download=False, file_url=str(existing))
        return ArchiveDecision(download=True)

    async def save_stream(
        self,
        *,
        url: str,
        deduplication_key: str | None,
        expected_type: str | None,
        headers: Mapping[str, str] | None = None,
        chunks: AsyncIterator[bytes],
    ) -> ArchiveWriteResult:
        digests = plan_digests(headers)
        target_dir = (
            _dedup_path(
                self.storage_dir, deduplication_key, expected_type
            ).parent
            if deduplication_key
            else self.storage_dir
        )
        await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(_clear_stale_stream_tmps, target_dir)

        tmp: IO[bytes] = await asyncio.to_thread(
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
                    await asyncio.to_thread(
                        _hash_and_write, digests, tmp, chunk
                    )
                await asyncio.to_thread(_finalize_temp, tmp)
            finally:
                # Sync, not ``to_thread``: an await in a cleanup path can
                # itself be cancelled, which would leak the fd and leave
                # the buffered tail unwritten under the final name.
                tmp.close()
            logger.info(
                "save_stream: chunks done url=%s elapsed=%.1fs chunks=%d "
                "bytes=%d",
                url,
                time.monotonic() - start,
                chunk_count,
                bytes_total,
            )
            verify_download(
                url=url,
                headers=headers,
                digests=digests,
                bytes_total=bytes_total,
            )
            sha_hex = digests["sha256"].hexdigest()
            if deduplication_key:
                final_path = _dedup_path(
                    self.storage_dir, deduplication_key, expected_type
                )
                await asyncio.to_thread(os.replace, tmp.name, final_path)
            else:
                final_path = await asyncio.to_thread(
                    _publish_content_addressed,
                    tmp.name,
                    self.storage_dir,
                    sha_hex,
                    expected_type,
                )
        except BaseException:
            # Best-effort cleanup — use sync os.unlink so this stays safe
            # under cancellation (an await here could itself be cancelled).
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return ArchiveWriteResult(
            file_url=str(final_path),
            file_size=bytes_total,
            content_hash=sha_hex,
        )
