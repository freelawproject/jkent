"""Zstd compression for stored run-database responses.

This module provides zstd compression and decompression for HTTP responses,
with support for per-step trained dictionaries for better compression
ratios on similar content.

Compression is done with zstd (Zstandard) which offers excellent compression
ratios and fast decompression speeds. Dictionary-based compression can
significantly improve compression of similar content (like HTML from the
same website).

What lives here is the part that needs no database: the zstd calls
themselves, the tuning constants, and the two batch helpers that run
off-loop in a worker thread. The manager-facing half — dictionary lookups,
per-response compression, and the train/recompress compaction pass — is
``CompressionMixin``, in
:mod:`jkent.driver.database_engine.sql_manager._compression`, mixed into
:class:`~jkent.driver.database_engine.sql_manager.SQLManager`.

Dictionaries are cached on the manager, precomputed for
:data:`DEFAULT_COMPRESSION_LEVEL`: a ``ZstdCompressor`` built on a plain
dictionary re-digests it on every single compression, which is the larger
half of the cost on short bodies. Compression therefore happens at that
one level, and the level knob survives only on :func:`compress` itself,
for callers that bring their own dictionary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final, NamedTuple

import zstandard as zstd

from jkent.contracts import require

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

logger = logging.getLogger(__name__)

#: Nanoseconds per second. The zstd leaves are timed on the integer ``_ns``
#: clocks (no float rounding mid-measurement) and converted once, here, at
#: the point of record — the metrics themselves are documented in seconds.
NS_PER_SECOND: Final = 1_000_000_000

#: Default compression level (3 is a good balance of speed/ratio).
DEFAULT_COMPRESSION_LEVEL: Final = 3

#: Default dictionary size (112640 bytes = 110KB, zstd's default).
DEFAULT_DICT_SIZE: Final = 112640

#: Rows per ``CompressionMixin.recompress_responses`` batch. Bounds both
#: the bytes held in memory (old + new compressed content per row) and how
#: long any single write transaction holds the run's DB lock during
#: compaction.
RECOMPRESS_CHUNK_SIZE: Final = 500

#: Cap on cumulative decompressed sample bytes fed to dictionary training,
#: as a multiple of the dictionary size. zstd's guidance is that ~100x the
#: dictionary's size in training data saturates what training can use; past
#: that, extra samples only cost memory (they are all held at once).
TRAIN_SAMPLE_BYTE_FACTOR: Final = 100

#: Rows per ``CompressionMixin._collect_training_samples`` batch. Bounds
#: the compressed bytes resident while the sample is gathered: only one
#: batch's bodies are held, rather than every sampled body at once.
TRAIN_SAMPLE_BATCH_SIZE: Final = 50

#: Fewest decompressed sample bytes training is attempted on. A dictionary
#: learned from less is worthless, and zstandard 0.25's trainer segfaults on
#: some such corpora (eight one-byte samples) instead of raising.
MIN_TRAIN_SAMPLE_BYTES: Final = 1024


class CompressedContent(NamedTuple):
    """``CompressionMixin.compress_response`` output: the compressed bytes
    and the id of the dictionary they were compressed with (None when no
    dictionary was available)."""

    data: bytes
    dict_id: int | None


@require(
    lambda level: 1 <= level <= 22,  # pyrefly: ignore[implicit-any-lambda]
    "compression level is within zstd's documented 1-22 range",
)
def compress(
    data: bytes,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    dictionary: zstd.ZstdCompressionDict | None = None,
) -> bytes:
    """Compress data using zstd.

    Args:
        data: The data to compress.
        level: Compression level (1-22, default 3). Ignored for the
            compression parameters a dictionary precomputed by
            ``CompressionMixin``'s dictionary lookups pins (see the module
            docstring); every in-tree caller passes those dictionaries at
            :data:`DEFAULT_COMPRESSION_LEVEL` anyway.
        dictionary: Optional pre-trained dictionary for better compression.

    Returns:
        Compressed data bytes.
    """
    return zstd.ZstdCompressor(level=level, dict_data=dictionary).compress(
        data
    )


def decompress(
    data: bytes,
    dictionary: zstd.ZstdCompressionDict | None = None,
) -> bytes:
    """Decompress zstd-compressed data.

    Args:
        data: The compressed data to decompress.
        dictionary: Dictionary used for compression (must match).

    Returns:
        Decompressed data bytes.
    """
    return zstd.ZstdDecompressor(dict_data=dictionary).decompress(data)


def _load_dictionary(blob: bytes) -> zstd.ZstdCompressionDict:
    """Build a cacheable dictionary from a stored blob.

    Precomputed for :data:`DEFAULT_COMPRESSION_LEVEL` before it is shared:
    otherwise every ``ZstdCompressor`` built on it re-digests the ~110KB
    dictionary, which on short bodies costs an order of magnitude more than
    the compression itself (and :func:`_recompress_chunk` pays it
    ``chunk_size`` times a chunk). The precomputed ``ZSTD_CDict`` is
    const, so the worker threads that share it only read it — but the
    precompute itself mutates, so it must happen here, on the loop, before
    the object reaches the cache and any thread.

    The cost is memory: a precomputed dictionary measures ~525KB resident
    for a :data:`DEFAULT_DICT_SIZE` blob. That puts us at less than 15MB
    for all of our current scrape runs, and less than 8MB for most.
    """
    dict_obj = zstd.ZstdCompressionDict(blob)
    dict_obj.precompute_compress(level=DEFAULT_COMPRESSION_LEVEL)
    return dict_obj


def _decompress_samples(
    rows: Sequence[tuple[bytes, int | None]],
    dictionaries: dict[int, zstd.ZstdCompressionDict],
    step: str,
    budget: int,
) -> list[bytes | bytearray | memoryview[int]]:
    """Decompress one batch of training samples; skip failures; stop at
    ``budget``.

    Runs in a worker thread — ``len(rows)`` back-to-back zstd calls would
    stall co-resident workers if run on the loop. A sample whose dictionary
    is missing or whose bytes fail to decompress is skipped with a warning.
    Collection stops once ``budget`` decompressed bytes are gathered, which
    is the caller's remaining share of ``max_sample_bytes``: past that
    point extra samples don't improve the dictionary, they only hold
    memory (all samples are resident at once).

    The element union of the returned list is wider than what this function
    produces — every sample is ``bytes`` — because ``zstd.train_dictionary``
    declares an invariant ``list[ByteString]`` parameter, to which a
    ``list[bytes]`` is not assignable.
    """
    samples: list[bytes | bytearray | memoryview[int]] = []
    total = 0
    for compressed, comp_dict_id in rows:
        dictionary = (
            dictionaries.get(comp_dict_id)
            if comp_dict_id is not None
            else None
        )
        try:
            content = decompress(compressed, dictionary=dictionary)
        except Exception:
            # Skip samples that fail to decompress, but make the skip
            # visible.
            logger.warning(
                "train_compression_dict: skipping undecompressable sample "
                "for step '%s' (dict_id=%s)",
                step,
                comp_dict_id,
                exc_info=True,
            )
            continue
        samples.append(content)
        total += len(content)
        if total >= budget:
            break
    return samples


class RecompressStats(NamedTuple):
    """What ``CompressionMixin.recompress_responses`` rewrote — and didn't.

    ``skipped_count`` is the rows the pass selected but did not rewrite:
    content that failed to decompress or recompress, plus rows a concurrent
    writer changed between the chunk's read and its guarded write. A
    nonzero value means the pass did not fully converge; the warnings in
    the log carry the per-row detail.
    """

    recompressed_count: int
    total_original_bytes: int
    total_compressed_bytes: int
    skipped_count: int = 0


class _OffDictionaryRow(NamedTuple):
    """One stored body read for recompression, with its store stamp.

    ``response_created_at`` and ``content_size_compressed`` are what the
    guarded UPDATE compares (see :class:`_PendingUpdate`).
    """

    request_id: int
    compressed: bytes
    dict_id: int | None
    response_created_at: datetime | None
    content_size_compressed: int | None


class _PendingUpdate(NamedTuple):
    """One recompressed row awaiting its guarded UPDATE.

    ``response_created_at`` and ``old_size`` are kept so the write can guard
    against a concurrent writer having changed the row between the chunk's
    read and its write. Every body write goes through ``store_response``,
    which restamps ``response_created_at`` in the same UPDATE, so the stamp
    moves whenever the body does — unlike ``retry_count``, which a retry
    bumps before its re-store, and which autowait re-stores and re-runs
    after an interrupted run never bump at all. The stamp has millisecond
    resolution; the compressed size covers two stores landing in the same
    millisecond, short of both also compressing to the same length.
    """

    request_id: int
    response_created_at: datetime | None
    old_size: int | None
    new_compressed: bytes
    original_size: int
    new_size: int


class _ChunkResult(NamedTuple):
    """One :func:`_recompress_chunk` pass: the rows it recompressed and
    the rows it could not.

    ``skipped`` counts only content that failed to decompress or recompress;
    rows lost to a concurrent writer are counted later, at the guarded
    UPDATE."""

    updates: list[_PendingUpdate]
    skipped: int


def _recompress_chunk(
    rows: Sequence[_OffDictionaryRow],
    old_dictionaries: dict[int, zstd.ZstdCompressionDict],
    dictionary: zstd.ZstdCompressionDict,
    step: str,
) -> _ChunkResult:
    """Decompress-and-recompress one chunk of rows; skip failures.

    Runs in a worker thread — up to ``chunk_size`` back-to-back zstd calls
    would stall co-resident workers if run on the loop, and compaction runs
    while the scrape is live. Returns the pending guarded UPDATEs plus the
    count of rows skipped because their content failed to process.
    """
    updates: list[_PendingUpdate] = []
    skipped = 0
    for request_id, compressed, old_dict_id, stamp, old_size in rows:
        old_dictionary = (
            old_dictionaries.get(old_dict_id)
            if old_dict_id is not None
            else None
        )
        try:
            # Decompress with the old dictionary, re-compress with the new
            # one.
            content = decompress(compressed, dictionary=old_dictionary)
            new_compressed = compress(content, dictionary=dictionary)
        except Exception:
            # Skip responses that fail to process, but make the skip
            # visible.
            skipped += 1
            logger.warning(
                "recompress_responses: skipping request %s for "
                "step '%s' (old dict_id=%s)",
                request_id,
                step,
                old_dict_id,
                exc_info=True,
            )
            continue
        updates.append(
            _PendingUpdate(
                request_id,
                stamp,
                old_size,
                new_compressed,
                len(content),
                len(new_compressed),
            )
        )
    return _ChunkResult(updates, skipped)
