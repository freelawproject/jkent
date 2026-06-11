"""Zstd compression for stored run-database responses.

This module provides zstd compression and decompression for HTTP responses,
with support for per-continuation trained dictionaries for better compression
ratios on similar content.

Compression is done with zstd (Zstandard) which offers excellent compression
ratios and fast decompression speeds. Dictionary-based compression can
significantly improve compression of similar content (like HTML from the
same website).

Locking: the read paths (dictionary lookups, and therefore
:func:`compress_response`/:func:`decompress_response`) use plain sessions —
WAL-mode readers never take SQLite's write lock, so they need no
serialization. Only the two mutating entry points,
:func:`train_compression_dict` and :func:`recompress_responses`, write, and
both require the run's shared ``db_lock`` so their transactions serialize
with every other writer on the database.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Final, NamedTuple, cast
from weakref import WeakKeyDictionary

import sqlalchemy as sa
import zstandard as zstd
from sqlalchemy import select

from jkent import observability as obs
from jkent.contracts import require
from jkent.driver.database_engine.database import write_session
from jkent.driver.database_engine.models import (
    CompressionDict,
    Request,
    RequestStatus,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import CursorResult, Row
    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)

#: Default compression level (3 is a good balance of speed/ratio).
DEFAULT_COMPRESSION_LEVEL: Final = 3

#: Default dictionary size (112640 bytes = 110KB, zstd's default).
DEFAULT_DICT_SIZE: Final = 112640

#: Rows per :func:`recompress_responses` batch. Bounds both the bytes held in
#: memory (old + new compressed content per row) and how long any single
#: write transaction holds the run's DB lock during compaction.
RECOMPRESS_CHUNK_SIZE: Final = 500

#: Cap on cumulative decompressed sample bytes fed to dictionary training,
#: as a multiple of the dictionary size. zstd's guidance is that ~100x the
#: dictionary's size in training data saturates what training can use; past
#: that, extra samples only cost memory (they are all held at once).
TRAIN_SAMPLE_BYTE_FACTOR: Final = 100


class DictEntry(NamedTuple):
    """A trained dictionary: its ``compression_dicts`` row id and the
    loaded zstd object."""

    dict_id: int
    dictionary: zstd.ZstdCompressionDict


class CompressedContent(NamedTuple):
    """:func:`compress_response` output: the compressed bytes and the id
    of the dictionary they were compressed with (None when no dictionary
    was available)."""

    data: bytes
    dict_id: int | None


class _DictCache:
    """Per-database in-memory cache of compression dictionaries.

    Without it, every stored response after compaction re-reads the ~110KB
    dictionary blob and rebuilds a ``ZstdCompressionDict`` from it.

    Dictionary rows are immutable once written — retraining mints a new
    row/version — so ``by_id`` entries never invalidate. ``latest`` maps a
    continuation to its newest :class:`DictEntry` and also caches the
    negative "no dictionary yet" result (the pre-compaction common case);
    both are dropped by :func:`train_compression_dict`, the only in-process
    writer. A run database has a single writing process, so no external
    training can slip past the negative cache.
    """

    __slots__ = ("by_id", "latest")

    def __init__(self) -> None:
        self.by_id: dict[int, zstd.ZstdCompressionDict] = {}
        self.latest: dict[str, DictEntry | None] = {}


# Keyed weakly by the factory's bound engine: one cache per open database,
# dropped with the engine. The engine rather than the factory, because two
# factories over one engine must share the cache — otherwise a train through
# one could leave the other's cached negative ("no dictionary yet") alive
# for the rest of the run. Two databases can reuse the same dict_id with
# different bytes, so a process-global id-keyed cache would still be wrong.
_dict_caches: WeakKeyDictionary[object, _DictCache] = WeakKeyDictionary()


def _cache_for(session_factory: async_sessionmaker) -> _DictCache:
    # A factory built without a bind (a shape init_database never produces)
    # keys by the factory itself rather than failing.
    key = session_factory.kw.get("bind") or session_factory
    cache = _dict_caches.get(key)
    if cache is None:
        cache = _dict_caches[key] = _DictCache()
    return cache


@require(
    lambda level: 1 <= level <= 22,
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
        level: Compression level (1-22, default 3).
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


async def get_compression_dict(
    session_factory: async_sessionmaker,
    continuation: str,
) -> DictEntry | None:
    """Get the latest compression dictionary for a continuation.

    Served from the per-database :class:`_DictCache` after the first lookup —
    including the negative result — so the steady-state store path never
    touches the DB for its dictionary. :func:`train_compression_dict`
    invalidates the entry when it mints a new version.

    A pure read: uses a plain session (WAL readers don't take the write
    lock), so it neither needs nor accepts the run's ``db_lock``.

    Args:
        session_factory: Async session factory.
        continuation: The continuation method name.

    Returns:
        The latest :class:`DictEntry`, or None if no dictionary exists.
    """
    cache = _cache_for(session_factory)
    if continuation in cache.latest:
        return cache.latest[continuation]

    async with session_factory() as session:
        result = await session.execute(
            select(CompressionDict.id, CompressionDict.dictionary_data)
            .where(CompressionDict.continuation == continuation)
            .order_by(CompressionDict.version.desc())
            .limit(1)
        )
        row = result.first()

    if row is None:
        cache.latest[continuation] = None
        return None
    dict_obj = zstd.ZstdCompressionDict(row[1])
    cache.by_id[row[0]] = dict_obj
    entry = DictEntry(row[0], dict_obj)
    cache.latest[continuation] = entry
    return entry


async def get_dict_by_id(
    session_factory: async_sessionmaker,
    dict_id: int,
) -> zstd.ZstdCompressionDict | None:
    """Get a compression dictionary by its ID.

    Dictionary rows are immutable, so a hit caches forever in the
    per-database :class:`_DictCache`. A miss is not cached: a dangling
    dict_id is an error path, not a steady state. A pure read (see
    :func:`get_compression_dict` for why no lock is involved).

    Args:
        session_factory: Async session factory.
        dict_id: The dictionary ID.

    Returns:
        The dictionary or None if not found.
    """
    cache = _cache_for(session_factory)
    cached = cache.by_id.get(dict_id)
    if cached is not None:
        return cached

    async with session_factory() as session:
        result = await session.execute(
            select(CompressionDict.dictionary_data).where(
                CompressionDict.id == dict_id
            )
        )
        row = result.first()

    if row is None:
        return None
    dict_obj = zstd.ZstdCompressionDict(row[0])
    cache.by_id[dict_id] = dict_obj
    return dict_obj


async def compress_response(
    session_factory: async_sessionmaker,
    content: bytes,
    continuation: str,
    level: int = DEFAULT_COMPRESSION_LEVEL,
) -> CompressedContent:
    """Compress response content, using dictionary if available.

    Attempts to use a trained dictionary for the continuation if one exists.
    Falls back to standard compression if no dictionary is available.

    Args:
        session_factory: Async session factory.
        content: The response content to compress.
        continuation: The continuation method name (for dictionary lookup).
        level: Compression level (1-22, default 3).

    Returns:
        :class:`CompressedContent`; its ``dict_id`` is None if no dictionary
        was used.
    """
    # Try to get a dictionary for this continuation
    dict_result = await get_compression_dict(session_factory, continuation)

    # Time the synchronous zstd call on both wall and on-loop CPU clocks: the
    # gap between the two, plus the loop-lag metric, is what tells us whether
    # this compression is blocking co-resident workers. compress() has no
    # await, so time.thread_time() over it is this call's CPU alone (no other
    # task can be co-scheduled onto this thread mid-call).
    labels = obs.current_labels()
    wall0 = time.monotonic()
    cpu0 = time.thread_time()
    if dict_result is not None:
        dict_id, dictionary = dict_result
        compressed = compress(content, level=level, dictionary=dictionary)
    else:
        dict_id = None
        compressed = compress(content, level=level)
    inst = obs.instruments()
    inst.compression_duration.record(
        time.monotonic() - wall0, {**labels, "kind": "compress"}
    )
    inst.request_cpu.record(
        time.thread_time() - cpu0, {**labels, "phase": obs.Phase.COMPRESS}
    )
    if content:
        inst.compression_ratio.record(len(compressed) / len(content), labels)
    return CompressedContent(compressed, dict_id)


async def decompress_response(
    session_factory: async_sessionmaker,
    compressed: bytes,
    dict_id: int | None,
) -> bytes:
    """Decompress response content, using dictionary if one was used.

    Args:
        session_factory: Async session factory.
        compressed: The compressed data.
        dict_id: The dictionary ID used for compression (or None).

    Returns:
        Decompressed data bytes.
    """
    dictionary = None
    if dict_id is not None:
        dictionary = await get_dict_by_id(session_factory, dict_id)
        if dictionary is None:
            raise ValueError(f"Dictionary {dict_id} not found in database")

    return decompress(compressed, dictionary=dictionary)


def _decompress_samples(
    rows: Sequence[Row[Any]],
    dictionaries: dict[int, zstd.ZstdCompressionDict],
    continuation: str,
    max_sample_bytes: int,
) -> list[bytes | bytearray | memoryview[int]]:
    """Decompress training samples; skip failures; stop at the byte cap.

    Runs in a worker thread — up to ``sample_limit`` back-to-back zstd calls
    would stall co-resident workers if run on the loop. A sample whose
    dictionary is missing or whose bytes fail to decompress is skipped with
    a warning. Collection stops once ``max_sample_bytes`` of decompressed
    content is gathered: past that point extra samples don't improve the
    dictionary, they only hold memory (all samples are resident at once).
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
                "for continuation '%s' (dict_id=%s)",
                continuation,
                comp_dict_id,
                exc_info=True,
            )
            continue
        samples.append(content)
        total += len(content)
        if total >= max_sample_bytes:
            break
    return samples


async def train_compression_dict(
    session_factory: async_sessionmaker,
    continuation: str,
    sample_limit: int = 1000,
    dict_size: int = DEFAULT_DICT_SIZE,
    *,
    db_lock: asyncio.Lock,
    max_sample_bytes: int | None = None,
) -> int:
    """Train a zstd compression dictionary from stored responses.

    Samples responses for the given continuation, trains a zstd dictionary,
    and stores it as a new version in the compression_dicts table.

    Args:
        session_factory: Async session factory.
        continuation: The continuation method name to train dictionary for.
        sample_limit: Maximum number of responses to sample (default 1000).
        dict_size: Size of dictionary to train (default 112640 bytes).
        db_lock: The run's shared writer lock. Required: a mutating entry
            point that ran outside it would race the run's own workers.
        max_sample_bytes: Cap on cumulative decompressed sample bytes held
            for training. Defaults to ``TRAIN_SAMPLE_BYTE_FACTOR *
            dict_size``, past which more data stops improving the
            dictionary — the cap keeps a large-bodied continuation from
            ballooning memory for no training benefit.

    Returns:
        The ID of the newly created dictionary.

    Raises:
        ValueError: If no responses found for continuation or training fails.
    """
    compaction_started = time.monotonic()
    if max_sample_bytes is None:
        max_sample_bytes = TRAIN_SAMPLE_BYTE_FACTOR * dict_size
    # Sampling is a pure read; no need to hold the writer lock for it.
    async with session_factory() as session:
        result = await session.execute(
            select(
                Request.content_compressed,
                Request.compression_dict_id,
            )
            .where(
                Request.continuation == continuation,
                # COMPLETED only: the worker stores a body on rows that
                # did not succeed too (a transient debug snapshot before a
                # retry, the observed error response on a persistent HTTP
                # failure, which stays there). Sampling those would train
                # a continuation's dictionary partly on error pages.
                # ``SQLManager.resolved_response_count`` applies the same
                # filter so the compactor threshold counts what this
                # samples.
                Request.status == RequestStatus.COMPLETED,
                Request.response_status_code.isnot(None),
                Request.content_compressed.isnot(None),
            )
            .order_by(sa.func.random())
            .limit(sample_limit)
        )
        rows = result.all()

    if not rows:
        raise ValueError(
            f"No responses found for continuation '{continuation}'"
        )

    # Resolve the dictionaries the samples were compressed with up front
    # (cached per-database), so the off-loop batch below needs no event-loop
    # access.
    dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
    for comp_dict_id in {row[1] for row in rows if row[1] is not None}:
        dict_obj = await get_dict_by_id(session_factory, comp_dict_id)
        if dict_obj is not None:
            dictionaries[comp_dict_id] = dict_obj

    samples = await asyncio.to_thread(
        _decompress_samples,
        rows,
        dictionaries,
        continuation,
        max_sample_bytes,
    )

    if not samples:
        raise ValueError(
            f"Could not decompress any samples for continuation '{continuation}'"
        )

    # Train the dictionary off-loop: over up to ``sample_limit`` bodies this
    # is a long synchronous call, and unlike per-response compression it is
    # not something the on-loop CPU metrics need to observe in place. zstd
    # raises ZstdError (e.g. too few/small samples); surface it as the
    # ValueError this function documents.
    try:
        dictionary_data = await asyncio.to_thread(
            zstd.train_dictionary, dict_size, samples
        )
    except zstd.ZstdError as exc:
        raise ValueError(
            f"Failed to train dictionary for continuation '{continuation}': "
            f"{exc}"
        ) from exc

    async with write_session(session_factory, db_lock) as session:
        # Next version for this continuation; max() over no rows is NULL.
        version_result = await session.execute(
            select(sa.func.max(CompressionDict.version)).where(
                CompressionDict.continuation == continuation
            )
        )
        next_version = (version_result.scalar_one() or 0) + 1

        # Store the new dictionary
        new_dict = CompressionDict(
            continuation=continuation,
            version=next_version,
            dictionary_data=dictionary_data.as_bytes(),
            sample_count=len(samples),
        )
        session.add(new_dict)
        await session.flush()
        dict_id = new_dict.id
        await session.commit()

    # Bookkeeping happens after the lock is released: the dictionary is
    # already committed, so a failure here must not make the caller believe
    # training failed (a retry would mint a redundant version), and the
    # metric record must not extend — or, if it raises, strand — the lock.
    #
    # The new version supersedes whatever the cache holds for this
    # continuation (including a cached "no dictionary yet").
    _cache_for(session_factory).latest.pop(continuation, None)

    obs.instruments().compaction_duration.record(
        time.monotonic() - compaction_started,
        {**obs.current_labels(), "step": continuation, "kind": "train"},
    )
    return dict_id


class RecompressStats(NamedTuple):
    """What :func:`recompress_responses` actually rewrote — and didn't.

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


class _PendingUpdate(NamedTuple):
    """One recompressed row awaiting its guarded UPDATE.

    ``old_compressed`` is kept so the write can guard against a concurrent
    writer having changed the row between the chunk's read and its write.
    The guard compares the blob itself, not ``compression_dict_id``: a
    rewrite that stored different content under the same dictionary (or
    none at all) would slip past an id-only guard and be clobbered with
    stale bytes here.
    """

    request_id: int
    old_compressed: bytes
    new_compressed: bytes
    original_size: int
    new_size: int


def _recompress_chunk(
    rows: Sequence[Row[Any]],
    old_dictionaries: dict[int, zstd.ZstdCompressionDict],
    dictionary: zstd.ZstdCompressionDict,
    level: int,
    continuation: str,
) -> tuple[list[_PendingUpdate], int]:
    """Decompress-and-recompress one chunk of rows; skip failures.

    Runs in a worker thread — up to ``chunk_size`` back-to-back zstd calls
    would stall co-resident workers if run on the loop, and compaction runs
    while the scrape is live. Returns the pending guarded UPDATEs plus the
    count of rows skipped because their content failed to process.
    """
    updates: list[_PendingUpdate] = []
    skipped = 0
    for request_id, compressed, old_dict_id in rows:
        old_dictionary = (
            old_dictionaries.get(old_dict_id)
            if old_dict_id is not None
            else None
        )
        try:
            # Decompress with the old dictionary, re-compress with the new
            # one.
            content = decompress(compressed, dictionary=old_dictionary)
            new_compressed = compress(
                content, level=level, dictionary=dictionary
            )
        except Exception:
            # Skip responses that fail to process, but make the skip
            # visible.
            skipped += 1
            logger.warning(
                "recompress_responses: skipping request %s for "
                "continuation '%s' (old dict_id=%s)",
                request_id,
                continuation,
                old_dict_id,
                exc_info=True,
            )
            continue
        updates.append(
            _PendingUpdate(
                request_id,
                compressed,
                new_compressed,
                len(content),
                len(new_compressed),
            )
        )
    return updates, skipped


async def recompress_responses(
    session_factory: async_sessionmaker,
    continuation: str,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    dict_id: int | None = None,
    *,
    db_lock: asyncio.Lock,
    chunk_size: int = RECOMPRESS_CHUNK_SIZE,
) -> RecompressStats:
    """Re-compress responses using a dictionary for a continuation.

    Decompresses responses for the continuation and re-compresses them using
    the specified or latest trained dictionary. This can significantly
    improve compression ratios after training a new dictionary.

    Rows already compressed against the target dictionary are excluded, so a
    second pass over an already-compacted continuation (e.g. a resumed run
    re-crossing the threshold) is a no-op rather than a silent full rewrite
    of the table through the WAL.

    Work proceeds in id-ordered chunks of ``chunk_size`` rows — read a page,
    recompress it, write it back in one guarded transaction, move on — so
    peak memory is one chunk's worth of bodies (not the whole continuation,
    old and new at once) and no single transaction holds the run's DB lock
    for the whole rewrite. The id ordering also makes an interrupted pass
    deterministic and resumable: rows already moved to the target dictionary
    are simply skipped next time.

    Args:
        session_factory: Async session factory.
        continuation: The continuation method name.
        level: Compression level for re-compression (default 3).
        dict_id: Specific dictionary ID to use. If None, uses the latest.
        db_lock: The run's shared writer lock. Required: a mutating entry
            point that ran outside it would race the run's own workers.
        chunk_size: Rows per read-recompress-write batch.

    Returns:
        :class:`RecompressStats` — counts and byte totals for the rows
        actually rewritten, plus how many selected rows were skipped.

    Raises:
        ValueError: If no dictionary exists for this continuation or dict_id.
    """
    compaction_started = time.monotonic()

    # Get the dictionary to use
    if dict_id is not None:
        dictionary = await get_dict_by_id(session_factory, dict_id)
        if dictionary is None:
            raise ValueError(f"No dictionary found with id {dict_id}.")
        target_dict_id = dict_id
    else:
        dict_result = await get_compression_dict(session_factory, continuation)
        if dict_result is None:
            raise ValueError(
                f"No dictionary found for continuation '{continuation}'. "
                "Train a dictionary first using train_compression_dict()."
            )
        target_dict_id, dictionary = dict_result

    recompressed_count = 0
    total_original = 0
    total_compressed = 0
    skipped_count = 0
    last_id = 0

    while True:
        # Reads use a plain session: the guarded UPDATE below (not the
        # transaction shape) is what protects against concurrent writers.
        async with session_factory() as session:
            result = await session.execute(
                select(
                    Request.id,
                    Request.content_compressed,
                    Request.compression_dict_id,
                )
                .where(
                    Request.continuation == continuation,
                    Request.response_status_code.isnot(None),
                    Request.content_compressed.isnot(None),
                    # Skip rows already on the target dictionary. The NULL
                    # arm matters: a bare ``!=`` would drop the
                    # pre-dictionary rows, which are exactly the ones a
                    # first pass most needs to recompress.
                    sa.or_(
                        Request.compression_dict_id.is_(None),
                        Request.compression_dict_id != target_dict_id,
                    ),
                    Request.id > last_id,
                )
                .order_by(Request.id)
                .limit(chunk_size)
            )
            rows = result.all()

        if not rows:
            break
        last_id = rows[-1][0]

        # Resolve the chunk's old dictionaries on-loop (get_dict_by_id is
        # cached per-database, so a shared old_dict_id — the common case for
        # a single continuation — is fetched once), then hand the zstd work
        # to a thread.
        old_dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
        for old_dict_id in {row[2] for row in rows if row[2] is not None}:
            dict_obj = await get_dict_by_id(session_factory, old_dict_id)
            if dict_obj is not None:
                old_dictionaries[old_dict_id] = dict_obj

        updates, chunk_skipped = await asyncio.to_thread(
            _recompress_chunk,
            rows,
            old_dictionaries,
            dictionary,
            level,
            continuation,
        )
        skipped_count += chunk_skipped

        # Persist the chunk in a single transaction. Each UPDATE guards on
        # the content we read (content_compressed unchanged), so a row a
        # concurrent writer touched between read and write is skipped rather
        # than clobbered with stale bytes. Counts/totals reflect only rows
        # actually written.
        if updates:
            async with write_session(session_factory, db_lock) as session:
                for update in updates:
                    # UPDATE returns a CursorResult; the typed execute()
                    # overload only promises Result, which has no rowcount.
                    result = cast(
                        "CursorResult[Any]",
                        await session.execute(
                            sa.update(Request)
                            .where(
                                Request.id == update.request_id,
                                Request.content_compressed
                                == update.old_compressed,
                            )
                            .values(
                                content_compressed=update.new_compressed,
                                content_size_original=update.original_size,
                                content_size_compressed=update.new_size,
                                compression_dict_id=target_dict_id,
                            )
                        ),
                    )
                    if result.rowcount == 0:
                        skipped_count += 1
                        logger.warning(
                            "recompress_responses: request %s changed "
                            "concurrently; skipping to avoid overwriting "
                            "newer content",
                            update.request_id,
                        )
                        continue
                    recompressed_count += 1
                    total_original += update.original_size
                    total_compressed += update.new_size
                await session.commit()

        if len(rows) < chunk_size:
            break

    obs.instruments().compaction_duration.record(
        time.monotonic() - compaction_started,
        {**obs.current_labels(), "step": continuation, "kind": "recompress"},
    )
    return RecompressStats(
        recompressed_count, total_original, total_compressed, skipped_count
    )
