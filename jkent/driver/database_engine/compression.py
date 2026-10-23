"""Zstd compression for stored run-database responses.

This module provides zstd compression and decompression for HTTP responses,
with support for per-step trained dictionaries for better compression
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
both hold the manager's ``lock`` so their transactions serialize with every
other writer on the database.

Dictionaries are cached on the manager
(:class:`~jkent.driver.database_engine.sql_manager._base.CompressionDictCache`),
so the steady-state store path never touches the database for its
dictionary. The cache holds them *precomputed* for
:data:`DEFAULT_COMPRESSION_LEVEL`: a ``ZstdCompressor`` built on a plain
dictionary re-digests it on every single compression, which is the larger
half of the cost on short bodies. Compression therefore happens at that
one level, and the level knob survives only on :func:`compress` itself,
for callers that bring their own dictionary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Final, NamedTuple

import sqlalchemy as sa
import zstandard as zstd
from sqlalchemy import select

from jkent import observability as obs
from jkent.contracts import require
from jkent.driver.database_engine.database import (
    execute_rowcount,
    write_session,
)
from jkent.driver.database_engine.models import (
    CompressionDict,
    Request,
)
from jkent.driver.database_engine.sql_manager._base import DictEntry
from jkent.driver.database_engine.stored_body import (
    stored_body_clauses,
    training_sample_clauses,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)

#: Nanoseconds per second. The zstd leaves are timed on the integer ``_ns``
#: clocks (no float rounding mid-measurement) and converted once, here, at
#: the point of record — the metrics themselves are documented in seconds.
NS_PER_SECOND: Final = 1_000_000_000

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

#: Rows per :func:`_collect_training_samples` batch. Bounds the compressed
#: bytes resident while the sample is gathered: only one batch's bodies are
#: held, rather than every sampled body at once.
TRAIN_SAMPLE_BATCH_SIZE: Final = 50

#: Fewest decompressed sample bytes training is attempted on. A dictionary
#: learned from less is worthless, and zstandard 0.25's trainer segfaults on
#: some such corpora (eight one-byte samples) instead of raising.
MIN_TRAIN_SAMPLE_BYTES: Final = 1024


class CompressedContent(NamedTuple):
    """:func:`compress_response` output: the compressed bytes and the id
    of the dictionary they were compressed with (None when no dictionary
    was available)."""

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
            :func:`get_compression_dict`/:func:`get_dict_by_id` pins (see
            the module docstring); every in-tree caller passes those
            dictionaries at :data:`DEFAULT_COMPRESSION_LEVEL` anyway.
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


async def get_compression_dict(
    db: SQLManager,
    step: str,
) -> DictEntry | None:
    """Get the latest compression dictionary for a step.

    Served from the manager's ``compression_dict_cache`` after the first
    lookup —
    including the negative result — so the steady-state store path never
    touches the DB for its dictionary, and never re-digests it either (see
    :func:`_load_dictionary`). :func:`train_compression_dict` invalidates
    the entry when it mints a new version.

    A pure read: uses a plain session (WAL readers don't take the write
    lock), so it does not take the manager's ``lock``.

    Args:
        db: The run database.
        step: The step method name.

    Returns:
        The latest :class:`DictEntry`, or None if no dictionary exists.
    """
    cache = db.compression_dict_cache
    if step in cache.latest:
        return cache.latest[step]

    async with db.session_factory() as session:
        result = await session.execute(
            select(CompressionDict.id, CompressionDict.dictionary_data)
            .where(CompressionDict.step == step)
            .order_by(CompressionDict.version.desc())
            .limit(1)
        )
        row = result.first()

    if row is None:
        cache.latest[step] = None
        return None
    dict_obj = _load_dictionary(row[1])
    cache.by_id[row[0]] = dict_obj
    entry = DictEntry(row[0], dict_obj)
    cache.latest[step] = entry
    return entry


async def get_dict_by_id(
    db: SQLManager,
    dict_id: int,
) -> zstd.ZstdCompressionDict | None:
    """Get a compression dictionary by its ID.

    Dictionary rows are immutable, so a hit caches forever in the
    manager's ``compression_dict_cache``, precomputed for compression (see
    :func:`_load_dictionary`). A miss is not cached: a dangling dict_id is
    an error path, not a steady state. A pure read (see
    :func:`get_compression_dict` for why no lock is involved).

    Args:
        db: The run database.
        dict_id: The dictionary ID.

    Returns:
        The dictionary or None if not found.
    """
    cache = db.compression_dict_cache
    cached = cache.by_id.get(dict_id)
    if cached is not None:
        return cached

    async with db.session_factory() as session:
        result = await session.execute(
            select(CompressionDict.dictionary_data).where(
                CompressionDict.id == dict_id
            )
        )
        row = result.first()

    if row is None:
        return None
    dict_obj = _load_dictionary(row[0])
    cache.by_id[dict_id] = dict_obj
    return dict_obj


async def compress_response(
    db: SQLManager,
    content: bytes,
    step: str,
) -> CompressedContent:
    """Compress response content, using dictionary if available.

    Attempts to use a trained dictionary for the step if one exists.
    Falls back to standard compression if no dictionary is available.

    Args:
        db: The run database.
        content: The response content to compress.
        step: The step method name (for dictionary lookup).

    Returns:
        :class:`CompressedContent`; its ``dict_id`` is None if no dictionary
        was used.
    """
    # Try to get a dictionary for this step
    dict_result = await get_compression_dict(db, step)

    # Time the synchronous zstd call on both wall and on-loop CPU clocks: the
    # gap between the two, plus the loop-lag metric, is what tells us whether
    # this compression is blocking co-resident workers. compress() has no
    # await, so time.thread_time_ns() over it is this call's CPU alone (no
    # other task can be co-scheduled onto this thread mid-call).
    #
    # Everything either clock would otherwise enclose — the label snapshot,
    # the instrument lookup — is hoisted above the first reading, and the two
    # closing readings are taken back to back, so neither measurement
    # includes the other's bookkeeping.
    labels = obs.current_labels()
    inst = obs.instruments()
    wall0 = time.monotonic_ns()
    cpu0 = time.thread_time_ns()
    if dict_result is not None:
        dict_id, dictionary = dict_result
        compressed = compress(content, dictionary=dictionary)
    else:
        dict_id = None
        compressed = compress(content)
    wall_ns = time.monotonic_ns() - wall0
    cpu_ns = time.thread_time_ns() - cpu0
    inst.compression_duration.record(
        wall_ns / NS_PER_SECOND, {**labels, "kind": "compress"}
    )
    inst.request_cpu.record(
        cpu_ns / NS_PER_SECOND, {**labels, "phase": obs.Phase.COMPRESS}
    )
    if content:
        inst.compression_ratio.record(len(compressed) / len(content), labels)
    return CompressedContent(compressed, dict_id)


async def decompress_response(
    db: SQLManager,
    compressed: bytes,
    dict_id: int | None,
) -> bytes:
    """Decompress response content, using dictionary if one was used.

    Timed on the same two clocks as :func:`compress_response`, and for the
    same reason: this is the other synchronous zstd leaf on the loop. It is
    the cheaper of the pair, but it is the one a read-heavy consumer runs
    per response, so a regression here is worth seeing rather than
    inferring.

    Args:
        db: The run database.
        compressed: The compressed data.
        dict_id: The dictionary ID used for compression (or None).

    Returns:
        Decompressed data bytes.
    """
    dictionary = None
    if dict_id is not None:
        dictionary = await get_dict_by_id(db, dict_id)
        if dictionary is None:
            raise ValueError(f"Dictionary {dict_id} not found in database")

    labels = obs.current_labels()
    inst = obs.instruments()
    wall0 = time.monotonic_ns()
    cpu0 = time.thread_time_ns()
    content = decompress(compressed, dictionary=dictionary)
    wall_ns = time.monotonic_ns() - wall0
    cpu_ns = time.thread_time_ns() - cpu0
    inst.compression_duration.record(
        wall_ns / NS_PER_SECOND, {**labels, "kind": "decompress"}
    )
    inst.request_cpu.record(
        cpu_ns / NS_PER_SECOND, {**labels, "phase": obs.Phase.DECOMPRESS}
    )
    return content


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


async def _collect_training_samples(
    db: SQLManager,
    step: str,
    sample_ids: Sequence[int],
    max_sample_bytes: int,
) -> list[bytes | bytearray | memoryview[int]]:
    """Fetch and decompress ``sample_ids`` in batches, up to the byte cap.

    The sample is chosen as a list of ids (see
    :func:`train_compression_dict`), and the bodies behind it are read
    :data:`TRAIN_SAMPLE_BATCH_SIZE` at a time. Only the current batch's
    compressed bytes are resident, so peak memory is that batch plus the
    decompressed samples kept so far — bounded by ``max_sample_bytes`` —
    rather than every sampled body at once, which on a site with large
    bodies is the difference between tens of megabytes and gigabytes.

    Each batch reads in its own short session: nothing here holds a read
    transaction open across the decompression threads.

    Batching also means the dictionaries a batch's rows were compressed
    with are resolved as that batch needs them. Sampling usually stops
    well short of ``sample_ids``, and the ids past the stop cost nothing.
    """
    samples: list[bytes | bytearray | memoryview[int]] = []
    collected = 0
    for start in range(0, len(sample_ids), TRAIN_SAMPLE_BATCH_SIZE):
        batch_ids = sample_ids[start : start + TRAIN_SAMPLE_BATCH_SIZE]
        async with db.session_factory() as session:
            result = await session.execute(
                select(
                    Request.content_compressed,
                    Request.compression_dict_id,
                ).where(Request.id.in_(batch_ids))
            )
            # ``content_compressed`` is NOT NULL across
            # ``training_sample_clauses``, which selected these ids.
            # Restating it narrows the fetched tuples to
            # ``(bytes, int | None)`` for the type checker, which is what
            # lets `_decompress_samples` take a precise row type instead
            # of ``Row[Any]``.
            rows = [
                (content, sample_dict_id)
                for content, sample_dict_id in result.tuples().all()
                if content is not None
            ]

        # Resolve this batch's dictionaries on-loop (get_dict_by_id is
        # cached per-database, so the single shared dict_id a step
        # normally has is fetched once for the whole walk), then hand the
        # zstd work to a thread.
        dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
        for comp_dict_id in {row[1] for row in rows if row[1] is not None}:
            dict_obj = await get_dict_by_id(db, comp_dict_id)
            if dict_obj is not None:
                dictionaries[comp_dict_id] = dict_obj

        batch = await asyncio.to_thread(
            _decompress_samples,
            rows,
            dictionaries,
            step,
            max_sample_bytes - collected,
        )
        samples.extend(batch)
        collected += sum(len(sample) for sample in batch)
        if collected >= max_sample_bytes:
            break
    return samples


async def train_compression_dict(
    db: SQLManager,
    step: str,
    sample_limit: int = 1000,
    dict_size: int = DEFAULT_DICT_SIZE,
    *,
    max_sample_bytes: int | None = None,
) -> int:
    """Train a zstd compression dictionary from stored responses.

    Samples responses for the given step, trains a zstd dictionary,
    and stores it as a new version in the compression_dicts table.

    Args:
        db: The run database.
        step: The step method name to train dictionary for.
        sample_limit: Maximum number of responses to sample (default 1000).
        dict_size: Size of dictionary to train (default 112640 bytes).
        max_sample_bytes: Cap on cumulative decompressed sample bytes held
            for training. Defaults to ``TRAIN_SAMPLE_BYTE_FACTOR *
            dict_size``, past which more data stops improving the
            dictionary. Collection stops at the cap, so the sampled bodies
            past it are never read.

    Returns:
        The ID of the newly created dictionary.

    Raises:
        ValueError: If no responses found for step or training fails.
    """
    compaction_started = time.monotonic_ns()
    if max_sample_bytes is None:
        max_sample_bytes = TRAIN_SAMPLE_BYTE_FACTOR * dict_size

    # Choose the sample as a list of ids, not as bodies: this is the only
    # query that has to touch every candidate row, and taking just the
    # ``id`` column keeps it to a few kilobytes however large the bodies
    # are. :func:`_collect_training_samples` then reads the bodies behind
    # these ids in batches, and stops at ``max_sample_bytes``.
    #
    # A pure read; no need to hold the writer lock for it.
    async with db.session_factory() as session:
        result = await session.execute(
            select(Request.id)
            .where(*training_sample_clauses(step))
            .order_by(sa.func.random())
            .limit(sample_limit)
        )
        sample_ids = list(result.scalars().all())

    if not sample_ids:
        raise ValueError(f"No responses found for step '{step}'")

    samples = await _collect_training_samples(
        db, step, sample_ids, max_sample_bytes
    )

    if not samples:
        raise ValueError(f"Could not decompress any samples for step '{step}'")
    sample_bytes = sum(len(sample) for sample in samples)
    if sample_bytes < MIN_TRAIN_SAMPLE_BYTES:
        raise ValueError(
            f"too little sample data to train a dictionary for step "
            f"'{step}': {sample_bytes} bytes, need {MIN_TRAIN_SAMPLE_BYTES}"
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
            f"Failed to train dictionary for step '{step}': {exc}"
        ) from exc

    async with write_session(db.session_factory, db.lock) as session:
        # Next version for this step. max() over no rows is NULL, so
        # coalesce in SQL: the stubs type the scalar as ``int`` either way,
        # and this makes that true rather than papering over a NULL with
        # ``or 0`` on a value the checker already believes is an ``int``.
        version_result = await session.execute(
            select(
                sa.func.coalesce(sa.func.max(CompressionDict.version), 0)
            ).where(CompressionDict.step == step)
        )
        next_version = version_result.scalar_one() + 1

        # Store the new dictionary
        new_dict = CompressionDict(
            step=step,
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
    # step (including a cached "no dictionary yet"). Drop the stale entry
    # before building the replacement, so a failure to load the dictionary
    # leaves the cache merely cold rather than wrong.
    #
    # Seeding the cache rather than only invalidating it saves the
    # immediately following recompression pass a read and a precompute:
    # the compactor trains and then recompresses against this very
    # dictionary, so the next lookup is guaranteed and its answer is
    # already in hand.
    cache = db.compression_dict_cache
    cache.latest.pop(step, None)
    trained = _load_dictionary(dictionary_data.as_bytes())
    cache.by_id[dict_id] = trained
    cache.latest[step] = DictEntry(dict_id, trained)

    obs.instruments().compaction_duration.record(
        (time.monotonic_ns() - compaction_started) / NS_PER_SECOND,
        {**obs.current_labels(), "step": step, "kind": "train"},
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


class _ChunkResult(NamedTuple):
    """One :func:`_recompress_chunk` pass: the rows it recompressed and
    the rows it could not.

    ``skipped`` counts only content that failed to decompress or recompress;
    rows lost to a concurrent writer are counted later, at the guarded
    UPDATE."""

    updates: list[_PendingUpdate]
    skipped: int


def _recompress_chunk(
    rows: Sequence[tuple[int, bytes, int | None]],
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
                compressed,
                new_compressed,
                len(content),
                len(new_compressed),
            )
        )
    return _ChunkResult(updates, skipped)


def _off_dictionary(
    step: str, dict_id: int
) -> tuple[sa.ColumnElement[bool], ...]:
    """``step``'s stored bodies, of any status, not yet on ``dict_id``.

    The NULL arm matters: a bare ``!=`` would drop the pre-dictionary rows,
    which are exactly the ones a first pass most needs to recompress.
    """
    return (
        *stored_body_clauses(step),
        sa.or_(
            Request.compression_dict_id.is_(None),
            Request.compression_dict_id != dict_id,
        ),
    )


async def count_off_dictionary(db: SQLManager, step: str, dict_id: int) -> int:
    """How many rows :func:`recompress_responses` would select for ``step``."""
    async with db.session_factory() as session:
        result = await session.execute(
            select(sa.func.count())
            .select_from(Request)
            .where(*_off_dictionary(step, dict_id))
        )
        return result.scalar_one()


async def recompress_responses(
    db: SQLManager,
    step: str,
    dict_id: int | None = None,
    *,
    chunk_size: int = RECOMPRESS_CHUNK_SIZE,
) -> RecompressStats:
    """Re-compress responses using a dictionary for a step.

    Decompresses responses for the step and re-compresses them using
    the specified or latest trained dictionary. This can significantly
    improve compression ratios after training a new dictionary.

    Rows already compressed against the target dictionary are excluded, so a
    second pass over an already-compacted step is a no-op and an interrupted
    pass resumes where it stopped.

    Work proceeds in id-ordered chunks of ``chunk_size`` rows — read a page,
    recompress it, write it back in one guarded transaction, move on — so
    peak memory is one chunk's worth of bodies (not the whole step,
    old and new at once) and no single transaction holds the run's DB lock
    for the whole rewrite.

    Args:
        db: The run database.
        step: The step method name.
        dict_id: Specific dictionary ID to use. If None, uses the latest.
        chunk_size: Rows per read-recompress-write batch.

    Returns:
        :class:`RecompressStats` — counts and byte totals for the rows
        actually rewritten, plus how many selected rows were skipped.

    Raises:
        ValueError: If no dictionary exists for this step or dict_id, or
            dict_id names a dictionary trained for another step.
    """
    compaction_started = time.monotonic_ns()

    # Get the dictionary to use
    if dict_id is not None:
        # Another step's dictionary would compress fine, and stamp every
        # row with an id this step's own compaction then treats as off.
        async with db.session_factory() as session:
            dict_step = (
                await session.execute(
                    select(CompressionDict.step).where(
                        CompressionDict.id == dict_id
                    )
                )
            ).scalar()
        if dict_step is not None and dict_step != step:
            raise ValueError(
                f"Dictionary {dict_id} was trained for step '{dict_step}', "
                f"not '{step}'."
            )
        dictionary = await get_dict_by_id(db, dict_id)
        if dictionary is None:
            raise ValueError(f"No dictionary found with id {dict_id}.")
        target_dict_id = dict_id
    else:
        dict_result = await get_compression_dict(db, step)
        if dict_result is None:
            raise ValueError(
                f"No dictionary found for step '{step}'. "
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
        async with db.session_factory() as session:
            result = await session.execute(
                select(
                    Request.id,
                    Request.content_compressed,
                    Request.compression_dict_id,
                )
                .where(
                    *_off_dictionary(step, target_dict_id),
                    Request.id > last_id,
                )
                .order_by(Request.id)
                .limit(chunk_size)
            )
            page = result.tuples().all()

        if not page:
            break
        last_id = page[-1][0]

        # ``content_compressed`` is NOT NULL across ``_off_dictionary``;
        # restating it narrows the rows to ``(int, bytes, int | None)`` for
        # the type checker. Pagination and the short-page exit below both
        # read ``page``, not this list, so narrowing cannot stall or
        # truncate the walk.
        rows = [
            (request_id, content, row_dict_id)
            for request_id, content, row_dict_id in page
            if content is not None
        ]

        # Resolve the chunk's old dictionaries on-loop (get_dict_by_id is
        # cached per-database, so a shared old_dict_id — the common case for
        # a single step — is fetched once), then hand the zstd work
        # to a thread.
        old_dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
        for old_dict_id in {row[2] for row in rows if row[2] is not None}:
            dict_obj = await get_dict_by_id(db, old_dict_id)
            if dict_obj is not None:
                old_dictionaries[old_dict_id] = dict_obj

        updates, chunk_skipped = await asyncio.to_thread(
            _recompress_chunk,
            rows,
            old_dictionaries,
            dictionary,
            step,
        )
        skipped_count += chunk_skipped

        # Persist the chunk in a single transaction. Each UPDATE guards on
        # the content we read (content_compressed unchanged), so a row a
        # concurrent writer touched between read and write is skipped rather
        # than clobbered with stale bytes. Counts/totals reflect only rows
        # actually written.
        if updates:
            async with write_session(db.session_factory, db.lock) as session:
                for update in updates:
                    written = await execute_rowcount(
                        session,
                        sa.update(Request)
                        .where(
                            Request.id == update.request_id,
                            Request.content_compressed
                            == update.old_compressed,
                        )
                        .values(
                            {
                                Request.content_compressed: (
                                    update.new_compressed
                                ),
                                Request.content_size_original: (
                                    update.original_size
                                ),
                                Request.content_size_compressed: (
                                    update.new_size
                                ),
                                Request.compression_dict_id: target_dict_id,
                            }
                        ),
                    )
                    if written == 0:
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

        if len(page) < chunk_size:
            break

    obs.instruments().compaction_duration.record(
        (time.monotonic_ns() - compaction_started) / NS_PER_SECOND,
        {**obs.current_labels(), "step": step, "kind": "recompress"},
    )
    return RecompressStats(
        recompressed_count, total_original, total_compressed, skipped_count
    )
