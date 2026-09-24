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
(:class:`~jkent.driver.database_engine.sql_manager._base.DictCache`), so the
steady-state store path never touches the database for its dictionary.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Final, NamedTuple

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

    from sqlalchemy.engine import Row

    from jkent.driver.database_engine.sql_manager import SQLManager

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
    db: SQLManager,
    step: str,
) -> DictEntry | None:
    """Get the latest compression dictionary for a step.

    Served from the manager's ``dict_cache`` after the first lookup —
    including the negative result — so the steady-state store path never
    touches the DB for its dictionary. :func:`train_compression_dict`
    invalidates the entry when it mints a new version.

    A pure read: uses a plain session (WAL readers don't take the write
    lock), so it does not take the manager's ``lock``.

    Args:
        db: The run database.
        step: The step method name.

    Returns:
        The latest :class:`DictEntry`, or None if no dictionary exists.
    """
    cache = db.dict_cache
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
    dict_obj = zstd.ZstdCompressionDict(row[1])
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
    manager's ``dict_cache``. A miss is not cached: a dangling
    dict_id is an error path, not a steady state. A pure read (see
    :func:`get_compression_dict` for why no lock is involved).

    Args:
        db: The run database.
        dict_id: The dictionary ID.

    Returns:
        The dictionary or None if not found.
    """
    cache = db.dict_cache
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
    dict_obj = zstd.ZstdCompressionDict(row[0])
    cache.by_id[dict_id] = dict_obj
    return dict_obj


async def compress_response(
    db: SQLManager,
    content: bytes,
    step: str,
    level: int = DEFAULT_COMPRESSION_LEVEL,
) -> CompressedContent:
    """Compress response content, using dictionary if available.

    Attempts to use a trained dictionary for the step if one exists.
    Falls back to standard compression if no dictionary is available.

    Args:
        db: The run database.
        content: The response content to compress.
        step: The step method name (for dictionary lookup).
        level: Compression level (1-22, default 3).

    Returns:
        :class:`CompressedContent`; its ``dict_id`` is None if no dictionary
        was used.
    """
    # Try to get a dictionary for this step
    dict_result = await get_compression_dict(db, step)

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
    db: SQLManager,
    compressed: bytes,
    dict_id: int | None,
) -> bytes:
    """Decompress response content, using dictionary if one was used.

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

    return decompress(compressed, dictionary=dictionary)


def _decompress_samples(
    rows: Sequence[Row[Any]],
    dictionaries: dict[int, zstd.ZstdCompressionDict],
    step: str,
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
                "for step '%s' (dict_id=%s)",
                step,
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
            dictionary. The compressed rows (up to ``sample_limit``) are
            fetched in full regardless.

    Returns:
        The ID of the newly created dictionary.

    Raises:
        ValueError: If no responses found for step or training fails.
    """
    compaction_started = time.monotonic()
    if max_sample_bytes is None:
        max_sample_bytes = TRAIN_SAMPLE_BYTE_FACTOR * dict_size
    # Sampling is a pure read; no need to hold the writer lock for it.
    async with db.session_factory() as session:
        result = await session.execute(
            select(
                Request.content_compressed,
                Request.compression_dict_id,
            )
            .where(*training_sample_clauses(step))
            .order_by(sa.func.random())
            .limit(sample_limit)
        )
        rows = result.all()

    if not rows:
        raise ValueError(f"No responses found for step '{step}'")

    # Resolve the dictionaries the samples were compressed with up front
    # (cached per-database), so the off-loop batch below needs no event-loop
    # access.
    dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
    for comp_dict_id in {row[1] for row in rows if row[1] is not None}:
        dict_obj = await get_dict_by_id(db, comp_dict_id)
        if dict_obj is not None:
            dictionaries[comp_dict_id] = dict_obj

    samples = await asyncio.to_thread(
        _decompress_samples,
        rows,
        dictionaries,
        step,
        max_sample_bytes,
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
        # Next version for this step; max() over no rows is NULL.
        version_result = await session.execute(
            select(sa.func.max(CompressionDict.version)).where(
                CompressionDict.step == step
            )
        )
        next_version = (version_result.scalar_one() or 0) + 1

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
    # step (including a cached "no dictionary yet").
    db.dict_cache.latest.pop(step, None)

    obs.instruments().compaction_duration.record(
        time.monotonic() - compaction_started,
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


def _recompress_chunk(
    rows: Sequence[Row[Any]],
    old_dictionaries: dict[int, zstd.ZstdCompressionDict],
    dictionary: zstd.ZstdCompressionDict,
    level: int,
    step: str,
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
    return updates, skipped


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
    level: int = DEFAULT_COMPRESSION_LEVEL,
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
        level: Compression level for re-compression (default 3).
        dict_id: Specific dictionary ID to use. If None, uses the latest.
        chunk_size: Rows per read-recompress-write batch.

    Returns:
        :class:`RecompressStats` — counts and byte totals for the rows
        actually rewritten, plus how many selected rows were skipped.

    Raises:
        ValueError: If no dictionary exists for this step or dict_id, or
            dict_id names a dictionary trained for another step.
    """
    compaction_started = time.monotonic()

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
            rows = result.all()

        if not rows:
            break
        last_id = rows[-1][0]

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
            level,
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

        if len(rows) < chunk_size:
            break

    obs.instruments().compaction_duration.record(
        time.monotonic() - compaction_started,
        {**obs.current_labels(), "step": step, "kind": "recompress"},
    )
    return RecompressStats(
        recompressed_count, total_original, total_compressed, skipped_count
    )
