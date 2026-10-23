"""CompressionMixin - zstd compression of stored bodies, and compaction.

The manager-facing half of
:mod:`jkent.driver.database_engine.compression`: everything that needs a
database — the dictionary cache lookups, the per-response
compress/decompress wrappers around them, and the two compaction entry
points that train a step's dictionary and move its stored bodies onto it.
The pure zstd calls and the off-loop batch helpers they hand work to stay
in that module.

Locking: the read paths (dictionary lookups, and therefore
:meth:`~CompressionMixin.compress_response` /
:meth:`~CompressionMixin.decompress_response`) use plain sessions —
WAL-mode readers never take SQLite's write lock, so they need no
serialization. Only the two mutating entry points,
:meth:`~CompressionMixin.train_compression_dict` and
:meth:`~CompressionMixin.recompress_responses`, write, and both go through
``_write_session`` so their transactions serialize with every other writer
on the database.

Dictionaries are cached on the manager
(:class:`~jkent.driver.database_engine.sql_manager._base.CompressionDictCache`),
so the steady-state store path never touches the database for its
dictionary. The cache holds them *precomputed* for
:data:`~jkent.driver.database_engine.compression.DEFAULT_COMPRESSION_LEVEL`;
see :func:`~jkent.driver.database_engine.compression._load_dictionary` for
why, and for what that costs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import sqlalchemy as sa
import zstandard as zstd
from sqlalchemy import select

from jkent import observability as obs
from jkent.driver.database_engine.compression import (
    DEFAULT_DICT_SIZE,
    MIN_TRAIN_SAMPLE_BYTES,
    NS_PER_SECOND,
    RECOMPRESS_CHUNK_SIZE,
    TRAIN_SAMPLE_BATCH_SIZE,
    TRAIN_SAMPLE_BYTE_FACTOR,
    CompressedContent,
    RecompressStats,
    _decompress_samples,
    _load_dictionary,
    _recompress_chunk,
    compress,
    decompress,
)
from jkent.driver.database_engine.database import execute_rowcount
from jkent.driver.database_engine.models import CompressionDict, Request
from jkent.driver.database_engine.sql_manager._base import (
    DictEntry,
    SQLManagerBase,
)
from jkent.driver.database_engine.stored_body import (
    stored_body_clauses,
    training_sample_clauses,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

logger = logging.getLogger(__name__)


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


class CompressionMixin(SQLManagerBase):
    """Compressing stored bodies, and compacting a step onto a dictionary."""

    async def get_compression_dict(self, step: str) -> DictEntry | None:
        """Get the latest compression dictionary for a step.

        Served from the manager's ``compression_dict_cache`` after the
        first lookup — including the negative result — so the steady-state
        store path never touches the DB for its dictionary, and never
        re-digests it either (see
        :func:`~jkent.driver.database_engine.compression._load_dictionary`).
        :meth:`train_compression_dict` invalidates the entry when it mints
        a new version.

        A pure read: uses a plain session (WAL readers don't take the write
        lock), so it does not take the manager's ``lock``.

        Args:
            step: The step method name.

        Returns:
            The latest :class:`DictEntry`, or None if no dictionary exists.
        """
        cache = self.compression_dict_cache
        if step in cache.latest:
            return cache.latest[step]

        async with self.session_factory() as session:
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
        self, dict_id: int
    ) -> zstd.ZstdCompressionDict | None:
        """Get a compression dictionary by its ID.

        Dictionary rows are immutable, so a hit caches forever in the
        manager's ``compression_dict_cache``, precomputed for compression
        (see
        :func:`~jkent.driver.database_engine.compression._load_dictionary`).
        A miss is not cached: a dangling dict_id is an error path, not a
        steady state. A pure read (see :meth:`get_compression_dict` for
        why no lock is involved).

        Args:
            dict_id: The dictionary ID.

        Returns:
            The dictionary or None if not found.
        """
        cache = self.compression_dict_cache
        cached = cache.by_id.get(dict_id)
        if cached is not None:
            return cached

        async with self.session_factory() as session:
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
        self, content: bytes, step: str
    ) -> CompressedContent:
        """Compress response content, using dictionary if available.

        Attempts to use a trained dictionary for the step if one exists.
        Falls back to standard compression if no dictionary is available.

        Args:
            content: The response content to compress.
            step: The step method name (for dictionary lookup).

        Returns:
            :class:`CompressedContent`; its ``dict_id`` is None if no
            dictionary was used.
        """
        # Try to get a dictionary for this step
        dict_result = await self.get_compression_dict(step)

        # Time the synchronous zstd call on both wall and on-loop CPU clocks:
        # the gap between the two, plus the loop-lag metric, is what tells us
        # whether this compression is blocking co-resident workers. compress()
        # has no await, so time.thread_time_ns() over it is this call's CPU
        # alone (no other task can be co-scheduled onto this thread mid-call).
        #
        # Everything either clock would otherwise enclose — the label snapshot,
        # the instrument lookup — is hoisted above the first reading, and the
        # two closing readings are taken back to back, so neither measurement
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
            inst.compression_ratio.record(
                len(compressed) / len(content), labels
            )
        return CompressedContent(compressed, dict_id)

    async def decompress_response(
        self, compressed: bytes, dict_id: int | None
    ) -> bytes:
        """Decompress response content, using dictionary if one was used.

        Timed on the same two clocks as :meth:`compress_response`, and
        for the same reason: this is the other synchronous zstd leaf on the
        loop. It is the cheaper of the pair, but it is the one a read-heavy
        consumer runs per response, so a regression here is worth seeing
        rather than inferring.

        Args:
            compressed: The compressed data.
            dict_id: The dictionary ID used for compression (or None).

        Returns:
            Decompressed data bytes.
        """
        dictionary = None
        if dict_id is not None:
            dictionary = await self.get_dict_by_id(dict_id)
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

    async def _collect_training_samples(
        self,
        step: str,
        sample_ids: Sequence[int],
        max_sample_bytes: int,
    ) -> list[bytes | bytearray | memoryview[int]]:
        """Fetch and decompress ``sample_ids`` in batches, up to the byte cap.

        The sample is chosen as a list of ids (see
        :meth:`train_compression_dict`), and the bodies behind it are read
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
            async with self.session_factory() as session:
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
                dict_obj = await self.get_dict_by_id(comp_dict_id)
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
        self,
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

        # Sample some random ids (just ids to limit memory use)

        async with self.session_factory() as session:
            result = await session.execute(
                select(Request.id)
                .where(*training_sample_clauses(step))
                .order_by(sa.func.random())
                .limit(sample_limit)
            )
            sample_ids = list(result.scalars().all())

        if not sample_ids:
            raise ValueError(f"No responses found for step '{step}'")

        samples = await self._collect_training_samples(
            step, sample_ids, max_sample_bytes
        )

        if not samples:
            raise ValueError(
                f"Could not decompress any samples for step '{step}'"
            )
        sample_bytes = sum(len(sample) for sample in samples)
        if sample_bytes < MIN_TRAIN_SAMPLE_BYTES:
            raise ValueError(
                f"too little sample data to train a dictionary for "
                f"step '{step}': {sample_bytes} bytes, need "
                f"{MIN_TRAIN_SAMPLE_BYTES}"
            )

        # Train the dictionary off-loop: over up to ``sample_limit`` bodies
        # this is a long synchronous call, and unlike per-response compression
        # it is not something the on-loop CPU metrics need to observe in place.
        # zstd raises ZstdError (e.g. too few/small samples); surface it as the
        # ValueError this function documents.
        try:
            dictionary_data = await asyncio.to_thread(
                zstd.train_dictionary, dict_size, samples
            )
        except zstd.ZstdError as exc:
            raise ValueError(
                f"Failed to train dictionary for step '{step}': {exc}"
            ) from exc

        async with self._write_session() as session:
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
        cache = self.compression_dict_cache
        cache.latest.pop(step, None)
        trained = _load_dictionary(dictionary_data.as_bytes())
        cache.by_id[dict_id] = trained
        cache.latest[step] = DictEntry(dict_id, trained)

        obs.instruments().compaction_duration.record(
            (time.monotonic_ns() - compaction_started) / NS_PER_SECOND,
            {**obs.current_labels(), "step": step, "kind": "train"},
        )
        return dict_id

    async def count_off_dictionary(self, step: str, dict_id: int) -> int:
        """How many rows :meth:`recompress_responses` would select for
        ``step``."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(sa.func.count())
                .select_from(Request)
                .where(*_off_dictionary(step, dict_id))
            )
            return result.scalar_one()

    async def _iter_off_dictionary_chunks(
        self,
        step: str,
        target_dict_id: int,
        chunk_size: int,
    ) -> AsyncIterator[list[tuple[int, bytes, int | None]]]:
        """Walk ``step``'s rows not yet on ``target_dict_id``, a page at
        a time.

        Keyset pagination on ``id``, not a streaming cursor, and deliberately
        so. The caller rewrites the rows this walk yields, and the rewrite is
        what takes them out of :func:`_off_dictionary` — a cursor held open
        across those writes would be iterating a predicate its own consumer is
        invalidating. Re-reading the predicate per page is also what makes an
        interrupted pass resumable: the next run simply finds the rows the
        last one never reached.

        Each page reads in its own short session, so nothing holds a read
        transaction open across the recompression threads or the write
        transactions between pages. Reads use a plain session: the caller's
        guarded UPDATE, not the transaction shape, is what protects against
        concurrent writers.

        The walk ends on the first short page. A page's rows may be fewer than
        ``chunk_size`` after the NOT NULL narrowing below, so that decision
        reads the page as fetched, not the yielded list, and cannot stall or
        truncate the walk.
        """
        last_id = 0
        while True:
            async with self.session_factory() as session:
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
                return
            last_id = page[-1][0]

            # ``content_compressed`` is NOT NULL across ``_off_dictionary``;
            # restating it narrows the rows to ``(int, bytes, int | None)`` for
            # the type checker.
            yield [
                (request_id, content, row_dict_id)
                for request_id, content, row_dict_id in page
                if content is not None
            ]

            if len(page) < chunk_size:
                return

    async def recompress_responses(
        self,
        step: str,
        dict_id: int | None = None,
        *,
        chunk_size: int = RECOMPRESS_CHUNK_SIZE,
    ) -> RecompressStats:
        """Re-compress responses using a dictionary for a step.

        Decompresses responses for the step and re-compresses them using
        the specified or latest trained dictionary. This can significantly
        improve compression ratios after training a new dictionary.

        Rows already compressed against the target dictionary are
        excluded, so a second pass over an already-compacted step is a
        no-op and an interrupted pass resumes where it stopped.

        Work proceeds in id-ordered chunks of ``chunk_size`` rows — read
        a page, recompress it, write it back in one guarded transaction,
        move on — so peak memory is one chunk's worth of bodies (not the
        whole step, old and new at once) and no single transaction holds
        the run's DB lock for the whole rewrite.
        :meth:`_iter_off_dictionary_chunks` is the read half of that
        walk.

        Args:
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
            async with self.session_factory() as session:
                dict_step = (
                    await session.execute(
                        select(CompressionDict.step).where(
                            CompressionDict.id == dict_id
                        )
                    )
                ).scalar()
            if dict_step is not None and dict_step != step:
                raise ValueError(
                    f"Dictionary {dict_id} was trained for step "
                    f"'{dict_step}', not '{step}'."
                )
            dictionary = await self.get_dict_by_id(dict_id)
            if dictionary is None:
                raise ValueError(f"No dictionary found with id {dict_id}.")
            target_dict_id = dict_id
        else:
            dict_result = await self.get_compression_dict(step)
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

        async for rows in self._iter_off_dictionary_chunks(
            step, target_dict_id, chunk_size
        ):
            # Resolve the chunk's old dictionaries on-loop (get_dict_by_id is
            # cached per-database, so a shared old_dict_id — the common case
            # for a single step — is fetched once), then hand the zstd work to
            # a thread.
            old_dictionaries: dict[int, zstd.ZstdCompressionDict] = {}
            for old_dict_id in {row[2] for row in rows if row[2] is not None}:
                dict_obj = await self.get_dict_by_id(old_dict_id)
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
            # concurrent writer touched between read and write is skipped
            # rather than clobbered with stale bytes. Counts/totals reflect
            # only rows actually written.
            if updates:
                async with self._write_session() as session:
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
                                    Request.compression_dict_id: (
                                        target_dict_id
                                    ),
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

        obs.instruments().compaction_duration.record(
            (time.monotonic_ns() - compaction_started) / NS_PER_SECOND,
            {**obs.current_labels(), "step": step, "kind": "recompress"},
        )
        return RecompressStats(
            recompressed_count, total_original, total_compressed, skipped_count
        )
