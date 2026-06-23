"""Orchestration helpers: the parts that drive a scrape and stay transport-blind.

Where :class:`~jkent.driver.unified_driver.transport.Transport` owns *how* a
request runs, these stay storage-side — :class:`Compactor` compacts stored
responses, delegating every actual fetch to the transport. The run lifecycle
itself lives in :class:`~jkent.driver.unified_driver.run.ScrapeRun`; the
per-worker loop in :class:`~jkent.driver.unified_driver.worker.PoolWorker`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from jkent.driver.database_engine.compression import (
    DEFAULT_DICT_SIZE,
    get_compression_dict,
    recompress_responses,
    train_compression_dict,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


class Compactor:
    """Per-step compaction — counts in memory, trains once at the threshold.

    One Compactor is created for each scraper step that currently has fewer
    than ``threshold`` stored responses. The "no other work" is specifically
    *no DB polling*: it tracks the step's response count in memory (bumped by
    the run as each request for the step completes) instead of querying the
    database to decide when to act. On the call that reaches ``threshold`` it
    **owns** the one-shot job — train a zstd dictionary for the step from its
    stored responses and recompress them — then goes inert for the rest of the
    run.
    """

    THRESHOLD = 1000

    def __init__(
        self,
        step: str,
        session_factory: async_sessionmaker,
        *,
        db_lock: asyncio.Lock | None = None,
        threshold: int = THRESHOLD,
        count: int = 0,
        sample_limit: int | None = None,
        dict_size: int | None = None,
    ) -> None:
        self.step = step
        self.count = count
        self.threshold = threshold
        self._session_factory = session_factory
        self._db_lock = db_lock
        self._sample_limit = (
            sample_limit if sample_limit is not None else threshold
        )
        self._dict_size = dict_size
        self._done = False

    async def record_request(self) -> bool:
        """Count one completed request; train+recompress once at the threshold.

        Returns ``True`` on the single call that brings the count to
        ``threshold`` — having trained the dictionary and recompressed the
        step's responses on that call — and ``False`` every other time. Once
        it has fired it is inert: later calls neither count nor act.
        """
        if self._done:
            return False
        self.count += 1
        if self.count >= self.threshold:
            # Claim the one-shot job before the first await: record_request has
            # no await between the top guard and here, so setting _done now
            # makes check-and-claim atomic against the event loop. Concurrent
            # workers on the same step that cross the threshold during
            # _train_and_compact would otherwise each re-train and re-compress
            # (the shared db_lock serializes but does not dedupe them). With the
            # flag set first, only the first caller acts.
            self._done = True
            await self._train_and_compact()
            return True
        return False

    @property
    def done(self) -> bool:
        """Whether the train+recompress has already happened."""
        return self._done

    async def _train_and_compact(self) -> None:
        """Train a dictionary for the step and recompress its responses.

        Idempotent: if a dictionary already exists for the step the compaction
        is already done — skip the train+recompress rather than minting a
        redundant version. This covers a Compactor seeded at/above the
        threshold on a resumed run (its first ``record_request`` would
        otherwise re-train over a step that was already compacted).
        """
        existing = await get_compression_dict(
            self._session_factory, self.step, self._db_lock
        )
        if existing is not None:
            return
        dict_size = (
            self._dict_size
            if self._dict_size is not None
            else DEFAULT_DICT_SIZE
        )
        dict_id = await train_compression_dict(
            self._session_factory,
            self.step,
            sample_limit=self._sample_limit,
            dict_size=dict_size,
            db_lock=self._db_lock,
        )
        await recompress_responses(
            self._session_factory,
            self.step,
            dict_id=dict_id,
            db_lock=self._db_lock,
        )
