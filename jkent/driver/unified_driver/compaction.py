"""Per-step response compaction.

:class:`Compactor` trains a zstd dictionary for a step once it has enough
stored bodies and recompresses them; :class:`Compactors` is the run's
registry of them, built at ``open``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from jkent.driver.database_engine.compression import (
    DEFAULT_DICT_SIZE,
    count_off_dictionary,
    get_compression_dict,
    recompress_responses,
    train_compression_dict,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from jkent.data_types import BaseScraper
    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)


class Compactor:
    """Per-step compaction — counts in memory, trains once at the threshold.

    One Compactor is created for each scraper step that currently has fewer
    than ``threshold`` stored responses and no dictionary. It tracks the
    step's response count in memory (:meth:`record_request`) instead of
    polling the database. On the call that reaches ``threshold`` it **owns**
    the one-shot job — train a zstd dictionary for the step from its stored
    responses and recompress them — then goes inert for the rest of the run.
    """

    THRESHOLD = 1000

    def __init__(
        self,
        step: str,
        db: SQLManager,
        *,
        threshold: int = THRESHOLD,
        count: int = 0,
        sample_limit: int | None = None,
        dict_size: int | None = None,
    ) -> None:
        self.step = step
        self.count = count
        self.threshold = threshold
        self._db = db
        self._sample_limit = (
            sample_limit if sample_limit is not None else threshold
        )
        self._dict_size = dict_size
        self._done = False

    @classmethod
    async def for_step(cls, step: str, db: SQLManager) -> Compactor | None:
        """The compactor ``step`` needs at run start, or None if it needs none.

        A step that already has a dictionary needs none; any of its bodies
        still off that dictionary — a pass interrupted between chunks, or a
        body stored behind the recompress cursor — are recompressed now. A
        step whose resolved-response count is already at/over
        :attr:`THRESHOLD` is compacted now and needs none after. Any other
        step gets a compactor seeded with its current count.
        """
        existing = await get_compression_dict(db, step)
        if existing is not None:
            remaining = await count_off_dictionary(db, step, existing.dict_id)
            if remaining:
                logger.info(
                    "Resuming compaction of step %r: %d rows off its "
                    "dictionary",
                    step,
                    remaining,
                )
                await recompress_responses(db, step, dict_id=existing.dict_id)
            return None
        compactor = cls(
            step,
            db,
            threshold=cls.THRESHOLD,
            count=await db.resolved_response_count(step),
        )
        if compactor.count >= compactor.threshold:
            await compactor._train_and_compact()
            return None
        return compactor

    async def record_request(self) -> bool:
        """Count one completed request; train+recompress once at the threshold.

        The worker calls this once per request of the step that completed
        with a stored response — not for archive requests, and not for
        failures.

        Returns ``True`` on the single call that brings the count to
        ``threshold`` — having trained the dictionary and recompressed the
        step's responses on that call — and ``False`` every other time,
        including that call when the step cannot be trained on. Once it has
        fired it is inert: later calls neither count nor act.
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
            # (the shared db lock serializes but does not dedupe them). With the
            # flag set first, only the first caller acts.
            self._done = True
            return await self._train_and_compact()
        return False

    @property
    def done(self) -> bool:
        """Whether the train+recompress has already happened."""
        return self._done

    async def _train_and_compact(self) -> bool:
        """Train a dictionary for the step and recompress its responses.

        Returns whether it did. A step with a dictionary already is left
        alone. A step zstd cannot train on (no stored bodies, or a corpus too
        small or uniform) is logged and left uncompressed: compaction is an
        optimization, and its failure must reach neither the worker nor the
        run's open. The next open tries again.
        """
        existing = await get_compression_dict(self._db, self.step)
        if existing is not None:
            return False
        dict_size = (
            self._dict_size
            if self._dict_size is not None
            else DEFAULT_DICT_SIZE
        )
        try:
            dict_id = await train_compression_dict(
                self._db,
                self.step,
                sample_limit=self._sample_limit,
                dict_size=dict_size,
            )
        except ValueError as exc:
            logger.warning(
                "Step %r left uncompacted: %s", self.step, exc, exc_info=True
            )
            return False
        await recompress_responses(self._db, self.step, dict_id=dict_id)
        return True


class Compactors:
    """The run's per-step compactor registry.

    Built once at ``open`` from the scraper's steps (:meth:`for_scraper`);
    the worker reports each completed request to :meth:`record`, which is a
    no-op for a step that needs no compaction. An empty registry is the null
    object a worker gets when constructed without a run.
    """

    def __init__(self, by_step: Mapping[str, Compactor] | None = None) -> None:
        self._by_step: dict[str, Compactor] = dict(by_step or {})

    @classmethod
    async def for_scraper(
        cls, scraper: BaseScraper[Any], db: SQLManager
    ) -> Compactors:
        """One :meth:`Compactor.for_step` per scraper step that needs one."""
        by_step: dict[str, Compactor] = {}
        for step_info in scraper.list_steps():
            compactor = await Compactor.for_step(step_info.name, db)
            if compactor is not None:
                by_step[step_info.name] = compactor
        return cls(by_step)

    def for_step(self, step: str) -> Compactor | None:
        """The compactor tracking ``step``, or None if it needs no compaction."""
        return self._by_step.get(step)

    async def record(self, step: str) -> None:
        """Count one completed request toward ``step``'s compactor, if any."""
        compactor = self._by_step.get(step)
        if compactor is not None:
            await compactor.record_request()

    def __iter__(self) -> Iterator[str]:
        return iter(self._by_step)

    def __len__(self) -> int:
        return len(self._by_step)
