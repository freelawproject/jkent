"""Shared parameter models for scraper @entry functions.

These Pydantic BaseModel subclasses define common parameter types
that scrapers can use in their @entry-decorated entry points.

Example::

    from jkent.common.param_models import DateRange, SpeculativeRange

    @entry(Docket)
    def search_by_date(self, date_range: DateRange) -> Generator[...]:
        ...

    @entry(Docket)
    def fetch_by_id(self, rid: SpeculativeRange) -> Request:
        return Request(url=f"/docket/{rid.min}", ...)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Self

from pydantic import BaseModel, model_validator
from pydantic.json_schema import SkipJsonSchema

from jkent.common.speculative import (
    PersistedSpeculative,
    Speculative,
    persisted_key_ref,
    resolve_persisted_cursor,
)


class DateRange(BaseModel):
    """Date range with start and end bounds.

    Both bounds are inclusive. Used for filtering by date range
    in scraper entry points.

    Attributes:
        start: Start date (inclusive).
        end: End date (inclusive).
    """

    start: date
    end: date

    @model_validator(mode="after")
    def _end_not_before_start(self) -> DateRange:
        if self.end < self.start:
            raise ValueError(
                f"end ({self.end}) must be >= start ({self.start})"
            )
        return self


class SpeculativeRange(BaseModel, Speculative):
    """Speculative parameter for sequential integer ID probing.

    Subclasses the ``Speculative`` ABC.  Use as a parameter type
    on an ``@entry`` function to enable automatic speculation.

    ``seed_range()`` returns ``range(min, soft_max)`` — those IDs are
    enqueued immediately as speculative requests. If ``should_advance``
    is True, the driver continues opening new probes beyond ``soft_max``
    until ``gap`` consecutive failures accrue.

    Attributes:
        min: Starting integer ID (the floor, inclusive).
        soft_max: Exclusive upper bound of the initial seed range. IDs
            ``[min, soft_max)`` are always enqueued. Beyond that, probing
            only continues if ``should_advance`` is True.
        should_advance: Whether to push the speculation ceiling past
            ``soft_max`` on success. Set False for backfills of a known
            finite set of IDs.
        gap: Max consecutive failures beyond the highest success before
            speculation stops. Also the size of the initial advance
            window enqueued when ``should_advance`` is True. Set 0 to
            disable the advance window entirely.

    Example::

        @entry(CaseData)
        def fetch_case(self, rid: SpeculativeRange) -> Request:
            return Request(
                request=HTTPRequestParams(url=f"/case/{rid.min}"),
                step=self.parse_case,
            )

        # seed_params: [{"fetch_case": {"rid": {"min": 1, "soft_max": 2, "gap": 20}}}]
    """

    min: int = 1
    soft_max: int = 2
    should_advance: bool = True
    gap: int = 10

    @model_validator(mode="after")
    def _min_positive(self) -> SpeculativeRange:
        if self.min <= 0:
            raise ValueError(f"min ({self.min}) must be > 0")
        return self

    @model_validator(mode="after")
    def _soft_max_above_min(self) -> SpeculativeRange:
        if self.soft_max < self.min:
            raise ValueError(
                f"soft_max ({self.soft_max}) must be >= min ({self.min})"
            )
        return self

    def seed_range(self) -> range:
        """The integer IDs to seed immediately.

        Returns:
            ``range(min, soft_max)`` — ``min`` inclusive, ``soft_max``
            exclusive.
        """
        return range(self.min, self.soft_max)

    def from_int(self, n: int) -> Self:
        """Re-base this range on a single ID.

        Args:
            n: The ID the new range starts at.

        Returns:
            A copy with ``min`` set to *n* and ``soft_max`` raised to at
            least ``n + 1``; every other field is carried over.
        """
        # model_copy preserves every other field and returns the same concrete
        # type, so subclasses adding fields (e.g. juriscraper's
        # YearlySpeculativeRange/CourtRange) inherit this as-is.
        return self.model_copy(
            update={"min": n, "soft_max": max(self.soft_max, n + 1)}
        )

    def max_gap(self) -> int:
        """Consecutive failures tolerated past the highest success.

        Returns:
            The configured ``gap``.
        """
        return self.gap


class PersistedSpeculativeRange(SpeculativeRange, PersistedSpeculative):
    """A :class:`SpeculativeRange` that can be seeded from a stored cursor.

    Any ordinary input validates exactly like ``SpeculativeRange``. A
    ``"[key]"`` string instead resolves against the active
    :func:`~jkent.common.speculative.persisted_cursor_store`, and the
    template remembers ``key`` as :attr:`persist_key`. Subclasses parse out
    to their own type, so a subclass adding fields (a court, a year) needs
    those fields in the stored value.

    After the run, :meth:`resume_after` gives the cursor the next run should
    start from, and :meth:`stored_value` what to write for it: ``min`` and ``soft_max`` both at the ID past the highest
    success, so the next run seeds nothing and opens its advance window
    there. A run with no successes leaves ``min`` where it was.

    Example::

        # store: {"cal-dockets": {"min": 1200, "soft_max": 1200, "gap": 20}}
        # seed_params: [{"fetch_docket": {"rid": "[cal-dockets]"}}]
        # highest success 1247 -> resume_after(1247).min == 1248

    Attributes:
        persist_key: The cursor key this range was seeded from, or None for
            a literal seed. Not part of the entry's parameter schema, and
            not part of the value written back to the store.
    """

    persist_key: SkipJsonSchema[str | None] = None

    @model_validator(mode="before")
    def _resolve_persisted(cls, data: Any) -> Any:
        key = persisted_key_ref(data)
        if key is None:
            return data
        return {**resolve_persisted_cursor(key), "persist_key": key}

    def resume_after(self, highest_successful_id: int | None) -> Self:
        """The cursor the next run should start from.

        Args:
            highest_successful_id: This run's highest success, or None.

        Returns:
            A copy with ``min`` and ``soft_max`` at
            ``max(min, highest_successful_id + 1)``; every other field,
            ``persist_key`` included, is carried over.
        """
        start = (
            self.min
            if highest_successful_id is None
            else max(self.min, highest_successful_id + 1)
        )
        return self.model_copy(update={"min": start, "soft_max": start})

    def stored_value(self) -> dict[str, Any]:
        """The value to store under ``persist_key``.

        Returns:
            Every field but ``persist_key``, JSON-compatible.
        """
        return self.model_dump(mode="json", exclude={"persist_key"})
