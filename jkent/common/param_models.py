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

from pydantic import BaseModel

from jkent.common.speculative import Speculative


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
                continuation=self.parse_case,
            )

        # seed_params: [{"fetch_case": {"rid": {"min": 1, "soft_max": 1, "gap": 20}}}]
    """

    min: int
    soft_max: int = 0
    should_advance: bool = True
    gap: int = 10

    def seed_range(self) -> range:
        return range(self.min, self.soft_max)

    def from_int(self, n: int) -> SpeculativeRange:
        # model_copy preserves every other field and returns the same concrete
        # type, so subclasses (e.g. YearlySpeculativeRange) inherit this as-is.
        return self.model_copy(update={"min": n})

    def max_gap(self) -> int:
        return self.gap


class YearlySpeculativeRange(SpeculativeRange):
    """Speculative parameter for year-partitioned integer ID probing.

    Like ``SpeculativeRange`` but adds a ``year`` field for scrapers that
    partition IDs by year (e.g. docket numbers of the form ``2025-00123``).
    ``seed_range``/``from_int``/``max_gap`` are inherited unchanged —
    ``from_int`` uses ``model_copy``, so the ``year`` is carried through.

    Supply one template per year via ``seed_params``.

    Attributes:
        year: The calendar year for this partition.
        min: Starting integer ID (the floor, inclusive).
        soft_max: Exclusive upper bound of the initial seed range.
        should_advance: Whether to push the speculation ceiling past
            ``soft_max`` on success.
        gap: Max consecutive failures before stopping.

    Example::

        @entry(CaseData)
        def fetch_case(self, case_id: YearlySpeculativeRange) -> Request:
            return Request(
                request=HTTPRequestParams(
                    url=f"/cases/{case_id.year}/{case_id.min}"
                ),
                continuation=self.parse_case,
            )

        # seed_params: [
        #     {"fetch_case": {"case_id": {"year": 2024, "min": 1, "soft_max": 4000, "gap": 0}}},
        #     {"fetch_case": {"case_id": {"year": 2025, "min": 1, "soft_max": 1, "gap": 15}}},
        # ]
    """

    year: int
