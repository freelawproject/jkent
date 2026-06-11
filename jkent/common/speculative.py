"""Speculative base class for entry point parameter models.

Defines an abstract base class that Pydantic BaseModel classes can
inherit to power the speculation system. Instead of jkent owning the
speculation configuration classes, scraper authors define their own
Pydantic models with speculation semantics.

The base class has four members:

- ``should_advance``: attribute (field or property) controlling whether
  the driver attempts to push the speculation ceiling past ``seed_range``
  on success.
- ``seed_range()``: returns the integer IDs to seed immediately. Every
  resulting request is enqueued as ``is_speculative=True`` with its own
  ``speculation_id``.
- ``from_int(n)``: builds a new template for a specific integer ID,
  preserving all non-range configuration.
- ``max_gap()``: the consecutive-failure ceiling (and initial window
  size when ``should_advance`` is True).

Example::

    class DocketId(BaseModel, Speculative):
        year: int
        min: int
        soft_max: int = 0
        should_advance: bool = True
        gap: int = 3

        def seed_range(self) -> range:
            return range(self.min, self.soft_max)

        def from_int(self, n: int) -> DocketId:
            return DocketId(
                year=self.year, min=n, soft_max=self.soft_max,
                should_advance=self.should_advance, gap=self.gap,
            )

        def max_gap(self) -> int:
            return self.gap
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Self


class Speculative(ABC):
    """Abstract base for entry-point parameter models that support speculation.

    When an ``@entry`` function has a parameter whose type subclasses this
    ABC, the driver detects it automatically and runs the speculation
    loop: seeding ``seed_range()`` as speculative requests, optionally
    opening an adaptive advance window, tracking success/failure, and
    stopping once ``max_gap()`` consecutive failures accrue beyond the
    highest observed success.

    Attributes:
        should_advance: Whether the driver should extend the speculation
            ceiling past ``seed_range().stop`` on success. When False the
            driver enqueues ``seed_range()`` only and stops.

    Methods:
        seed_range: Returns the integer IDs to seed immediately. A
            ``range`` (``start`` inclusive, ``stop`` exclusive, step=1).
            An empty range is valid and means "no initial seeding — rely
            entirely on the advance window".
        from_int: Creates a new instance for integer ID *n*, preserving
            all other fields (year, config, etc.) from the template.
        max_gap: Maximum consecutive failures beyond the highest success
            before the speculation stops. Also controls the size of the
            initial advance window enqueued when ``should_advance`` is
            True. Returning 0 disables the advance window entirely.
    """

    should_advance: bool

    @abstractmethod
    def seed_range(self) -> range: ...

    @abstractmethod
    def from_int(self, n: int) -> Self: ...

    @abstractmethod
    def max_gap(self) -> int: ...
