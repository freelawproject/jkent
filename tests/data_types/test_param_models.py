"""Shared ``@entry`` parameter models: their bounds checks and the
``Speculative`` hooks ``SpeculativeRange`` implements."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from jkent.common.param_models import DateRange, SpeculativeRange


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min": 0}, "must be > 0"),
        ({"min": -3, "soft_max": 5}, "must be > 0"),
        ({"min": 5, "soft_max": 4}, "must be >= min"),
    ],
)
def test_speculative_range_rejects_bad_bounds(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SpeculativeRange.model_validate(kwargs)


def test_speculative_range_allows_empty_seed() -> None:
    """``soft_max == min`` seeds nothing and is valid."""
    assert list(SpeculativeRange(min=5, soft_max=5).seed_range()) == []


def test_seed_range_is_min_to_soft_max_exclusive() -> None:
    assert SpeculativeRange(min=3, soft_max=6).seed_range() == range(3, 6)


def test_max_gap_is_gap() -> None:
    assert SpeculativeRange(gap=17).max_gap() == 17


@pytest.mark.parametrize(
    ("soft_max", "n", "expected_soft_max"),
    [
        (10, 4, 10),  # n inside the seed range: soft_max kept
        (10, 9, 10),
        (10, 10, 11),  # at/after soft_max: raised to n + 1
        (10, 25, 26),
    ],
)
def test_from_int_moves_min_and_keeps_soft_max_above_it(
    soft_max: int, n: int, expected_soft_max: int
) -> None:
    rid = SpeculativeRange(
        min=1, soft_max=soft_max, gap=3, should_advance=False
    )
    moved = rid.from_int(n)
    assert (moved.min, moved.soft_max) == (n, expected_soft_max)
    assert (moved.gap, moved.should_advance) == (3, False)


def test_from_int_preserves_subclass_and_its_fields() -> None:
    class _YearRange(SpeculativeRange):
        year: int

    moved = _YearRange(year=2024, min=1, soft_max=2).from_int(40)
    assert type(moved) is _YearRange
    assert (moved.year, moved.min, moved.soft_max) == (2024, 40, 41)


def test_date_range_rejects_end_before_start() -> None:
    with pytest.raises(ValidationError, match="must be >= start"):
        DateRange(start=date(2024, 5, 2), end=date(2024, 5, 1))


def test_date_range_allows_a_single_day() -> None:
    day = date(2024, 5, 1)
    assert DateRange(start=day, end=day).end == day
