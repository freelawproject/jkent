"""Shared ``@entry`` parameter models: their bounds checks, the
``Speculative`` hooks ``SpeculativeRange`` implements, and
``PersistedSpeculativeRange``'s ``[key]`` resolution and write-back."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from jkent.common.param_models import (
    DateRange,
    PersistedSpeculativeRange,
    SpeculativeRange,
)
from jkent.common.speculative import (
    PERSISTED_CURSORS,
    persisted_cursor_store,
    persisted_keys_in,
)


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


# --- PersistedSpeculativeRange --------------------------------------------


class _CourtRange(PersistedSpeculativeRange):
    court_id: str


_STORE = {
    "base": {"min": 5, "soft_max": 10, "gap": 20},
    "cal": {"court_id": "cal", "min": 300, "soft_max": 300, "gap": 100},
}


def test_key_reference_resolves_and_remembers_its_key() -> None:
    with persisted_cursor_store(_STORE):
        r = PersistedSpeculativeRange.model_validate("  [base]  ")
    assert type(r) is PersistedSpeculativeRange
    assert (r.min, r.soft_max, r.gap) == (5, 10, 20)
    assert r.persist_key == "base"


def test_key_reference_parses_to_the_subclass() -> None:
    with persisted_cursor_store(_STORE):
        r = _CourtRange.model_validate("[cal]")
    assert type(r) is _CourtRange
    assert r.court_id == "cal"


def test_stored_json_string_is_accepted() -> None:
    with persisted_cursor_store({"k": '{"min": 3, "soft_max": 3}'}):
        assert PersistedSpeculativeRange.model_validate("[k]").min == 3


def test_literal_seed_has_no_key() -> None:
    r = PersistedSpeculativeRange.model_validate({"min": 1})
    assert r.persist_key is None


def test_key_reaches_a_nested_field() -> None:
    class Params(BaseModel):
        docket: _CourtRange

    with persisted_cursor_store(_STORE):
        params = Params.model_validate({"docket": "[cal]"})
    assert params.docket.persist_key == "cal"


@pytest.mark.parametrize(
    ("store", "value"),
    [
        (None, "[cal]"),  # no store active
        (_STORE, "[nope]"),  # key absent
        ({"x": {"min": 1}}, "[x]"),  # stored value lacks court_id
        ({"x": [1, 2]}, "[x]"),  # stored value not a mapping
        (_STORE, "cal"),  # not a reference: ordinary validation rejects it
    ],
)
def test_unresolvable_reference_fails_validation(
    store: dict[str, object] | None, value: str
) -> None:
    if store is None:
        assert PERSISTED_CURSORS.get() is None
        with pytest.raises(ValidationError):
            _CourtRange.model_validate(value)
        return
    with persisted_cursor_store(store), pytest.raises(ValidationError):
        _CourtRange.model_validate(value)


def test_store_is_restored_on_exit() -> None:
    with persisted_cursor_store(_STORE):
        assert PERSISTED_CURSORS.get() is _STORE
    assert PERSISTED_CURSORS.get() is None


def test_key_survives_from_int_and_a_json_round_trip() -> None:
    with persisted_cursor_store(_STORE):
        r = _CourtRange.model_validate("[cal]")
    probe = r.from_int(305)
    assert probe.persist_key == "cal"
    # What a resume validates: the stored template, with no store active.
    assert _CourtRange.model_validate_json(r.model_dump_json()) == r


def test_persist_key_is_not_in_the_entry_schema() -> None:
    schema = PersistedSpeculativeRange.model_json_schema()
    assert "persist_key" not in schema["properties"]


@pytest.mark.parametrize(
    ("highest", "expected"),
    [(None, 300), (299, 300), (300, 301), (412, 413)],
)
def test_resume_after_starts_past_the_highest_success(
    highest: int | None, expected: int
) -> None:
    with persisted_cursor_store(_STORE):
        r = _CourtRange.model_validate("[cal]")
    nxt = r.resume_after(highest)
    assert (nxt.min, nxt.soft_max) == (expected, expected)
    assert (nxt.court_id, nxt.gap, nxt.persist_key) == ("cal", 100, "cal")
    # The stored value is what "[cal]" resolves to next run.
    assert "persist_key" not in nxt.stored_value()
    with persisted_cursor_store({"cal": nxt.stored_value()}):
        assert _CourtRange.model_validate("[cal]") == nxt


def test_persisted_keys_in_finds_every_reference() -> None:
    seed = [
        {"a": {"rid": "[one]"}},
        {"b": {"rid": {"min": 1}, "other": ["[two]", "plain"]}},
        {"c": {"rid": "[one]"}},
    ]
    assert persisted_keys_in(seed) == {"one", "two"}
    assert persisted_keys_in(None) == set()
