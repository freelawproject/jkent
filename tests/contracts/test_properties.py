"""Hypothesis bridge for the CrossHair property harness.

Drives the property functions in ``properties.py`` (whose icontract
postconditions state the intended behavior) with generated inputs, plus
the counterexample that originally exposed each bug pinned via
``@example`` so a regression fails deterministically.
"""

from __future__ import annotations

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from tests.contracts import properties

pytestmark = pytest.mark.generative

_field = st.tuples(st.text(max_size=5), st.text(max_size=5))


@given(st.lists(_field, max_size=5))
@example([("a", "2"), ("a", "1")])
def test_dedup_key_ignores_form_field_order(fields):
    properties.dedup_key_ignores_form_field_order(fields)


@given(
    st.lists(
        st.tuples(st.text(max_size=3), st.integers() | st.text(max_size=3)),
        max_size=4,
    )
)
@example([("k", 0), ("k", "")])
def test_dedup_key_total_over_declared_params(params):
    properties.dedup_key_total_over_declared_params(params)


@given(st.binary(max_size=16))
@example(b"")
def test_dedup_key_deterministic_for_file_bodies(content):
    properties.dedup_key_deterministic_for_file_bodies(content)


@given(st.text(max_size=8))
@example("a&b")
def test_resolve_url_preserves_query_values(value):
    properties.resolve_url_preserves_query_values(value)


@given(st.dictionaries(st.text(max_size=5), st.text(max_size=5), max_size=4))
@example({})
def test_replay_body_agrees_with_queue_body(form):
    properties.replay_body_agrees_with_queue_body(form)


@given(st.text(max_size=12))
@example("//div/text()")
def test_waitability_ignores_positional_predicate(selector):
    properties.waitability_ignores_positional_predicate(selector)


@given(st.binary(max_size=64), st.integers(), st.binary(max_size=32))
@example(b"", 0, b"")
def test_compression_round_trips(data, level_seed, dictionary):
    properties.compression_round_trips(data, level_seed, dictionary)


# The queue privileges JSON by design: JSON-shaped bytes load back as
# their decoded value, everything else byte-exact (see _canonical_body).
@given(st.binary(max_size=32))
@example(b'{"a": 1}')  # JSON object: decoded on load
@example(b'{"a":1}')  # same value, different spacing: equal in JSON space
@example(b"123")  # JSON scalar: changes type on load
@example(b"0")  # falsy JSON scalar: must survive, not collapse to None
@example(b"")  # empty body: dropped by the loader's truthiness check
@example(b"not json")  # non-JSON bytes: byte-exact round trip
def test_queue_body_round_trips(data):
    properties.queue_body_round_trips(data)
