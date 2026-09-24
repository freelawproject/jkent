"""Hypothesis bridge for the CrossHair property harness.

Drives the property functions in ``properties.py`` (whose icontract
postconditions state the intended behavior) with generated inputs, plus
the counterexample that originally exposed each bug pinned via
``@example`` so a regression fails deterministically.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    RESERVED_RATE_LIMIT_NAMES,
)
from tests.contracts import properties

pytestmark = pytest.mark.generative


@given(
    st.lists(
        st.tuples(st.text(max_size=3), st.integers() | st.text(max_size=3)),
        max_size=4,
    )
)
@example([("k", 0), ("k", "")])
def test_dedup_key_total_over_declared_params(
    params: list[tuple[str, int | str]],
):
    properties.dedup_key_total_over_declared_params(params)


@given(st.binary(max_size=16))
@example(b"")
def test_dedup_key_deterministic_for_file_bodies(content: bytes):
    properties.dedup_key_deterministic_for_file_bodies(content)


@given(st.text(max_size=8))
@example("a&b")
def test_resolve_url_preserves_query_values(value: str):
    properties.resolve_url_preserves_query_values(value)


@given(st.dictionaries(st.text(max_size=5), st.text(max_size=5), max_size=4))
@example({})
def test_replay_body_agrees_with_queue_body(form: dict[str, str]):
    properties.replay_body_agrees_with_queue_body(form)


@given(st.text(max_size=12))
@example("//div/text()")
def test_waitability_ignores_positional_predicate(selector: str):
    properties.waitability_ignores_positional_predicate(selector)


@given(st.binary(max_size=64), st.integers(), st.binary(max_size=32))
@example(b"", 0, b"")
def test_compression_round_trips(
    data: bytes, level_seed: int, dictionary: bytes
):
    properties.compression_round_trips(data, level_seed, dictionary)


@given(st.binary(max_size=32))
@example(b'{"q": "smith"}')  # JSON-shaped raw body: stays bytes
@example(b"123")  # JSON scalar: stays bytes, not int
@example(b"{}")
@example(b"")
@example(b"not json")
def test_queue_body_round_trips(data: bytes):
    properties.queue_body_round_trips(data)


# --- parse_retry_after ------------------------------------------------------

_RETRY_AFTER = "retry-after"
_retry_after_key = st.lists(
    st.booleans(), min_size=len(_RETRY_AFTER), max_size=len(_RETRY_AFTER)
).map(
    lambda bits: "".join(
        c.upper() if bit else c for c, bit in zip(_RETRY_AFTER, bits)
    )
)
_retry_after_value = (
    st.text()
    | st.floats().map(str)
    | st.sampled_from(
        [
            "nan",
            "inf",
            "-inf",
            "1e400",
            " 5 ",
            "Wed, 21 Oct 2099 07:28:00 GMT",  # RFC 7231 http-date, future
            "Wed, 21 Oct 2015 07:28:00 GMT",  # http-date in the past
        ]
    )
)


@given(_retry_after_key, _retry_after_value)
@example("Retry-After", "nan")
@example("Retry-After", "inf")
@example("Retry-After", "-inf")
@example("Retry-After", "1e400")
def test_retry_after_is_finite_and_clamped(key: str, value: str):
    properties.retry_after_is_finite_and_clamped(key, value)


# --- HttpxTransport client pool ---------------------------------------------

# The default lane is drawn by name too: it is the one lane with a special
# case (with ``verify=True`` it is the main client, not a pooled one).
_lane = st.text(alphabet="xca:True", max_size=8) | st.just(DEFAULT_RATE_LIMIT)
_verify = st.sampled_from([True, False, *properties.CA_DIR_NAMES])


@given(st.lists(st.tuples(_lane, _verify), max_size=4))
@example([("x", "ca:True"), ("x:ca", True)])
def test_httpx_client_pool_keys_on_lane_and_verify(
    pairs: list[tuple[str, bool | str]],
):
    properties.httpx_client_pool_keys_on_lane_and_verify(pairs)


# --- header merge -----------------------------------------------------------

_header_name = st.sampled_from(
    ["Authorization", "authorization", "X-A", "x-a"]
)
_headers = st.dictionaries(_header_name, st.text(max_size=3), max_size=3)


@given(_headers, _headers)
@example({"Authorization": "a"}, {"authorization": "b"})
@example({"Authorization": "a", "authorization": "b"}, {})
def test_permanent_header_merge_agrees_with_merge_headers(
    permanent: dict[str, str], explicit: dict[str, str]
):
    properties.permanent_header_merge_agrees_with_merge_headers(
        permanent, explicit
    )


# --- dedup key params spelling ----------------------------------------------


@given(
    st.dictionaries(st.text(max_size=4), st.text(max_size=3) | st.integers())
)
@example({"a b": "1", "a": "2"})
def test_dedup_key_params_dict_matches_sorted_pair_list(
    fields: dict[str, str | int],
):
    properties.dedup_key_params_dict_matches_sorted_pair_list(fields)


@pytest.mark.parametrize("field", ["params", "data"])
@given(
    pairs=st.lists(
        st.tuples(st.text(max_size=3), st.text(max_size=3) | st.integers()),
        max_size=4,
    )
)
@example(pairs=[("a", "1"), ("a", "2")])
@example(pairs=[("a", "2"), ("a", "1")])
@example(pairs=[])
def test_dedup_key_pair_list_keeps_order_and_duplicates(
    field: str, pairs: list[tuple[str, str | int]]
):
    properties.dedup_key_pair_list_keeps_order_and_duplicates(field, pairs)


_query_pairs = st.lists(
    st.tuples(st.text(min_size=1, max_size=3), st.text(max_size=3)),
    max_size=3,
)


@given(
    base_url=st.sampled_from(
        [
            "http://example.com/x",
            "http://example.com/x?page=2",
            "http://example.com/a%20b/?q=a%26b",
        ]
    ),
    pairs=_query_pairs,
)
@example(base_url="http://example.com/x", pairs=[("x", "1")])
def test_dedup_key_ignores_query_spelling(
    base_url: str, pairs: list[tuple[str, str]]
):
    properties.dedup_key_ignores_query_spelling(base_url, pairs)


@given(_query_pairs.filter(bool))
@example([("k", "v")])
def test_dedup_key_url_cannot_impersonate_params(
    pairs: list[tuple[str, str]],
):
    properties.dedup_key_url_cannot_impersonate_params(pairs)


# --- RateLimitTable ---------------------------------------------------------

_lane_name = st.text(min_size=1).filter(
    lambda name: name not in RESERVED_RATE_LIMIT_NAMES
)
_lane_names = st.lists(_lane_name, max_size=4)


@given(_lane_names)
@example([])
@example(["downloads", "search"])
def test_rate_limit_table_codes_round_trip(names: list[str]):
    properties.rate_limit_table_codes_round_trip(names)


@given(_lane_names, st.integers(), st.text())
@example([], 2, "downloads")
@example(["downloads"], -1, "default")
def test_rate_limit_table_rejects_foreign_values(
    names: list[str], code: int, name: str
):
    properties.rate_limit_table_rejects_foreign_values(names, code, name)


# --- _requote_uri -----------------------------------------------------------

_URI_ALPHABET = "AZaz09-._~%0123456789abcdefABCDEF&=/?# \u00e9\u4e2d\U0001f600"


@given(st.text(alphabet=_URI_ALPHABET, max_size=12))
@example("q=a%26b")
@example("%")
@example("%2")
@example("%%41")
@example("100%")
def test_requote_uri_normalizes_escapes(uri: str):
    properties.requote_uri_normalizes_escapes(uri)


# --- timestamps -------------------------------------------------------------

_offsets = st.integers(-1439, 1439).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)
_aware = st.datetimes(
    min_value=datetime(1900, 1, 1),
    max_value=datetime(2100, 1, 1),
    timezones=_offsets,
)
_naive = st.datetimes(
    min_value=datetime(1900, 1, 1), max_value=datetime(2100, 1, 1)
)
_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@given(_aware, _aware, st.booleans(), st.booleans())
@example(_T0, _T0 + timedelta(microseconds=1), True, True)  # six vs six
@example(_T0, _T0 + timedelta(microseconds=1), False, False)  # three vs three
@example(_T0, _T0 + timedelta(milliseconds=1), True, False)  # ms apart, mixed
@example(  # same millisecond, mixed: the prefix inversion
    _T0 + timedelta(microseconds=1),
    _T0 + timedelta(microseconds=900),
    True,
    False,
)
def test_timestamp_text_orders_chronologically(
    a: datetime, b: datetime, a_from_python: bool, b_from_python: bool
):
    properties.timestamp_text_orders_chronologically(
        a, b, a_from_python, b_from_python
    )


@given(_aware)
@example(
    datetime(2026, 1, 1, 0, 0, 0, 123456, tzinfo=timezone(timedelta(hours=5)))
)
def test_utc_datetime_round_trips(value: datetime):
    properties.utc_datetime_round_trips(value)


@given(_naive)
@example(datetime(2026, 1, 1))
def test_utc_datetime_rejects_naive(value: datetime):
    properties.utc_datetime_rejects_naive(value)
