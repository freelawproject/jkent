"""The request queue's serialize -> deserialize fixed point.

``f = _deserialize_request . _insert_payload`` is what a request goes
through between ``enqueue`` and ``dequeue``. No database is needed to
exercise it: the insert payload wrapped as a :class:`DequeuedRow` is exactly
what ``dequeue_next_request`` hands back (``queue_body_round_trips`` in
``tests/contracts/properties.py`` uses the same seam).

Three laws over a deliberately wide ``Request`` strategy — every field the
queue stores, including the ones ``test_speculation``'s
``_round_trippable_requests`` leaves at their defaults because they are
lossy:

1. **Idempotence.** ``f(f(r)) == f(r)`` field by field. Whatever the queue
   normalizes (params folded into the URL, JSON values for JSON columns)
   must be normalized *once*: a second trip through the queue — a host's
   reseed, a replay re-enqueue — must not move the request again.
2. **By-value survival.** Every field ``queue.py`` documents as preserved
   comes back equal after one trip. Where the queue is documented lossy
   the test asserts the documented normal form instead of equality (see
   :func:`test_documented_fields_survive_one_trip` for the list).
3. **``verify`` encoding.** Any ``VerifyType`` round-trips, including a
   CA-bundle path spelled ``"true"`` or ``"false"``.

Plus the lane table's enqueue-time rejection of an undeclared lane, and the
``SkipDeduplicationCheck`` opt-out, which has no stored form of its own.
"""

from __future__ import annotations

import functools
import json
import string
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any
from urllib.parse import parse_qsl, urlparse

import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from pyrate_limiter import Duration, Rate

from jkent.common.exceptions import ScraperConfigError
from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    RateLimitTable,
)
from jkent.common.request import SkipDeduplicationCheck
from jkent.common.selectors import CSS, XPath
from jkent.common.via import ViaFormSubmit, ViaLink
from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.sql_manager import DequeuedRow

pytestmark = pytest.mark.generative

_URL = "https://example.com/"

# --- The lane table ---------------------------------------------------------

#: The one scraper-declared lane; stored as code 2 (after default=0, none=1).
_LANE = "slow"


class _LaneScraper:
    """Enough of a scraper for ``RateLimitTable.for_scraper``."""

    rate_limits: list[Rate] | None = None
    named_rate_limits = {_LANE: [Rate(1, Duration.SECOND)]}


_TABLE = RateLimitTable.for_scraper(_LaneScraper)


def _queue() -> RequestQueueDB:
    """A queue with the lane table and no database (none is needed)."""
    return RequestQueueDB(None, rate_limits=_TABLE)  # type: ignore[arg-type]


def _trip(request: Request) -> Request:
    """``f``: the insert payload read back as a dequeued row.

    ``_insert_payload`` (not bare ``serialize_request``) so the row carries
    the effective priority and dedup key the enqueue paths store.
    """
    stored = _queue()._insert_payload(request)
    row = DequeuedRow(**stored.model_dump(), id=1, preresolved=False)
    return _queue()._deserialize_request(row)


# --- Strategies -------------------------------------------------------------

# Text without surrogates: json/pydantic refuse to encode a lone surrogate,
# which would be a strategy artefact, not a queue property.
_text = st.text(alphabet=st.characters(codec="utf-8"), max_size=8)
_key = st.text(alphabet=string.ascii_lowercase + "_-", min_size=1, max_size=6)
_str_dict = st.dictionaries(_key, _text, max_size=3)
# Finite floats only: NaN breaks equality and json refuses infinities.
_finite = st.floats(
    min_value=0, max_value=10_000, allow_nan=False, allow_infinity=False
)
_json_scalar = st.none() | st.booleans() | st.integers() | _finite | _text
_json = st.recursive(
    _json_scalar,
    lambda inner: (
        st.lists(inner, max_size=3) | st.dictionaries(_text, inner, max_size=3)
    ),
    max_leaves=6,
)
_json_dict = st.dictionaries(_text, _json, max_size=3)

# URL pieces: %-escapes (valid and stray), non-ASCII, spaces, delimiters.
_path = st.text(
    alphabet=string.ascii_letters + string.digits + "/-._~%éñ日 ",
    max_size=10,
)
_query = st.text(
    alphabet=string.ascii_letters + string.digits + "=&%+;é ",
    max_size=10,
)


@st.composite
def _urls(draw: st.DrawFn) -> str:
    url = "https://example.com/" + draw(_path)
    if draw(st.booleans()):
        url += "?" + draw(_query)
    if draw(st.booleans()):
        url += "#" + draw(_key)
    return url


_param_value = _text | st.integers() | st.lists(_text, max_size=3)
_params = (
    st.none()
    | st.dictionaries(_text, _param_value, max_size=3)
    | st.lists(st.tuples(_text, _param_value), max_size=3)
)

# JSON-shaped bytes sit alongside arbitrary bytes: a raw body must come
# back as bytes even when it parses as JSON.
_json_bytes = st.sampled_from(
    [b"", b"{}", b"[]", b"0", b"null", b'""', b"false", b'{"a": 1}', b"[1]"]
)
_data = (
    st.none()
    | st.dictionaries(_key, _text | st.integers(), max_size=3)
    | st.lists(st.tuples(_key, _text), max_size=3)
    | _json_bytes
    | st.binary(max_size=8)
)

_timeout = st.none() | _finite | st.tuples(_finite, _finite)
# A CA-bundle path, including the spellings of the two bools.
_verify_path = st.text(
    alphabet=string.ascii_letters + "/._-", max_size=10
) | st.sampled_from(["true", "false"])
_verify = st.booleans() | _verify_path

_selector = st.builds(CSS, _key) | st.builds(XPath, _key.map("//".__add__))
_via = st.none() | st.one_of(
    st.builds(ViaLink, selector=_selector, description=_text),
    st.builds(
        ViaFormSubmit,
        form_selector=_selector,
        submit_selector=st.none() | _key,
        field_data=st.dictionaries(
            _key, _text | st.lists(_text, max_size=2), max_size=3
        ),
        description=_text,
    ),
)


def parse_step(_response: Any) -> None:
    """A callable step: the queue stores its ``__name__``."""


_step: st.SearchStrategy[str | Callable[..., Any]] = _key | st.just(parse_step)


@st.composite
def _permanents(draw: st.DrawFn) -> dict[str, Any]:
    """A JSON-able ``permanent`` whose headers/cookies (if any) are str->str.

    ``Request.__post_init__`` merges those two keys into the request's
    headers/cookies, and the row model types both as ``dict[str, str]``.
    """
    permanent = draw(
        st.dictionaries(
            _text.filter(lambda k: k not in ("headers", "cookies")),
            _json,
            max_size=2,
        )
    )
    if draw(st.booleans()):
        permanent["headers"] = draw(_str_dict)
    if draw(st.booleans()):
        permanent["cookies"] = draw(_str_dict)
    return permanent


@st.composite
def _requests(draw: st.DrawFn) -> Request:
    """A ``Request`` varying every field the queue stores."""
    http = HTTPRequestParams(
        method=draw(st.sampled_from(HttpMethod)),
        url=draw(_urls()),
        params=draw(_params),
        data=draw(_data),
        json=draw(st.none() | _json),
        headers=draw(st.none() | _str_dict),
        cookies=draw(st.none() | _str_dict),
        timeout=draw(_timeout),
        verify=draw(_verify),
    )
    speculative = draw(st.booleans())
    return Request(
        request=http,
        step=draw(_step),
        current_location=draw(st.just("") | _urls()),
        accumulated_data=draw(_json_dict),
        priority=draw(st.none() | st.integers(0, 9)),
        permanent=draw(_permanents()),
        is_speculative=speculative,
        speculation_tracking_id=draw(st.integers(1, 100))
        if speculative
        else None,
        speculative_index=draw(st.integers(0, 100)) if speculative else None,
        via=draw(_via),
        rate_limit=draw(st.sampled_from([None, "none", "default", _LANE])),
        reseedable=draw(st.none() | st.booleans()),
        nonnavigating=draw(st.booleans()),
        archive=draw(st.booleans()),
        expected_type=draw(st.none() | st.sampled_from(["pdf", "audio"])),
    )


# --- Law 1: idempotence -----------------------------------------------------


def _field_diffs(left: Any, right: Any) -> dict[str, tuple[Any, Any]]:
    """``{field: (left, right)}`` for every dataclass field that differs."""
    return {
        f.name: (getattr(left, f.name), getattr(right, f.name))
        for f in fields(left)
        if getattr(left, f.name) != getattr(right, f.name)
    }


@given(request=_requests())
# Pinned counterexample: a bytes body that is falsy *as JSON*. When the
# loader guessed at JSON, the first trip decoded b"{}" to {} and the second
# trip's truthiness gate stored {} as NULL, so the body vanished.
@example(
    request=Request(
        request=HTTPRequestParams(HttpMethod.POST, _URL, data=b"{}"),
        step="s",
    )
)
def test_round_trip_is_a_fixed_point_after_one_application(
    request: Request,
) -> None:
    """``f(f(r)) == f(r)`` on every ``Request`` and ``HTTPRequestParams`` field.

    The queue's normalizations (params -> URL, JSON columns -> JSON values,
    ``"default"`` -> ``None``, archive priority) must all land in one trip.
    """
    once = _trip(request)
    twice = _trip(once)

    assert _field_diffs(once.request, twice.request) == {}
    outer = _field_diffs(once, twice)
    outer.pop("request", None)
    assert outer == {}


# --- Law 2: by-value survival / documented normal forms ---------------------


def _expected_url(request: Request, stored_url: str) -> None:
    """The documented fold of ``params`` into the URL.

    Independent of ``serialize_url_and_body``: the folded URL must agree
    with the original on every component but the query, and its query
    must *parse* to the original pairs followed by the params flattened the
    way a browser repeats a list-valued key (``doseq``).
    """
    http = request.request
    if not http.params:
        assert stored_url == http.url
        return
    original = urlparse(http.url)
    folded = urlparse(stored_url)
    assert folded._replace(query="") == original._replace(query="")

    # The strategy never draws raw-bytes params; narrow for the type checker.
    assert not isinstance(http.params, bytes)
    items = (
        http.params.items() if isinstance(http.params, dict) else http.params
    )
    flattened: list[tuple[str, str]] = []
    for key, value in items:
        if isinstance(value, list):
            flattened.extend((key, item) for item in value)
        else:
            flattened.append((key, str(value)))
    expected = parse_qsl(original.query, keep_blank_values=True) + flattened
    assert parse_qsl(folded.query, keep_blank_values=True) == expected


def _oracle_body(data: Any) -> Any:
    """The documented normal form of ``data`` after one trip.

    bytes survive verbatim, JSON-shaped or not. A truthy dict comes back as
    its JSON value and a truthy pair list as pairs of JSON values; falsy
    non-bytes data (``None``, ``{}``, ``[]``) stores as NULL and reads back
    ``None``.
    """
    if isinstance(data, bytes):
        return data
    if not data:
        return None
    if isinstance(data, dict):
        return _json_value(data)
    return [tuple(_json_value(list(pair))) for pair in data]


def _json_value(value: Any) -> Any:
    """What a JSON column is specified to give back: the JSON value."""
    return json.loads(json.dumps(value))


@given(request=_requests())
def test_documented_fields_survive_one_trip(request: Request) -> None:
    """Every by-value field survives; documented-lossy fields normalize.

    By value: method, headers, cookies, timeout, verify, step (a callable
    by its ``__name__``), current_location, accumulated_data, permanent,
    via, reseedable, archive, and the effective priority and dedup key.

    Documented normal forms, asserted instead of equality:

    * ``url``/``params``: params are folded into the URL and ``params``
      comes back ``None`` (``_deserialize_request``'s note).
    * ``data``: see :func:`_oracle_body`.
    * ``json``: the JSON value (tuples become lists).
    * ``rate_limit``: ``"default"`` is the unset value and decodes to
      ``None`` (``RateLimitTable.decode``).
    * ``nonnavigating``/``expected_type``: the row stores one
      ``RequestType``; archive wins, and ``expected_type`` is
      archive-only. The speculation fields survive on every type.
    """
    once = _trip(request)
    http = request.request

    assert once.request.method == http.method
    _expected_url(request, once.request.url)
    assert once.request.params is None
    assert once.request.data == _oracle_body(http.data)
    assert once.request.json == (
        None if http.json is None else _json_value(http.json)
    )
    assert once.request.headers == http.headers
    assert once.request.cookies == http.cookies
    assert once.request.timeout == http.timeout
    assert once.request.verify == http.verify

    step = request.step
    assert once.step == (step if isinstance(step, str) else step.__name__)
    assert once.current_location == request.current_location
    assert once.accumulated_data == _json_value(request.accumulated_data)
    assert once.permanent == _json_value(request.permanent)
    assert once.priority == request.effective_priority
    assert once.deduplication_key == request.effective_deduplication_key
    assert once.via == request.via
    assert once.rate_limit == (
        None if request.rate_limit == "default" else request.rate_limit
    )
    assert once.reseedable == request.reseedable

    assert once.archive == request.archive
    assert once.nonnavigating == (
        request.nonnavigating and not request.archive
    )
    assert once.expected_type == (
        request.expected_type if request.archive else None
    )
    # Stored for every request type, so restored for every one: a
    # speculative probe that is a side fetch or a download is still a probe.
    assert once.is_speculative == request.is_speculative
    assert once.speculation_tracking_id == request.speculation_tracking_id
    assert once.speculative_index == request.speculative_index


# --- Law 3: verify encoding -------------------------------------------------


@given(request=_requests(), verify=_verify)
@example(
    request=Request(request=HTTPRequestParams(HttpMethod.GET, _URL), step="s"),
    verify="false",
)
def test_verify_round_trips(request: Request, verify: bool | str) -> None:
    """Any ``VerifyType`` survives, including a path spelled ``"false"``.

    A CA bundle literally named ``"false"`` must not read back as ``False``,
    which would silently disable TLS verification.
    """
    request = replace(request, request=replace(request.request, verify=verify))
    assert _trip(request).request.verify == verify


# --- Lane table: an undeclared lane fails at enqueue -----------------------


@given(
    request=_requests(),
    # Anything but the table's lanes: the scraper's one plus the two
    # framework lanes every table carries.
    lane=_key.filter(
        lambda s: s not in (_LANE, DEFAULT_RATE_LIMIT, NO_RATE_LIMIT)
    ),
)
def test_undeclared_lane_raises_at_serialize(
    request: Request, lane: str
) -> None:
    """A lane the table does not declare is rejected when serialized.

    ``RateLimitTable.encode`` raises ``ValueError`` so the failure lands at
    enqueue, on the step that yielded the request (``Request.rate_limit``
    docs: "an unknown one fails at enqueue").
    """
    request = replace(request, rate_limit=lane)
    with pytest.raises(ValueError, match="Unknown rate limit"):
        _queue().serialize_request(request)


# --- A callable step with no name -------------------------------------------


def test_nameless_callable_step_raises_at_serialize() -> None:
    """A step callable without ``__name__`` is refused at enqueue.

    The row stores the step by name and the worker resolves it with
    ``getattr(scraper, name)``. A ``functools.partial`` has no ``__name__``;
    its ``repr`` used to be stored instead, which no scraper attribute is
    called, so the failure surfaced only at dequeue, far from the yield.
    """
    request = Request(
        request=HTTPRequestParams(HttpMethod.GET, _URL),
        step=functools.partial(parse_step),
    )
    with pytest.raises(ScraperConfigError, match="__name__"):
        _queue().serialize_request(request)


# --- Dedup opt-out -----------------------------------------------------------


@given(request=_requests())
def test_dedup_opt_out_survives_one_trip(request: Request) -> None:
    """``SkipDeduplicationCheck`` still means "never deduplicate" after a trip.

    The opt-out stores as a NULL ``deduplication_key``. On the way back
    that NULL must not read as "unset" — an unset key is re-hashed from the
    request, and the worker consults ``effective_deduplication_key`` on the
    dequeued request (the archive handler's ``should_download`` /
    ``save_stream`` key on it), so the request the worker runs must still
    answer ``None``.
    """
    request = replace(request, deduplication_key=SkipDeduplicationCheck())
    assert request.effective_deduplication_key is None

    assert _trip(request).effective_deduplication_key is None
