"""CrossHair property harness for contracts the codebase should satisfy.

Each function here is a *property*: it takes inputs, exercises production
code, and states the expected relationship as an icontract postcondition.
Nothing in this module runs in production. CrossHair explores the input
space symbolically and reports counterexamples::

    uv run crosshair check --analysis_kind=icontract \
        tests/contracts/properties.py

This harness uses icontract directly, so the command above needs no
setup. The contracts on *production* functions go through the
dev-time gate in ``jkent.contracts`` and are inert unless
``JKENT_ENFORCE_CONTRACTS=1`` is set (the test suite sets it in
``conftest.py``); prefix the command with it to CrossHair-check a
production module's own contracts.

Cross-call properties (permutation invariance, two implementations
agreeing) cannot be expressed as a postcondition on the production
function itself — a postcondition sees one call. So each harness makes
the calls it needs and returns both sides; the postcondition compares
them.

These properties all hold now; each was originally falsifiable, and the
per-function notes record the bug its counterexample exposed. They run
as regression guards via the Hypothesis bridge in ``test_properties.py``
(which pins each original counterexample as an ``@example``), and can be
re-explored symbolically with CrossHair using the command above.

Known CrossHair artifacts on this harness (not code bugs — both were
checked against concrete execution):

- ``resolve_url_preserves_query_values`` may report a ``ValueError``
  from ``fromhex`` on non-BMP characters (e.g. ``'\\U00010000'``);
  the concrete round-trip is fine — it's CrossHair's symbolic model
  of ``urllib.parse.unquote``.
- Long runs may report ``RecursionError`` or ``NotDeterministic`` —
  the symbolic interpreter blowing its own stack or tripping over its
  icontract integration, not the code under test. Re-run the reported
  input concretely before treating it as a bug.
"""

from __future__ import annotations

import io
import json
from urllib.parse import parse_qs, quote, urlparse

import icontract

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    _generate_deduplication_key,
)
from jkent.driver.database_engine.compression import compress, decompress
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.unified_driver.continuation import can_playwright_wait

_URL = "http://example.com/x"
# A pre-supplied dedup key keeps Request.__post_init__ from hashing the
# (symbolic) request, which would force CrossHair to concretize early.
_KEY = "0" * 64


@icontract.ensure(
    lambda result: result[0] == result[1],
    "the dedup key ignores the order of form fields",
)
def dedup_key_ignores_form_field_order(
    fields: list[tuple[str, str]],
) -> tuple[str, str]:
    """Same multiset of form fields, same key.

    Guards a fixed bug: ``data`` lists were sorted by first element
    only, so duplicate field names with values in different orders
    produced different keys (counterexample
    ``[("a", "2"), ("a", "1")]``).
    """
    forward = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.POST, _URL, data=list(fields))
    )
    backward = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.POST, _URL, data=list(reversed(fields)))
    )
    return (forward, backward)


@icontract.ensure(
    lambda result: len(result) == 64,
    "key generation is total over the declared QueryParams type",
)
def dedup_key_total_over_declared_params(
    params: list[tuple[str, int | str]],
) -> str:
    """Any value matching the QueryParams alias must produce a key.

    Guards a fixed bug: ``sorted()`` over the full tuples raised
    TypeError when two entries shared a name and carried values of
    uncomparable types (counterexample ``[("k", 0), ("k", "")]``).
    Params now sort by ``repr``, which is total.
    """
    return _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, _URL, params=list(params))
    )


@icontract.ensure(
    lambda result: result[0] == result[1],
    "the dedup key is a function of the body content, not object identity",
)
def dedup_key_deterministic_for_file_bodies(
    content: bytes,
) -> tuple[str, str]:
    """Two file-like bodies with identical bytes get identical keys.

    Guards a fixed bug: the fallback branch was
    ``str(request_params.data)``, which for a BytesIO rendered its
    memory address; seekable streams now key on their content. Both
    file objects are kept alive together — back-to-back temporaries
    can reuse the same address and mask a regression.
    """
    body_a = io.BytesIO(content)
    body_b = io.BytesIO(content)
    first = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.POST, _URL, data=body_a)
    )
    second = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.POST, _URL, data=body_b)
    )
    return (first, second)


@icontract.ensure(
    lambda result: result[0] == result[1],
    "URL normalization preserves query-string semantics",
)
def resolve_url_preserves_query_values(
    value: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """resolve_url must not change what the query parses to.

    Guards a fixed bug: a blanket unquote/quote normalization turned a
    percent-encoded delimiter inside a value into a live delimiter —
    ``q=a%26b`` became ``q=a&b`` — and space-as-plus into a literal
    plus. ``_requote_uri`` now only decodes unreserved escapes.
    """
    url = _URL + "?q=" + quote(value, safe="")
    request = Request(
        request=HTTPRequestParams(HttpMethod.GET, url),
        continuation="step",
        deduplication_key=_KEY,
    )
    resolved = request.resolve_url("http://example.com/")
    expected = parse_qs(urlparse(url).query, keep_blank_values=True)
    actual = parse_qs(urlparse(resolved).query, keep_blank_values=True)
    return (expected, actual)


@icontract.ensure(
    lambda result: result[0] == result[1],
    "the queue stores the (url, body) replay's key derivation expects",
)
def replay_body_agrees_with_queue_body(
    form: dict[str, str],
) -> tuple[tuple[str, bytes | None], tuple[str, bytes | None]]:
    """The contract stated in serialize_url_and_body's docstring.

    Replay's fallback key derivation and the queue's write path both go
    through ``serialize_url_and_body``, so they cannot drift from each
    other — what can still regress is what that one function stores. So
    this pins the queue's stored ``(url, body)`` against an *independent*
    spec rather than re-running the same serializer: a POST with falsy
    form data (e.g. ``{}``) stores body None; truthy form data stores its
    ``json.dumps`` bytes. Guards a fixed bug at ``form={}`` where the body
    was serialized as ``b"{}"`` instead of the queue's stored None.
    """
    http_request = HTTPRequestParams(HttpMethod.POST, _URL, data=dict(form))
    request = Request(
        request=http_request,
        continuation="step",
        deduplication_key=_KEY,
    )
    stored = RequestQueueDB().serialize_request(request)
    queue_side = (stored["url"], stored["body"])
    # Independent oracle — deliberately NOT serialize_url_and_body, so a
    # regression in that shared function is caught rather than mirrored.
    expected_body = json.dumps(dict(form)).encode() if form else None
    spec_side = (_URL, expected_body)
    return (queue_side, spec_side)


@icontract.ensure(
    lambda result: result[0] == result[1],
    "a positional predicate does not change what kind of node a "
    "selector targets",
)
def waitability_ignores_positional_predicate(
    selector: str,
) -> tuple[bool, bool]:
    """``s`` and ``s[1]`` target the same node kind, so same answer.

    Guards a fixed bug: the text-node check was a bare
    ``endswith("/text()")``, so ``//div/text()[1]`` was wrongly
    reported waitable. Trailing predicates are now stripped before
    the node-kind checks.
    """
    return (
        can_playwright_wait(selector, "xpath"),
        can_playwright_wait(selector + "[1]", "xpath"),
    )


@icontract.ensure(
    lambda result: result[0] == result[1],
    "decompress inverts compress when the same dictionary is supplied",
)
def compression_round_trips(
    data: bytes, level_seed: int, dictionary: bytes
) -> tuple[bytes, bytes]:
    """Stored response content must come back byte-identical.

    ``dictionary=b""`` exercises the no-dictionary path (the production
    code's ``if dictionary:`` treats empty bytes as absent on both
    sides). ``level_seed`` is folded into zstd's documented 1-22 range.
    """
    level = 1 + abs(level_seed) % 22
    compressed = compress(data, level=level, dictionary=dictionary)
    return (data, decompress(compressed, dictionary=dictionary))


# Row layout consumed by RequestQueueDB._deserialize_request
# (queue.py): the serialize dict's fields in column order, with
# id/priority/dedup_key interleaved where the SELECT puts them.
_ROW_FIELDS_AFTER_URL = (
    "headers_json",
    "cookies_json",
    "body",
    "continuation",
    "current_location",
    "accumulated_data_json",
    "permanent_json",
    "expected_type",
)
_ROW_FIELDS_AFTER_SPECULATION = (
    "verify",
    "via_json",
    "bypass_rate_limit",
)
_ROW_FIELDS_TAIL = (
    "timeout_json",
    "json_data",
    "files_json",
    "auth_json",
    "allow_redirects",
    "proxies_json",
    "stream",
    "cert_json",
    "archive_hash_header",
    "reseedable",
)


def _oracle_restored(data: bytes) -> object:
    """What the queue's loader is *specified* to return for a bytes body.

    An independent reference — deliberately NOT the production loader — so
    the round trip compares production against a spec instead of against
    itself. The queue privileges JSON: bytes that parse as JSON decode to
    their object; ``b""`` and other non-JSON / non-UTF-8 bytes survive
    verbatim (the loader gates on ``is not None``, not truthiness, so an
    empty body stays ``b""`` rather than collapsing to None).
    """
    try:
        return json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return data


def _comparable(value: object) -> object:
    """Canonical form for comparison that keeps bytes distinct from objects.

    Decoded values become canonical JSON text (``sort_keys`` for dict-order
    stability, text form so ``NaN`` compares equal to itself); bytes stay
    tagged. Crucially — unlike a loader — this never decodes bytes into an
    object, so a body that comes back undecoded is distinguishable from one
    decoded correctly, and a loader regression is caught rather than masked.
    """
    if isinstance(value, bytes):
        return ("bytes", value)
    return ("json", json.dumps(value, sort_keys=True))


@icontract.ensure(
    lambda result: result[0] == result[1],
    "a request body survives the queue's store/load round trip — "
    "compared as JSON when it parses as JSON, byte-exact otherwise",
)
def queue_body_round_trips(data: bytes) -> tuple[object, object]:
    """Serialize a bytes-bodied request to a row and read it back.

    The queue deliberately privileges JSON: a bytes body that parses as
    JSON is loaded back in decoded form (``b'{"a": 1}'`` → the dict
    ``{"a": 1}``), so the round trip is specified to preserve the
    body's *JSON value* for JSON-shaped bytes and the exact bytes for
    everything else. The expected side is computed by the independent
    :func:`_oracle_restored`; both sides are then put in
    :func:`_comparable` form (which keeps bytes distinct from decoded
    objects, so a loader regression is caught, not masked).
    """
    request = Request(
        request=HTTPRequestParams(HttpMethod.POST, _URL, data=data),
        continuation="step",
        deduplication_key=_KEY,
    )
    queue = RequestQueueDB()
    stored = queue.serialize_request(request)
    row_fields: list[object] = [
        1,
        stored["request_type"],
        stored["method"],
        stored["url"],
    ]
    row_fields += [stored[f] for f in _ROW_FIELDS_AFTER_URL]
    # priority lives on the row, not the serialize dict
    row_fields += [0, stored["is_speculative"], stored["speculation_id"]]
    row_fields += [stored[f] for f in _ROW_FIELDS_AFTER_SPECULATION]
    row_fields.append(request.deduplication_key)
    row_fields += [stored[f] for f in _ROW_FIELDS_TAIL]
    restored = queue._deserialize_request(tuple(row_fields))
    return (
        _comparable(_oracle_restored(data)),
        _comparable(restored.request.data),
    )
