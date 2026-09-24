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

import asyncio
import atexit
import contextlib
import io
import json
import math
import os
import re
import shutil
import sqlite3
import ssl
import tempfile
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote, urlencode, urlparse

import icontract
import zstandard as zstd
from pyrate_limiter import Rate
from sqlalchemy.dialects import sqlite as _sqlite_dialect

from jkent.common.headers import merge_headers
from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    RESERVED_RATE_LIMIT_NAMES,
    RateLimitTable,
)
from jkent.common.request import _generate_deduplication_key, _requote_uri
from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.database_engine.compression import compress, decompress
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.sql_manager import DequeuedRow
from jkent.driver.database_engine.timestamps import (
    TIMESTAMP_FORMAT,
    UtcDateTime,
)
from jkent.driver.unified_driver.steps import can_playwright_wait
from jkent.driver.unified_driver.transport import (
    MAX_RETRY_AFTER_S,
    parse_retry_after,
)
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)

_URL = "http://example.com/x"
# A pre-supplied dedup key keeps Request.__post_init__ from hashing the
# (symbolic) request, which would force CrossHair to concretize early.
_KEY = "0" * 64


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        len(result) == 64
    ),
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
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
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
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
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
        step="step",
        deduplication_key=_KEY,
    )
    resolved = request.resolve_url("http://example.com/")
    expected = parse_qs(urlparse(url).query, keep_blank_values=True)
    actual = parse_qs(urlparse(resolved).query, keep_blank_values=True)
    return (expected, actual)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "the queue stores the (url, body) replay's key derivation expects",
)
def replay_body_agrees_with_queue_body(
    form: dict[str, str],
) -> tuple[tuple[str, bytes | None], tuple[str, bytes | None]]:
    """The contract stated in serialize_url_and_body's docstring.

    The queue's write path goes through ``serialize_url_and_body`` (as
    does jent's replay key derivation), so what can regress is what that
    one function stores. So
    this pins the queue's stored ``(url, body)`` against an *independent*
    spec rather than re-running the same serializer: a POST with falsy
    form data (e.g. ``{}``) stores body None; truthy form data stores its
    ``json.dumps`` bytes. Guards a fixed bug at ``form={}`` where the body
    was serialized as ``b"{}"`` instead of the queue's stored None.
    """
    http_request = HTTPRequestParams(HttpMethod.POST, _URL, data=dict(form))
    request = Request(
        request=http_request,
        step="step",
        deduplication_key=_KEY,
    )
    # (de)serialize need no db.
    stored = RequestQueueDB(None).serialize_request(request)  # type: ignore[arg-type]
    queue_side = (stored.url, stored.body)
    # Independent oracle — deliberately NOT serialize_url_and_body, so a
    # regression in that shared function is caught rather than mirrored.
    expected_body = json.dumps(dict(form)).encode() if form else None
    spec_side = (_URL, expected_body)
    return (queue_side, spec_side)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
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
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "decompress inverts compress when the same dictionary is supplied",
)
def compression_round_trips(
    data: bytes, level_seed: int, dictionary: bytes
) -> tuple[bytes, bytes]:
    """Stored response content must come back byte-identical.

    ``dictionary=b""`` exercises the no-dictionary path; non-empty bytes
    are wrapped as a raw-content ``ZstdCompressionDict``, matching how the
    production code only ever hands ``compress``/``decompress`` a built
    dictionary object (or None). ``level_seed`` is folded into zstd's
    documented 1-22 range.
    """
    level = 1 + abs(level_seed) % 22
    dict_obj = zstd.ZstdCompressionDict(dictionary) if dictionary else None
    compressed = compress(data, level=level, dictionary=dict_obj)
    return (data, decompress(compressed, dictionary=dict_obj))


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        isinstance(result[1], bytes) and result[0] == result[1]
    ),
    "a bytes request body survives the queue's store/load round trip "
    "byte-exact",
)
def queue_body_round_trips(data: bytes) -> tuple[bytes, object]:
    """Serialize a bytes-bodied request to a row and read it back.

    A raw body is sent verbatim, so it must come back verbatim — including
    bytes that happen to parse as JSON (``b'{"q": "smith"}'``, ``b"123"``).

    Guards a fixed bug: the body column did not record whether it held
    raw bytes or JSON-encoded form data, so the loader guessed, and a raw
    JSON-shaped body came back as a dict that the transport then sent as
    form fields.
    """
    request = Request(
        request=HTTPRequestParams(HttpMethod.POST, _URL, data=data),
        step="step",
        deduplication_key=_KEY,
    )
    queue = RequestQueueDB(None)  # type: ignore[arg-type]  # no db needed
    stored = queue.serialize_request(request)
    # ``id``/``preresolved`` are the queue-state columns the row carries
    # beyond the insert payload.
    restored = queue._deserialize_request(
        DequeuedRow(**stored.model_dump(), id=1, preresolved=False)
    )
    return (data, restored.request.data)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result is None
        or (
            isinstance(result, float)
            and math.isfinite(result)
            and 0.0 <= result <= MAX_RETRY_AFTER_S
        )
    ),
    "a parsed Retry-After is None or a finite float in [0, MAX_RETRY_AFTER_S]",
)
def retry_after_is_finite_and_clamped(key: str, value: str) -> float | None:
    """``parse_retry_after`` yields nothing or a usable, bounded delay.

    The function's contract (its docstring, and what both consumers — the
    retry scheduler and the rate limiter's global pause — rely on) is that
    the value is clamped to ``[0, MAX_RETRY_AFTER_S]`` or ``None``. The
    header name is looked up case-insensitively, so *key* is any casing of
    ``retry-after``.

    Guards a fixed bug: ``float("nan")`` parses without error, and
    ``min(max(nan, 0.0), 300.0)`` is ``nan`` — every comparison against
    NaN is False, so both clamps pass it through. Counterexample
    ``{"Retry-After": "nan"}`` returned NaN, which then poisoned any
    ``max()`` / ``<`` the scheduler did with it.
    """
    return parse_retry_after({key: value})


# --- HttpxTransport client pool ---------------------------------------------

#: Relative directory names an str ``verify`` may take in
#: :func:`httpx_client_pool_keys_on_lane_and_verify`. They are created
#: under a private temp root the property ``chdir``s into, so httpx (which
#: treats a str ``verify`` as a CA path and validates it eagerly) accepts
#: them. A *directory* is used because ``capath`` is scanned lazily, so it
#: needs no certificate inside; ``ca:True`` exists so a lane can collide
#: with it in the pool's ``f"{lane}:{verify}"`` key.
CA_DIR_NAMES: tuple[str, ...] = ("ca", "ca:True")
_ca_root: str | None = None


def _ca_root_dir() -> str:
    """The temp root holding :data:`CA_DIR_NAMES`, made once per process.

    Removed at interpreter exit: the property runs many examples per
    session, and each xdist worker would otherwise leave its own root.
    """
    global _ca_root
    root = _ca_root
    if root is None:
        root = tempfile.mkdtemp(prefix="jkent-capath-")
        atexit.register(lambda: shutil.rmtree(root, ignore_errors=True))
        for name in CA_DIR_NAMES:
            os.mkdir(os.path.join(root, name))
        _ca_root = root
    return root


def _observed_verify(client: object) -> bool | str:
    """What SSL verification a built httpx client actually carries.

    Reaches into httpx's private transport/pool for the ``SSLContext``
    (httpx 0.28 keeps it on ``AsyncHTTPTransport._pool._ssl_context``).
    ``verify=False`` is ``CERT_NONE``; ``verify=True`` loads httpx's default
    bundle (a non-empty store); a str ``verify`` naming a directory builds a
    ``capath`` context whose store is empty until a handshake. Two different
    directories are indistinguishable this way, so the property also
    requires distinct pairs to get distinct client objects.
    """
    ctx: ssl.SSLContext = client._transport._pool._ssl_context  # type: ignore[attr-defined]
    if ctx.verify_mode == ssl.CERT_NONE:
        return False
    if ctx.cert_store_stats()["x509"] == 0:
        return "capath"
    return True


def _expected_verify(verify: bool | str) -> bool | str:
    return "capath" if isinstance(verify, str) else verify


@icontract.ensure(
    lambda result: all(  # pyrefly: ignore[implicit-any-lambda]
        (pair_a == pair_b) == (id_a == id_b)
        for pair_a, id_a, _ in result
        for pair_b, id_b, _ in result
    ),
    "one client object per distinct (lane, verify) pair — never shared",
)
@icontract.ensure(
    lambda result: all(  # pyrefly: ignore[implicit-any-lambda]
        observed == _expected_verify(pair[1]) for pair, _, observed in result
    ),
    "the client handed back for a pair carries that pair's verify",
)
def httpx_client_pool_keys_on_lane_and_verify(
    pairs: list[tuple[str, bool | str]],
) -> list[tuple[tuple[str, bool | str], int, bool | str]]:
    """``_client_for`` is a function of the *pair*, not of a joined string.

    Asks the transport for a client per ``(lane, verify)`` pair (a str
    ``verify`` is one of :data:`CA_DIR_NAMES`, resolved relative to a temp
    root the property ``chdir``s into) and returns, per pair, the identity
    of the client it got and the verification the client actually carries.
    The default lane with ``verify=True`` is the main client opened by
    ``open()``; that is just the pair's client like any other and needs no
    special-casing here.

    Guards a fixed bug: the pool was keyed on ``f"{lane}:{verify}"``, so
    pairs that differ collided once a lane or a verify path contained
    ``:``. Counterexample ``[("x", "ca:True"), ("x:ca", True)]`` — both
    keyed as ``"x:ca:True"``, so the second request was served the first
    pair's client and silently used ``verify="ca:True"`` where
    ``verify=True`` (real certificate verification) was asked for.
    """

    async def run() -> list[tuple[tuple[str, bool | str], int, bool | str]]:
        transport = HttpxTransport()
        await transport.open()
        try:
            out = []
            for lane, verify in pairs:
                client = transport._client_for(verify, lane)
                out.append(
                    ((lane, verify), id(client), _observed_verify(client))
                )
            return out
        finally:
            await transport.aclose()

    # The names must stay relative (see CA_DIR_NAMES), hence the chdir; it is
    # process-wide, but an xdist worker runs one test at a time.
    # Not contextlib.chdir: that is 3.11+ and the package supports 3.10.
    previous = os.getcwd()
    os.chdir(_ca_root_dir())
    try:
        return asyncio.run(run())
    finally:
        os.chdir(previous)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        len(set(result[1])) == len(result[1])
    ),
    "merge_headers names each header once, whatever the case of either side",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "the request's merged headers name each header once, as merge_headers "
    "would",
)
def permanent_header_merge_agrees_with_merge_headers(
    permanent: dict[str, str], explicit: dict[str, str]
) -> tuple[list[str], list[str]]:
    """``Request`` merges permanent headers the way the wire path merges.

    ``Request.__post_init__`` folds ``permanent["headers"]`` under the
    request's own headers via ``_merge_permanent_into_request``;
    ``merge_headers`` (the transport seam) is the stated rule for layering
    headers: an explicit header replaces a base header of the same name
    *regardless of case*, so exactly one value per name goes on the wire.
    Both sides are reduced to the sorted list of lowercased header names,
    so a name surviving twice under two spellings shows up as a duplicate.

    Guards two fixed bugs: ``_merge_permanent_into_request`` was a
    case-sensitive ``dict.update`` (counterexample
    ``P={"Authorization": "a"}, E={"authorization": "b"}``), and
    ``merge_headers`` folded case across its two sides but not within one
    (``P={"Authorization": "a", "authorization": "b"}, E={}``).
    """
    request = Request(
        request=HTTPRequestParams(
            HttpMethod.GET, _URL, headers=dict(explicit)
        ),
        step="step",
        deduplication_key=_KEY,
        permanent={"headers": dict(permanent)},
    )
    request_side = sorted(
        name.lower() for name in request.request.headers or {}
    )
    wire_side = sorted(
        name.lower() for name in merge_headers(permanent, explicit)
    )
    return (request_side, wire_side)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "params as a dict and as its name-sorted pair list get the same dedup key",
)
def dedup_key_params_dict_matches_sorted_pair_list(
    fields: dict[str, str | int],
) -> tuple[str, str]:
    """A dict and the pair list that spells the same query key the same.

    A dict is an unordered mapping of unique names, so the pair list that
    means the same query is its items in name order; a scraper switching
    between the two spellings must not defeat deduplication.

    Guards a fixed bug: the dict branch sorted ``items()`` by name while
    the list branch sorted by ``repr`` of the whole tuple, and the two
    orders differ whenever a name is a prefix of another followed by a
    character that sorts below ``'``. Counterexample
    ``{"a b": "1", "a": "2"}`` — by name ``("a", ...)`` comes first; by
    repr ``"('a b', ...)"`` comes first because a space sorts below the
    closing quote. Pair lists are no longer sorted at all (see
    :func:`dedup_key_pair_list_keeps_order_and_duplicates`).
    """
    as_dict = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, _URL, params=dict(fields))
    )
    as_list = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, _URL, params=sorted(fields.items()))
    )
    return (as_dict, as_list)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["reversed_same_key"] == result["reversed_same_list"]
    ),
    "reordering a pair list changes the key iff it changes the list",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["duplicated_same_key"] is False
    ),
    "repeating a pair changes the key",
)
def dedup_key_pair_list_keeps_order_and_duplicates(
    field: str, pairs: list[tuple[str, str | int]]
) -> dict[str, bool]:
    """A ``params`` or ``data`` pair list is identified by order and repeats.

    Both are sent exactly as given — ``urlencode(pairs, doseq=True)`` for
    the query, the form encoder for the body — so
    ``[("a", "1"), ("a", "2")]`` and ``[("a", "2"), ("a", "1")]`` are
    different requests on the wire and must not deduplicate each other;
    nor may ``[("a", "1")]`` and ``[("a", "1"), ("a", "1")]``. Duplicate
    names are legitimate HTTP.

    Guards a fixed bug: both branches sorted their pairs (by ``repr``),
    collapsing every ordering of the same multiset onto one key.
    """

    def key(value: list[tuple[str, str | int]]) -> str:
        if field == "params":
            request = HTTPRequestParams(HttpMethod.GET, _URL, params=value)
        else:
            request = HTTPRequestParams(HttpMethod.POST, _URL, data=value)
        return _generate_deduplication_key(request)

    pairs = list(pairs)
    duplicated: list[tuple[str, str | int]] = [*pairs, *pairs[:1]]
    if not pairs:
        duplicated = [("a", "1")]
    return {
        "reversed_same_key": key(pairs) == key(pairs[::-1]),
        "reversed_same_list": pairs == pairs[::-1],
        "duplicated_same_key": key(pairs) == key(duplicated),
    }


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "a query spelled in the URL and the same query spelled as params get "
    "the same dedup key",
)
def dedup_key_ignores_query_spelling(
    base_url: str, pairs: list[tuple[str, str]]
) -> tuple[str, str]:
    """The key is a function of the stored request, not its spelling.

    The queue stores ``params`` folded into the URL, so ``/a?x=1`` and
    ``/a`` with ``params=[("x", "1")]`` are one row's worth of request and
    must get one key — else two spellings of one request get two rows.

    Guards a fixed bug: the key hashed the request as spelled, and joined
    its parts with an unescaped ``?``/``|``, so a URL containing the
    rendered params text also collided with the params spelling.
    """
    in_url = base_url + ("&" if "?" in base_url else "?") + urlencode(pairs)
    spelled_in_url = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, in_url if pairs else base_url)
    )
    spelled_as_params = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, base_url, params=list(pairs))
    )
    return (spelled_in_url, spelled_as_params)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] != result[1]
    ),
    "a URL that contains another request's rendered params does not "
    "collide with it",
)
def dedup_key_url_cannot_impersonate_params(
    pairs: list[tuple[str, str]],
) -> tuple[str, str]:
    """A literal ``?<rendered params>`` in a URL is not those params.

    Guards a fixed bug: the key rendered params as ``str(list)`` after a
    bare ``?``, so ``/a?[('k', 'v')]`` and ``/a`` with
    ``params=[("k", "v")]`` hashed the same text.
    """
    as_params = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, _URL, params=list(pairs))
    )
    as_literal = _generate_deduplication_key(
        HTTPRequestParams(HttpMethod.GET, f"{_URL}?{list(pairs)}")
    )
    return (as_params, as_literal)


# --- RateLimitTable ---------------------------------------------------------


def _table_for(names: list[str]) -> RateLimitTable:
    """A table for a stand-in scraper declaring *names* as lanes.

    ``for_scraper`` reads ``named_rate_limits`` through ``getattr``, so a
    bare class stands in for a ``BaseScraper`` subclass. Duplicate names
    collapse in the mapping, as they would in a class body.
    """
    stand_in = type(
        "StandIn",
        (),
        {"named_rate_limits": {name: [Rate(1, 1000)] for name in names}},
    )
    return RateLimitTable.for_scraper(stand_in)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["decoded"] == result["names_or_none"]
    ),
    "decode inverts encode for every lane name (default reads back as None)",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["encoded"] == result["codes"]
    ),
    "encode inverts decode for every valid code",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["head"] == RESERVED_RATE_LIMIT_NAMES
    ),
    "the framework's two lanes always hold codes 0 and 1",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        len(set(result["names"])) == len(result["names"])
    ),
    "lane names are unique, so codes are unambiguous",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        set(result["names"]) == result["rate_names"]
    ),
    "every lane has rates and every rates entry is a lane",
)
def rate_limit_table_codes_round_trip(names: list[str]) -> dict[str, object]:
    """Codes are a bijection onto the table's lanes, reserved pair first."""
    table = _table_for(names)
    codes = list(range(len(table.names)))
    return {
        "names": table.names,
        "head": table.names[:2],
        "rate_names": set(table.rates),
        "decoded": [table.decode(table.encode(name)) for name in table.names],
        "names_or_none": [
            None if name == DEFAULT_RATE_LIMIT else name
            for name in table.names
        ],
        "encoded": [table.encode(table.decode(code)) for code in codes],
        "codes": codes,
    }


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "decode accepts exactly the codes the table holds, else ValueError",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[2] == result[3]
    ),
    "encode accepts exactly the declared names, else ValueError",
)
def rate_limit_table_rejects_foreign_values(
    names: list[str], code: int, name: str
) -> tuple[bool, bool, bool, bool]:
    """Out-of-range codes and undeclared names are loud, in-range ones quiet.

    Returns ``(decode_accepted, code_in_range, encode_accepted,
    name_declared)``.
    """
    table = _table_for(names)
    try:
        table.decode(code)
        decode_accepted = True
    except ValueError:
        decode_accepted = False
    try:
        table.encode(name)
        encode_accepted = True
    except ValueError:
        encode_accepted = False
    return (
        decode_accepted,
        0 <= code < len(table.names),
        encode_accepted,
        name in table.names,
    )


# --- _requote_uri -----------------------------------------------------------

#: Escapes of URL delimiters that must pass through normalization verbatim:
#: decoding any of them would change the URL's structure.
ENCODED_DELIMITERS: tuple[str, ...] = ("%26", "%2F", "%3D", "%3F", "%23")
_STRAY_PERCENT = re.compile(r"%(?![0-9A-Fa-f]{2})")


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["once"] == result["twice"]
    ),
    "_requote_uri is idempotent",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["strays_out"] == 0
    ),
    "every % in the output starts a valid escape",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["delimiters_in"] == result["delimiters_out"]
    ),
    "encoded delimiters survive unchanged, one for one",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["pct25_out"] == result["pct25_in"] + result["strays_in"]
    ),
    "each stray % becomes exactly one %25 and nothing else does",
)
def requote_uri_normalizes_escapes(uri: str) -> dict[str, object]:
    """The laws in ``_requote_uri``'s docstring, checked independently.

    A stray ``%`` is one not followed by two hex digits, counted here by a
    regex rather than the function's own scanner.
    """
    once = _requote_uri(uri)
    twice = _requote_uri(once)
    return {
        "once": once,
        "twice": twice,
        "strays_in": len(_STRAY_PERCENT.findall(uri)),
        "strays_out": len(_STRAY_PERCENT.findall(once)),
        "delimiters_in": {d: uri.count(d) for d in ENCODED_DELIMITERS},
        "delimiters_out": {d: once.count(d) for d in ENCODED_DELIMITERS},
        "pct25_in": uri.count("%25"),
        "pct25_out": once.count("%25"),
    }


# --- timestamps -------------------------------------------------------------

_SQLITE = _sqlite_dialect.dialect()
# The dialect-adapted type: the real SQLAlchemy path a mapped column goes
# through, with the SQLite DATETIME string format underneath.
_UTC_DATETIME = UtcDateTime().dialect_impl(_SQLITE)


def _python_timestamp_text(value: datetime) -> str:
    """The text a Python-written aware datetime lands in the column as.

    ``UtcDateTime`` normalises to naive UTC and renders ``TIMESTAMP_FORMAT``
    itself — three fractional digits, the microsecond floored to the
    millisecond — so the text has the same width as a server-written one.
    """
    process = _UTC_DATETIME.bind_processor(_SQLITE)
    assert process is not None, "UtcDateTime always binds through a processor"
    text = process(value)
    assert isinstance(text, str)
    return text


def _sqlite_timestamp_text(value: datetime) -> str:
    """The text SQLite's ``server_default`` writes for the same instant.

    ``strftime(TIMESTAMP_FORMAT, 'now')`` renders SQLite's clock, which is
    millisecond-resolution (its current-time reads truncate microseconds),
    with ``%f``'s three fractional digits. So the instant is floored to the
    millisecond before SQLite renders it — handing SQLite six digits would
    make it *round*, which is not what ``'now'`` does.
    """
    utc = value.astimezone(timezone.utc).replace(tzinfo=None)
    floored = utc.replace(microsecond=utc.microsecond // 1000 * 1000)
    # closing(), not the connection's own ``with`` (a transaction scope).
    with contextlib.closing(sqlite3.connect(":memory:")) as conn:
        return conn.execute(
            "SELECT strftime(?, ?)", (TIMESTAMP_FORMAT, floored.isoformat(" "))
        ).fetchone()[0]


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result["earlier_text"] <= result["later_text"]
    ),
    "timestamp text sorts in chronological order, whichever writer "
    "produced each side",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        not result["distinct_millis"]
        or result["earlier_text"] < result["later_text"]
    ),
    "timestamps in distinct milliseconds sort strictly, whichever writer "
    "produced each side",
)
def timestamp_text_orders_chronologically(
    a: datetime, b: datetime, a_from_python: bool, b_from_python: bool
) -> dict[str, object]:
    """``ORDER BY created_at`` on the raw text agrees with the clock.

    The module docstring of ``timestamps`` promises that the stored text
    sorts lexicographically in chronological order whichever writer
    produced it. Each side here is rendered by the writer its flag selects
    (Python through ``UtcDateTime``, SQLite through ``strftime``); ``a``/
    ``b`` are ordered inside the property so the postconditions read
    earlier <= later, strictly when the two instants fall in different
    milliseconds (both writers floor to the millisecond, so instants inside
    one legitimately render identically).

    Guards a fixed bug: Python-written values used to carry six fractional
    digits while SQLite wrote three, and a three-digit rendering is a
    *prefix* of the six-digit rendering of any instant in the same
    millisecond, so the prefix sorted first. Counterexample
    ``a=…:00.000001`` written from Python and ``b=…:00.000900`` written by
    SQLite — ``b`` is later, but ``"…:00.000" < "…:00.000001"``. Both
    writers now render three digits.
    """
    earlier, later = (a, b) if a <= b else (b, a)
    earlier_from_python, later_from_python = (
        (a_from_python, b_from_python)
        if a <= b
        else (b_from_python, a_from_python)
    )
    render = {True: _python_timestamp_text, False: _sqlite_timestamp_text}
    return {
        "earlier_text": render[earlier_from_python](earlier),
        "later_text": render[later_from_python](later),
        "distinct_millis": _floor_millis(earlier) != _floor_millis(later),
    }


def _floor_millis(value: datetime) -> datetime:
    """The instant floored to the millisecond, in UTC — what either writer stores."""
    utc = value.astimezone(timezone.utc)
    return utc.replace(microsecond=utc.microsecond // 1000 * 1000)


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0] == result[1]
    ),
    "an aware datetime reads back as the same instant, in UTC",
)
@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result[0].tzinfo is timezone.utc
    ),
    "what is read back is aware UTC",
)
def utc_datetime_round_trips(value: datetime) -> tuple[datetime, datetime]:
    """``UtcDateTime`` bind then result gives ``value`` in UTC, to the millisecond.

    The stored text carries three fractional digits, so the microsecond is
    floored on the way in; that is the documented precision of every
    timestamp column, not a loss particular to Python-written values.
    """
    text = _python_timestamp_text(value)
    process = _UTC_DATETIME.result_processor(_SQLITE, None)
    assert process is not None, "UtcDateTime always reads through a processor"
    restored = process(text)
    assert isinstance(restored, datetime)
    return (restored, _floor_millis(value))


@icontract.ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result is True
    ),
    "a naive datetime is rejected with ValueError",
)
def utc_datetime_rejects_naive(value: datetime) -> bool:
    """A naive value cannot be made correct, so it must not be stored."""
    try:
        _python_timestamp_text(value)
    except ValueError:
        return True
    return False
