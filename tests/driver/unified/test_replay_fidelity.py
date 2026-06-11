"""Fidelity rig for ``ReplayTransport`` (the v1 replay backend).

Hypothesis generates a run database (request rows + stored responses), then we
point a ``ReplayTransport`` at that file DB and assert it returns the stored
response for each request. The generated DB is the oracle.

Replay matches a request by its **deduplication key**, which is
``hash(method + url + sorted params + data/json)`` (see
``data_types._generate_deduplication_key``) — nothing else. So two properties:

1. **Fidelity** — for every stored row, resolving its request returns exactly
   that response (status code, raw content bytes, response headers, final URL).
2. **Invisibility** — fields that are *not* part of the dedup key do not change
   what replay returns. Holding method + url + params + body fixed while
   varying these yields the same stored response:
       - request headers
       - cookies
       - continuation
       - timeout
   (Also invisible but not exercised here: priority, accumulated_data,
   permanent, bypass_rate_limit, request_id / parent_request_id.)

VISIBLE (they form the dedup key): HTTP method, url, query params, and body
(data/json).

Replay does **not** classify status codes (unlike ``HttpxTransport``): it
returns the stored status verbatim, so a stored 500 comes back as a
``Response(500)`` and is never raised — the fidelity property below covers
error statuses to pin that down.

Beyond fidelity/invisibility, this module pins four more ReplayTransport
properties:

3. **Idempotency / replay-of-replay** — resolving the same stored request
   repeatedly is stable, and a resolved response re-materialized into a fresh DB
   and replayed again is identical.
4. **Concurrent fetch** — ``asyncio.gather`` of distinct ``resolve`` calls on one
   transport (and a two-source-DB variant) shows no cross-corruption; each call
   returns its own stored content/status/url. (``ReplayTransport`` runs
   ``index.fetch_response`` via ``asyncio.to_thread``, so this pins thread-safety.)
5. **Null dedup-key fallback** — a stored row with ``deduplication_key = NULL`` is
   matched via the URL+body fallback and still resolves to its response.
6. **Archive path verbatim** — ``resolve_archive`` streams the SOURCE file path
   (no copy); the streamed bytes equal the on-disk file and ``finish_archiving``
   leaves it in place (replay's file belongs to the source DB).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.replay.source_index import serialize_url_and_body
from jkent.driver.unified_driver import (
    QueuedRequest,
    ReplayMiss,
    ReplayTransport,
)
from jkent.driver.unified_driver.compression import compress

# --- strategies ----------------------------------------------------------

_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=8
)
_str_map = st.dictionaries(_text, _text, max_size=3)
_methods = st.sampled_from(
    [
        HttpMethod.GET,
        HttpMethod.POST,
        HttpMethod.PUT,
        HttpMethod.HEAD,
        HttpMethod.DELETE,
    ]
)


@st.composite
def _ingredients(draw: st.DrawFn) -> dict[str, Any]:
    """Everything for one stored row except its (uniquely assigned) URL."""
    return {
        "method": draw(_methods),
        "params": draw(st.none() | _str_map),
        "json": draw(
            st.none()
            | st.dictionaries(_text, st.integers() | _text, max_size=3)
        ),
        "req_headers": draw(st.none() | _str_map),
        # Error codes included on purpose: replay returns the stored status
        # verbatim and never classifies (contrast HttpxTransport), so a stored
        # 500 comes back as Response(500), not a raised exception.
        "status": draw(st.sampled_from([200, 201, 202, 404, 500, 503])),
        "content": draw(st.binary(max_size=300)),
        "resp_headers": draw(_str_map),
    }


@st.composite
def _invisible(draw: st.DrawFn) -> dict[str, Any]:
    """Variation across fields that must NOT change replay's answer."""
    return {
        "req_headers": draw(st.none() | _str_map),
        "cookies": draw(st.none() | _str_map),
        "continuation": draw(st.sampled_from(["parse", "detail", "other"])),
        "timeout": draw(
            st.none() | st.floats(min_value=0.1, max_value=60, allow_nan=False)
        ),
    }


# --- materialization -----------------------------------------------------


@dataclass
class _Entry:
    request: Request
    status: int
    content: bytes
    resp_headers: dict[str, str]
    response_url: str


def _assemble(ing: dict[str, Any], index: int) -> _Entry:
    """Build an entry with a URL unique to ``index`` (→ unique dedup key)."""
    url = f"https://replay.test/r{index}"
    request = Request(
        request=HTTPRequestParams(
            method=ing["method"],
            url=url,
            params=ing["params"],
            json=ing["json"],
            headers=ing["req_headers"],
        ),
        continuation="parse",
    )
    return _Entry(
        request=request,
        status=ing["status"],
        content=ing["content"],
        resp_headers=ing["resp_headers"],
        response_url=f"{url}/final",
    )


_INSERT = """
INSERT INTO requests (
    status, priority, queue_counter, method, url, body, continuation,
    current_location, deduplication_key, request_type, response_status_code,
    response_url, response_headers_json, content_compressed,
    content_size_original, content_size_compressed, compression_dict_id,
    completed_at_ns, created_at_ns)
VALUES ('completed', 9, ?, ?, ?, ?, ?, '', ?, 'navigating', ?, ?, ?, ?, ?, ?,
    NULL, ?, ?)
"""


def _materialize(template: Path, dest: Path, entries: list[_Entry]) -> Path:
    """Copy the schema template and insert one requests row per entry."""
    shutil.copy(template, dest)
    conn = sqlite3.connect(str(dest))
    try:
        for i, e in enumerate(entries):
            url, body = serialize_url_and_body(e.request.request)
            compressed = compress(e.content)
            assert isinstance(e.request.deduplication_key, str)
            conn.execute(
                _INSERT,
                (
                    i + 1,
                    e.request.request.method.value,
                    url,
                    body,
                    e.request.continuation,
                    e.request.deduplication_key,
                    e.status,
                    e.response_url,
                    json.dumps(e.resp_headers),
                    compressed,
                    len(e.content),
                    len(compressed),
                    i + 1,
                    i + 1,
                ),
            )
        conn.commit()
        # Fold the WAL into the main file so the read-only SourceIndex sees it.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return dest


# --- properties ----------------------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(ingredients=st.lists(_ingredients(), min_size=1, max_size=6))
def test_replay_returns_each_stored_response(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    ingredients: list[dict[str, Any]],
) -> None:
    entries = [_assemble(ing, i) for i, ing in enumerate(ingredients)]
    dest = tmp_path_factory.mktemp("run") / "run.db"
    _materialize(schema_template, dest, entries)

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            for e in entries:
                resp = await transport.resolve(
                    handle, QueuedRequest(request=e.request, request_id=1)
                )
                assert resp.status_code == e.status
                assert resp.content == e.content
                assert resp.headers == e.resp_headers
                assert resp.url == e.response_url
                assert resp.request is e.request
        finally:
            await transport.aclose()

    asyncio.run(run())


@pytest.mark.generative
@settings(deadline=None)
@given(
    ingredients=_ingredients(),
    variations=st.lists(_invisible(), min_size=1, max_size=5),
)
def test_invisible_fields_do_not_change_resolution(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    ingredients: dict[str, Any],
    variations: list[dict[str, Any]],
) -> None:
    entry = _assemble(ingredients, 0)
    dest = tmp_path_factory.mktemp("run") / "run.db"
    _materialize(schema_template, dest, [entry])
    base = entry.request.request  # method + url + params + json define the key

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            for v in variations:
                variant = Request(
                    request=HTTPRequestParams(
                        method=base.method,
                        url=base.url,
                        params=base.params,
                        json=base.json,
                        headers=v["req_headers"],
                        cookies=v["cookies"],
                        timeout=v["timeout"],
                    ),
                    continuation=v["continuation"],
                )
                # Same method+url+params+body → same dedup key as the
                # stored row.
                assert (
                    variant.deduplication_key
                    == entry.request.deduplication_key
                )
                resp = await transport.resolve(
                    handle, QueuedRequest(request=variant, request_id=1)
                )
                assert resp.content == entry.content
                assert resp.status_code == entry.status
                assert resp.url == entry.response_url
        finally:
            await transport.aclose()

    asyncio.run(run())


async def test_unstored_request_raises_miss(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    dest = tmp_path_factory.mktemp("run") / "empty.db"
    _materialize(schema_template, dest, [])

    transport = ReplayTransport([dest])
    await transport.open()
    handle = await transport.acquire(0)
    try:
        req = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://replay.test/missing"
            ),
            continuation="parse",
        )
        with pytest.raises(ReplayMiss):
            await transport.resolve(
                handle, QueuedRequest(request=req, request_id=1)
            )
    finally:
        await transport.aclose()


# --- idempotency / replay-of-replay --------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(ingredients=st.lists(_ingredients(), min_size=1, max_size=4))
def test_resolve_is_idempotent_and_replay_of_replay_stable(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    ingredients: list[dict[str, Any]],
) -> None:
    """Same request resolves identically N times; re-stored + replayed is stable."""
    entries = [_assemble(ing, i) for i, ing in enumerate(ingredients)]
    work = tmp_path_factory.mktemp("idem")
    dest = work / "run.db"
    _materialize(schema_template, dest, entries)

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            # (a) Resolving the same stored request repeatedly is identical.
            for e in entries:
                queued = QueuedRequest(request=e.request, request_id=1)
                first = await transport.resolve(handle, queued)
                for _ in range(3):
                    again = await transport.resolve(handle, queued)
                    assert again.status_code == first.status_code
                    assert again.content == first.content
                    assert again.headers == first.headers
                    assert again.url == first.url
        finally:
            await transport.aclose()

        # (b) Replay-of-replay: re-store each resolved response into a fresh DB,
        # replay it, and assert it matches the original stored row.
        replayed = [
            _Entry(
                request=e.request,
                status=e.status,
                content=e.content,
                resp_headers=e.resp_headers,
                response_url=e.response_url,
            )
            for e in entries
        ]
        dest2 = work / "run2.db"
        _materialize(schema_template, dest2, replayed)
        transport2 = ReplayTransport([dest2])
        await transport2.open()
        handle2 = await transport2.acquire(0)
        try:
            for e in entries:
                resp = await transport2.resolve(
                    handle2, QueuedRequest(request=e.request, request_id=1)
                )
                assert resp.status_code == e.status
                assert resp.content == e.content
                assert resp.headers == e.resp_headers
                assert resp.url == e.response_url
        finally:
            await transport2.aclose()

    asyncio.run(run())


# --- concurrent fetch ----------------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(ingredients=st.lists(_ingredients(), min_size=2, max_size=6))
def test_concurrent_resolve_no_cross_corruption(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    ingredients: list[dict[str, Any]],
) -> None:
    """gather()-ed resolves on one transport each return THEIR stored row."""
    entries = [_assemble(ing, i) for i, ing in enumerate(ingredients)]
    dest = tmp_path_factory.mktemp("concurrent") / "run.db"
    _materialize(schema_template, dest, entries)

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            results = await asyncio.gather(
                *(
                    transport.resolve(
                        handle, QueuedRequest(request=e.request, request_id=1)
                    )
                    for e in entries
                )
            )
            for e, resp in zip(entries, results, strict=True):
                assert resp.status_code == e.status
                assert resp.content == e.content
                assert resp.url == e.response_url
        finally:
            await transport.aclose()

    asyncio.run(run())


@pytest.mark.generative
@settings(deadline=None)
@given(
    left=st.lists(_ingredients(), min_size=1, max_size=3),
    right=st.lists(_ingredients(), min_size=1, max_size=3),
)
def test_concurrent_resolve_across_two_source_dbs(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> None:
    """Concurrent resolves spanning two source DBs stay un-corrupted."""
    work = tmp_path_factory.mktemp("two_db")
    # Disjoint URL ranges so each entry has a unique dedup key across DBs.
    left_entries = [_assemble(ing, i) for i, ing in enumerate(left)]
    right_entries = [_assemble(ing, 100 + i) for i, ing in enumerate(right)]
    db_a = work / "a.db"
    db_b = work / "b.db"
    _materialize(schema_template, db_a, left_entries)
    _materialize(schema_template, db_b, right_entries)
    entries = left_entries + right_entries

    async def run() -> None:
        transport = ReplayTransport([db_a, db_b])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            results = await asyncio.gather(
                *(
                    transport.resolve(
                        handle, QueuedRequest(request=e.request, request_id=1)
                    )
                    for e in entries
                )
            )
            for e, resp in zip(entries, results, strict=True):
                assert resp.status_code == e.status
                assert resp.content == e.content
                assert resp.url == e.response_url
        finally:
            await transport.aclose()

    asyncio.run(run())


# --- null dedup-key fallback ---------------------------------------------


_INSERT_NULL_DEDUP = """
INSERT INTO requests (
    status, priority, queue_counter, method, url, body, continuation,
    current_location, deduplication_key, request_type, response_status_code,
    response_url, response_headers_json, content_compressed,
    content_size_original, content_size_compressed, compression_dict_id,
    completed_at_ns, created_at_ns)
VALUES ('completed', 9, 1, ?, ?, ?, 'parse', '', NULL, 'navigating', ?, ?, ?,
    ?, ?, ?, NULL, 1, 1)
"""


def _materialize_null_dedup(template: Path, dest: Path, entry: _Entry) -> None:
    """Materialize a single row whose ``deduplication_key`` column is NULL."""
    shutil.copy(template, dest)
    conn = sqlite3.connect(str(dest))
    try:
        url, body = serialize_url_and_body(entry.request.request)
        compressed = compress(entry.content)
        conn.execute(
            _INSERT_NULL_DEDUP,
            (
                entry.request.request.method.value,
                url,
                body,
                entry.status,
                entry.response_url,
                json.dumps(entry.resp_headers),
                compressed,
                len(entry.content),
                len(compressed),
            ),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


@pytest.mark.generative
@settings(deadline=None)
@given(ingredients=_ingredients())
def test_null_dedup_key_resolved_via_url_body_fallback(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    ingredients: dict[str, Any],
) -> None:
    """A NULL-dedup_key row is matched by the URL+body fallback key."""
    entry = _assemble(ingredients, 0)
    dest = tmp_path_factory.mktemp("null_dedup") / "run.db"
    _materialize_null_dedup(schema_template, dest, entry)

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            resp = await transport.resolve(
                handle, QueuedRequest(request=entry.request, request_id=1)
            )
            assert resp.status_code == entry.status
            assert resp.content == entry.content
            assert resp.headers == entry.resp_headers
            assert resp.url == entry.response_url
        finally:
            await transport.aclose()

    asyncio.run(run())


# --- archive path verbatim (no copy) -------------------------------------


def _materialize_archive(
    template: Path, dest: Path, file_path: Path, content: bytes, url: str
) -> Request:
    """Record one archive row + archived_files row pointing at ``file_path``."""
    shutil.copy(template, dest)
    file_path.write_bytes(content)
    request = Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse",
    )
    assert isinstance(request.deduplication_key, str)
    conn = sqlite3.connect(str(dest))
    try:
        ser_url, body = serialize_url_and_body(request.request)
        cur = conn.execute(
            """
            INSERT INTO requests (
                status, priority, queue_counter, method, url, body,
                continuation, current_location, deduplication_key,
                request_type, response_status_code, response_url,
                completed_at_ns, created_at_ns)
            VALUES ('completed', 9, 1, 'GET', ?, ?, 'parse', '', ?,
                'archive', 200, ?, 1, 1)
            """,
            (ser_url, body, request.deduplication_key, url),
        )
        conn.execute(
            "INSERT INTO archived_files (request_id, file_path, original_url) "
            "VALUES (?, ?, ?)",
            (cur.lastrowid, str(file_path), url),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return request


@pytest.mark.generative
@settings(deadline=None)
@given(content=st.binary(max_size=400))
def test_archive_streams_source_file_verbatim_without_deleting(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    content: bytes,
) -> None:
    """resolve_archive streams the SOURCE file; finish_archiving leaves it intact."""
    work = tmp_path_factory.mktemp("archive_verbatim")
    dest = work / "run.db"
    file_path = work / "archive.bin"
    url = "https://archive.test/file"
    request = _materialize_archive(
        schema_template, dest, file_path, content, url
    )

    async def run() -> None:
        transport = ReplayTransport([dest])
        await transport.open()
        handle = await transport.acquire(0)
        try:
            stream = await transport.resolve_archive(
                handle, QueuedRequest(request=request, request_id=1)
            )
            streamed = bytearray()
            async for chunk in stream:
                streamed.extend(chunk)
            assert bytes(streamed) == content
            assert bytes(streamed) == file_path.read_bytes()
            assert stream.status_code == 200

            # finish_archiving must NOT delete the source-owned file.
            await transport.finish_archiving(stream)
            assert file_path.exists()
            assert file_path.read_bytes() == content
        finally:
            await transport.aclose()

    asyncio.run(run())
