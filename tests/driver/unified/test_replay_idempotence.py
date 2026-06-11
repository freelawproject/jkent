"""Run-level generative rig for replay idempotence (replay-of-replay).

``test_replay_fidelity`` pins the ``ReplayTransport`` in isolation; this rig
pins the property the replay *workflow* promises end to end: replaying a
replay's output DB changes nothing. Hypothesis generates a small site — a
tree of JSON pages with parsed-data items, archive-file leaves, links that
miss (never stored), links carrying POST bodies, reseedable flags, and scripted
parse failures — materializes it as an original source DB, then drives real
``ReplayRun``s over it:

    source ──replay──▶ out₁ ──replay──▶ out₂ ──replay──▶ out₃

Properties:

1. **Idempotence** — without in-step transients, ``project(out₂) ==
   project(out₁)``: same request rows (statuses, methods, bodies, dedup
   keys, parent links), same response statuses/headers/content, same
   results, same archived files, same errors. Misses stay pending, parse
   failures stay pending-with-content, completions stay completed.
2. **Model oracle** — ``out₁`` matches a pure-Python model of the site:
   clean pages complete and emit their data items, archives complete and
   emit their marker datum, missing links and parse-failure pages end
   pending. Failures localize to the first replay rather than the diff.
3. **Fixpoint** — with in-step ``TransientException``s enabled (whose stub
   walks to the nearest reseedable anchor and finalize prunes the anchor's
   descendants), one replay step may not be idempotent — the anchor keeps
   its stored response, so the next replay re-serves and re-expands it.
   The chain must still reach a fixpoint by the second replay:
   ``project(out₃) == project(out₂)``.
4. **Archive verbatim chain** — every replay generation's
   ``archived_files`` rows reference the ORIGINAL source DB's file path,
   and the file bytes are untouched (replay never copies an archive).

The projection deliberately excludes per-run identity and timing — row ids,
``queue_counter``, ``started_at``/``started_at_ns``, ``created_at_ns``,
``completed_at_ns``, ``response_created_at``, retry/backoff counters, and
compression internals (``content_size_compressed``, ``compression_dict_id``;
content is compared decompressed). Those legitimately differ between runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.common.decorators import entry, step
from jkent.common.exceptions import TransientException
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.database_engine.compression import decompress
from jkent.driver.replay.source_index import serialize_url_and_body
from jkent.driver.unified_driver.compression import compress
from jkent.driver.unified_driver.replay_run import ReplayRun

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.generative

_BASE = "https://idem.test"


def _node_url(i: int) -> str:
    return f"{_BASE}/n{i}"


# --- the scraper (fixed code; behavior driven by generated content) --------


def _child_request(spec: dict[str, Any]) -> Request:
    """The one constructor for a node's request.

    Used by BOTH the scraper (from a parent page's content) and the
    source-DB materializer, so the dedup key on the lookup side and the
    index side agree by construction.
    """
    params = HTTPRequestParams(
        method=HttpMethod(spec["method"]),
        url=spec["url"],
        json=spec["payload"],
    )
    if spec["archive"]:
        return Request(
            request=params,
            continuation="parse_archive",
            archive=True,
            expected_type="bin",
            reseedable=spec["reseedable"],
        )
    return Request(
        request=params, continuation="parse", reseedable=spec["reseedable"]
    )


_ROOT_SPEC: dict[str, Any] = {
    "url": _node_url(0),
    "method": "GET",
    "payload": None,
    "archive": False,
    "reseedable": None,
}


class SiteScraper(BaseScraper[dict]):
    """Pages are JSON: data items, child link specs, and failure markers."""

    BASE_URL = _BASE

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        yield _child_request(_ROOT_SPEC)

    @step
    def parse(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        doc = json.loads(response.content)
        if doc["marker"] == "transient":
            raise TransientException("scripted replay transient")
        if doc["marker"] == "boom":
            raise ValueError("scripted parse failure")
        for item in doc["data"]:
            yield ParsedData(data=item)
        for child in doc["children"]:
            yield _child_request(child)

    @step
    def parse_archive(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        yield ParsedData(data={"archived": response.request.request.url})


_SCRAPER_NAME = f"{SiteScraper.__module__}:{SiteScraper.__name__}"


# --- generated sites --------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    """One site node; the link fields describe the request that reaches it."""

    parent: int | None
    kind: str  # "page" | "archive" | "missing"
    method: str
    payload: dict[str, Any] | None
    reseedable: bool | None
    marker: str | None  # None | "boom" | "transient" (pages only)
    status: int
    data: tuple[dict[str, Any], ...]
    headers: dict[str, str]
    file_content: bytes  # archives only


_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=8
)
_str_map = st.dictionaries(_text, _text, max_size=3)
_payloads = st.dictionaries(_text, _text | st.integers(), max_size=2)
_data_items = st.lists(_payloads, max_size=2)


@st.composite
def _sites(draw: st.DrawFn, *, allow_transient: bool) -> list[_Node]:
    """A reachable site tree of 1..6 nodes.

    Node 0 is the root: always a stored, marker-free page (the entry
    point). Only clean stored pages can be parents — a marker page raises
    before yielding children and archive/missing nodes yield none, so
    restricting parents keeps every generated node reachable and the
    model oracle total.
    """
    count = draw(st.integers(min_value=1, max_value=6))
    markers = [None, None, "boom"] + (["transient"] if allow_transient else [])
    nodes: list[_Node] = []
    clean_pages: list[int] = []
    for i in range(count):
        if i == 0:
            parent: int | None = None
            kind, method, payload, reseedable = "page", "GET", None, None
            marker = None
        else:
            parent = draw(st.sampled_from(clean_pages))
            kind = draw(
                st.sampled_from(["page", "page", "archive", "missing"])
            )
            method = draw(st.sampled_from(["GET", "POST"]))
            payload = draw(st.none() | _payloads)
            reseedable = draw(st.sampled_from([None, False, True]))
            marker = draw(st.sampled_from(markers)) if kind == "page" else None
        nodes.append(
            _Node(
                parent=parent,
                kind=kind,
                method=method,
                payload=payload,
                reseedable=reseedable,
                marker=marker,
                status=draw(st.sampled_from([200, 404, 500])),
                data=tuple(draw(_data_items)) if kind == "page" else (),
                headers=draw(_str_map),
                file_content=draw(st.binary(max_size=64))
                if kind == "archive"
                else b"",
            )
        )
        if kind == "page" and marker is None:
            clean_pages.append(i)
    return nodes


def _spec_for(nodes: list[_Node], i: int) -> dict[str, Any]:
    """The link spec for node ``i`` (what its parent's content carries)."""
    if i == 0:
        return _ROOT_SPEC
    node = nodes[i]
    return {
        "url": _node_url(i),
        "method": node.method,
        "payload": node.payload,
        "archive": node.kind == "archive",
        "reseedable": node.reseedable,
    }


def _page_content(nodes: list[_Node], i: int) -> bytes:
    """Render page ``i`` as the JSON document the scraper parses."""
    children = [
        _spec_for(nodes, j) for j, node in enumerate(nodes) if node.parent == i
    ]
    doc = {
        "marker": nodes[i].marker,
        "data": list(nodes[i].data),
        "children": children,
    }
    return json.dumps(doc, sort_keys=True).encode()


# --- source-DB materialization ----------------------------------------------

_INSERT_PAGE = """
INSERT INTO requests (
    status, priority, queue_counter, method, url, body, continuation,
    current_location, deduplication_key, request_type, response_status_code,
    response_url, response_headers_json, content_compressed,
    content_size_original, content_size_compressed, compression_dict_id,
    completed_at_ns, created_at_ns)
VALUES ('completed', 9, ?, ?, ?, ?, 'parse', '', ?, 'navigating', ?, ?, ?, ?,
    ?, ?, NULL, ?, ?)
"""

_INSERT_ARCHIVE = """
INSERT INTO requests (
    status, priority, queue_counter, method, url, body, continuation,
    current_location, deduplication_key, request_type, expected_type,
    response_status_code, response_url, response_headers_json,
    completed_at_ns, created_at_ns)
VALUES ('completed', 9, ?, ?, ?, ?, 'parse_archive', '', ?, 'archive', 'bin',
    ?, ?, ?, ?, ?)
"""


def _materialize_site(
    template: Path, dest: Path, files_dir: Path, nodes: list[_Node]
) -> dict[int, Path]:
    """Write the original source DB + archive files; return node → file."""
    shutil.copy(template, dest)
    files_dir.mkdir(exist_ok=True)
    archive_files: dict[int, Path] = {}
    conn = sqlite3.connect(str(dest))
    try:
        conn.execute(
            "INSERT INTO run_metadata (id, scraper_name, base_delay, jitter, "
            "num_workers, max_backoff_time) VALUES (1, ?, 0, 0, 1, 0)",
            (_SCRAPER_NAME,),
        )
        for i, node in enumerate(nodes):
            if node.kind == "missing":
                continue
            request = _child_request(_spec_for(nodes, i))
            ser_url, body = serialize_url_and_body(request.request)
            assert isinstance(request.deduplication_key, str)
            if node.kind == "archive":
                cur = conn.execute(
                    _INSERT_ARCHIVE,
                    (
                        i + 1,
                        node.method,
                        ser_url,
                        body,
                        request.deduplication_key,
                        node.status,
                        _node_url(i),
                        json.dumps(node.headers),
                        i + 1,
                        i + 1,
                    ),
                )
                file_path = files_dir / f"n{i}.bin"
                file_path.write_bytes(node.file_content)
                archive_files[i] = file_path
                conn.execute(
                    "INSERT INTO archived_files (request_id, file_path, "
                    "original_url, expected_type, file_size, content_hash) "
                    "VALUES (?, ?, ?, 'bin', ?, ?)",
                    (
                        cur.lastrowid,
                        str(file_path),
                        _node_url(i),
                        len(node.file_content),
                        hashlib.sha256(node.file_content).hexdigest(),
                    ),
                )
            else:
                content = _page_content(nodes, i)
                compressed = compress(content)
                conn.execute(
                    _INSERT_PAGE,
                    (
                        i + 1,
                        node.method,
                        ser_url,
                        body,
                        request.deduplication_key,
                        node.status,
                        _node_url(i),
                        json.dumps(node.headers),
                        compressed,
                        len(content),
                        len(compressed),
                        i + 1,
                        i + 1,
                    ),
                )
        conn.commit()
        # Fold the WAL into the main file for the read-only SourceIndex.
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    return archive_files


# --- driving the replay chain ------------------------------------------------


def _replay_chain(workdir: Path, source: Path, generations: int) -> list[Path]:
    """Replay ``source``, then each output, ``generations`` times."""
    outputs: list[Path] = []

    async def go() -> None:
        src = source
        for gen in range(1, generations + 1):
            out = workdir / f"out{gen}.db"
            run = ReplayRun(
                SiteScraper(),
                out,
                source_db_paths=[src],
                miss_policy="stub",
                resume=False,
                num_workers=1,
            )
            await run.open(setup_signal_handlers=False)
            try:
                await run.run()
            finally:
                await run.aclose()
            outputs.append(out)
            src = out

    asyncio.run(go())
    return outputs


# --- the semantic projection -------------------------------------------------


def _canon(raw: str | None) -> str | None:
    return None if raw is None else json.dumps(json.loads(raw), sort_keys=True)


def _content_of(row: sqlite3.Row) -> bytes | None:
    blob = row["content_compressed"]
    if blob is None:
        return None
    return blob if blob == b"" else decompress(blob)


def _project(db_path: Path) -> dict[str, Any]:
    """Everything a replay must preserve, keyed run-independently by URL.

    Identity/timing columns are deliberately absent — see module docstring.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM requests").fetchall()
        url_of = {row["id"]: row["url"] for row in rows}
        requests = {}
        for row in rows:
            assert row["compression_dict_id"] is None, (
                "rig assumes no trained compression dict"
            )
            requests[row["url"]] = {
                "status": row["status"],
                "method": row["method"],
                "body": row["body"],
                "json_data": _canon(row["json_data"]),
                "continuation": row["continuation"],
                "request_type": row["request_type"],
                "expected_type": row["expected_type"],
                "deduplication_key": row["deduplication_key"],
                "parent_url": url_of.get(row["parent_request_id"]),
                "reseedable": row["reseedable"],
                "response_status_code": row["response_status_code"],
                "response_url": row["response_url"],
                "response_headers": json.loads(row["response_headers_json"])
                if row["response_headers_json"]
                else {},
                "content": _content_of(row),
                "content_size_original": row["content_size_original"],
            }
        results = sorted(
            (url_of[r["request_id"]], r["result_type"], _canon(r["data_json"]))
            for r in conn.execute("SELECT * FROM results")
        )
        archives = {
            url_of[a["request_id"]]: {
                "file_path": a["file_path"],
                "original_url": a["original_url"],
                "expected_type": a["expected_type"],
                "file_size": a["file_size"],
                "content_hash": a["content_hash"],
                "file_bytes": Path(a["file_path"]).read_bytes(),
            }
            for a in conn.execute("SELECT * FROM archived_files")
        }
        errors = sorted(
            (
                url_of.get(e["request_id"]),
                e["error_type"],
                e["message"],
                e["is_resolved"],
            )
            for e in conn.execute("SELECT * FROM errors")
        )
    finally:
        conn.close()
    return {
        "requests": requests,
        "results": results,
        "archives": archives,
        "errors": errors,
    }


# --- the model oracle for the FIRST replay (no transients) -------------------


def _expected_first_replay(
    nodes: list[_Node],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """url → expected status, and the expected (url, canonical-data) results."""
    statuses: dict[str, str] = {}
    results: list[tuple[str, str]] = []
    for i, node in enumerate(nodes):
        url = _node_url(i)
        if node.kind == "missing" or node.marker is not None:
            statuses[url] = "pending"
        else:
            statuses[url] = "completed"
        if node.kind == "archive":
            results.append(
                (url, json.dumps({"archived": url}, sort_keys=True))
            )
        elif node.kind == "page" and node.marker is None:
            results.extend(
                (url, json.dumps(item, sort_keys=True)) for item in node.data
            )
    return statuses, sorted(results)


# --- properties ---------------------------------------------------------------


@settings(deadline=None)
@given(nodes=_sites(allow_transient=False))
def test_replay_of_replay_is_idempotent(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    nodes: list[_Node],
) -> None:
    """Without in-step transients, the second replay changes nothing."""
    workdir = tmp_path_factory.mktemp("idem")
    source = workdir / "source.db"
    archive_files = _materialize_site(
        schema_template, source, workdir / "files", nodes
    )
    out1, out2 = _replay_chain(workdir, source, generations=2)
    proj1, proj2 = _project(out1), _project(out2)

    # Property 2 — the first replay matches the site model.
    statuses, expected_results = _expected_first_replay(nodes)
    assert {u: r["status"] for u, r in proj1["requests"].items()} == statuses
    assert [(u, d) for u, _t, d in proj1["results"]] == expected_results
    for i, node in enumerate(nodes):
        row = proj1["requests"][_node_url(i)]
        if node.kind == "page":
            # Stored content replays verbatim — including for a 'boom' page,
            # whose response is stored before its step raises and must
            # survive the stub → pending finalization (that retained content
            # is exactly what makes the second replay re-serve it).
            assert row["content"] == _page_content(nodes, i)
            assert row["response_status_code"] == node.status
            assert row["response_headers"] == node.headers
        elif node.kind == "archive":
            assert row["content"] is None
            assert proj1["archives"][_node_url(i)]["file_path"] == str(
                archive_files[i]
            )
            assert (
                proj1["archives"][_node_url(i)]["file_bytes"]
                == node.file_content
            )
        else:  # missing: stubbed → pending with nothing fetched
            assert row["response_status_code"] is None
            assert row["content"] is None

    # Property 4 — the second generation still references the ORIGINAL
    # files, byte-identical (replay never copies an archive).
    for i, path in archive_files.items():
        assert proj2["archives"][_node_url(i)]["file_path"] == str(path)
        assert path.read_bytes() == nodes[i].file_content

    # Property 1 — the headline: replay-of-replay is a no-op.
    assert proj2 == proj1


@settings(deadline=None)
@given(nodes=_sites(allow_transient=True))
def test_replay_chain_reaches_fixpoint_by_second_replay(
    schema_template: Path,
    tmp_path_factory: pytest.TempPathFactory,
    nodes: list[_Node],
) -> None:
    """With transients (reseedable-walk stubs + descendant pruning), the chain
    must stabilize by the second replay: replay #3 reproduces replay #2.

    A transient stubs its nearest reseedable anchor, which keeps its stored
    response; the next replay re-serves the anchor and re-expands its
    pruned subtree (as pending misses), so out₂ may differ from out₁ — but
    out₃ must equal out₂.
    """
    workdir = tmp_path_factory.mktemp("fixpoint")
    source = workdir / "source.db"
    _materialize_site(schema_template, source, workdir / "files", nodes)
    _out1, out2, out3 = _replay_chain(workdir, source, generations=3)
    assert _project(out3) == _project(out2)
