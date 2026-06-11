"""Generative + integration rig for the replay error-free subsystem.

Three layers, previously untested end to end:

1. **Pruning-plan model oracle** (generative): Hypothesis draws request
   forests — arbitrary parent links, reseedable flags, and per-row error
   states — which are materialized into a real source DB. A pure-Python
   model recomputes the expected mode-3 plan; ``compute_pruning_plan``
   must match it exactly. This also finally *executes* the dev-time
   contracts on ``error_pruning`` (armed via conftest) and the mode-3
   ``SourceIndex`` walk methods.

2. **Mode-aware index build** (same generated forests): for each of the
   three ``MatchMode``s the ``ReplayTransport`` must serve / miss
   exactly the rows the model predicts — ``curr-error-free`` serves
   errored rows and records them as retry-eligible parents,
   ``prev-error-free`` misses them, ``desc-error-free`` misses every
   pruned ancestor and re-seeds exactly the anchors.

3. **Miss-policy routing** (integration): a real ``ReplayRun`` driven
   over seeded output rows pins the ``ReplayWorker`` taxonomy — hit →
   completed; miss → stub/skip/raise; in-step ``TransientException`` →
   reseedable-walk stub; children of a retry-eligible parent force-missed
   unless ``trust_subtree_after_retry``; ``aclose`` flips stubs to
   pending.

Error states model the retry-eligible gate: only an *unresolved*
``HTMLStructuralAssumptionException``-class error counts; resolved or
transient-type errors must not prune.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from hypothesis import given
from hypothesis import strategies as st

from jkent.common.decorators import entry, step
from jkent.common.exceptions import RequestFailedHalt, TransientException
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.replay.error_pruning import compute_pruning_plan
from jkent.driver.replay.errors import ReplayScraperMismatchError
from jkent.driver.replay.source_index import SourceIndex
from jkent.driver.unified_driver.compression import compress
from jkent.driver.unified_driver.replay_run import ReplayRun
from jkent.driver.unified_driver.transport import QueuedRequest
from jkent.driver.unified_driver.transport.replay_transport import (
    ReplayMiss,
    ReplayTransport,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

pytestmark = pytest.mark.generative

# Per-node error states (the retry-eligible gate under test):
_NO_ERROR = "none"
_ELIGIBLE = "eligible-unresolved"  # structural, unresolved -> prunes
_RESOLVED = "eligible-resolved"  # structural but resolved -> must not
_INELIGIBLE = "ineligible-unresolved"  # transient-type -> must not


@dataclass(frozen=True)
class _Node:
    """One generated source-DB request row."""

    parent: int | None  # index into the forest, or None for a root
    reseedable: bool | None
    error: str


@st.composite
def _forests(draw: st.DrawFn) -> list[_Node]:
    """A forest of 1..10 rows; node i may only point at an earlier node."""
    size = draw(st.integers(min_value=1, max_value=10))
    nodes: list[_Node] = []
    for i in range(size):
        parent = draw(st.none() | st.integers(0, i - 1)) if i > 0 else None
        reseedable = draw(st.sampled_from([None, False, True]))
        error = draw(
            st.sampled_from([_NO_ERROR, _ELIGIBLE, _RESOLVED, _INELIGIBLE])
        )
        nodes.append(_Node(parent=parent, reseedable=reseedable, error=error))
    return nodes


def _key(i: int) -> str:
    return f"replay-rig-key-{i}"


def _url(i: int) -> str:
    return f"https://replay-rig.test/{i}"


def _materialize_forest(
    template: Path,
    dest: Path,
    nodes: list[_Node],
    *,
    scraper_name: str | None = None,
) -> None:
    """Insert the forest as completed, response-bearing source rows.

    Node ``i`` becomes request id ``i + 1``. Every row passes the index
    inclusion gate (status code + compressed content); error states
    become ``errors`` rows with the matching type/resolution.
    """
    shutil.copy(template, dest)
    conn = sqlite3.connect(str(dest))
    try:
        if scraper_name is not None:
            conn.execute(
                "INSERT INTO run_metadata (id, scraper_name, base_delay, "
                "jitter, num_workers, max_backoff_time) "
                "VALUES (1, ?, 0, 0, 1, 0)",
                (scraper_name,),
            )
        for i, node in enumerate(nodes):
            content = f"<html>row {i}</html>".encode()
            compressed = compress(content)
            conn.execute(
                """
                INSERT INTO requests (
                    id, status, priority, queue_counter, method, url,
                    continuation, current_location, deduplication_key,
                    request_type, parent_request_id, reseedable,
                    response_status_code, response_url,
                    response_headers_json, content_compressed,
                    content_size_original, content_size_compressed,
                    completed_at_ns, created_at_ns)
                VALUES (?, 'completed', 9, ?, 'GET', ?, 'parse', '', ?,
                        'navigating', ?, ?, 200, ?, '{}', ?, ?, ?, 1, 1)
                """,
                (
                    i + 1,
                    i + 1,
                    _url(i),
                    _key(i),
                    None if node.parent is None else node.parent + 1,
                    node.reseedable,
                    _url(i),
                    compressed,
                    len(content),
                    len(compressed),
                ),
            )
            if node.error != _NO_ERROR:
                error_type = (
                    "TransientException"
                    if node.error == _INELIGIBLE
                    else "HTMLStructuralAssumptionException"
                )
                conn.execute(
                    "INSERT INTO errors (request_id, error_type, "
                    "error_class, message, request_url, is_resolved) "
                    "VALUES (?, ?, 'x.Y', 'scripted', ?, ?)",
                    (
                        i + 1,
                        error_type,
                        _url(i),
                        1 if node.error == _RESOLVED else 0,
                    ),
                )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _expected_plan(
    nodes: list[_Node],
) -> tuple[list[tuple[int, int]], set[int]]:
    """The model oracle: (anchor multiset, excluded ids) in request ids.

    For each eligible-unresolved row, walk to the first reseedable=True
    ancestor (the row itself counts) or the root; the anchor is
    re-seeded, and every node from the errored row up to the anchor
    inclusive is excluded. Duplicate anchors (two errored rows sharing
    one) appear once per errored row, mirroring the implementation.
    """
    anchors: list[tuple[int, int]] = []
    excluded: set[int] = set()
    for i, node in enumerate(nodes):
        if node.error != _ELIGIBLE:
            continue
        chain: list[int] = []
        current: int | None = i
        while current is not None:
            chain.append(current)
            current = nodes[current].parent
        depth = next(
            (d for d, j in enumerate(chain) if nodes[j].reseedable is True),
            len(chain) - 1,
        )
        anchors.append((chain[depth] + 1, depth))
        excluded.update(j + 1 for j in chain[: depth + 1])
    return anchors, excluded


def _request_for(i: int) -> Request:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=_url(i)),
        continuation="parse",
        deduplication_key=_key(i),
    )


class _ForestWorkdir:
    """Per-example DB paths under one session temp dir."""

    def __init__(self, workdir: Path, schema_template: Path) -> None:
        self.workdir = workdir
        self.schema_template = schema_template
        self._counter = itertools.count()

    def materialize(self, nodes: list[_Node]) -> Path:
        dest = self.workdir / f"src-{next(self._counter)}.db"
        _materialize_forest(self.schema_template, dest, nodes)
        return dest

    def fresh_path(self, stem: str) -> Path:
        return self.workdir / f"{stem}-{next(self._counter)}.db"


@pytest.fixture(scope="module")
def forest_workdir(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> _ForestWorkdir:
    return _ForestWorkdir(
        tmp_path_factory.mktemp("replay_error_free"), schema_template
    )


@given(nodes=_forests())
def test_pruning_plan_matches_model(
    forest_workdir: _ForestWorkdir, nodes: list[_Node]
) -> None:
    """``compute_pruning_plan`` equals the pure-Python oracle exactly."""
    src = forest_workdir.materialize(nodes)
    index = SourceIndex(source_db_paths=[src])
    try:
        plan = compute_pruning_plan(index)
    finally:
        index.close()

    expected_anchors, expected_excluded = _expected_plan(nodes)
    assert sorted(plan.anchors[0]) == sorted(expected_anchors)
    assert plan.excluded_request_ids[0] == expected_excluded


@given(nodes=_forests())
def test_modes_serve_and_miss_exactly_what_the_model_predicts(
    forest_workdir: _ForestWorkdir, nodes: list[_Node]
) -> None:
    """Per mode: each row resolves or misses exactly as modeled.

    - curr-error-free serves everything, recording errored rows as
      retry-eligible parents at resolve time;
    - prev-error-free misses exactly the eligible-unresolved rows;
    - desc-error-free misses exactly the pruned ancestor sets and
      re-seeds exactly the anchors.
    """
    src = forest_workdir.materialize(nodes)
    eligible = {i for i, n in enumerate(nodes) if n.error == _ELIGIBLE}
    expected_anchors, expected_excluded = _expected_plan(nodes)

    async def drive_mode(mode: str) -> tuple[set[int], set[int], list[dict]]:
        """Resolve every row; return (hits, misses, anchor seed rows)."""
        transport = ReplayTransport([src], mode=mode)  # type: ignore[arg-type]
        await transport.open()
        try:
            handle = await transport.acquire(0)
            hits: set[int] = set()
            misses: set[int] = set()
            retry_parents: set[int] = set()
            for i in range(len(nodes)):
                queued = QueuedRequest(
                    request=_request_for(i), request_id=100 + i
                )
                try:
                    response = await transport.resolve(handle, queued)
                except ReplayMiss:
                    misses.add(i)
                    continue
                assert isinstance(response, Response)
                assert response.content == f"<html>row {i}</html>".encode()
                hits.add(i)
                if transport.is_retry_eligible_parent(100 + i):
                    retry_parents.add(i)
            seeds = list(transport.seed_anchor_rows())
            assert retry_parents == (eligible & hits), (
                "retry-eligible recording must match the errored hits"
            )
            return hits, misses, seeds
        finally:
            await transport.aclose()

    all_rows = set(range(len(nodes)))

    hits, misses, seeds = asyncio.run(drive_mode("curr-error-free"))
    assert hits == all_rows and misses == set() and seeds == []

    hits, misses, seeds = asyncio.run(drive_mode("prev-error-free"))
    assert misses == eligible and hits == all_rows - eligible
    assert seeds == []

    hits, misses, seeds = asyncio.run(drive_mode("desc-error-free"))
    excluded_indices = {rid - 1 for rid in expected_excluded}
    assert misses == excluded_indices
    assert hits == all_rows - excluded_indices
    assert sorted(row["deduplication_key"] for row in seeds) == sorted(
        _key(rid - 1) for rid, _depth in expected_anchors
    )


# --- Layer 3: ReplayRun / ReplayWorker miss-policy routing -----------------

_TRANSIENT_MARKER = "/transient"
_CHILD_INDEX = 1  # forest position of the child row in the parent/child tests


class ReplayRigScraper(BaseScraper[dict]):
    """Scripted steps: transient on marker URLs, child-yield on row 0."""

    BASE_URL = "http://127.0.0.1"
    yields_child = False

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        if _TRANSIENT_MARKER in response.url:
            raise TransientException("scripted replay transient")
        if self.yields_child and response.url == _url(0):
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=_url(_CHILD_INDEX)
                ),
                continuation="parse",
                deduplication_key=_key(_CHILD_INDEX),
            )


def _scraper_name(scraper: BaseScraper[Any]) -> str:
    return f"{scraper.__class__.__module__}:{scraper.__class__.__name__}"


async def _seed_output(
    run: ReplayRun, rows: list[tuple[str, str]]
) -> list[int]:
    """Insert pending output rows as (url, dedup_key); return their ids."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        for counter, (url, key) in enumerate(rows, start=1):
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location, deduplication_key)
                    VALUES ('pending', 9, :qc, 'GET', :url, 'parse', '',
                            :key)
                    """
                ),
                {"qc": counter, "url": url, "key": key},
            )
        await session.commit()
        result = await session.execute(
            sa.text("SELECT id FROM requests ORDER BY id")
        )
        return [row[0] for row in result]


async def _output_statuses(run: ReplayRun) -> dict[str, str]:
    """Map output-row url -> status."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        result = await session.execute(
            sa.text("SELECT url, status FROM requests")
        )
        return {row[0]: row[1] for row in result}


def _make_replay_run(
    forest_workdir: _ForestWorkdir,
    src: Path,
    scraper: BaseScraper[Any],
    **kwargs: Any,
) -> ReplayRun:
    return ReplayRun(
        scraper,
        forest_workdir.fresh_path("out"),
        source_db_paths=[src],
        resume=False,
        num_workers=1,
        **kwargs,
    )


def _stored_root() -> list[_Node]:
    """A single clean stored row."""
    return [_Node(parent=None, reseedable=None, error=_NO_ERROR)]


_MISSING_URL = "https://replay-rig.test/not-stored"
_MISSING_KEY = "replay-rig-key-not-stored"


@pytest.mark.parametrize("policy", ["stub", "skip"])
async def test_miss_policy_terminal_states(
    forest_workdir: _ForestWorkdir, policy: str
) -> None:
    """Hit completes; a miss is stubbed->pending (stub) or deleted (skip)."""
    scraper = ReplayRigScraper()
    src = forest_workdir.fresh_path("src")
    _materialize_forest(
        forest_workdir.schema_template,
        src,
        _stored_root(),
        scraper_name=_scraper_name(scraper),
    )
    run = _make_replay_run(forest_workdir, src, scraper, miss_policy=policy)
    await run.open(setup_signal_handlers=False)
    try:
        await _seed_output(
            run, [(_url(0), _key(0)), (_MISSING_URL, _MISSING_KEY)]
        )
        await run.run()
        statuses = await _output_statuses(run)
        assert statuses[_url(0)] == "completed"  # the hit
        if policy == "stub":
            assert statuses[_MISSING_URL] == "stubbed"
        else:
            assert _MISSING_URL not in statuses  # skip deletes the row
    finally:
        await run.aclose()

    # aclose finalizes: stubs become pending for a downstream re-fetch run.
    if policy == "stub":
        conn = sqlite3.connect(str(run.db_path))
        try:
            row = conn.execute(
                "SELECT status FROM requests WHERE url = ?", (_MISSING_URL,)
            ).fetchone()
        finally:
            conn.close()
        assert row is not None and row[0] == "pending"


async def test_miss_policy_raise_halts_the_run(
    forest_workdir: _ForestWorkdir,
) -> None:
    """``miss_policy="raise"`` surfaces a miss as a run-halting failure."""
    scraper = ReplayRigScraper()
    src = forest_workdir.fresh_path("src")
    _materialize_forest(
        forest_workdir.schema_template,
        src,
        _stored_root(),
        scraper_name=_scraper_name(scraper),
    )
    run = _make_replay_run(forest_workdir, src, scraper, miss_policy="raise")
    await run.open(setup_signal_handlers=False)
    try:
        await _seed_output(run, [(_MISSING_URL, _MISSING_KEY)])
        with pytest.raises(RequestFailedHalt):
            await run.run()
    finally:
        await run.aclose()


async def test_scraper_class_mismatch_refuses_to_open(
    forest_workdir: _ForestWorkdir,
) -> None:
    """A source DB recorded by a different scraper class is rejected."""
    src = forest_workdir.fresh_path("src")
    _materialize_forest(
        forest_workdir.schema_template,
        src,
        _stored_root(),
        scraper_name="somewhere.else:OtherScraper",
    )
    transport = ReplayTransport([src], scraper=ReplayRigScraper())
    with pytest.raises(ReplayScraperMismatchError) as excinfo:
        await transport.open()
    # The error names both the expected class and the offending DB.
    assert "ReplayRigScraper" in str(excinfo.value)
    assert "OtherScraper" in str(excinfo.value)


@pytest.mark.parametrize("policy", ["stub", "skip"])
async def test_transient_in_step_routes_by_policy(
    forest_workdir: _ForestWorkdir, policy: str
) -> None:
    """A step's TransientException stubs the row (its own anchor at the
    root) under ``stub``, and deletes it under ``skip``."""
    scraper = ReplayRigScraper()
    transient_url = _url(0) + _TRANSIENT_MARKER

    # Materialize a row whose URL carries the transient marker.
    src = forest_workdir.fresh_path("src")
    _materialize_forest(
        forest_workdir.schema_template,
        src,
        _stored_root(),
        scraper_name=_scraper_name(scraper),
    )
    conn = sqlite3.connect(str(src))
    try:
        conn.execute(
            "UPDATE requests SET url = ?, response_url = ? WHERE id = 1",
            (transient_url, transient_url),
        )
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()

    run = _make_replay_run(forest_workdir, src, scraper, miss_policy=policy)
    await run.open(setup_signal_handlers=False)
    try:
        await _seed_output(run, [(transient_url, _key(0))])
        await run.run()
        statuses = await _output_statuses(run)
        if policy == "stub":
            assert statuses[transient_url] == "stubbed"
        else:
            assert transient_url not in statuses  # skip deletes the row
    finally:
        await run.aclose()


def _parent_child_forest() -> list[_Node]:
    """Row 0: stored parent with an unresolved structural error.

    Row 1: clean stored child of row 0.
    """
    return [
        _Node(parent=None, reseedable=None, error=_ELIGIBLE),
        _Node(parent=0, reseedable=None, error=_NO_ERROR),
    ]


@pytest.mark.parametrize("trust", [False, True])
async def test_children_of_retry_eligible_parent(
    forest_workdir: _ForestWorkdir, trust: bool
) -> None:
    """curr-error-free: a retried parent's children are force-missed...

    ...unless ``trust_subtree_after_retry`` is set, in which case they
    resolve normally. The parent itself always resolves (that is the
    point of curr-error-free: re-execute the step on the stored body).
    """

    class ChildYieldingScraper(ReplayRigScraper):
        yields_child = True

    scraper = ChildYieldingScraper()
    src = forest_workdir.fresh_path("src")
    _materialize_forest(
        forest_workdir.schema_template,
        src,
        _parent_child_forest(),
        scraper_name=_scraper_name(scraper),
    )
    run = _make_replay_run(
        forest_workdir,
        src,
        scraper,
        miss_policy="stub",
        mode="curr-error-free",
        trust_subtree_after_retry=trust,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await _seed_output(run, [(_url(0), _key(0))])
        await run.run()
        statuses = await _output_statuses(run)
        assert statuses[_url(0)] == "completed"  # parent re-executed
        if trust:
            assert statuses[_url(_CHILD_INDEX)] == "completed"
        else:
            assert statuses[_url(_CHILD_INDEX)] == "stubbed"
    finally:
        await run.aclose()
