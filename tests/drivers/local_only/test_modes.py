"""Mode-specific behavior of LocalOnlyDriver.

* prev-error-free: previously-errored rows fall through to the miss
  policy (we use ``stub`` here so they end up as pending in the output).
* curr-error-free: the continuation gets re-executed against the stored
  response; on success, children become pending in the output by default
  (no source lookup); with ``--trust-subtree-after-retry``, children go
  through normal lookup.
* desc-error-free: a pre-pass walks each errored row up the parent chain
  to the nearest ``hateoas=True`` ancestor and seeds that ancestor as
  pending in the output. The error-pruning module is tested standalone
  here, since the full end-to-end requires a HATEOAS-aware fixture
  scraper.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlalchemy as sa

from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.local_only_driver.error_pruning import (
    _pick_anchor_depth,
    compute_pruning_plan,
)
from kent.driver.local_only_driver.source_index import SourceIndex
from kent.driver.persistent_driver.sql_manager import SQLManager
from tests.drivers.local_only.conftest import (
    make_bug_court_scraper,
    read_result_data_json,
    run_with_persistent_driver,
)


async def _inject_unresolved_structural_error(
    db_path: Path, url_substring: str
) -> int:
    """Inject an unresolved HTMLStructuralAssumptionException for a request.

    Returns the request_id we attached the error to.
    """
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT id FROM requests WHERE url LIKE ? LIMIT 1",
        (f"%{url_substring}%",),
    ).fetchone()
    if row is None:
        conn.close()
        raise AssertionError(
            f"No request matching url like %{url_substring}% in {db_path}"
        )
    rid = row[0]
    conn.execute(
        """
        INSERT INTO errors (request_id, error_type, error_class, message,
                            request_url, is_resolved)
        VALUES (?, 'HTMLStructuralAssumptionException',
                'HTMLStructuralAssumptionException',
                'injected for test', '', 0)
        """,
        (rid,),
    )
    conn.commit()
    conn.close()
    return rid


async def _status_counts(db_path: Path) -> dict[str, int]:
    async with (
        SQLManager.open(db_path) as sql,
        sql._session_factory() as session,
    ):
        rows = (
            await session.execute(
                sa.text(
                    "SELECT status, COUNT(*) FROM requests GROUP BY status"
                )
            )
        ).all()
    return {r[0]: r[1] for r in rows}


@pytest.mark.asyncio
async def test_prev_error_free_stubs_previously_errored_rows(
    bug_court_server, tmp_path: Path
) -> None:
    """prev-error-free excludes errored rows; they become stubs in output."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    # Pretend the list page errored.
    await _inject_unresolved_structural_error(source_db, "/cases")

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="stub",
        mode="prev-error-free",
        num_workers=1,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    counts = await _status_counts(out_db)
    # The list page was excluded → stubbed as pending; its subtree never
    # ran, so no ParsedData results were produced.
    assert counts.get("pending", 0) >= 1
    assert (await read_result_data_json(out_db)) == []


@pytest.mark.asyncio
async def test_curr_error_free_retries_continuation(
    bug_court_server, tmp_path: Path
) -> None:
    """curr-error-free serves the response and re-runs the continuation."""
    source_db = tmp_path / "source.db"
    out_db = tmp_path / "out.db"
    await run_with_persistent_driver(
        make_bug_court_scraper(bug_court_server.url), source_db
    )
    # Inject an error on a single detail row. The error is "in the past";
    # the current scraper code now succeeds. So curr-error-free should
    # serve the response and produce ParsedData. But the row's children
    # (none for parse_detail) are blocked from lookup. With
    # trust_subtree_after_retry=False (default), that doesn't matter
    # because parse_detail doesn't yield child requests.
    await _inject_unresolved_structural_error(source_db, "/cases/")

    async with LocalOnlyDriver.open(
        scraper=make_bug_court_scraper(bug_court_server.url),
        db_path=out_db,
        source_db_paths=[source_db],
        miss_policy="raise",
        mode="curr-error-free",
        num_workers=2,
        enable_monitor=False,
    ) as driver:
        await driver.run(setup_signal_handlers=False)

    # All detail pages should still produce ParsedData (parse_detail
    # succeeds on the current code).
    results = await read_result_data_json(out_db)
    assert results, "curr-error-free retry did not produce any results"


# ---------------------------------------------------------------------------
# error-pruning unit tests (HATEOAS-aware ancestor walk)
# ---------------------------------------------------------------------------


def test_pick_anchor_depth_stops_at_first_true() -> None:
    chain = [(1, None), (2, False), (3, True), (4, None)]
    assert _pick_anchor_depth(chain) == 2


def test_pick_anchor_depth_walks_past_false_and_none() -> None:
    chain = [(1, False), (2, None), (3, True)]
    assert _pick_anchor_depth(chain) == 2


def test_pick_anchor_depth_falls_back_to_root() -> None:
    chain = [(1, None), (2, False), (3, None)]
    assert _pick_anchor_depth(chain) == 2  # the root (last element)


def test_compute_pruning_plan_excludes_anchor_and_descendants(
    tmp_path: Path,
) -> None:
    """A 3-deep chain with hateoas=True in the middle prunes correctly."""
    import zstandard as zstd

    db = tmp_path / "errored.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY,
            deduplication_key TEXT,
            response_status_code INTEGER,
            content_compressed BLOB,
            url TEXT,
            method TEXT,
            body BLOB,
            continuation TEXT,
            parent_request_id INTEGER,
            completed_at_ns INTEGER,
            created_at_ns INTEGER,
            request_type TEXT,
            hateoas BOOLEAN
        );
        CREATE TABLE errors (
            id INTEGER PRIMARY KEY,
            request_id INTEGER,
            error_type TEXT,
            is_resolved BOOLEAN DEFAULT 0
        );
        CREATE TABLE archived_files (id INTEGER PRIMARY KEY);
        CREATE TABLE compression_dicts (
            id INTEGER PRIMARY KEY,
            dictionary_data BLOB
        );
        CREATE TABLE run_metadata (id INTEGER PRIMARY KEY, scraper_name TEXT);
        """
    )
    compressed = zstd.ZstdCompressor().compress(b"<x/>")

    # 1 (root, hateoas=None) -> 2 (hateoas=True) -> 3 (hateoas=False, errored)
    for rid, parent, hateoas in [
        (1, None, None),
        (2, 1, True),
        (3, 2, False),
    ]:
        conn.execute(
            "INSERT INTO requests "
            "(id, deduplication_key, response_status_code, content_compressed, "
            "url, method, continuation, parent_request_id, completed_at_ns, "
            "created_at_ns, request_type, hateoas) "
            "VALUES (?, ?, 200, ?, 'http://x/', 'GET', 'p', ?, 1, 1, "
            "'navigating', ?)",
            (rid, f"K{rid}", compressed, parent, hateoas),
        )
    conn.execute(
        "INSERT INTO errors (request_id, error_type) VALUES (3, "
        "'HTMLStructuralAssumptionException')"
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(source_db_paths=[db])
    try:
        plan = compute_pruning_plan(idx)
        # Anchor for errored row 3 is row 2 (the hateoas=True one).
        assert plan.anchors[0] == [(2, 1)]
        # Excluded: row 2 (anchor) and row 3 (errored descendant).
        # Row 1 (above the anchor) stays in the index.
        assert plan.excluded_request_ids[0] == {2, 3}
    finally:
        idx.close()
