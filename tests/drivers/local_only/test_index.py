"""Unit tests for :class:`SourceIndex`.

These poke the index directly using small synthetic source DBs so they
run without a scraper or worker loop. They cover the consolidation
policy: most recent ``completed_at_ns`` → most recent ``created_at_ns``
→ earlier-specified DB wins ties.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import zstandard as zstd

from kent.driver.local_only_driver.source_index import SourceIndex


def _create_minimal_db(path: Path) -> sqlite3.Connection:
    """Open a fresh SQLite DB with just the columns SourceIndex reads."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY,
            deduplication_key TEXT,
            response_status_code INTEGER,
            content_compressed BLOB,
            compression_dict_id INTEGER,
            response_headers_json TEXT,
            response_url TEXT,
            url TEXT,
            method TEXT,
            headers_json TEXT,
            cookies_json TEXT,
            body BLOB,
            continuation TEXT,
            current_location TEXT,
            accumulated_data_json TEXT,
            permanent_json TEXT,
            expected_type TEXT,
            verify TEXT,
            bypass_rate_limit INTEGER DEFAULT 0,
            request_type TEXT,
            parent_request_id INTEGER,
            completed_at_ns INTEGER,
            created_at_ns INTEGER,
            priority INTEGER DEFAULT 9,
            hateoas BOOLEAN
        );
        CREATE TABLE errors (
            id INTEGER PRIMARY KEY,
            request_id INTEGER,
            error_type TEXT,
            error_class TEXT,
            message TEXT,
            request_url TEXT,
            is_resolved BOOLEAN DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE archived_files (
            id INTEGER PRIMARY KEY,
            request_id INTEGER,
            file_path TEXT,
            original_url TEXT,
            content_hash TEXT,
            created_at TEXT
        );
        CREATE TABLE compression_dicts (
            id INTEGER PRIMARY KEY,
            continuation TEXT,
            version INTEGER,
            dictionary_data BLOB,
            sample_count INTEGER
        );
        CREATE TABLE run_metadata (
            id INTEGER PRIMARY KEY,
            scraper_name TEXT,
            status TEXT
        );
        """
    )
    return conn


def _insert_completed_request(
    conn: sqlite3.Connection,
    *,
    dedup_key: str,
    completed_at_ns: int,
    created_at_ns: int,
    content: bytes = b"<html></html>",
    request_type: str = "navigating",
) -> int:
    """Insert a fully-completed request row and return its rowid."""
    compressed = zstd.ZstdCompressor().compress(content)
    cur = conn.execute(
        """
        INSERT INTO requests (
            deduplication_key, response_status_code, content_compressed,
            response_headers_json, response_url, url, method, continuation,
            completed_at_ns, created_at_ns, request_type
        ) VALUES (?, 200, ?, '{}', 'http://x/', 'http://x/', 'GET',
                  'parse', ?, ?, ?)
        """,
        (
            dedup_key,
            compressed,
            completed_at_ns,
            created_at_ns,
            request_type,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def test_multi_db_resolution_picks_most_recent_completion(
    tmp_path: Path,
) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    conn_a = _create_minimal_db(db_a)
    conn_b = _create_minimal_db(db_b)
    _insert_completed_request(
        conn_a, dedup_key="K", completed_at_ns=100, created_at_ns=50
    )
    _insert_completed_request(
        conn_b, dedup_key="K", completed_at_ns=200, created_at_ns=70
    )
    conn_a.close()
    conn_b.close()

    idx = SourceIndex(source_db_paths=[db_a, db_b])
    try:
        entry = idx.lookup("K")
        assert entry is not None
        assert entry.source_db_idx == 1
    finally:
        idx.close()


def test_tiebreaker_higher_created_at_wins(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    conn_a = _create_minimal_db(db_a)
    conn_b = _create_minimal_db(db_b)
    _insert_completed_request(
        conn_a, dedup_key="K", completed_at_ns=100, created_at_ns=80
    )
    _insert_completed_request(
        conn_b, dedup_key="K", completed_at_ns=100, created_at_ns=50
    )
    conn_a.close()
    conn_b.close()

    idx = SourceIndex(source_db_paths=[db_a, db_b])
    try:
        entry = idx.lookup("K")
        assert entry is not None
        assert entry.source_db_idx == 0  # A wins on created_at_ns
    finally:
        idx.close()


def test_tiebreaker_earlier_db_wins_on_full_tie(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    conn_a = _create_minimal_db(db_a)
    conn_b = _create_minimal_db(db_b)
    _insert_completed_request(
        conn_a, dedup_key="K", completed_at_ns=100, created_at_ns=50
    )
    _insert_completed_request(
        conn_b, dedup_key="K", completed_at_ns=100, created_at_ns=50
    )
    conn_a.close()
    conn_b.close()

    idx = SourceIndex(source_db_paths=[db_a, db_b])
    try:
        entry = idx.lookup("K")
        assert entry is not None
        assert entry.source_db_idx == 0
    finally:
        idx.close()


def test_content_gate_excludes_rows_missing_response(tmp_path: Path) -> None:
    """Rows with response_status_code set but no content are not indexed."""
    db = tmp_path / "incomplete.db"
    conn = _create_minimal_db(db)
    conn.execute(
        """
        INSERT INTO requests (
            deduplication_key, response_status_code, content_compressed,
            url, method, continuation, completed_at_ns, created_at_ns,
            request_type
        ) VALUES ('K', 200, NULL, 'http://x/', 'GET', 'p', 1, 1, 'navigating')
        """
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(source_db_paths=[db])
    try:
        assert idx.lookup("K") is None
    finally:
        idx.close()


def test_retry_eligible_flag_set_for_unresolved_structural_error(
    tmp_path: Path,
) -> None:
    db = tmp_path / "errored.db"
    conn = _create_minimal_db(db)
    rid = _insert_completed_request(
        conn, dedup_key="K", completed_at_ns=1, created_at_ns=1
    )
    conn.execute(
        """
        INSERT INTO errors (request_id, error_type, error_class, message,
                            request_url, is_resolved)
        VALUES (?, 'HTMLStructuralAssumptionException',
                'HTMLStructuralAssumptionException', 'parse broke',
                'http://x/', 0)
        """,
        (rid,),
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(source_db_paths=[db])
    try:
        entry = idx.lookup("K")
        assert entry is not None
        assert entry.retry_eligible is True
    finally:
        idx.close()


def test_resolved_errors_do_not_set_retry_eligible(tmp_path: Path) -> None:
    """A resolved error in the source DB doesn't keep the row out of the index."""
    db = tmp_path / "resolved.db"
    conn = _create_minimal_db(db)
    rid = _insert_completed_request(
        conn, dedup_key="K", completed_at_ns=1, created_at_ns=1
    )
    conn.execute(
        """
        INSERT INTO errors (request_id, error_type, error_class, message,
                            request_url, is_resolved)
        VALUES (?, 'HTMLStructuralAssumptionException',
                'HTMLStructuralAssumptionException', 'old',
                'http://x/', 1)
        """,
        (rid,),
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(source_db_paths=[db])
    try:
        entry = idx.lookup("K")
        assert entry is not None
        assert entry.retry_eligible is False
    finally:
        idx.close()


def test_exclude_retry_eligible_drops_them_from_index(tmp_path: Path) -> None:
    """prev-error-free mode rebuilds with exclude_retry_eligible=True."""
    db = tmp_path / "errored.db"
    conn = _create_minimal_db(db)
    rid = _insert_completed_request(
        conn, dedup_key="K", completed_at_ns=1, created_at_ns=1
    )
    conn.execute(
        """
        INSERT INTO errors (request_id, error_type, error_class, message,
                            request_url, is_resolved)
        VALUES (?, 'HTMLStructuralAssumptionException',
                'HTMLStructuralAssumptionException', 'parse broke',
                'http://x/', 0)
        """,
        (rid,),
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(
        source_db_paths=[db],
        exclude_retry_eligible=True,
    )
    try:
        assert idx.lookup("K") is None
    finally:
        idx.close()


def test_lookup_skipdedup_returns_none(tmp_path: Path) -> None:
    """A None dedup_key is the SkipDeduplicationCheck signal: always miss."""
    db = tmp_path / "a.db"
    conn = _create_minimal_db(db)
    _insert_completed_request(
        conn, dedup_key="K", completed_at_ns=1, created_at_ns=1
    )
    conn.close()
    idx = SourceIndex(source_db_paths=[db])
    try:
        assert idx.lookup(None) is None
    finally:
        idx.close()


def test_pre_migration_db_without_hateoas_column_reads_as_none(
    tmp_path: Path,
) -> None:
    """A source DB built before the v21 migration has no hateoas column.

    fetch_parent_chain must handle that gracefully (treats every node as
    None) without raising.
    """
    db = tmp_path / "old.db"
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
            request_type TEXT
        );
        CREATE TABLE errors (
            id INTEGER PRIMARY KEY,
            request_id INTEGER,
            error_type TEXT,
            is_resolved BOOLEAN DEFAULT 0
        );
        CREATE TABLE compression_dicts (
            id INTEGER PRIMARY KEY,
            dictionary_data BLOB
        );
        CREATE TABLE archived_files (id INTEGER PRIMARY KEY);
        CREATE TABLE run_metadata (id INTEGER PRIMARY KEY, scraper_name TEXT);
        """
    )
    compressed = zstd.ZstdCompressor().compress(b"<x/>")
    conn.execute(
        """
        INSERT INTO requests (deduplication_key, response_status_code,
                              content_compressed, url, method, continuation,
                              parent_request_id, completed_at_ns,
                              created_at_ns, request_type)
        VALUES ('A', 200, ?, 'http://x/', 'GET', 'p', NULL, 1, 1, 'navigating')
        """,
        (compressed,),
    )
    conn.commit()
    conn.close()

    idx = SourceIndex(source_db_paths=[db])
    try:
        chain = idx.fetch_parent_chain(0, 1)
        assert chain == [(1, None)]
    finally:
        idx.close()
