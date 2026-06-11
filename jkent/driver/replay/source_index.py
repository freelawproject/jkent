"""SQLite-backed index over multiple source DBs for replay.

Maps each fulfillable ``deduplication_key`` to the winning source row,
applying the consolidation policy: most recent ``completed_at_ns`` →
most recent ``created_at_ns`` → lower source-list position.

Inclusion gate: a source row enters the index iff it has both
``response_status_code`` and ``content_compressed`` set. Rows missing
content (pending, in_progress, network-failed) are naturally absent and
fall through to the driver's miss policy.

The ``retry_eligible`` flag is True for source rows that have an
unresolved structural / validation error against them — used by
``curr-error-free`` mode to re-execute the continuation against the
stored response.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode, urlparse, urlunparse

from jkent.contracts import ensure
from jkent.driver.database_engine.compression import decompress

if TYPE_CHECKING:
    from collections.abc import Iterable

    from jkent.data_types import HTTPRequestParams


# Error types that signal "the parser broke on a valid response" — these are
# fixable by editing scraper code, so curr-error-free mode retries them.
# Other error types (TransientException, network errors) are not retried by
# replay — they fall through to the miss policy for re-fetching.
_RETRY_ELIGIBLE_ERROR_TYPES = frozenset(
    {
        "HTMLStructuralAssumptionException",
        "DataFormatAssumptionException",
    }
)


# Namespace prefix for the fallback routing key. Keeps the fallback key
# space disjoint from `_generate_deduplication_key`'s SHA256 hex output,
# so even if a real dedup_key happens to match what the fallback would
# produce, the two indexing schemes don't collide.
_FALLBACK_KEY_NAMESPACE = b"local-only-fallback\x00"


@ensure(
    lambda result: len(result) == 64
    and set(result) <= set("0123456789abcdef"),
    "fallback key shares the sha256-hex shape of real dedup keys",
)
def fallback_replay_key(url: str, body: bytes | None) -> str:
    """Routing key for source rows whose ``deduplication_key`` is NULL.

    The persistence layer urlencodes ``HTTPRequestParams.params`` into the
    stored ``url`` and turns ``data`` into ``body`` bytes. This function hashes
    that
    *post-serialization* form so the index-build side (reading source
    rows) and the lookup side (re-serializing a yielded Request) can
    agree on a key without round-tripping through `HTTPRequestParams`.

    Returned key shares the ``len(sha256-hex) == 64`` shape as a real
    dedup_key but is namespaced internally, so it can be stored in the
    same `source_index.dedup_key` column without collision risk.
    """
    h = hashlib.sha256()
    h.update(_FALLBACK_KEY_NAMESPACE)
    h.update(url.encode("utf-8", errors="replace"))
    h.update(b"\x00")
    if body is not None:
        h.update(
            body if isinstance(body, bytes) else str(body).encode("utf-8")
        )
    return h.hexdigest()


@ensure(
    lambda result, http_request: bool(http_request.params)
    or http_request.data is not None
    or result == (http_request.url, None),
    "a request with no params and no data serializes to itself",
)
def serialize_url_and_body(
    http_request: HTTPRequestParams,
) -> tuple[str, bytes | None]:
    """Reproduce the URL + body form ``_queue._serialize_request`` stores.

    Used by :func:`fallback_replay_key_for_request` so lookup-side keys
    match index-side keys derived from the source DB's stored columns.
    """
    url = http_request.url
    if http_request.params:
        parsed = urlparse(url)
        if isinstance(http_request.params, bytes):
            query = http_request.params.decode()
        else:
            query = urlencode(http_request.params)
        if parsed.query:
            query = parsed.query + "&" + query
        url = urlunparse(parsed._replace(query=query))

    # Mirror the queue's body expression exactly: bytes pass through
    # (even b""), any other *falsy* data (None, {}, []) is stored as
    # None, and only truthy structures are JSON-encoded. Checking
    # `is None` here used to serialize data={} as b"{}" while the
    # queue stored None — diverging keys, so replay always missed.
    body: bytes | None
    if isinstance(http_request.data, bytes):
        body = http_request.data
    elif http_request.data:
        body = json.dumps(http_request.data).encode()
    else:
        body = None

    return url, body  # type: ignore


def fallback_replay_key_for_request(
    http_request: HTTPRequestParams,
) -> str:
    """Lookup-side fallback key for a yielded ``HTTPRequestParams``."""
    url, body = serialize_url_and_body(http_request)
    return fallback_replay_key(url, body)


@dataclass(frozen=True)
class IndexEntry:
    """A resolved source-DB row that can fulfill a yielded Request."""

    dedup_key: str
    source_db_idx: int
    request_id: int
    retry_eligible: bool


@dataclass(frozen=True)
class FetchedResponse:
    """Materialized response data read from a source DB at lookup time."""

    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str


@dataclass(frozen=True)
class FetchedArchive:
    """Materialized archive metadata for an archive request replay."""

    status_code: int
    headers: dict[str, str]
    file_path: str
    url: str


class SourceIndex:
    """Routing index across one or more source DBs.

    Build with :meth:`build`. Look up a yielded request's dedup_key with
    :meth:`lookup`. Fetch the response payload with :meth:`fetch_response`
    or :meth:`fetch_archive`. Close source-DB connections via
    :meth:`close`.

    The index DB is its own SQLite (``:memory:`` by default, or a file
    path). Source DBs are opened read-only via ``mode=ro`` URI; they are
    never written to.
    """

    def __init__(
        self,
        *,
        source_db_paths: list[Path],
        index_db_path: Path | None = None,
        excluded_request_ids: dict[int, set[int]] | None = None,
        exclude_retry_eligible: bool = False,
    ) -> None:
        """Open source DBs and build the routing index.

        Args:
            source_db_paths: Source DB paths in priority order (earlier
                wins ties).
            index_db_path: Path for the index SQLite, or None for an
                in-memory index.
            excluded_request_ids: Optional per-source-db set of request_id
                values to *exclude* from the index. Used by mode 3 (
                ``desc-error-free``) to drop reseedable-anchor rows so they
                become natural misses and get stubbed for re-fetch.
            exclude_retry_eligible: When True, source rows whose unresolved
                error is parser-side are also excluded from the index.
                Used by mode 1 (``prev-error-free``) so retry-eligible
                rows fall through to the miss policy instead of being
                served.
        """
        self.source_db_paths = list(source_db_paths)
        self._excluded = excluded_request_ids or {}
        self._exclude_retry_eligible = exclude_retry_eligible
        self._index_conn = sqlite3.connect(
            str(index_db_path) if index_db_path is not None else ":memory:"
        )
        self._index_conn.execute(
            """
            CREATE TABLE source_index (
                dedup_key TEXT PRIMARY KEY,
                source_db_idx INTEGER NOT NULL,
                request_id INTEGER NOT NULL,
                completed_at_ns INTEGER NOT NULL,
                created_at_ns INTEGER NOT NULL,
                retry_eligible INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._source_conns: list[sqlite3.Connection] = []
        # One lock per source connection. sqlite3 with check_same_thread=False
        # accepts cross-thread use, but not *concurrent* use — the worker
        # thread pool will hammer fetch_response/fetch_archive on the same
        # connection unless we serialize. Lock per-DB so unrelated source
        # DBs can still be read in parallel.
        self._source_conn_locks: list[threading.Lock] = []
        for path in self.source_db_paths:
            # Open read-only; URI mode is required to set mode=ro.
            conn = sqlite3.connect(
                f"file:{path}?mode=ro&immutable=0",
                uri=True,
                check_same_thread=False,
            )
            self._source_conns.append(conn)
            self._source_conn_locks.append(threading.Lock())
        self._build()

    @classmethod
    def build(
        cls,
        source_db_paths: list[Path],
        *,
        index_db_path: Path | None = None,
        excluded_request_ids: dict[int, set[int]] | None = None,
    ) -> SourceIndex:
        """Construct a SourceIndex (alias for the constructor).

        Provided so callers read fluently:
        ``SourceIndex.build([a, b]).lookup(dedup_key)``.
        """
        return cls(
            source_db_paths=source_db_paths,
            index_db_path=index_db_path,
            excluded_request_ids=excluded_request_ids,
        )

    def _build(self) -> None:
        """Scan every source DB and populate ``source_index``.

        Resolution policy on duplicate dedup_key: higher
        ``completed_at_ns`` wins → tiebreak on higher ``created_at_ns``
        → tiebreak on lower ``source_db_idx``.
        """
        retry_eligible_by_db: list[set[int]] = []
        for conn in self._source_conns:
            retry_eligible_by_db.append(_retry_eligible_request_ids(conn))

        for db_idx, conn in enumerate(self._source_conns):
            excluded = self._excluded.get(db_idx, set())
            retry_ids = retry_eligible_by_db[db_idx]
            # Inclusion gate: a row is fulfillable iff it has a response
            # AND either (a) the response content is stored inline (normal
            # responses) or (b) it's an archive request — those store the
            # file on disk and the response body in the DB is NULL by
            # design.
            #
            # Rows are allowed to have a NULL deduplication_key: that
            # path covers source DBs whose original yield went through a
            # code path that didn't auto-populate the key (or whose
            # scraper used SkipDeduplicationCheck). For those rows we
            # derive a stable fallback key from the stored URL + body
            # — the lookup side computes the same fallback when its
            # probe with the yielded request's own key misses. Rows that
            # *do* have a real dedup_key keep it (overrides are preserved).
            cur = conn.execute(
                """
                SELECT id, deduplication_key, completed_at_ns, created_at_ns,
                       url, body
                FROM requests
                WHERE response_status_code IS NOT NULL
                  AND (
                      content_compressed IS NOT NULL
                      OR request_type = 'archive'
                  )
                """
            )
            for row in cur:
                (
                    request_id,
                    dedup_key,
                    completed_ns,
                    created_ns,
                    row_url,
                    row_body,
                ) = row
                if request_id in excluded:
                    continue
                if self._exclude_retry_eligible and request_id in retry_ids:
                    continue
                if dedup_key is None:
                    dedup_key = fallback_replay_key(row_url, row_body)
                retry_eligible = 1 if request_id in retry_ids else 0
                completed_ns = completed_ns or 0
                created_ns = created_ns or 0
                # Insert; if a row already exists, decide using the policy.
                existing = self._index_conn.execute(
                    "SELECT source_db_idx, request_id, completed_at_ns, "
                    "created_at_ns FROM source_index WHERE dedup_key = ?",
                    (dedup_key,),
                ).fetchone()
                if existing is None:
                    self._index_conn.execute(
                        "INSERT INTO source_index "
                        "(dedup_key, source_db_idx, request_id, "
                        "completed_at_ns, created_at_ns, retry_eligible) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            dedup_key,
                            db_idx,
                            request_id,
                            completed_ns,
                            created_ns,
                            retry_eligible,
                        ),
                    )
                else:
                    ex_idx, _ex_req_id, ex_completed, ex_created = existing
                    new_key = (completed_ns, created_ns, -db_idx)
                    ex_key = (ex_completed, ex_created, -ex_idx)
                    if new_key > ex_key:
                        self._index_conn.execute(
                            "UPDATE source_index SET source_db_idx = ?, "
                            "request_id = ?, completed_at_ns = ?, "
                            "created_at_ns = ?, retry_eligible = ? "
                            "WHERE dedup_key = ?",
                            (
                                db_idx,
                                request_id,
                                completed_ns,
                                created_ns,
                                retry_eligible,
                                dedup_key,
                            ),
                        )
        self._index_conn.commit()

    def lookup(self, dedup_key: str | None) -> IndexEntry | None:
        """Look up a dedup_key. Returns None on miss."""
        if dedup_key is None:
            return None
        row = self._index_conn.execute(
            "SELECT source_db_idx, request_id, retry_eligible "
            "FROM source_index WHERE dedup_key = ?",
            (dedup_key,),
        ).fetchone()
        if row is None:
            return None
        return IndexEntry(
            dedup_key=dedup_key,
            source_db_idx=row[0],
            request_id=row[1],
            retry_eligible=bool(row[2]),
        )

    def fetch_response(self, entry: IndexEntry) -> FetchedResponse:
        """Decompress the source row's stored response.

        Called from the worker pool via ``asyncio.to_thread``; serializes
        access to the source DB connection with a threading.Lock so
        concurrent workers can't interleave statement executions on the
        same SQLite handle.
        """
        conn = self._source_conns[entry.source_db_idx]
        lock = self._source_conn_locks[entry.source_db_idx]
        with lock:
            row = conn.execute(
                "SELECT content_compressed, compression_dict_id, "
                "response_headers_json, response_status_code, response_url, url "
                "FROM requests WHERE id = ?",
                (entry.request_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(
                    f"Index points at source_db_idx={entry.source_db_idx} "
                    f"request_id={entry.request_id} but the row is gone"
                )
            (
                content_compressed,
                dict_id,
                headers_json,
                status_code,
                response_url,
                url,
            ) = row
            dictionary: bytes | None = None
            if dict_id is not None:
                dict_row = conn.execute(
                    "SELECT dictionary_data FROM compression_dicts WHERE id = ?",
                    (dict_id,),
                ).fetchone()
                if dict_row is None:
                    raise RuntimeError(
                        f"compression_dict_id={dict_id} referenced by request "
                        f"{entry.request_id} not present in source DB "
                        f"(source_db_idx={entry.source_db_idx})"
                    )
                dictionary = dict_row[0]
        # Decompression is CPU-bound; do it outside the lock so a parallel
        # worker on a different source DB can read its own connection.
        content = decompress(content_compressed, dictionary=dictionary)
        headers: dict[str, str] = (
            json.loads(headers_json) if headers_json else {}
        )
        return FetchedResponse(
            status_code=status_code,
            headers=headers,
            content=content,
            url=response_url or url,
        )

    def fetch_archive(self, entry: IndexEntry) -> FetchedArchive | None:
        """Resolve the source row's archived-file path.

        Returns None if the source row has no archived_file companion;
        the caller should treat that as a miss (the original archive
        request was deferred via the archive_handler).
        """
        conn = self._source_conns[entry.source_db_idx]
        lock = self._source_conn_locks[entry.source_db_idx]
        with lock:
            row = conn.execute(
                "SELECT af.file_path, r.response_headers_json, "
                "r.response_status_code, COALESCE(r.response_url, r.url) "
                "FROM requests r LEFT JOIN archived_files af "
                "ON af.request_id = r.id WHERE r.id = ?",
                (entry.request_id,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        file_path, headers_json, status_code, url = row
        headers: dict[str, str] = (
            json.loads(headers_json) if headers_json else {}
        )
        return FetchedArchive(
            status_code=status_code or 200,
            headers=headers,
            file_path=file_path,
            url=url,
        )

    def all_scraper_names(self) -> list[tuple[Path, str | None]]:
        """Return the recorded scraper_name from each source DB's metadata.

        Returns one ``(path, scraper_name)`` tuple per source DB, in the
        order they were passed. ``scraper_name`` is None if the source
        DB's ``run_metadata`` table is empty.
        """
        out: list[tuple[Path, str | None]] = []
        for path, conn in zip(
            self.source_db_paths, self._source_conns, strict=True
        ):
            try:
                row = conn.execute(
                    "SELECT scraper_name FROM run_metadata "
                    "ORDER BY id ASC LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError:
                row = None
            out.append((path, row[0] if row else None))
        return out

    def iter_errored_rows(
        self, source_db_idx: int
    ) -> Iterable[tuple[int, int | None]]:
        """Yield (request_id, parent_request_id) for each errored row.

        "Errored" = has at least one unresolved ``errors`` row whose
        ``error_type`` indicates a parser/validation problem. Used by
        mode 3 to drive the parent-walk.
        """
        conn = self._source_conns[source_db_idx]
        types_list = ",".join("?" * len(_RETRY_ELIGIBLE_ERROR_TYPES))
        cur = conn.execute(
            f"""
            SELECT DISTINCT r.id, r.parent_request_id
            FROM requests r
            INNER JOIN errors e ON e.request_id = r.id
            WHERE e.is_resolved = 0
              AND e.error_type IN ({types_list})
            """,
            tuple(_RETRY_ELIGIBLE_ERROR_TYPES),
        )
        yield from cur

    def fetch_parent_chain(
        self, source_db_idx: int, request_id: int
    ) -> list[tuple[int, bool | None]]:
        """Walk ``parent_request_id`` from request_id upward to the root.

        Returns a list of ``(request_id, reseedable)`` tuples beginning with
        the given row and ending at the root (the row whose
        ``parent_request_id`` is NULL). The ``reseedable`` value is read
        directly from the source's ``requests.reseedable`` column; None on
        pre-migration source DBs (column missing or value NULL).
        """
        conn = self._source_conns[source_db_idx]
        # Detect whether the reseedable column exists; pre-v21 source DBs
        # won't have it and we read as None.
        has_reseedable = _column_exists(conn, "requests", "reseedable")
        select_reseedable = "reseedable" if has_reseedable else "NULL"
        chain: list[tuple[int, bool | None]] = []
        current: int | None = request_id
        while current is not None:
            row = conn.execute(
                f"SELECT id, parent_request_id, {select_reseedable} "
                f"FROM requests WHERE id = ?",
                (current,),
            ).fetchone()
            if row is None:
                break
            chain.append((row[0], None if row[2] is None else bool(row[2])))
            current = row[1]
        return chain

    def fetch_entry_request_row(
        self, source_db_idx: int, request_id: int
    ) -> dict[str, object] | None:
        """Read the full Request row from a source DB for seeding.

        Used by mode 3: once the reseedable-anchor ancestor is chosen, we
        seed the output DB with an equivalent entry-point request.
        """
        conn = self._source_conns[source_db_idx]
        row = conn.execute(
            """
            SELECT priority, request_type, method, url, headers_json,
                   cookies_json, body, continuation, current_location,
                   accumulated_data_json, permanent_json, expected_type,
                   deduplication_key, verify, bypass_rate_limit,
                   via_json, timeout_json, json_data, files_json, auth_json,
                   allow_redirects, proxies_json, stream, cert_json
            FROM requests WHERE id = ?
            """,
            (request_id,),
        ).fetchone()
        if row is None:
            return None
        keys = [
            "priority",
            "request_type",
            "method",
            "url",
            "headers_json",
            "cookies_json",
            "body",
            "continuation",
            "current_location",
            "accumulated_data_json",
            "permanent_json",
            "expected_type",
            "deduplication_key",
            "verify",
            "bypass_rate_limit",
            "via_json",
            "timeout_json",
            "json_data",
            "files_json",
            "auth_json",
            "allow_redirects",
            "proxies_json",
            "stream",
            "cert_json",
        ]
        return dict(zip(keys, row, strict=True))

    def close(self) -> None:
        """Close all DB handles (source DBs and the index DB)."""
        for conn in self._source_conns:
            conn.close()
        self._index_conn.close()


def _retry_eligible_request_ids(conn: sqlite3.Connection) -> set[int]:
    """Set of request_ids in this DB whose unresolved error is parser-side."""
    types_list = ",".join("?" * len(_RETRY_ELIGIBLE_ERROR_TYPES))
    cur = conn.execute(
        f"""
        SELECT DISTINCT request_id FROM errors
        WHERE is_resolved = 0
          AND error_type IN ({types_list})
          AND request_id IS NOT NULL
        """,
        tuple(_RETRY_ELIGIBLE_ERROR_TYPES),
    )
    return {row[0] for row in cur}


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check whether ``column`` exists on ``table``.

    Needed because pre-migration source DBs predate the ``reseedable``
    column, and reading a non-existent column raises in raw sqlite3.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur)
