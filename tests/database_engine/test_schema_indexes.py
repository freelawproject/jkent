"""Evidence for which indexes in ``models.py`` actually earn their place.

Every index costs a write on every insert, so an index the planner never
chooses is pure overhead. These tests ask SQLite directly — ``sqlite_master``
for what exists, ``EXPLAIN QUERY PLAN`` for what gets used — rather than
reasoning about it.

They pin findings, not preferences: each one states what SQLite does today so
that changing an index (or a query) has to reckon with the evidence that
justified it.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite as sqlite_dialect

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Request

if TYPE_CHECKING:
    from pathlib import Path


def _sql(stmt: sa.sql.expression.ClauseElement) -> str:
    """Render *stmt* with values inlined, for ``EXPLAIN QUERY PLAN``."""
    return str(
        stmt.compile(
            dialect=sqlite_dialect.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


def _plan(conn: sqlite3.Connection, query: str) -> list[str]:
    """The ``EXPLAIN QUERY PLAN`` detail lines for *query*."""
    return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + query)]


def _indexes(conn: sqlite3.Connection, table: str) -> dict[str, str | None]:
    """``{index name: create sql}`` for *table*, including SQLite's own."""
    return {
        row[0]: row[1]
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = ?",
            (table,),
        )
    }


async def _seeded_requests(db_path: Path, rows: int = 2000) -> None:
    """A ``requests`` table with enough rows for the planner to care."""
    engine, session_factory = await init_database(db_path)
    try:
        async with session_factory() as session:
            await session.execute(
                sa.insert(Request),
                [
                    {
                        "status": RequestStatus.COMPLETED,
                        "priority": 9,
                        "queue_counter": i,
                        "method": HttpMethod.GET,
                        "url": f"https://example.com/{i}",
                        "continuation": "parse",
                        "current_location": "",
                        "deduplication_key": f"key-{i}",
                    }
                    for i in range(rows)
                ],
            )
            await session.commit()
    finally:
        await engine.dispose()


# --- [2] the dedup column relies on the UNIQUE constraint's own index ------


async def test_dedup_column_is_indexed_only_by_the_unique_constraint(
    tmp_path: Path,
) -> None:
    """``deduplication_key`` carries exactly one index: SQLite's own.

    SQLite backs every UNIQUE constraint with an automatic index
    (``sqlite_autoindex_<table>_<n>``), so a hand-declared index on the same
    single column is redundant — it costs a write per insert and the planner
    has no reason to prefer it. One was declared here and has been removed;
    this keeps it removed, and makes re-adding one a deliberate act that has to
    justify itself against :func:`test_dedup_lookups_are_covering_searches`.
    """
    db_path = tmp_path / "dedup.db"
    await _seeded_requests(db_path, rows=1)

    conn = sqlite3.connect(db_path)
    try:
        indexes = _indexes(conn, "requests")
        on_dedup = {
            name
            for name, ddl in indexes.items()
            if ddl is None or "deduplication_key" in ddl
        }
    finally:
        conn.close()

    assert on_dedup == {"sqlite_autoindex_requests_1"}, (
        "deduplication_key should be served by the UNIQUE constraint's "
        f"automatic index alone; found {sorted(on_dedup)}"
    )


def _dedup_queries() -> dict[str, str]:
    """The two dedup lookup shapes ``_requests.py`` issues.

    The point lookup from ``_find_by_dedup_key_in_session`` and the chunked
    batch from ``_find_existing_dedup_keys_in_session``.
    """
    return {
        "point": _sql(
            sa.select(Request.id).where(Request.deduplication_key == "key-5")
        ),
        "batch": _sql(
            sa.select(Request.deduplication_key).where(
                Request.deduplication_key.in_(["key-1", "key-2", "key-3"])
            )
        ),
    }


# jkent never runs ANALYZE (``database.py`` sets only journal_mode, busy_timeout
# and foreign_keys), so a real run database has no ``sqlite_stat1`` and the
# no-statistics plan is the one that ships. Both regimes are exercised anyway:
# the conclusion should not rest on which one a given database happens to be in.
_STATS_REGIMES = (False, True)


@pytest.mark.parametrize("analyzed", _STATS_REGIMES)
async def test_dedup_lookups_are_covering_searches(
    tmp_path: Path, analyzed: bool
) -> None:
    """Both dedup lookups ride the UNIQUE autoindex, with or without statistics.

    The evidence that dropping the hand-declared index cost nothing: neither
    query degrades to a table scan, and both are still *covering* — SQLite
    answers them out of the index without touching the row.
    """
    db_path = tmp_path / "dedup.db"
    await _seeded_requests(db_path)

    conn = sqlite3.connect(db_path)
    try:
        if analyzed:
            conn.execute("ANALYZE")
        for label, query in _dedup_queries().items():
            plan = _plan(conn, query)
            assert plan == [
                "SEARCH requests USING COVERING INDEX "
                "sqlite_autoindex_requests_1 (deduplication_key=?)"
            ], (
                f"{label} lookup is no longer a covering autoindex search: {plan}"
            )
    finally:
        conn.close()
