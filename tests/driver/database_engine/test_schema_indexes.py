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
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Base, Request
from jkent.driver.database_engine.sql_manager._requests import next_pending_id

if TYPE_CHECKING:
    from pathlib import Path

    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )


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


async def _seeded_requests(
    session_factory: async_sessionmaker[AsyncSession], rows: int = 2000
) -> None:
    """A ``requests`` table with enough rows for the planner to care."""
    async with session_factory() as session:
        await session.execute(
            sa.insert(Request),
            [
                {
                    "status": RequestStatus.COMPLETED,
                    "priority": 9,
                    "method": HttpMethod.GET,
                    "url": f"https://example.com/{i}",
                    "step": "parse",
                    "current_location": "",
                    "deduplication_key": f"key-{i}",
                }
                for i in range(rows)
            ],
        )
        await session.commit()


# --- [2] UNIQUE columns rely on the constraint's own index ----------------

#: ``(table, columns)`` of every UNIQUE constraint the models declare.
_UNIQUE_CONSTRAINTS = sorted(
    (table.name, tuple(column.name for column in constraint.columns))
    for table in Base.metadata.tables.values()
    for constraint in table.constraints
    if isinstance(constraint, sa.UniqueConstraint)
)


def _index_columns(
    conn: sqlite3.Connection, table: str
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """``{index name: (origin, columns)}`` for *table*, SQLite's own included.

    ``origin`` is ``PRAGMA index_list``'s: ``u`` for a UNIQUE constraint's
    automatic index, ``c`` for a declared ``CREATE INDEX``.
    """
    return {
        name: (
            origin,
            tuple(
                row[2] for row in conn.execute(f"PRAGMA index_info('{name}')")
            ),
        )
        for _seq, name, _unique, origin, _partial in conn.execute(
            f"PRAGMA index_list('{table}')"
        )
    }


@pytest.mark.parametrize(
    ("table", "columns"),
    _UNIQUE_CONSTRAINTS,
    ids=[f"{t}({','.join(c)})" for t, c in _UNIQUE_CONSTRAINTS],
)
async def test_unique_columns_are_indexed_only_by_the_constraint(
    initialized_db: object, db_path: Path, table: str, columns: tuple[str, ...]
) -> None:
    """A UNIQUE constraint's columns carry exactly one index: SQLite's own.

    SQLite backs every UNIQUE constraint with an automatic index
    (``sqlite_autoindex_<table>_<n>``), which also serves lookups on any
    leading prefix of its columns. A declared index on the same columns, or
    on a leading prefix of them, is redundant — it costs a write per insert
    and the planner has no reason to prefer it. Re-adding one is a
    deliberate act that has to justify itself with a query plan (as
    :func:`test_dedup_lookup_is_a_covering_search` does for the dedup key).
    """
    conn = sqlite3.connect(db_path)
    try:
        serving = {
            name: origin
            for name, (origin, indexed) in _index_columns(conn, table).items()
            if indexed == columns[: len(indexed)]
        }
    finally:
        conn.close()

    assert list(serving.values()) == ["u"], (
        f"{table}({', '.join(columns)}) should be served by its UNIQUE "
        f"constraint's automatic index alone; found {sorted(serving)}"
    )


#: The dedup point lookup ``_find_by_dedup_key_in_session`` issues.
_DEDUP_QUERY = _sql(
    sa.select(Request.id).where(Request.deduplication_key == "key-5")
)


# jkent never runs ANALYZE (``database.py`` sets only journal_mode, busy_timeout
# and foreign_keys), so a real run database has no ``sqlite_stat1`` and the
# no-statistics plan is the one that ships. Both regimes are exercised anyway:
# the conclusion should not rest on which one a given database happens to be in.
_STATS_REGIMES = (False, True)


@pytest.mark.parametrize("analyzed", _STATS_REGIMES)
async def test_dedup_lookup_is_a_covering_search(
    initialized_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
    db_path: Path,
    analyzed: bool,
) -> None:
    """The dedup lookup rides the UNIQUE autoindex, with or without statistics.

    It does not degrade to a table scan, and it is *covering* — SQLite answers
    it out of the index without touching the row.
    """
    await _seeded_requests(initialized_db[1])

    conn = sqlite3.connect(db_path)
    try:
        if analyzed:
            conn.execute("ANALYZE")
        assert _plan(conn, _DEDUP_QUERY) == [
            "SEARCH requests USING COVERING INDEX "
            "sqlite_autoindex_requests_1 (deduplication_key=?)"
        ]
    finally:
        conn.close()


@pytest.mark.parametrize("analyzed", _STATS_REGIMES)
async def test_dequeue_order_is_served_by_the_status_priority_index(
    initialized_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
    db_path: Path,
    analyzed: bool,
) -> None:
    """Priority then FIFO-by-id needs no sort, with or without statistics."""
    await _seeded_requests(initialized_db[1])

    conn = sqlite3.connect(db_path)
    try:
        if analyzed:
            conn.execute("ANALYZE")
        plan = _plan(conn, _sql(next_pending_id()))
        assert plan == [
            "SEARCH requests USING INDEX idx_requests_status_priority "
            "(status=?)"
        ], plan
    finally:
        conn.close()
