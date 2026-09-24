"""Pin the schema DDL that ``create_all`` emits.

``create_all`` is the only schema authority — there is no migration runner —
so the table definitions in ``models.py`` are what lands on disk in every run
database and replay corpus. A change to how those models are declared must not
change the resulting SQLite schema.

The snapshot is taken from ``sqlite_master`` of a freshly initialized database
rather than from compiled ``CreateTable`` objects: it is the artifact that
actually persists, and it captures indexes as well as tables.

Regenerate deliberately, never to make a red test green::

    JKENT_UPDATE_SCHEMA_SNAPSHOT=1 uv run pytest tests/driver/database_engine/test_schema_ddl.py
"""

from __future__ import annotations

import os
from pathlib import Path

import sqlalchemy as sa

from jkent.driver.database_engine.database import init_database

SNAPSHOT_PATH = Path(__file__).parent / "schema_snapshot.sql"


async def _dump_schema(db_path: Path) -> str:
    """Return the normalized ``sqlite_master`` DDL of a fresh database."""
    engine, session_factory = await init_database(db_path)
    try:
        async with session_factory() as session:
            rows = (
                await session.execute(
                    sa.text(
                        "SELECT type, name, sql FROM sqlite_master "
                        "WHERE sql IS NOT NULL ORDER BY type, name"
                    )
                )
            ).all()
    finally:
        await engine.dispose()

    # SQLAlchemy emits trailing whitespace inside CREATE TABLE bodies; strip it
    # so the snapshot diffs on substance rather than on formatting.
    chunks = []
    for kind, name, sql in rows:
        body = "\n".join(line.rstrip() for line in sql.splitlines())
        chunks.append(f"-- {kind}: {name}\n{body};\n")
    return "\n".join(chunks)


async def test_schema_ddl_matches_snapshot(tmp_path: Path) -> None:
    """The emitted schema is byte-for-byte what the snapshot records."""
    actual = await _dump_schema(tmp_path / "snapshot.db")

    if os.environ.get("JKENT_UPDATE_SCHEMA_SNAPSHOT"):
        SNAPSHOT_PATH.write_text(actual)
        return

    expected = SNAPSHOT_PATH.read_text()
    assert actual == expected, (
        "Emitted schema drifted from tests/driver/database_engine/schema_snapshot.sql. "
        "Regenerate the snapshot only when the schema change is intended: "
        "JKENT_UPDATE_SCHEMA_SNAPSHOT=1."
    )
