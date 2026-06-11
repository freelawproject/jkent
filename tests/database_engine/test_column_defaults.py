"""Why several columns carry both a Python ``default`` and a ``server_default``.

The pair looks redundant — same constant, written twice — but the two serve
different writers:

- ``server_default`` puts ``DEFAULT`` in the DDL, which is what lets a writer
  that bypasses the ORM omit the column. jent's replay seeding, this suite's
  fixtures, and any ad-hoc ``sqlite3`` session all do exactly that, and a
  ``NOT NULL`` column without a server default rejects them.
- ``default`` puts the value in the ``INSERT`` the ORM emits, so the row is
  fully specified by the statement rather than filled in by the database.

The risk of stating a constant twice is that the two copies drift, so the
central test here is that every such column agrees with itself: the value a
Python-side insert lands and the value the database fills in are the same. That
is what makes the duplication safe, and it holds for every column
automatically rather than needing a case per column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.models import Base, Request

if TYPE_CHECKING:
    from pathlib import Path

# The columns a bare ``INSERT`` has to supply no matter what: no default of
# either kind, and NOT NULL.
_REQUIRED_REQUEST_COLUMNS = {
    "queue_counter": 1,
    "method": HttpMethod.GET.code,
    "url": "https://example.com/x",
    "continuation": "parse",
}


def _doubly_defaulted_columns() -> list[sa.Column[Any]]:
    """Every mapped column carrying both a client- and a server-side default."""
    return [
        column
        for table in Base.metadata.tables.values()
        for column in table.columns
        if column.default is not None and column.server_default is not None
    ]


def test_some_columns_do_carry_both_defaults() -> None:
    """Guard the guard: the parametrized test below has subjects.

    If the schema ever stops declaring both, the comparison test would pass
    vacuously; this makes that show up as a deliberate change instead.
    """
    names = {f"{c.table.name}.{c.name}" for c in _doubly_defaulted_columns()}
    assert "requests.status" in names
    assert "requests.priority" in names
    assert len(names) > 5, f"expected a handful of such columns, found {names}"


@pytest.mark.parametrize(
    "column",
    _doubly_defaulted_columns(),
    ids=lambda c: f"{c.table.name}.{c.name}",
)
def test_client_and_server_defaults_state_the_same_value(
    column: sa.Column[Any],
) -> None:
    """The two defaults on a column agree.

    Compared as rendered text because that is the only form they share: the
    Python side holds a value (or a coded-enum member), the server side holds a
    SQL literal. Both are normalised through the column's own bind processing,
    so a coded enum's ``PENDING`` and its DDL ``1`` compare equal.
    """
    default, server_default = column.default, column.server_default
    # Narrowed rather than asserted away: ``_doubly_defaulted_columns`` only
    # yields columns that have both, but a callable or SQL-expression default
    # carries no fixed value for this comparison to make sense of.
    if not isinstance(default, sa.ColumnDefault) or not default.is_scalar:
        pytest.skip("default is not a plain scalar value")
    assert isinstance(server_default, sa.DefaultClause)

    client_value = default.arg
    bind = column.type.bind_processor(sqlite_dialect.dialect())
    stored = bind(client_value) if bind is not None else client_value

    server_sql = str(server_default.arg)
    assert server_sql.strip("'") == str(stored), (
        f"{column.table.name}.{column.name}: python default {client_value!r} "
        f"stores as {stored!r}, but the DDL default is {server_sql!r} — the "
        "two have drifted"
    )


async def test_server_default_lets_a_raw_insert_omit_the_column(
    tmp_path: Path,
) -> None:
    """A non-ORM ``INSERT`` may skip the defaulted columns and still land.

    This is the property jent's replay seeding and this suite's raw-SQL
    fixtures depend on. Dropping ``server_default`` would turn every one of
    those inserts into a ``NOT NULL`` failure — see
    :func:`test_without_a_server_default_a_raw_insert_fails`.
    """
    engine, session_factory = await init_database(tmp_path / "defaults.db")
    try:
        async with session_factory() as session:
            columns = ", ".join(_REQUIRED_REQUEST_COLUMNS)
            binds = ", ".join(f":{name}" for name in _REQUIRED_REQUEST_COLUMNS)
            await session.execute(
                sa.text(f"INSERT INTO requests ({columns}) VALUES ({binds})"),
                _REQUIRED_REQUEST_COLUMNS,
            )
            await session.commit()

            row = (
                await session.execute(
                    sa.select(
                        Request.status,
                        Request.priority,
                        Request.request_type,
                        Request.allow_redirects,
                        Request.retry_count,
                    )
                )
            ).one()
    finally:
        await engine.dispose()

    assert row.status is RequestStatus.PENDING
    assert row.priority == 9
    assert row.request_type is RequestType.NAVIGATING
    assert row.allow_redirects is True
    assert row.retry_count == 0


async def test_orm_insert_sends_the_defaults_rather_than_reading_them_back(
    tmp_path: Path,
) -> None:
    """The Python-side default puts the value in the emitted ``INSERT``.

    What the client-side half buys: the statement fully specifies the row, so
    the values are the ones Python chose rather than whatever the schema
    happens to say. Without it the column is left out of the ``INSERT``
    entirely and comes back via ``RETURNING`` — see
    :func:`test_without_a_client_default_the_orm_omits_the_column`.
    """
    statements: list[str] = []
    engine, session_factory = await init_database(tmp_path / "emitted.db")
    sa.event.listen(
        engine.sync_engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, *rest: statements.append(statement),
    )
    try:
        async with session_factory() as session:
            session.add(
                Request(
                    queue_counter=1,
                    method=HttpMethod.GET,
                    url="https://example.com/x",
                    continuation="parse",
                )
            )
            await session.commit()
    finally:
        await engine.dispose()

    inserts = [s for s in statements if "INSERT INTO requests" in s]
    assert len(inserts) == 1, f"expected one insert, got {inserts}"
    for name in ("status", "priority", "request_type", "allow_redirects"):
        assert name in inserts[0], (
            f"{name} should be named in the emitted INSERT, since its "
            f"python-side default supplies a value:\n{inserts[0]}"
        )


# --- What each half is actually load-bearing for --------------------------
#
# Throwaway models on their own metadata, so the question ("what breaks without
# one of the two?") can be asked directly instead of inferred from the real
# schema, which always has both.


class _ProbeBase(DeclarativeBase):
    """Metadata for the default-behaviour probes only."""


class _ClientOnly(_ProbeBase):
    """A ``NOT NULL`` column defaulted in Python but not in the DDL."""

    __tablename__ = "client_only"

    id: Mapped[int] = mapped_column(primary_key=True)
    n: Mapped[int] = mapped_column(default=9)


class _ServerOnly(_ProbeBase):
    """A ``NOT NULL`` column defaulted in the DDL but not in Python."""

    __tablename__ = "server_only"

    id: Mapped[int] = mapped_column(primary_key=True)
    n: Mapped[int] = mapped_column(server_default=sa.text("9"))


async def _probe_engine() -> tuple[Any, async_sessionmaker]:
    """An in-memory engine holding just the probe tables."""
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as conn:
        await conn.run_sync(_ProbeBase.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_without_a_server_default_a_raw_insert_fails() -> None:
    """No ``server_default`` means no ``DEFAULT`` in the DDL, so raw SQL breaks.

    The concrete cost of dropping the server-side half: a writer that does not
    go through the ORM has no way to omit the column, because the schema itself
    no longer says what to put there.
    """
    engine, session_factory = await _probe_engine()
    try:
        ddl = None
        async with session_factory() as session:
            ddl = (
                await session.execute(
                    sa.text(
                        "SELECT sql FROM sqlite_master WHERE name = 'client_only'"
                    )
                )
            ).scalar_one()
            assert "DEFAULT" not in ddl, ddl

            with pytest.raises(sa.exc.IntegrityError, match="NOT NULL"):
                await session.execute(
                    sa.text("INSERT INTO client_only (id) VALUES (1)")
                )
                await session.commit()
    finally:
        await engine.dispose()


async def test_without_a_client_default_the_orm_omits_the_column() -> None:
    """No Python ``default`` means the ORM leaves the column to the database.

    The value still lands — the DDL default supplies it, and SQLAlchemy reads
    it back through ``RETURNING`` — so the server-side half alone is enough for
    correctness on the ORM path. The client-side half is what keeps the value
    in the statement rather than round-tripping it, which is the difference the
    pair actually buys.
    """
    statements: list[str] = []
    engine, session_factory = await _probe_engine()
    sa.event.listen(
        engine.sync_engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, *rest: statements.append(statement),
    )
    try:
        async with session_factory() as session:
            session.add(_ServerOnly())
            await session.commit()
            assert (
                await session.execute(sa.select(_ServerOnly.n))
            ).scalar_one() == 9
    finally:
        await engine.dispose()

    inserts = [s for s in statements if "INSERT INTO server_only" in s]
    assert len(inserts) == 1, inserts
    assert "DEFAULT VALUES" in inserts[0], (
        f"expected the column to be absent from the INSERT:\n{inserts[0]}"
    )
    assert "RETURNING id, n" in inserts[0], (
        f"expected the value to be read back rather than sent:\n{inserts[0]}"
    )
