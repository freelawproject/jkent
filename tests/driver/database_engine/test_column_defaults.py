"""Why several columns carry both a Python ``default`` and a ``server_default``.

The pair looks redundant — same constant, written twice — but the two serve
different writers:

- ``server_default`` puts ``DEFAULT`` in the DDL, which is what lets a writer
  that bypasses the ORM omit the column. A host's replay seeding, this suite's
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

from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite as sqlite_dialect

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.models import Base, Request

# A value for each column a bare ``INSERT`` has to supply (see
# ``_required_columns``), for every table that has doubly-defaulted columns.
# Tables in insert order: ``results.request_id`` names the requests row.
_REQUIRED_VALUES: dict[str, dict[str, Any]] = {
    "requests": {
        "method": HttpMethod.GET.code,
        "url": "https://example.com/x",
        "step": "parse",
    },
    "results": {"request_id": 1, "result_type": "CaseData", "data_json": "{}"},
    "run_metadata": {
        "scraper_name": "S",
        "jitter": 0.0,
        "num_workers": 1,
        "max_backoff_time": 60.0,
    },
    "speculation_tracking": {"func_name": "f"},
}


def _required_columns(table: sa.Table) -> list[str]:
    """The columns a bare ``INSERT`` has to supply no matter what.

    NOT NULL with no default of either kind; the integer primary key is the
    rowid, which SQLite assigns.
    """
    return [
        column.name
        for column in table.columns
        if not column.nullable
        and column.default is None
        and column.server_default is None
        and not column.primary_key
    ]


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
        pytest.fail(
            f"{column.table.name}.{column.name}: a callable or SQL-expression "
            "default next to a constant server default is exactly the case "
            "where the two can differ per row"
        )
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
    initialized_db: tuple[Any, Any],
) -> None:
    """A non-ORM ``INSERT`` may skip the defaulted columns and still land.

    This is the property a host's replay seeding and this suite's raw-SQL
    fixtures depend on. Dropping ``server_default`` would turn every one of
    those inserts into a ``NOT NULL`` failure. Every doubly-defaulted column
    is omitted and must read back as its Python-side default.
    """
    _engine, session_factory = initialized_db
    doubly = _doubly_defaulted_columns()
    tables = {column.table.name for column in doubly}
    assert tables <= set(_REQUIRED_VALUES), (
        f"no required values for {sorted(tables - set(_REQUIRED_VALUES))}"
    )

    async with session_factory() as session:
        for name, values in _REQUIRED_VALUES.items():
            table = Base.metadata.tables[name]
            required = {c: values[c] for c in _required_columns(table)}
            columns = ", ".join(required)
            binds = ", ".join(f":{c}" for c in required)
            await session.execute(
                sa.text(f"INSERT INTO {name} ({columns}) VALUES ({binds})"),
                required,
            )
        await session.commit()

        for name in tables:
            defaulted = [c for c in doubly if c.table.name == name]
            row = (await session.execute(sa.select(*defaulted))).one()
            assert dict(
                zip((c.name for c in defaulted), row, strict=True)
            ) == {
                c.name: c.default.arg  # type: ignore[union-attr]
                for c in defaulted
            }, name


async def test_orm_insert_sends_the_defaults_rather_than_reading_them_back(
    initialized_db: tuple[Any, Any],
) -> None:
    """The Python-side default puts the value in the emitted ``INSERT``.

    What the client-side half buys: the statement fully specifies the row, so
    the values are the ones Python chose rather than whatever the schema
    happens to say. Without it the column is left out of the ``INSERT``
    entirely and comes back via ``RETURNING``.
    """
    statements: list[str] = []
    engine, session_factory = initialized_db

    def _record(
        conn: sa.Connection, cursor: object, statement: str, *rest: object
    ) -> None:
        statements.append(statement)

    sa.event.listen(engine.sync_engine, "before_cursor_execute", _record)
    async with session_factory() as session:
        session.add(
            Request(
                method=HttpMethod.GET,
                url="https://example.com/x",
                step="parse",
            )
        )
        await session.commit()

    inserts = [s for s in statements if "INSERT INTO requests" in s]
    assert len(inserts) == 1, f"expected one insert, got {inserts}"
    for name in ("status", "priority", "request_type", "rate_limit"):
        assert name in inserts[0], (
            f"{name} should be named in the emitted INSERT, since its "
            f"python-side default supplies a value:\n{inserts[0]}"
        )
