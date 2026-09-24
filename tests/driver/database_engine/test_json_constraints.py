"""The ``json_valid`` CHECK on every ``*_json`` column.

``*_json`` is a naming convention, and a convention the database cannot
enforce is one that eventually gets broken somewhere far from where it shows
up. These constraints move that failure to the INSERT.

The tests cover the invariant (no ``*_json`` column escapes a check) and the
three-valued-logic behaviour the constraint relies on, since a bare
``json_valid(col)`` accepting NULL is not obvious from reading it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Base

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager

    InsertRequest = Callable[..., Awaitable[int]]


def _json_columns() -> list[tuple[str, str]]:
    """Every ``(table, column)`` whose name marks it as holding JSON."""
    return [
        (table.name, column.name)
        for table in Base.metadata.tables.values()
        for column in table.columns
        if column.name.endswith("_json")
    ]


def _checked_columns() -> set[tuple[str, str]]:
    """Every ``(table, column)`` actually carrying a ``json_valid`` CHECK."""
    found: set[tuple[str, str]] = set()
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            if not isinstance(constraint, sa.CheckConstraint):
                continue
            text = str(constraint.sqltext)
            if text.startswith("json_valid("):
                found.add((table.name, text[len("json_valid(") : -1]))
    return found


def test_every_json_column_is_checked() -> None:
    """No ``*_json`` column ships without a validity constraint.

    The one failure mode of declaring these per table: a column added later
    quietly misses the list. This is what notices.
    """
    declared = set(_json_columns())
    assert declared, "expected the schema to have *_json columns at all"
    assert declared - _checked_columns() == set(), (
        "these *_json columns have no json_valid CHECK: "
        f"{sorted(declared - _checked_columns())}"
    )


def test_no_check_names_a_column_that_does_not_exist() -> None:
    """The constraints and the columns agree in the other direction too.

    ``json_checks`` takes column names as strings, so a typo would emit a
    constraint against a nonexistent column — which SQLite accepts at
    ``CREATE TABLE`` and only fails on at INSERT.
    """
    assert _checked_columns() - set(_json_columns()) == set(), (
        "these json_valid CHECKs name columns that are not *_json columns: "
        f"{sorted(_checked_columns() - set(_json_columns()))}"
    )


@pytest.mark.parametrize(
    ("value", "accepted"),
    [
        ('{"a": 1}', True),
        ("[1, 2]", True),
        ("60", True),  # timeout_json stores a bare number
        ('"text"', True),
        ("null", True),
        (None, True),  # json_valid(NULL) is NULL, so the CHECK passes
        ("", False),  # the empty string is not JSON — absence must be NULL
        ("not json", False),
        ("{unclosed", False),
    ],
    ids=lambda v: repr(v)[:16],
)
async def test_json_check_accepts_json_and_rejects_junk(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    value: str | None,
    accepted: bool,
) -> None:
    """What the constraint lets through, on a real column in a real database.

    ``timeout_json`` holding ``60`` and the NULL case are the two that decide
    whether this constraint is safe to apply at all: bare scalars are valid
    JSON, and an absent value stays absent rather than becoming ``'null'``.
    """
    request_id = await insert_request()
    async with sql_manager.session_factory() as session:
        # Raw SQL: the value must reach the CHECK unvalidated, bypassing
        # anything the ORM or the insert path would do to it first.
        update = sa.text("UPDATE requests SET headers_json = :v WHERE id = :i")
        if accepted:
            await session.execute(update, {"v": value, "i": request_id})
            await session.commit()
            stored = (
                await session.execute(
                    sa.text("SELECT headers_json FROM requests WHERE id = :i"),
                    {"i": request_id},
                )
            ).scalar_one()
            assert stored == value
        else:
            with pytest.raises(sa.exc.IntegrityError, match="CHECK"):
                await session.execute(update, {"v": value, "i": request_id})
                await session.commit()


async def test_the_real_write_path_satisfies_the_checks(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """A request carrying every JSON-bearing field inserts cleanly.

    The constraints are only worth having if the values jkent actually
    serializes pass them — including the awkward ones, like a ``timeout``
    tuple and a bare-number timeout.
    """
    request_id = await insert_request(
        method=HttpMethod.POST,
        url="https://example.com/search",
        headers_json='{"Accept": "text/html"}',
        cookies_json='{"session": "abc"}',
        body=b"q=x",
        accumulated_data_json='{"count": 1}',
        permanent_json='{"court": "ca1"}',
        deduplication_key="k-1",
        timeout_json="[60.0, 30.0]",
        json_data='{"query": "x"}',
    )
    assert request_id > 0

    async with sql_manager.session_factory() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM requests WHERE "
                    "json_valid(headers_json) AND json_valid(timeout_json)"
                )
            )
        ).scalar_one()
    assert row == 1


async def test_absent_json_stays_null_rather_than_empty(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """A request with no JSON fields leaves them NULL, which the checks allow.

    The complement of the case above, and the reason the constraint has no
    ``IS NULL`` arm: the write path already represents "nothing here" as NULL.
    """
    await insert_request()
    async with sql_manager.session_factory() as session:
        nulls = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM requests WHERE "
                    "headers_json IS NULL AND cookies_json IS NULL "
                    "AND timeout_json IS NULL AND status = :s"
                ),
                {"s": RequestStatus.PENDING.code},
            )
        ).scalar_one()
    assert nulls == 1
