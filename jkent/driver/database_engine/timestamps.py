"""One definition of how this schema writes and measures time.

Every timestamp column holds text in SQLite's default format extended to
milliseconds — ``2026-08-06 22:58:11.250``, UTC. That format is deliberate:

- It sorts lexicographically in chronological order, so ``ORDER BY created_at``
  and jent's "most recent wins" comparisons work on the raw column.
- It is the same shape SQLite's own ``CURRENT_TIMESTAMP`` produces, just with
  a fractional part, so a database read by hand still looks familiar.
- It is wall clock, which is what makes a timestamp comparable against one
  written by a different process, run, or machine.

Columns are mapped to SQLAlchemy's ``DateTime`` (see
``models.Base.type_annotation_map``), so they are read as ``datetime`` even
though what is stored is this text. Values written by ``server_default`` carry
three fractional digits, as ``strftime('%f')`` produces; values written from
Python carry six, as SQLAlchemy's SQLite ``DATETIME`` produces. That mix is
harmless — the fraction is a decimal expansion either way, so lexicographic
ordering still agrees with chronological ordering, and ``unixepoch`` accepts
both — but it is why the format above describes the floor rather than a
guarantee of exactly three digits.

That last point about wall clock is why the ``requests`` table no longer
carries a parallel set of ``*_at_ns`` columns. Those held
``time.monotonic_ns()``, whose epoch is the machine's boot: a difference
between two of them taken in one process is a valid duration, but the values
are meaningless anywhere else — including in jent's replay index, which
compares them *across source databases* to decide which capture is newer.

Durations are measured with :func:`epoch_seconds`, which uses SQLite's
``unixepoch(..., 'subsec')`` rather than ``julianday``. ``julianday`` returns a
day-scale float, so differencing two of them carries roughly 6 microseconds of
floating-point noise; ``unixepoch`` is second-scale and does not. The cost is a
version floor — see :func:`require_subsec_support`.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement

__all__ = [
    "MIN_SQLITE_VERSION",
    "NOW_SQL",
    "TIMESTAMP_FORMAT",
    "epoch_seconds",
    "now",
    "now_sql",
    "require_subsec_support",
]

#: ``strftime`` format for every timestamp column: SQLite's default plus
#: milliseconds. ``%f`` is seconds-with-fraction (``11.250``), not microseconds.
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%f"

#: The same thing as raw SQL, for ``server_default``.
NOW_SQL = f"strftime('{TIMESTAMP_FORMAT}', 'now')"

#: ``unixepoch``'s ``'subsec'`` modifier landed in SQLite 3.42 (2023-05).
MIN_SQLITE_VERSION = (3, 42, 0)

_subsec_checked = False


def now_sql() -> sa.TextClause:
    """The current timestamp, for a column's ``server_default``."""
    return sa.text(NOW_SQL)


def now() -> ColumnElement[str]:
    """The current timestamp, for the right-hand side of an ``UPDATE``."""
    return sa.func.strftime(TIMESTAMP_FORMAT, "now")


def epoch_seconds(column: Any) -> ColumnElement[float]:
    """*column* as Unix seconds carrying its fractional part.

    Subtract two of these for a duration in seconds. Requires the SQLite floor
    in :data:`MIN_SQLITE_VERSION`; :func:`require_subsec_support` is what makes
    a build that lacks it fail loudly instead of silently returning NULL.

    Not replaceable by subtracting the two mapped ``DateTime`` columns
    directly. That spelling compiles to a SQL ``-`` between two TEXT values,
    which SQLite coerces to 0 instead of rejecting, so the aggregate comes back
    as ``0.0`` — a wrong answer with no error anywhere to notice it.
    """
    return sa.func.unixepoch(column, "subsec")


def require_subsec_support() -> None:
    """Raise unless this SQLite understands ``unixepoch(..., 'subsec')``.

    Checked by behaviour rather than by version string, because an
    unrecognised modifier is not an error in SQLite — ``unixepoch(x, 'nosuch')``
    returns NULL. A build without ``'subsec'`` would therefore turn every
    duration into NULL and every average into "no data", which is the kind of
    failure that gets noticed months later in a dashboard. Runs once per
    process; the answer cannot change under us.

    Raises:
        RuntimeError: If the linked SQLite cannot compute sub-second epochs.
    """
    global _subsec_checked
    if _subsec_checked:
        return
    with sqlite3.connect(":memory:") as conn:
        got = conn.execute(
            "SELECT unixepoch('2026-01-01 00:00:00.500', 'subsec')"
        ).fetchone()[0]
    if got != 1767225600.5:
        raise RuntimeError(
            "jkent needs SQLite >= "
            f"{'.'.join(str(p) for p in MIN_SQLITE_VERSION)} for "
            "unixepoch(..., 'subsec'), which request-duration stats are "
            f"computed with. This interpreter links SQLite "
            f"{sqlite3.sqlite_version}, where the call returned {got!r} "
            "instead of 1767225600.5."
        )
    _subsec_checked = True
