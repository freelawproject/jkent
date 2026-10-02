"""One definition of how this schema writes and measures time.

Every timestamp column holds text in SQLite's default format extended to
milliseconds — ``2026-08-06 22:58:11.250``, UTC. That format is deliberate:

- It sorts lexicographically in chronological order, so ``ORDER BY created_at``
  and a replay index's "most recent wins" comparisons work on the raw
  column.
- It is the same shape SQLite's own ``CURRENT_TIMESTAMP`` produces, just with
  a fractional part, so a database read by hand still looks familiar.
- It is wall clock, which is what makes a timestamp comparable against one
  written by a different process, run, or machine.

Columns are mapped to :class:`UtcDateTime` (see
``models.Base.type_annotation_map``), so they are read as an aware ``datetime``
even though what is stored is this text, and a value written from Python is
either in UTC or rejected. Every writer renders exactly three fractional
digits: SQLite's ``strftime('%f')`` does by construction, and
:class:`UtcDateTime` floors a Python value to the millisecond and renders it
the same way rather than letting SQLAlchemy's SQLite ``DATETIME`` emit six.
One width is what makes ``ORDER BY`` on the raw text agree with the clock:
a three-digit rendering is a *prefix* of a six-digit rendering of any
instant in the same millisecond, so mixing widths sorted a later instant
first.

Durations are measured with :func:`epoch_seconds`, which uses SQLite's
``unixepoch(..., 'subsec')`` rather than ``julianday``. ``julianday`` returns a
day-scale float, so differencing two of them carries roughly 6 microseconds of
floating-point noise; ``unixepoch`` is second-scale and does not. The cost is a
version floor — see :func:`require_subsec_support`.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from typing_extensions import override

if TYPE_CHECKING:
    from sqlalchemy.engine.interfaces import Dialect
    from sqlalchemy.sql.elements import ColumnElement
    from sqlalchemy.sql.type_api import _BindProcessorType

__all__ = [
    "MIN_SQLITE_VERSION",
    "NOW_SQL",
    "TIMESTAMP_FORMAT",
    "UtcDateTime",
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


class UtcDateTime(sa.TypeDecorator[datetime]):
    """``DateTime`` that keeps a timestamp column honest about being UTC.

    The stored text carries no offset, and SQLAlchemy's SQLite ``DateTime``
    formats whatever ``datetime`` it is handed field by field — so a value
    carrying ``tzinfo`` has its offset dropped on the way in, recording an
    instant hours away from the one the caller meant with nothing to show it
    happened. That is what this type exists to prevent: it is the only place the
    "UTC" in every timestamp docstring is actually true rather than assumed.

    On the way in, an aware value is converted to UTC and a naive one is
    rejected. Naive cannot be *made* correct — there is no way to tell a caller
    who already normalised to UTC from one who passed local wall clock, and
    guessing either way corrupts half the callers silently. On the way out, the
    value comes back with ``timezone.utc`` attached, so what is read back is
    directly comparable with ``datetime.now(timezone.utc)`` and can be written
    to another column without tripping the check above.

    Values written by ``server_default``/``onupdate`` never reach this type —
    they are SQL, computed by SQLite from ``'now'``, which is UTC by definition.
    What this type writes is rendered to the same text they produce — the
    module's :data:`TIMESTAMP_FORMAT`, three fractional digits — so every row
    sorts by the clock whichever side wrote it (see the module docstring).
    """

    impl = sa.DateTime
    cache_ok = True

    @override
    def bind_processor(
        self, dialect: Dialect
    ) -> _BindProcessorType[datetime] | None:
        """Render the normalised UTC value as :data:`TIMESTAMP_FORMAT` text.

        Overrides the decorator's default composition (which would hand the
        datetime to the SQLite ``DATETIME`` type, and get six fractional
        digits) so the stored text has the same width as a server-written one.
        The microsecond is floored to the millisecond, as SQLite's own clock
        reads are.
        """

        def process(value: datetime | None) -> str | None:
            normalised = self.process_bind_param(value, dialect)
            if normalised is None:
                return None
            millis = normalised.microsecond // 1000
            return f"{normalised:%Y-%m-%d %H:%M:%S}.{millis:03d}"

        return process

    @override
    def process_bind_param(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Normalise *value* to naive UTC for storage.

        Raises:
            TypeError: If *value* is not a ``datetime`` (a plain ``date``
                included — a bare day is not a timestamp).
            ValueError: If *value* has no ``tzinfo``.
        """
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError(
                "timestamp columns take a datetime, got "
                f"{type(value).__name__}: {value!r}"
            )
        if value.tzinfo is None:
            raise ValueError(
                f"timestamp columns are UTC, but {value!r} is naive; pass an "
                "aware datetime (datetime.now(timezone.utc)) so the instant "
                "meant is unambiguous"
            )
        # Stripped after conversion because the stored format has nowhere to
        # put an offset; keeping one would only be dropped a layer down.
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @override
    def process_result_value(
        self, value: datetime | None, dialect: Dialect
    ) -> datetime | None:
        """Return *value* as an aware UTC ``datetime``."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


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
    with closing(sqlite3.connect(":memory:")) as conn:
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
