"""Closed vocabularies for the run-database's enumerated columns.

Several columns in :mod:`jkent.driver.database_engine.models` carry a small
fixed set of values. This module names those sets so they live in one place
instead of being repeated as bare literals across the query layer.

Each vocabulary is a :class:`~jkent.common.coded_enum.CodedEnum`: the member is
handled in Python as its label (a ``str`` subclass, so ``row.status ==
"pending"`` and ``f"{row.status}"`` behave as they did when the column was a
plain ``str``) and stored in the database as the small integer in its
``.code``. :class:`CodedEnumType` is the SQLAlchemy type that maps between the
two.

Integer storage makes the column opaque to anything reading the database
without this module — a raw ``SELECT status FROM requests`` yields ``1``, not
``pending``. Two things offset that:

- :meth:`CodedEnum.from_code` is the supported decode path for raw readers,
  and is what sibling tooling should use rather than hard-coding integers.
- :func:`code_check` emits a ``CHECK`` constraint listing the valid codes, so
  the schema still records the size and shape of each vocabulary even though it
  can no longer record the names. Without it an integer column would document
  nothing at all.

Codes are a storage format: they may be appended, never renumbered or recycled.
See :mod:`jkent.common.coded_enum`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.types import TypeDecorator

from jkent.common.coded_enum import CodedEnum

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect

__all__ = [
    "CodedEnumType",
    "RequestStatus",
    "RequestType",
    "RunStatus",
    "SpeculationOutcome",
    "code_check",
]


class RequestStatus(CodedEnum):
    """Lifecycle state of a row in the ``requests`` queue."""

    #: Queued and eligible for a worker to claim.
    PENDING = (1, "pending")
    #: Claimed by a worker; set atomically by ``dequeue_next_request``. A run
    #: that dies mid-flight leaves rows here, and the next startup's
    #: ``restore_queue`` resets them to ``PENDING``.
    IN_PROGRESS = (2, "in_progress")
    #: Continuation ran to completion.
    COMPLETED = (3, "completed")
    #: Terminal failure — retries exhausted, or a persistent error.
    FAILED = (4, "failed")
    #: Parked without being dropped: the circuit breaker holds pending work
    #: here so a tripped run can be resumed rather than restarted.
    HELD = (5, "held")
    #: Replay-only. ``jent``'s ``ReplayStorage`` marks a request whose
    #: response was missing from the corpus so a downstream ``jkent run``
    #: re-fetches it. No live run ever writes this value, but the vocabulary
    #: has to admit it or the column's ``CHECK`` constraint would reject
    #: replay's own writes.
    STUBBED = (6, "stubbed")


class RequestType(CodedEnum):
    """Which driver path a request takes, derived from the scraper's flags."""

    #: Default. Advances the scraper's browsing state; the only kind that
    #: participates in speculation.
    NAVIGATING = (1, "navigating")
    #: ``Request(nonnavigating=True)`` — a side fetch (XHR, API call) that
    #: must not disturb the current location.
    NON_NAVIGATING = (2, "non_navigating")
    #: ``Request(archive=True)`` — a file download routed to the archive
    #: handler and recorded in ``archived_files``.
    ARCHIVE = (3, "archive")


class SpeculationOutcome(CodedEnum):
    """How a speculative request resolved, for tuning the next ceiling.

    NULL on non-speculative requests.
    """

    #: The speculated resource existed; raises the template's ceiling.
    SUCCESS = (1, "success")
    #: The miss that stopped this template's probing.
    STOPPED = (2, "stopped")
    #: Never dispatched — the template had already stopped when this
    #: request came up for dispatch.
    SKIPPED = (3, "skipped")


class RunStatus(CodedEnum):
    """State of the run as a whole, in the single ``run_metadata`` row."""

    #: Bootstrapped but never started.
    CREATED = (1, "created")
    #: A driver holds this database. Left stale by a crash, which is
    #: harmless: only a database no run has reopened is observed here.
    RUNNING = (2, "running")
    #: Queue drained normally.
    COMPLETED = (3, "completed")
    #: The run raised; the exception is in ``error_message``.
    ERROR = (4, "error")
    #: Stopped short on purpose — stop event, or a budget cutoff.
    INTERRUPTED = (5, "interrupted")


class CodedEnumType(TypeDecorator):
    """Stores a :class:`CodedEnum` as its integer ``.code``.

    Binds a member — or the label string the query layer still passes in some
    places — to the integer, and returns the member on load. An unknown value
    raises rather than persisting: this is the only validation an integer
    column gets, since a ``CHECK`` on codes cannot catch a *valid* code that
    means the wrong thing.

    Raw integers are deliberately *not* accepted on bind. A caller holding an
    integer got it from outside this mapping (a raw ``SELECT``, another
    process), and silently trusting it is how a code from the wrong vocabulary
    lands in a column; :meth:`CodedEnum.from_code` is the way in.
    """

    impl = sa.Integer
    cache_ok = True

    def __init__(self, enum_class: type[CodedEnum], **kwargs: Any) -> None:
        """Bind this type to *enum_class*."""
        self.enum_class = enum_class
        super().__init__(**kwargs)

    def process_bind_param(self, value: Any, dialect: Dialect) -> int | None:
        """Member or label in, integer code out."""
        if value is None:
            return None
        if isinstance(value, self.enum_class):
            return value.code
        if isinstance(value, str):
            try:
                return self.enum_class(value).code
            except ValueError:
                raise LookupError(
                    f"{value!r} is not a valid {self.enum_class.__name__}; "
                    f"expected one of {[m.value for m in self.enum_class]}"
                ) from None
        raise LookupError(
            f"cannot store {value!r} ({type(value).__name__}) as "
            f"{self.enum_class.__name__}; pass a member or its label "
            f"(use {self.enum_class.__name__}.from_code() for a raw code)"
        )

    def process_result_value(
        self, value: Any, dialect: Dialect
    ) -> CodedEnum | None:
        """Integer code in, member out."""
        if value is None:
            return None
        return self.enum_class.from_code(value)

    def __repr__(self) -> str:
        return f"CodedEnumType({self.enum_class.__name__})"


def code_check(
    column: str, enum_class: type[CodedEnum], name: str
) -> sa.CheckConstraint:
    """A ``CHECK`` restricting *column* to *enum_class*'s codes.

    The vocabulary's only trace in the schema itself, now that the column is an
    integer. Generated from the enum so it cannot drift from the Python
    definition.

    Args:
        column: Column name to constrain.
        enum_class: The vocabulary whose codes are permitted.
        name: Constraint name as it appears in the DDL.

    Returns:
        The constraint, for inclusion in ``__table_args__``.
    """
    codes = ", ".join(str(c) for c in enum_class.codes())
    return sa.CheckConstraint(f"{column} IN ({codes})", name=name)
