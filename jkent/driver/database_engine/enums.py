"""Closed vocabularies for the run-database's enumerated columns.

Several columns in :mod:`jkent.driver.database_engine.models` carry a small
fixed set of values. This module names those sets so they live in one place
instead of being repeated as bare literals across the query layer.

Each vocabulary is a :class:`~jkent.common.coded_enum.CodedEnum`: the member is
handled in Python as its label (a ``str`` subclass, so ``row.status ==
"pending"`` and ``f"{row.status}"`` compare and format as the label) and stored in the database as the small integer in its
``.code``. :class:`CodedEnumType` is the SQLAlchemy type that maps between the
two.

Integer storage makes the column opaque to anything reading the database
without this module — a raw ``SELECT status FROM requests`` yields ``1``, not
``pending``. Two things offset that:

- :meth:`CodedEnum.from_code` is the supported decode path for raw readers,
  and is what sibling tooling should use rather than hard-coding integers.
- :func:`code_check` emits a ``CHECK`` constraint listing the valid codes, so
  the schema still records the size and shape of each vocabulary even though it
  cannot record the names.

Codes are a storage format: they may be appended, never renumbered or recycled.
See :mod:`jkent.common.coded_enum`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.types import TypeDecorator
from typing_extensions import override

from jkent.common.coded_enum import CodedEnum

# Declared with the exceptions that carry it — the raise sites need it and
# ``common`` cannot import this package — and re-exported here so the models
# and query layer reach every stored vocabulary through one module.
from jkent.common.exceptions import TransientKind

if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect

__all__ = [
    "CodedEnumType",
    "ErrorType",
    "RequestStatus",
    "RequestType",
    "RunStatus",
    "SelectorType",
    "SpeculationOutcome",
    "TransientKind",
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
    #: Step ran to completion.
    COMPLETED = (3, "completed")
    #: Terminal failure — retries exhausted, or a persistent error.
    FAILED = (4, "failed")
    #: Replay-only.
    STUBBED = (6, "stubbed")

    # The status *groups* the query layer and stats reason about. Every
    # status must sit in exactly one of active / terminal / parked —
    # ``tests/driver/database_engine/test_enums.py`` fails on a member that
    # does not, so a new status cannot slip past ``count_active_requests``
    # or the stats aggregates unnoticed. Classmethods rather than class
    # attributes: an assignment in an Enum body mints a member, and every
    # type checker understands a classmethod.

    @classmethod
    def active(cls) -> frozenset[RequestStatus]:
        """Work the run still owns: queued or claimed.

        What ``ScrapeRun.status()`` and ``count_active_requests`` mean by
        "active".
        """
        return _ACTIVE

    @classmethod
    def terminal(cls) -> frozenset[RequestStatus]:
        """Settled for good; nothing will move these rows again."""
        return _TERMINAL

    @classmethod
    def dequeuable(cls) -> frozenset[RequestStatus]:
        """What ``dequeue_next_request`` may claim."""
        return _DEQUEUABLE

    @classmethod
    def parked(cls) -> frozenset[RequestStatus]:
        """Neither active nor settled.

        ``STUBBED`` is replay's intermediate state, resolved to ``PENDING``
        by ``finalize_stubs``.
        """
        return _PARKED


_ACTIVE = frozenset({RequestStatus.PENDING, RequestStatus.IN_PROGRESS})
_TERMINAL = frozenset({RequestStatus.COMPLETED, RequestStatus.FAILED})
_DEQUEUABLE = frozenset({RequestStatus.PENDING})
_PARKED = frozenset({RequestStatus.STUBBED})


class RequestType(CodedEnum):
    """Which driver path a request takes, derived from the scraper's flags."""

    #: Default. Advances the scraper's browsing state.
    NAVIGATING = (1, "navigating")
    #: ``Request(nonnavigating=True)`` — a side fetch (XHR, API call) that
    #: must not disturb the current location.
    NON_NAVIGATING = (2, "non_navigating")
    #: ``Request(archive=True)`` — a file download routed to the archive
    #: handler and recorded in ``archived_files``.
    ARCHIVE = (3, "archive")


class SpeculationOutcome(CodedEnum):
    """How a speculative request resolved, for tuning the next ceiling.

    NULL on non-speculative requests, and on a probe whose tracking row no
    loaded template owns.
    """

    #: The probe found something; raises the template's ceiling. Its step
    #: ran.
    HIT = (1, "hit")
    #: The miss that stopped this template's probing. Stored like a
    #: :attr:`MISS`.
    STOPPED = (2, "stopped")
    #: Dequeued after its template had stopped; never fetched, so no
    #: response is stored.
    TERMINATED_EARLY = (3, "terminated_early")
    #: The probe found nothing: a persistent HTTP code, or a 2xx the
    #: scraper's ``actually_successful`` rejects. Its response is stored and
    #: its step does not run.
    MISS = (4, "miss")


class RunStatus(CodedEnum):
    """State of the run as a whole, in the single ``run_metadata`` row."""

    #: Bootstrapped but never started.
    CREATED = (1, "created")
    #: A driver holds this database. Left stale by a crash, which is
    #: harmless: only a database no run has reopened is observed here.
    RUNNING = (2, "running")
    #: Queue drained normally.
    COMPLETED = (3, "completed")
    #: The run raised, or its error budget ran out; ``error_message`` says
    #: which. Resumable like any other unfinished run.
    ERROR = (4, "error")
    #: Stopped short on purpose — the stop event, a cancellation, or an
    #: interrupt.
    INTERRUPTED = (5, "interrupted")


class ErrorType(CodedEnum):
    """Bucket a raised exception was classified into.

    The labels are exactly what ``errors.classify_error`` returns, since that
    function's result is bound straight into ``errors.error_type``.
    """

    #: ``HTMLStructuralAssumptionException`` — a selector matched the wrong
    #: number of elements. Populates the ``selector*``/``expected_*``/
    #: ``actual_count`` columns.
    STRUCTURAL = (1, "structural")
    #: ``DataFormatAssumptionException`` — a record failed its model's
    #: validation. Populates ``model_name`` and the validation columns.
    VALIDATION = (2, "validation")
    #: ``TransientException`` — worth retrying (HTTP 5xx, timeouts).
    TRANSIENT = (3, "transient")
    #: ``PersistentException`` — retrying will not help.
    PERSISTENT = (4, "persistent")
    #: Anything ``classify_error`` did not recognize. Not a category so much
    #: as an admission, but the column is NOT NULL and a stored error with no
    #: bucket at all would be worse.
    UNKNOWN = (5, "unknown")


class SelectorType(CodedEnum):
    """Grammar a selector recorded on a structural error was written in.

    The labels are :attr:`jkent.common.selectors.Selector.grammar`, so a
    writer passes ``selector.grammar`` straight through.
    """

    #: :class:`jkent.common.selectors.CSS`.
    CSS = (1, "css")
    #: :class:`jkent.common.selectors.XPath`.
    XPATH = (2, "xpath")


class CodedEnumType(TypeDecorator[CodedEnum]):
    """Stores a :class:`CodedEnum` as its integer ``.code``.

    Binds a member to the integer, and returns the member on load. Only a
    member of the bound enum is accepted: a label string or a raw integer got
    there from outside this mapping (a raw ``SELECT``, another process, a
    literal), and silently trusting it is how a value from the wrong
    vocabulary lands in a column. The pydantic row models coerce labels at
    the SQLManager boundary; :meth:`CodedEnum.from_code` decodes raw codes.
    """

    impl = sa.Integer
    cache_ok = True

    def __init__(self, enum_class: type[CodedEnum], **kwargs: Any) -> None:
        """Bind this type to *enum_class*."""
        self.enum_class = enum_class
        super().__init__(**kwargs)

    @override
    def process_bind_param(
        self, value: CodedEnum | None, dialect: Dialect
    ) -> int | None:
        """Member in, integer code out."""
        if value is None:
            return None
        if isinstance(value, self.enum_class):
            return value.code
        raise LookupError(
            f"cannot store {value!r} ({type(value).__name__}) as "
            f"{self.enum_class.__name__}; pass a member "
            f"(use {self.enum_class.__name__}.from_code() for a raw code)"
        )

    @override
    def process_result_value(
        self, value: int | None, dialect: Dialect
    ) -> CodedEnum | None:
        """Integer code in, member out."""
        if value is None:
            return None
        return self.enum_class.from_code(value)

    @override
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
    codes = ", ".join(map(str, enum_class.codes()))
    return sa.CheckConstraint(f"{column} IN ({codes})", name=name)
