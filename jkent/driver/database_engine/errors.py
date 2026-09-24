"""Error classification and the error record type for the unified driver.

This module turns a raised exception into the columns the ``errors`` table
holds: :func:`describe_error` lifts an exception's structured attributes into
:class:`ErrorDetails`, and :class:`ErrorRecord` is the read-side DTO the CLI
and web transport serialize. The database access itself lives on
:class:`~jkent.driver.database_engine.sql_manager._errors.ErrorsMixin`.

Error Types:
- structural: HTMLStructuralAssumptionException (selector issues)
- validation: DataFormatAssumptionException (Pydantic validation failures)
- transient: TransientException subclasses (HTTP errors, timeouts)
- persistent: PersistentException subclasses (retrying will not help)
"""

from __future__ import annotations

import traceback as tb
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Annotated, Any

from pydantic import AliasChoices, Field

from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    HTTPResponseAssumptionException,
    PersistentException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    ScraperAssumptionException,
    TransientException,
)
from jkent.common.serialization import (
    JsonColumn,
    dump_json,
    dump_json_or_none,
)
from jkent.driver.database_engine.enums import (
    ErrorType,
    SelectorType,
    TransientKind,
)
from jkent.driver.database_engine.models import Error, RowModel

#: Stand-in for ``errors.request_url`` when neither the exception nor the
#: caller supplied one. The column is NOT NULL.
UNKNOWN_URL = "unknown"


class ErrorRecord(RowModel):
    """Error record from database for listing and display.

    Built straight from an ``errors`` row (``ErrorRecord.model_validate(row)``
    — the field names are the column names, with the two ``*_json`` columns
    decoded into ``validation_errors`` / ``failed_doc``), and re-validates
    from its own ``model_dump_json`` for the web transport.

    Attributes:
        id: Database ID of the error.
        request_id: ID of the request that caused this error (if any).
        error_type: Classification as an :class:`ErrorType` member.
        error_class: Full exception class name.
        message: Human-readable error message.
        request_url: URL that triggered the error.
        context_json: JSON-encoded error context.
        selector: For structural errors - the selector that failed.
        selector_type: For structural errors - the :class:`SelectorType`
            grammar the selector was written in.
        expected_min: For structural errors - minimum expected count.
        expected_max: For structural errors - maximum expected count.
        actual_count: For structural errors - actual count found.
        model_name: For validation errors - the Pydantic model name.
        validation_errors: For validation errors - list of error dicts.
        failed_doc: For validation errors - the document that failed.
        status_code: For transient errors - HTTP status code.
        timeout_seconds: For transient errors - timeout duration.
        kind: For bare transient errors - which subsystem failed.
        traceback: Full Python stack trace.
        created_at: When the error was recorded. Required: the column is
            typed nullable only because a server default fills it, and a
            record whose creation time we don't know is not worth keeping.
    """

    id: int
    request_id: int | None
    error_type: ErrorType
    error_class: str
    message: str
    request_url: str
    context_json: str | None
    selector: str | None
    selector_type: SelectorType | None
    expected_min: int | None
    expected_max: int | None
    actual_count: int | None
    model_name: str | None
    validation_errors: Annotated[list[dict[str, Any]] | None, JsonColumn] = (
        Field(
            validation_alias=AliasChoices(
                "validation_errors", "validation_errors_json"
            )
        )
    )
    failed_doc: Annotated[dict[str, Any] | None, JsonColumn] = Field(
        validation_alias=AliasChoices("failed_doc", "failed_doc_json")
    )
    status_code: int | None
    timeout_seconds: float | None
    kind: TransientKind | None
    traceback: str | None
    created_at: datetime


@dataclass(frozen=True)
class ErrorDetails:
    """The columns an exception's own attributes supply.

    Every field maps to a column on :class:`~...models.Error`, so
    :func:`describe_error` can be unpacked straight into the row. Fields the
    exception type does not carry stay ``None`` (NULL).
    """

    error_type: ErrorType
    request_url: str | None = None
    context_json: str | None = None
    selector: str | None = None
    selector_type: SelectorType | None = None
    expected_min: int | None = None
    expected_max: int | None = None
    actual_count: int | None = None
    model_name: str | None = None
    validation_errors_json: str | None = None
    failed_doc_json: str | None = None
    status_code: int | None = None
    timeout_seconds: float | None = None
    kind: TransientKind | None = None


def _context_json(exc: ScraperAssumptionException) -> str | None:
    """JSON of an assumption exception's context dict (``'{}'`` when empty)."""
    return dump_json_or_none(exc.context, bytes_as_repr=True)


def describe_error(exc: Exception) -> ErrorDetails:
    """Lift an exception's structured attributes into error columns.

    One dispatch over the exception hierarchy covers both the classification
    and the type-specific columns, so the two cannot disagree about what an
    exception is.

    The arms are ordered most-specific first: the structural and validation
    cases are ``ScraperAssumptionException`` subclasses and the HTTP/timeout
    cases are ``TransientException`` / ``PersistentException`` subclasses, so
    the category arms at the bottom must not shadow them.

    Args:
        exc: The exception to describe.

    Returns:
        The :class:`ErrorDetails` for *exc*. Anything unrecognized is
        ``ErrorType.UNKNOWN`` with no type-specific columns.
    """
    match exc:
        case HTMLStructuralAssumptionException():
            return ErrorDetails(
                error_type=ErrorType.STRUCTURAL,
                request_url=exc.request_url,
                context_json=_context_json(exc),
                selector=exc.selector,
                selector_type=SelectorType(exc.selector_type),
                expected_min=exc.expected_min,
                expected_max=exc.expected_max,
                actual_count=exc.actual_count,
            )
        case DataFormatAssumptionException():
            return ErrorDetails(
                error_type=ErrorType.VALIDATION,
                request_url=exc.request_url,
                context_json=_context_json(exc),
                model_name=exc.model_name,
                validation_errors_json=dump_json(
                    exc.errors, bytes_as_repr=True
                ),
                failed_doc_json=dump_json(exc.failed_doc, bytes_as_repr=True),
            )
        case ScraperAssumptionException():
            return ErrorDetails(
                error_type=ErrorType.PERSISTENT,
                request_url=exc.request_url,
                context_json=_context_json(exc),
            )
        case HTTPResponseAssumptionException():
            return ErrorDetails(
                error_type=ErrorType.TRANSIENT,
                request_url=exc.url,
                status_code=exc.status_code,
            )
        case RequestTimeoutException():
            return ErrorDetails(
                error_type=ErrorType.TRANSIENT,
                request_url=exc.url,
                timeout_seconds=exc.timeout_seconds,
            )
        case PersistentHTTPResponseException():
            return ErrorDetails(
                error_type=ErrorType.PERSISTENT,
                request_url=exc.url,
                status_code=exc.status_code,
            )
        case TransientException():
            # The bare-transient arm: no status, no deadline — ``kind`` and
            # ``url`` are the only structure such a failure has, which is
            # exactly why they exist.
            return ErrorDetails(
                error_type=ErrorType.TRANSIENT,
                request_url=exc.url,
                kind=exc.kind,
            )
        case PersistentException():
            return ErrorDetails(error_type=ErrorType.PERSISTENT)
        case _:
            return ErrorDetails(error_type=ErrorType.UNKNOWN)


def classify_error(exc: Exception) -> ErrorType:
    """Classify an exception into an error type.

    Args:
        exc: The exception to classify.

    Returns:
        The :class:`ErrorType` member the exception falls into. Members are
        ``str`` subclasses, so callers comparing against ``"structural"`` and
        friends keep working.
    """
    return describe_error(exc).error_type


def error_message(exc: BaseException) -> str:
    """``exc``'s message, or its class name when it has none.

    What a request row's ``last_error`` stores: a bare ``TimeoutError()``
    reads as ``"TimeoutError"``, not as an empty reason.
    """
    return str(exc) or type(exc).__name__


def build_error(
    exc: Exception,
    request_id: int | None = None,
    request_url: str | None = None,
) -> Error:
    """Build an unpersisted ``errors`` row for an exception.

    Args:
        exc: The exception to record.
        request_id: ID of the request that caused this error (if known).
        request_url: URL that triggered the error. Overrides the URL the
            exception carries; falls back to :data:`UNKNOWN_URL` when
            neither has one.

    Returns:
        The :class:`~...models.Error` instance, not yet added to a session.
    """
    details = describe_error(exc)
    return Error(
        **{
            **asdict(details),
            "request_url": request_url or details.request_url or UNKNOWN_URL,
        },
        request_id=request_id,
        error_class=f"{type(exc).__module__}.{type(exc).__name__}",
        message=str(exc),
        traceback="".join(
            tb.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )
