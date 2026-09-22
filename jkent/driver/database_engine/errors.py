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

import json
import traceback as tb
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel

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
from jkent.driver.database_engine.enums import ErrorType, SelectorType
from jkent.driver.database_engine.models import Error

#: Stand-in for ``errors.request_url`` when neither the exception nor the
#: caller supplied one. The column is NOT NULL.
UNKNOWN_URL = "unknown"


class ErrorRecord(BaseModel):
    """Error record from database for listing and display.

    A ``pydantic.BaseModel`` rather than a dataclass so ``model_dump_json``
    / ``model_validate_json`` handle the web transport, instead of a
    hand-maintained field list that drifts from the columns.

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
    validation_errors: list[dict[str, Any]] | None
    failed_doc: dict[str, Any] | None
    status_code: int | None
    timeout_seconds: float | None
    traceback: str | None
    created_at: datetime

    @classmethod
    def from_model(cls, error: Error) -> ErrorRecord:
        """Build a record from an ``errors`` row.

        Args:
            error: Error model instance.

        Returns:
            The record, with the JSON columns parsed back into objects.

        Raises:
            pydantic.ValidationError: If the row has no ``created_at``. The
                column's server default makes that unreachable; the cast
                below tells the type checker so, and pydantic still checks.
        """
        return cls(
            id=error.id,
            request_id=error.request_id,
            error_type=error.error_type,
            error_class=error.error_class,
            message=error.message,
            request_url=error.request_url,
            context_json=error.context_json,
            selector=error.selector,
            selector_type=error.selector_type,
            expected_min=error.expected_min,
            expected_max=error.expected_max,
            actual_count=error.actual_count,
            model_name=error.model_name,
            validation_errors=(
                json.loads(error.validation_errors_json)
                if error.validation_errors_json
                else None
            ),
            failed_doc=(
                json.loads(error.failed_doc_json)
                if error.failed_doc_json
                else None
            ),
            status_code=error.status_code,
            timeout_seconds=error.timeout_seconds,
            traceback=error.traceback,
            created_at=cast(datetime, error.created_at),
        )


@dataclass(frozen=True)
class ErrorDetails:
    """The columns an exception's own attributes supply.

    Every field maps to a column on :class:`~...models.Error`, so
    :func:`describe_error` can be unpacked straight into the row. Fields the
    exception type does not carry stay ``None``, which is exactly how the
    table stores them.
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


def _context_json(exc: ScraperAssumptionException) -> str | None:
    """JSON of an assumption exception's context dict, or None if empty.

    Dumped with ``default=str``: the context holds arbitrary scraped values.
    """
    return json.dumps(exc.context, default=str) if exc.context else None


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
                validation_errors_json=json.dumps(exc.errors, default=str),
                failed_doc_json=json.dumps(exc.failed_doc, default=str),
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
            return ErrorDetails(error_type=ErrorType.TRANSIENT)
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
        request_id=request_id,
        error_type=details.error_type,
        error_class=f"{type(exc).__module__}.{type(exc).__name__}",
        message=str(exc),
        request_url=request_url or details.request_url or UNKNOWN_URL,
        context_json=details.context_json,
        selector=details.selector,
        selector_type=details.selector_type,
        expected_min=details.expected_min,
        expected_max=details.expected_max,
        actual_count=details.actual_count,
        model_name=details.model_name,
        validation_errors_json=details.validation_errors_json,
        failed_doc_json=details.failed_doc_json,
        status_code=details.status_code,
        timeout_seconds=details.timeout_seconds,
        traceback="".join(
            tb.format_exception(type(exc), exc, exc.__traceback__)
        ),
    )
