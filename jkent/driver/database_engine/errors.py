"""Error tracking and storage for the unified driver.

This module provides functionality for capturing, storing, and querying
errors that occur during scraping. It supports all exception types from
the scraper_driver.common.exceptions module with type-specific details.

Error Types:
- structural: HTMLStructuralAssumptionException (selector issues)
- validation: DataFormatAssumptionException (Pydantic validation failures)
- transient: TransientException subclasses (HTTP errors, timeouts)
"""

from __future__ import annotations

import asyncio
import json
import traceback as tb
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import select

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
from jkent.driver.database_engine.models import Error, Request

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


@dataclass
class ErrorRecord:
    """Error record from database for listing and display.

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
        created_at: When the error was recorded.
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
    created_at: datetime | None

    def to_json(self) -> str:
        """Serialize to JSON for web transport."""
        return json.dumps(
            {
                "id": self.id,
                "request_id": self.request_id,
                "error_type": self.error_type,
                "error_class": self.error_class,
                "message": self.message,
                "request_url": self.request_url,
                "context_json": self.context_json,
                "selector": self.selector,
                "selector_type": self.selector_type,
                "expected_min": self.expected_min,
                "expected_max": self.expected_max,
                "actual_count": self.actual_count,
                "model_name": self.model_name,
                "validation_errors": self.validation_errors,
                "failed_doc": self.failed_doc,
                "status_code": self.status_code,
                "timeout_seconds": self.timeout_seconds,
                "traceback": self.traceback,
                "created_at": self.created_at.isoformat()
                if self.created_at
                else None,
            }
        )


def classify_error(exc: Exception) -> ErrorType:
    """Classify an exception into an error type.

    Args:
        exc: The exception to classify.

    Returns:
        The :class:`ErrorType` member the exception falls into. Members are
        ``str`` subclasses, so callers comparing against ``"structural"`` and
        friends keep working.
    """
    if isinstance(exc, HTMLStructuralAssumptionException):
        return ErrorType.STRUCTURAL
    elif isinstance(exc, DataFormatAssumptionException):
        return ErrorType.VALIDATION
    elif isinstance(exc, TransientException):
        return ErrorType.TRANSIENT
    elif isinstance(exc, PersistentException):
        return ErrorType.PERSISTENT
    else:
        return ErrorType.UNKNOWN


async def store_error(
    session_factory: async_sessionmaker,
    exc: Exception,
    request_id: int | None = None,
    request_url: str | None = None,
    *,
    db_lock: asyncio.Lock,
) -> int:
    """Store an error in the database.

    Extracts type-specific fields from the exception and stores them
    in the errors table.

    Args:
        session_factory: Async session factory.
        exc: The exception to store.
        request_id: ID of the request that caused this error (if known).
        request_url: URL that triggered the error (fallback if not in exception).
        db_lock: Shared asyncio lock for serializing SQLite access.
            Required and keyword-only: a per-call lock would not actually
            serialize concurrent writers, so the shared instance must be
            passed explicitly.

    Returns:
        The database ID of the stored error.
    """
    error_type = classify_error(exc)
    error_class = f"{type(exc).__module__}.{type(exc).__name__}"
    message = str(exc)

    traceback_str = "".join(
        tb.format_exception(type(exc), exc, exc.__traceback__)
    )

    if request_url is None:
        if isinstance(exc, ScraperAssumptionException):
            request_url = exc.request_url
        elif isinstance(
            exc,
            HTTPResponseAssumptionException
            | PersistentHTTPResponseException
            | RequestTimeoutException,
        ):
            request_url = exc.url
        else:
            request_url = "unknown"

    context_json = None
    if isinstance(exc, ScraperAssumptionException) and exc.context:
        context_json = json.dumps(exc.context, default=str)

    selector = None
    selector_type = None
    expected_min = None
    expected_max = None
    actual_count = None
    model_name = None
    validation_errors_json = None
    failed_doc_json = None
    status_code = None
    timeout_seconds = None

    if isinstance(exc, HTMLStructuralAssumptionException):
        selector = exc.selector
        selector_type = exc.selector_type
        expected_min = exc.expected_min
        expected_max = exc.expected_max
        actual_count = exc.actual_count

    elif isinstance(exc, DataFormatAssumptionException):
        model_name = exc.model_name
        validation_errors_json = json.dumps(exc.errors, default=str)
        failed_doc_json = json.dumps(exc.failed_doc, default=str)

    elif isinstance(
        exc,
        HTTPResponseAssumptionException | PersistentHTTPResponseException,
    ):
        status_code = exc.status_code

    elif isinstance(exc, RequestTimeoutException):
        timeout_seconds = exc.timeout_seconds

    error = Error(
        request_id=request_id,
        error_type=error_type,
        error_class=error_class,
        message=message,
        request_url=request_url,
        context_json=context_json,
        selector=selector,
        selector_type=selector_type,
        expected_min=expected_min,
        expected_max=expected_max,
        actual_count=actual_count,
        model_name=model_name,
        validation_errors_json=validation_errors_json,
        failed_doc_json=failed_doc_json,
        status_code=status_code,
        timeout_seconds=timeout_seconds,
        traceback=traceback_str,
    )

    async with db_lock, session_factory() as session:
        session.add(error)
        await session.flush()
        error_id = error.id
        await session.commit()

    return error_id if error_id else 0


def _error_model_to_record(error: Error) -> ErrorRecord:
    """Convert an Error model instance to an ErrorRecord.

    Args:
        error: Error model instance.

    Returns:
        ErrorRecord instance.
    """
    # Parse JSON fields
    validation_errors = (
        json.loads(error.validation_errors_json)
        if error.validation_errors_json
        else None
    )
    failed_doc = (
        json.loads(error.failed_doc_json) if error.failed_doc_json else None
    )

    return ErrorRecord(
        id=error.id,  # type: ignore[arg-type]
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
        validation_errors=validation_errors,
        failed_doc=failed_doc,
        status_code=error.status_code,
        timeout_seconds=error.timeout_seconds,
        traceback=error.traceback,
        created_at=error.created_at,
    )


async def get_error(
    session_factory: async_sessionmaker,
    error_id: int,
) -> ErrorRecord | None:
    """Get a single error by ID.

    Args:
        session_factory: Async session factory.
        error_id: The error ID to retrieve.

    Returns:
        ErrorRecord if found, None otherwise.
    """
    async with session_factory() as session:
        error = await session.get(Error, error_id)
        if error is None:
            return None
        return _error_model_to_record(error)


async def list_errors(
    session_factory: async_sessionmaker,
    error_type: ErrorType | str | None = None,
    continuation: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> list[ErrorRecord]:
    """List errors with optional filters.

    Args:
        session_factory: Async session factory.
        error_type: Filter by error type, as an :class:`ErrorType` member or
            its label — ``CodedEnumType`` binds either to the stored code.
        continuation: Filter by continuation method name (requires join with requests).
        offset: Number of records to skip.
        limit: Maximum records to return.

    Returns:
        List of ErrorRecord objects.
    """
    async with session_factory() as session:
        stmt = select(Error)

        if continuation:
            stmt = stmt.join(Request, Error.request_id == Request.id)
            stmt = stmt.where(Request.continuation == continuation)

        if error_type:
            stmt = stmt.where(Error.error_type == error_type)

        stmt = stmt.order_by(Error.created_at.desc())
        stmt = stmt.limit(limit).offset(offset)

        result = await session.execute(stmt)
        errors = result.scalars().all()

        return [_error_model_to_record(e) for e in errors]


async def count_errors(
    session_factory: async_sessionmaker,
    error_type: ErrorType | str | None = None,
    continuation: str | None = None,
) -> int:
    """Count errors with optional filters.

    Accepts the same filters as :func:`list_errors` so a paginated total
    matches the rows that listing would return.

    Args:
        session_factory: Async session factory.
        error_type: Filter by error type (member or label, as
            :func:`list_errors`).
        continuation: Filter by continuation method name (requires join with requests).

    Returns:
        Count of matching errors.
    """
    async with session_factory() as session:
        stmt = select(sa.func.count()).select_from(Error)

        if continuation:
            stmt = stmt.join(Request, Error.request_id == Request.id)
            stmt = stmt.where(Request.continuation == continuation)

        if error_type:
            stmt = stmt.where(Error.error_type == error_type)

        result = await session.execute(stmt)
        return result.scalar_one()
