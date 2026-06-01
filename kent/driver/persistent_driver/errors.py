"""Error tracking and storage for LocalDevDriver.

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
from sqlmodel import select

from kent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLResponseAssumptionException,
    HTMLStructuralAssumptionException,
    PersistentException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    ScraperAssumptionException,
    TransientException,
)
from kent.driver.persistent_driver.models import Error, Request

if TYPE_CHECKING:
    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


@dataclass
class ErrorRecord:
    """Error record from database for listing and display.

    Attributes:
        id: Database ID of the error.
        request_id: ID of the request that caused this error (if any).
        error_type: Classification (structural, validation, transient).
        error_class: Full exception class name.
        message: Human-readable error message.
        request_url: URL that triggered the error.
        context_json: JSON-encoded error context.
        selector: For structural errors - the selector that failed.
        selector_type: For structural errors - xpath or css.
        expected_min: For structural errors - minimum expected count.
        expected_max: For structural errors - maximum expected count.
        actual_count: For structural errors - actual count found.
        model_name: For validation errors - the Pydantic model name.
        validation_errors: For validation errors - list of error dicts.
        failed_doc: For validation errors - the document that failed.
        status_code: For transient errors - HTTP status code.
        timeout_seconds: For transient errors - timeout duration.
        traceback: Full Python stack trace.
        is_resolved: Whether this error has been resolved.
        resolved_at: When this error was resolved.
        resolution_notes: Notes about how the error was resolved.
        created_at: When the error was recorded.
    """

    id: int
    request_id: int | None
    error_type: str
    error_class: str
    message: str
    request_url: str
    context_json: str | None
    selector: str | None
    selector_type: str | None
    expected_min: int | None
    expected_max: int | None
    actual_count: int | None
    model_name: str | None
    validation_errors: list[dict[str, Any]] | None
    failed_doc: dict[str, Any] | None
    status_code: int | None
    timeout_seconds: float | None
    traceback: str | None
    is_resolved: bool
    resolved_at: datetime | None
    resolution_notes: str | None
    created_at: datetime

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
                "selector": self.selector,
                "selector_type": self.selector_type,
                "expected_min": self.expected_min,
                "expected_max": self.expected_max,
                "actual_count": self.actual_count,
                "model_name": self.model_name,
                "validation_errors": self.validation_errors,
                "status_code": self.status_code,
                "timeout_seconds": self.timeout_seconds,
                "traceback": self.traceback,
                "is_resolved": self.is_resolved,
                "resolved_at": self.resolved_at.isoformat()
                if self.resolved_at
                else None,
                "resolution_notes": self.resolution_notes,
                "created_at": self.created_at.isoformat()
                if self.created_at
                else None,
            }
        )


def classify_error(exc: Exception) -> str:
    """Classify an exception into an error type.

    Args:
        exc: The exception to classify.

    Returns:
        Error type string: 'structural', 'validation', or 'transient'.
    """
    if isinstance(exc, HTMLStructuralAssumptionException):
        return "structural"
    elif isinstance(exc, DataFormatAssumptionException):
        return "validation"
    elif isinstance(exc, TransientException):
        return "transient"
    elif isinstance(exc, PersistentException):
        return "persistent"
    else:
        return "unknown"


async def store_error(
    session_factory: ScopedSessionFactory,
    exc: Exception,
    request_id: int | None = None,
    request_url: str | None = None,
    db_lock: asyncio.Lock | None = None,
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

    Returns:
        The database ID of the stored error.
    """
    error_type = classify_error(exc)
    error_class = f"{type(exc).__module__}.{type(exc).__name__}"
    message = str(exc)

    # Capture traceback
    traceback_str = "".join(
        tb.format_exception(type(exc), exc, exc.__traceback__)
    )

    # Extract URL from exception if available
    if request_url is None:
        if isinstance(exc, ScraperAssumptionException):
            request_url = exc.request_url
        elif isinstance(
            exc,
            HTMLResponseAssumptionException
            | PersistentHTTPResponseException
            | RequestTimeoutException,
        ):
            request_url = exc.url
        else:
            request_url = "unknown"

    # Extract context if available
    context_json = None
    if isinstance(exc, ScraperAssumptionException) and exc.context:
        context_json = json.dumps(exc.context)

    # Type-specific fields
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
        validation_errors_json = json.dumps(exc.errors)
        failed_doc_json = json.dumps(exc.failed_doc)

    elif isinstance(
        exc,
        HTMLResponseAssumptionException | PersistentHTTPResponseException,
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

    lock: asyncio.Lock = db_lock or asyncio.Lock()
    async with lock, session_factory() as session:
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

    # Parse timestamps
    resolved_at_dt = None
    if error.resolved_at:
        resolved_at_dt = (
            datetime.fromisoformat(error.resolved_at)
            if isinstance(error.resolved_at, str)
            else error.resolved_at
        )

    created_at_dt = (
        datetime.fromisoformat(error.created_at)
        if isinstance(error.created_at, str)
        else error.created_at
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
        is_resolved=error.is_resolved,
        resolved_at=resolved_at_dt,
        resolution_notes=error.resolution_notes,
        created_at=created_at_dt,  # type: ignore[arg-type]
    )


async def get_error(
    session_factory: ScopedSessionFactory,
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
    session_factory: ScopedSessionFactory,
    error_type: str | None = None,
    continuation: str | None = None,
    unresolved_only: bool = True,
    offset: int = 0,
    limit: int = 50,
) -> list[ErrorRecord]:
    """List errors with optional filters.

    Args:
        session_factory: Async session factory.
        error_type: Filter by error type (structural, validation, transient).
        continuation: Filter by continuation method name (requires join with requests).
        unresolved_only: If True, only return unresolved errors.
        offset: Number of records to skip.
        limit: Maximum records to return.

    Returns:
        List of ErrorRecord objects.
    """
    async with session_factory() as session:
        stmt = select(Error)

        if continuation:
            stmt = stmt.join(Request, Error.request_id == Request.id)  # type: ignore[arg-type]
            stmt = stmt.where(Request.continuation == continuation)

        if error_type:
            stmt = stmt.where(Error.error_type == error_type)

        if unresolved_only:
            stmt = stmt.where(Error.is_resolved == sa.false())

        stmt = stmt.order_by(Error.created_at.desc())  # type: ignore[union-attr]
        stmt = stmt.limit(limit).offset(offset)

        result = await session.execute(stmt)
        errors = result.scalars().all()

        return [_error_model_to_record(e) for e in errors]


async def count_errors(
    session_factory: ScopedSessionFactory,
    error_type: str | None = None,
    unresolved_only: bool = True,
) -> int:
    """Count errors with optional filters.

    Args:
        session_factory: Async session factory.
        error_type: Filter by error type.
        unresolved_only: If True, only count unresolved errors.

    Returns:
        Count of matching errors.
    """
    async with session_factory() as session:
        stmt = select(sa.func.count()).select_from(Error)

        if error_type:
            stmt = stmt.where(Error.error_type == error_type)

        if unresolved_only:
            stmt = stmt.where(Error.is_resolved == sa.false())

        result = await session.execute(stmt)
        return result.scalar_one()


async def resolve_error(
    session_factory: ScopedSessionFactory,
    error_id: int,
    notes: str | None = None,
) -> bool:
    """Mark an error as resolved.

    Args:
        session_factory: Async session factory.
        error_id: The error ID to resolve.
        notes: Optional notes about how the error was resolved.

    Returns:
        True if error was found and updated, False if not found.
    """
    async with session_factory() as session:
        result = await session.execute(
            sa.update(Error)
            .where(Error.id == error_id, Error.is_resolved == sa.false())  # type: ignore[arg-type]
            .values(
                is_resolved=True,
                resolved_at=sa.text("CURRENT_TIMESTAMP"),
                resolution_notes=notes,
            )
        )
        await session.commit()
        return result.rowcount > 0  # type: ignore[attr-defined]
