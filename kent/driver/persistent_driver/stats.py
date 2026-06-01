"""Statistics dataclasses and queries for LocalDevDriver.

This module provides dataclasses for various statistics about the driver's
state and functions to query them from the database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlmodel import select

from kent.driver.persistent_driver.models import (
    Error,
    Request,
    Result,
    RunMetadata,
)

if TYPE_CHECKING:
    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


@dataclass
class QueueStats:
    """Statistics about the request queue.

    Attributes:
        pending: Number of pending requests.
        in_progress: Number of requests currently being processed.
        completed: Number of successfully completed requests.
        failed: Number of failed requests.
        held: Number of held (paused) requests.
        total: Total number of requests.
        by_continuation: Counts by continuation method name.
    """

    pending: int = 0
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    held: int = 0
    total: int = 0
    by_continuation: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pending": self.pending,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "held": self.held,
            "total": self.total,
            "by_continuation": self.by_continuation,
        }


@dataclass
class ThroughputStats:
    """Statistics about request throughput.

    Attributes:
        total_completed: Total requests completed.
        total_duration_seconds: Total duration from first to last request.
        requests_per_minute: Average requests completed per minute.
        average_response_time_seconds: Average time between start and completion.
    """

    total_completed: int = 0
    total_duration_seconds: float = 0.0
    requests_per_minute: float = 0.0
    average_response_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_completed": self.total_completed,
            "total_duration_seconds": self.total_duration_seconds,
            "requests_per_minute": self.requests_per_minute,
            "average_response_time_seconds": self.average_response_time_seconds,
        }


@dataclass
class CompressionStats:
    """Statistics about response compression.

    Attributes:
        total_responses: Total number of stored responses.
        total_original_bytes: Sum of original content sizes.
        total_compressed_bytes: Sum of compressed content sizes.
        compression_ratio: Overall compression ratio (original/compressed).
        dict_compressed_count: Number of responses using dictionary compression.
        no_dict_compressed_count: Number of responses without dictionary.
    """

    total_responses: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    compression_ratio: float = 1.0
    dict_compressed_count: int = 0
    no_dict_compressed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_responses": self.total_responses,
            "total_original_bytes": self.total_original_bytes,
            "total_compressed_bytes": self.total_compressed_bytes,
            "compression_ratio": self.compression_ratio,
            "dict_compressed_count": self.dict_compressed_count,
            "no_dict_compressed_count": self.no_dict_compressed_count,
        }


@dataclass
class ResultStats:
    """Statistics about scraped results.

    Attributes:
        total: Total number of results.
        valid: Number of valid results.
        invalid: Number of invalid results.
        by_type: Counts by result type (Pydantic model name).
    """

    total: int = 0
    valid: int = 0
    invalid: int = 0
    by_type: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "by_type": self.by_type,
        }


@dataclass
class ErrorStats:
    """Statistics about errors.

    Attributes:
        total: Total number of errors.
        unresolved: Number of unresolved errors.
        resolved: Number of resolved errors.
        by_type: Counts by error type (structural, validation, transient).
        by_continuation: Counts by continuation method name.
    """

    total: int = 0
    unresolved: int = 0
    resolved: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_continuation: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "unresolved": self.unresolved,
            "resolved": self.resolved,
            "by_type": self.by_type,
            "by_continuation": self.by_continuation,
        }


@dataclass
class DevDriverStats:
    """Combined statistics for LocalDevDriver.

    Attributes:
        queue: Queue statistics.
        throughput: Throughput statistics.
        compression: Compression statistics.
        results: Result statistics.
        errors: Error statistics.
        run_status: Current run status.
        scraper_name: Name of the scraper.
    """

    queue: QueueStats
    throughput: ThroughputStats
    compression: CompressionStats
    results: ResultStats
    errors: ErrorStats
    run_status: str = "unknown"
    scraper_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "queue": self.queue.to_dict(),
            "throughput": self.throughput.to_dict(),
            "compression": self.compression.to_dict(),
            "results": self.results.to_dict(),
            "errors": self.errors.to_dict(),
            "run_status": self.run_status,
            "scraper_name": self.scraper_name,
        }

    def to_json(self) -> str:
        """Serialize to JSON for transport."""
        return json.dumps(self.to_dict())


async def get_queue_stats(
    session_factory: ScopedSessionFactory,
) -> QueueStats:
    """Get statistics about the request queue.

    Args:
        session_factory: Async session factory.

    Returns:
        QueueStats instance with current queue state.
    """
    async with session_factory() as session:
        # Get counts by status
        result = await session.execute(
            select(Request.status, sa.func.count()).group_by(Request.status)
        )
        rows = result.all()

        stats = QueueStats()
        for status, count in rows:
            if status == "pending":
                stats.pending = count
            elif status == "in_progress":
                stats.in_progress = count
            elif status == "completed":
                stats.completed = count
            elif status == "failed":
                stats.failed = count
            elif status == "held":
                stats.held = count

        stats.total = (
            stats.pending
            + stats.in_progress
            + stats.completed
            + stats.failed
            + stats.held
        )

        # Get counts by continuation
        result = await session.execute(
            select(
                Request.continuation,
                Request.status,
                sa.func.count(),
            ).group_by(Request.continuation, Request.status)
        )
        rows = result.all()

        for continuation, status, count in rows:
            if continuation not in stats.by_continuation:
                stats.by_continuation[continuation] = {}
            stats.by_continuation[continuation][status] = count

    return stats


async def get_throughput_stats(
    session_factory: ScopedSessionFactory,
) -> ThroughputStats:
    """Get statistics about request throughput.

    Args:
        session_factory: Async session factory.

    Returns:
        ThroughputStats instance with throughput metrics.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(
                sa.func.count(),
                sa.func.min(Request.started_at),
                sa.func.max(Request.completed_at),
                sa.func.avg(
                    (
                        sa.func.julianday(Request.completed_at)
                        - sa.func.julianday(Request.started_at)
                    )
                    * 86400
                ),
            ).where(
                Request.status == "completed",
                Request.started_at.isnot(None),  # type: ignore[union-attr]
                Request.completed_at.isnot(None),  # type: ignore[union-attr]
            )
        )
        row = result.first()

        stats = ThroughputStats()
        if row and row[0] > 0:
            stats.total_completed = row[0]

            # Calculate duration from first to last request
            if row[1] and row[2]:
                duration_result = await session.execute(
                    select(
                        (
                            sa.func.julianday(sa.literal(row[2]))
                            - sa.func.julianday(sa.literal(row[1]))
                        )
                        * 86400
                    )
                )
                duration_row = duration_result.first()
                if duration_row and duration_row[0]:
                    stats.total_duration_seconds = duration_row[0]
                    if stats.total_duration_seconds > 0:
                        stats.requests_per_minute = (
                            stats.total_completed
                            / stats.total_duration_seconds
                        ) * 60

            if row[3]:
                stats.average_response_time_seconds = row[3]

    return stats


async def get_compression_stats(
    session_factory: ScopedSessionFactory,
) -> CompressionStats:
    """Get statistics about response compression.

    Args:
        session_factory: Async session factory.

    Returns:
        CompressionStats instance with compression metrics.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(  # type: ignore[call-overload]
                sa.func.count(),
                sa.func.coalesce(
                    sa.func.sum(Request.content_size_original), 0
                ),
                sa.func.coalesce(
                    sa.func.sum(Request.content_size_compressed), 0
                ),
                sa.func.sum(
                    sa.case(
                        (
                            Request.compression_dict_id.isnot(None),  # type: ignore[union-attr]
                            1,
                        ),
                        else_=0,
                    )
                ),
                sa.func.sum(
                    sa.case(
                        (Request.compression_dict_id.is_(None), 1),  # type: ignore[union-attr]
                        else_=0,
                    )
                ),
            ).where(
                Request.response_status_code.isnot(None),  # type: ignore[union-attr]
            )
        )
        row = result.first()

        stats = CompressionStats()
        if row:
            stats.total_responses = row[0]
            stats.total_original_bytes = row[1]
            stats.total_compressed_bytes = row[2]
            stats.dict_compressed_count = row[3]
            stats.no_dict_compressed_count = row[4]

            if stats.total_compressed_bytes > 0:
                stats.compression_ratio = (
                    stats.total_original_bytes / stats.total_compressed_bytes
                )

    return stats


async def get_result_stats(
    session_factory: ScopedSessionFactory,
) -> ResultStats:
    """Get statistics about scraped results.

    Args:
        session_factory: Async session factory.

    Returns:
        ResultStats instance with result metrics.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(
                sa.func.count(),
                sa.func.sum(
                    sa.case((Result.is_valid == sa.true(), 1), else_=0)  # type: ignore[arg-type]
                ),
                sa.func.sum(
                    sa.case((Result.is_valid == sa.false(), 1), else_=0)  # type: ignore[arg-type]
                ),
            )
        )
        row = result.first()

        stats = ResultStats()
        if row:
            stats.total = row[0]
            stats.valid = row[1] or 0
            stats.invalid = row[2] or 0

        # Get counts by type
        result = await session.execute(
            select(Result.result_type, sa.func.count()).group_by(
                Result.result_type
            )
        )
        rows = result.all()
        for result_type, count in rows:
            stats.by_type[result_type] = count

    return stats


async def get_error_stats(
    session_factory: ScopedSessionFactory,
) -> ErrorStats:
    """Get statistics about errors.

    Args:
        session_factory: Async session factory.

    Returns:
        ErrorStats instance with error metrics.
    """
    async with session_factory() as session:
        result = await session.execute(
            select(
                sa.func.count(),
                sa.func.sum(
                    sa.case((Error.is_resolved == sa.false(), 1), else_=0)  # type: ignore[arg-type]
                ),
                sa.func.sum(
                    sa.case((Error.is_resolved == sa.true(), 1), else_=0)  # type: ignore[arg-type]
                ),
            )
        )
        row = result.first()

        stats = ErrorStats()
        if row:
            stats.total = row[0]
            stats.unresolved = row[1] or 0
            stats.resolved = row[2] or 0

        # Get counts by type
        result = await session.execute(
            select(Error.error_type, sa.func.count()).group_by(
                Error.error_type
            )
        )
        rows = result.all()
        for error_type, count in rows:
            stats.by_type[error_type] = count

        # Get counts by continuation (via joined requests)
        result = await session.execute(
            select(Request.continuation, sa.func.count(Error.id))  # type: ignore[arg-type]
            .join(Request, Error.request_id == Request.id)  # type: ignore[arg-type]
            .group_by(Request.continuation)
        )
        rows = result.all()
        for continuation, count in rows:
            stats.by_continuation[continuation] = count

    return stats


async def get_stats(
    session_factory: ScopedSessionFactory,
) -> DevDriverStats:
    """Get all statistics for the LocalDevDriver.

    Args:
        session_factory: Async session factory.

    Returns:
        DevDriverStats instance with all statistics.
    """
    async with session_factory() as session:
        # Get run metadata
        result = await session.execute(
            select(RunMetadata.scraper_name, RunMetadata.status).where(
                RunMetadata.id == 1
            )
        )
        row = result.first()
        scraper_name = row[0] if row else ""
        run_status = row[1] if row else "unknown"

    return DevDriverStats(
        queue=await get_queue_stats(session_factory),
        throughput=await get_throughput_stats(session_factory),
        compression=await get_compression_stats(session_factory),
        results=await get_result_stats(session_factory),
        errors=await get_error_stats(session_factory),
        run_status=run_status,
        scraper_name=scraper_name,
    )
