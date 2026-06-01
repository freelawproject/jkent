"""REST API endpoints for viewing results within a run.

This module provides endpoints for:
- Listing results with filters
- Getting result details including full data
- Summary statistics by result type with valid/invalid counts
- JSONL export for bulk data download
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import select

from kent.driver.persistent_driver.web.app import (
    RunManager,
    get_run_manager,
)
from kent.driver.persistent_driver.web.routes._helpers import get_debugger

router = APIRouter(prefix="/api/runs/{run_id}/results", tags=["results"])


class ResultResponse(BaseModel):
    """Response model for a single result."""

    id: int
    request_id: int | None
    result_type: str
    is_valid: bool
    created_at: str | None


class ResultWithDataResponse(ResultResponse):
    """Response model for a result with full data."""

    data: dict[str, Any]
    validation_errors: list[dict[str, Any]] | None


class ResultListResponse(BaseModel):
    """Response model for listing results."""

    items: list[ResultResponse]
    total: int
    offset: int
    limit: int
    has_more: bool


class ResultTypeSummaryItem(BaseModel):
    """Summary stats for a single result type."""

    result_type: str
    valid_count: int
    invalid_count: int
    total_count: int


class ResultsSummaryResponse(BaseModel):
    """Response model for results summary statistics."""

    total_valid: int
    total_invalid: int
    total: int
    by_type: list[ResultTypeSummaryItem]


@router.get("", response_model=ResultListResponse)
async def list_results(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
    result_type: str | None = Query(None, description="Filter by result type"),
    is_valid: bool | None = Query(
        None, description="Filter by validation status"
    ),
    request_id: int | None = Query(None, description="Filter by request ID"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
) -> ResultListResponse:
    """List results for a run with optional filters.

    Args:
        run_id: The run identifier.
        result_type: Optional result type filter.
        is_valid: Optional validation status filter.
        request_id: Optional request ID filter.
        offset: Pagination offset.
        limit: Maximum number of results.

    Returns:
        Paginated list of results.
    """
    debugger = await get_debugger(run_id, manager)

    # LDDD only supports result_type and is_valid filters, not request_id
    # Fall back to SQL for request_id filtering
    if request_id is not None:
        page = await debugger.sql.list_results(
            result_type=result_type,
            is_valid=is_valid,
            request_id=request_id,
            offset=offset,
            limit=limit,
        )
    else:
        page = await debugger.list_results(
            result_type=result_type,
            is_valid=is_valid,
            offset=offset,
            limit=limit,
        )

    items = [
        ResultResponse(
            id=r.id,
            request_id=r.request_id,
            result_type=r.result_type,
            is_valid=r.is_valid,
            created_at=r.created_at,
        )
        for r in page.items
    ]

    return ResultListResponse(
        items=items,
        total=page.total,
        offset=page.offset,
        limit=page.limit,
        has_more=page.has_more,
    )


# NOTE: Literal path routes must be defined BEFORE parameterized routes
# to ensure FastAPI matches them correctly (e.g., /summary before /{result_id})


@router.get("/types/summary")
async def get_result_type_summary(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> dict[str, int]:
    """Get a summary of result counts by type.

    Args:
        run_id: The run identifier.

    Returns:
        Dictionary mapping result types to their counts.
    """
    from kent.driver.persistent_driver.models import Result

    debugger = await get_debugger(run_id, manager)

    async with debugger._session_factory() as session:
        result = await session.execute(
            select(Result.result_type, sa.func.count())
            .group_by(Result.result_type)
            .order_by(sa.func.count().desc())
        )
        rows = result.all()

    return {r[0]: r[1] for r in rows}


@router.get("/summary", response_model=ResultsSummaryResponse)
async def get_results_summary(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ResultsSummaryResponse:
    """Get summary statistics for results including valid/invalid counts by type.

    Args:
        run_id: The run identifier.

    Returns:
        Summary with total counts and breakdown by result type.
    """
    debugger = await get_debugger(run_id, manager)

    # Use LDDD's get_result_summary method
    summary = await debugger.get_result_summary()

    by_type: list[ResultTypeSummaryItem] = []
    total_valid = 0
    total_invalid = 0

    for result_type, counts in summary.items():
        valid_count = counts["valid"]
        invalid_count = counts["invalid"]
        total_count = counts["total"]

        by_type.append(
            ResultTypeSummaryItem(
                result_type=result_type,
                valid_count=valid_count,
                invalid_count=invalid_count,
                total_count=total_count,
            )
        )
        total_valid += valid_count
        total_invalid += invalid_count

    return ResultsSummaryResponse(
        total_valid=total_valid,
        total_invalid=total_invalid,
        total=total_valid + total_invalid,
        by_type=by_type,
    )


@router.get("/export.jsonl")
async def export_results_jsonl(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
    result_type: str | None = Query(None, description="Filter by result type"),
    is_valid: bool | None = Query(
        None, description="Filter by validation status"
    ),
) -> StreamingResponse:
    """Export results as JSONL (newline-delimited JSON) for bulk download.

    Each line is a valid JSON object containing result data. This format
    is efficient for large datasets and can be processed line-by-line.

    Args:
        run_id: The run identifier.
        result_type: Optional filter by result type.
        is_valid: Optional filter by validation status.

    Returns:
        Streaming JSONL response with Content-Disposition for download.
    """
    from kent.driver.persistent_driver.models import Result as ResultModel

    debugger = await get_debugger(run_id, manager)

    # Build query
    stmt = select(  # type: ignore[call-overload]
        ResultModel.id,
        ResultModel.request_id,
        ResultModel.result_type,
        ResultModel.data_json,
        ResultModel.is_valid,
        ResultModel.validation_errors_json,
        ResultModel.created_at,
    ).order_by(ResultModel.created_at.asc())  # type: ignore[union-attr]

    if result_type:
        stmt = stmt.where(ResultModel.result_type == result_type)
    if is_valid is not None:
        stmt = stmt.where(ResultModel.is_valid == is_valid)

    async def generate_jsonl() -> AsyncGenerator[bytes, None]:
        """Stream results as JSONL."""
        async with debugger._session_factory() as session:
            result = await session.execute(stmt)
            for row in result.all():
                (
                    result_id,
                    request_id_val,
                    rtype,
                    data_json,
                    valid,
                    errors_json,
                    created_at,
                ) = row

                # Parse JSON fields
                try:
                    data = json.loads(data_json) if data_json else {}
                except json.JSONDecodeError:
                    data = {}

                validation_errors = None
                if errors_json:
                    try:
                        validation_errors = json.loads(errors_json)
                    except json.JSONDecodeError:
                        pass

                record = {
                    "id": result_id,
                    "request_id": request_id_val,
                    "result_type": rtype,
                    "is_valid": bool(valid),
                    "data": data,
                    "validation_errors": validation_errors,
                    "created_at": created_at,
                }
                yield (json.dumps(record) + "\n").encode("utf-8")

    # Build filename with optional filters
    filename_parts = [run_id, "results"]
    if result_type:
        filename_parts.append(result_type)
    if is_valid is not None:
        filename_parts.append("valid" if is_valid else "invalid")
    filename = "-".join(filename_parts) + ".jsonl"

    return StreamingResponse(
        generate_jsonl(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# Parameterized route must come LAST to avoid matching literal paths like /summary
@router.get("/{result_id}", response_model=ResultWithDataResponse)
async def get_result(
    run_id: str,
    result_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ResultWithDataResponse:
    """Get full details for a specific result including data.

    Args:
        run_id: The run identifier.
        result_id: The result ID.

    Returns:
        Full result details including data and validation errors.

    Raises:
        HTTPException: 404 if result not found.
    """
    debugger = await get_debugger(run_id, manager)

    record = await debugger.get_result(result_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Result {result_id} not found in run '{run_id}'",
        )

    # Parse JSON data
    try:
        data = json.loads(record.data_json) if record.data_json else {}
    except json.JSONDecodeError:
        data = {}

    # Parse validation errors
    validation_errors = None
    if record.validation_errors_json:
        try:
            validation_errors = json.loads(record.validation_errors_json)
        except json.JSONDecodeError:
            validation_errors = None

    return ResultWithDataResponse(
        id=record.id,
        request_id=record.request_id,
        result_type=record.result_type,
        is_valid=record.is_valid,
        created_at=record.created_at,
        data=data,
        validation_errors=validation_errors,
    )
