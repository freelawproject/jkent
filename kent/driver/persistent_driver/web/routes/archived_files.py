"""REST API endpoints for viewing archived files within a run.

This module provides endpoints for:
- Listing archived files with filters
- Getting archived file details
- Getting archived file content
- Archived files statistics
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel
from sqlmodel import select

from kent.driver.persistent_driver.web.app import (
    RunManager,
    get_run_manager,
)
from kent.driver.persistent_driver.web.routes._helpers import get_debugger

router = APIRouter(
    prefix="/api/runs/{run_id}/archived-files", tags=["archived-files"]
)


class ArchivedFileResponse(BaseModel):
    """Response model for a single archived file record."""

    id: int
    request_id: int
    file_path: str
    original_url: str
    expected_type: str | None
    file_size: int | None
    content_hash: str | None
    created_at: str | None
    continuation: str | None


class ArchivedFileListResponse(BaseModel):
    """Response model for listing archived files."""

    items: list[ArchivedFileResponse]
    total: int
    offset: int
    limit: int
    has_more: bool


class ArchivedFilesStatsResponse(BaseModel):
    """Response model for archived files statistics."""

    total_files: int
    total_size: int
    total_size_human: str


async def _fetch_archived_file_row(
    run_id: str, file_id: int, manager: RunManager
) -> sa.Row:
    """Fetch a single archived file row joined with its request, or 404."""
    from kent.driver.persistent_driver.models import ArchivedFile
    from kent.driver.persistent_driver.models import Request as RequestModel

    debugger = await get_debugger(run_id, manager, read_only=True)

    async with debugger._session_factory() as session:
        stmt = (
            select(  # type: ignore[call-overload,misc]
                ArchivedFile.id,
                ArchivedFile.request_id,
                ArchivedFile.file_path,
                ArchivedFile.original_url,
                ArchivedFile.expected_type,
                ArchivedFile.file_size,
                ArchivedFile.content_hash,
                ArchivedFile.created_at,
                RequestModel.continuation,
            )
            .outerjoin(
                RequestModel, ArchivedFile.request_id == RequestModel.id
            )
            .where(ArchivedFile.id == file_id)
        )
        result = await session.execute(stmt)
        row = result.first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Archived file {file_id} not found in run '{run_id}'",
        )
    return row


def _format_size(size: int) -> str:
    """Format bytes to human-readable string."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.1f} GB"


@router.get("", response_model=ArchivedFileListResponse)
async def list_archived_files(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
    expected_type: str | None = Query(None, description="Filter by file type"),
    continuation: str | None = Query(
        None, description="Filter by continuation"
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
) -> ArchivedFileListResponse:
    """List archived files for a run with optional filters.

    Args:
        run_id: The run identifier.
        expected_type: Optional file type filter (pdf, audio, etc.).
        continuation: Optional continuation name filter.
        offset: Pagination offset.
        limit: Maximum number of results.

    Returns:
        Paginated list of archived files.
    """
    from kent.driver.persistent_driver.models import ArchivedFile
    from kent.driver.persistent_driver.models import Request as RequestModel

    debugger = await get_debugger(run_id, manager, read_only=True)

    async with debugger._session_factory() as session:
        # Build count query
        count_stmt = select(sa.func.count()).select_from(ArchivedFile)
        if continuation:
            count_stmt = count_stmt.join(
                RequestModel,
                ArchivedFile.request_id == RequestModel.id,  # type: ignore[arg-type]
            )
        if expected_type:
            count_stmt = count_stmt.where(
                ArchivedFile.expected_type == expected_type
            )
        if continuation:
            count_stmt = count_stmt.where(
                RequestModel.continuation == continuation
            )

        result = await session.execute(count_stmt)
        total = result.scalar_one()

        # Get paginated results
        stmt = (
            select(  # type: ignore[call-overload,misc]
                ArchivedFile.id,
                ArchivedFile.request_id,
                ArchivedFile.file_path,
                ArchivedFile.original_url,
                ArchivedFile.expected_type,
                ArchivedFile.file_size,
                ArchivedFile.content_hash,
                ArchivedFile.created_at,
                RequestModel.continuation,
            )
            .outerjoin(
                RequestModel, ArchivedFile.request_id == RequestModel.id
            )
            .order_by(ArchivedFile.created_at.desc())  # type: ignore[union-attr]
            .limit(limit)
            .offset(offset)
        )
        if expected_type:
            stmt = stmt.where(ArchivedFile.expected_type == expected_type)
        if continuation:
            stmt = stmt.where(RequestModel.continuation == continuation)

        result = await session.execute(stmt)
        rows = result.all()

    items = [
        ArchivedFileResponse(
            id=r[0],
            request_id=r[1],
            file_path=r[2],
            original_url=r[3],
            expected_type=r[4],
            file_size=r[5],
            content_hash=r[6],
            created_at=r[7],
            continuation=r[8],
        )
        for r in rows
    ]

    return ArchivedFileListResponse(
        items=items,
        total=total,
        offset=offset,
        limit=limit,
        has_more=offset + len(items) < total,
    )


@router.get("/stats", response_model=ArchivedFilesStatsResponse)
async def get_archived_files_stats(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ArchivedFilesStatsResponse:
    """Get statistics for archived files.

    Args:
        run_id: The run identifier.

    Returns:
        Archived files statistics.
    """
    from kent.driver.persistent_driver.models import ArchivedFile

    debugger = await get_debugger(run_id, manager, read_only=True)

    async with debugger._session_factory() as session:
        result = await session.execute(
            select(
                sa.func.count(),
                sa.func.coalesce(sa.func.sum(ArchivedFile.file_size), 0),
            )
        )
        row = result.first()

    total_files = row[0] if row else 0
    total_size = row[1] if row else 0

    return ArchivedFilesStatsResponse(
        total_files=total_files,
        total_size=total_size,
        total_size_human=_format_size(total_size),
    )


@router.get("/{file_id}", response_model=ArchivedFileResponse)
async def get_archived_file(
    run_id: str,
    file_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ArchivedFileResponse:
    """Get details for a specific archived file.

    Args:
        run_id: The run identifier.
        file_id: The archived file ID.

    Returns:
        Archived file details.

    Raises:
        HTTPException: 404 if file not found.
    """
    row = await _fetch_archived_file_row(run_id, file_id, manager)

    return ArchivedFileResponse(
        id=row[0],
        request_id=row[1],
        file_path=row[2],
        original_url=row[3],
        expected_type=row[4],
        file_size=row[5],
        content_hash=row[6],
        created_at=row[7],
        continuation=row[8],
    )


@router.get("/{file_id}/content")
async def get_archived_file_content(
    run_id: str,
    file_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> Response:
    """Get content of an archived file from disk.

    Args:
        run_id: The run identifier.
        file_id: The archived file ID.

    Returns:
        File content as raw bytes.

    Raises:
        HTTPException: 404 if file not found or file doesn't exist on disk.
    """
    row = await _fetch_archived_file_row(run_id, file_id, manager)

    file_path = row[2]
    expected_type = row[4]

    # Read file from disk
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File not found on disk: {file_path}",
        )

    content = path.read_bytes()

    # Determine content type
    content_type_map = {
        "pdf": "application/pdf",
        "audio": "audio/mpeg",
        "image": "image/jpeg",
        "html": "text/html",
    }
    content_type = content_type_map.get(
        expected_type or "", "application/octet-stream"
    )

    return Response(content=content, media_type=content_type)
