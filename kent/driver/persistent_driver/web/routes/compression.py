"""REST API endpoints for compression management within a run.

This module provides endpoints for:
- Training compression dictionaries
- Recompressing responses with new dictionaries
- Viewing compression statistics
"""

from __future__ import annotations

from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlmodel import select

from kent.driver.persistent_driver.web.app import (
    RunManager,
    get_run_manager,
)
from kent.driver.persistent_driver.web.routes._helpers import get_debugger

router = APIRouter(
    prefix="/api/runs/{run_id}/compression", tags=["compression"]
)


class TrainDictRequest(BaseModel):
    """Request model for training a compression dictionary."""

    continuation: str = Field(
        ..., description="Continuation to train dictionary for"
    )
    sample_limit: int | None = Field(
        default=None,
        ge=1,
        description="Maximum samples to use (auto-calculated if not specified)",
    )
    dict_size: int | None = Field(
        default=None,
        ge=1024,
        description="Dictionary size in bytes (auto-calculated if not specified)",
    )


class TrainDictResponse(BaseModel):
    """Response model for dictionary training."""

    dict_id: int
    continuation: str
    sample_count: int
    dict_size: int
    message: str


class RecompressRequest(BaseModel):
    """Request model for recompressing responses."""

    continuation: str = Field(..., description="Continuation to recompress")
    compression_level: int = Field(
        default=3, ge=1, le=22, description="Zstd compression level"
    )


class RecompressResponse(BaseModel):
    """Response model for recompression."""

    count: int
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    message: str


class CompressionStatsResponse(BaseModel):
    """Response model for compression statistics."""

    total_responses: int
    total_original_bytes: int
    total_compressed_bytes: int
    compression_ratio: float
    with_dict_count: int
    no_dict_count: int


class CompressionStatsByContinuationItem(BaseModel):
    """Compression stats for a single continuation/dictionary combination."""

    continuation: str
    dict_id: int | None = None
    dict_version: int | None = None
    response_count: int = 0
    total_original_bytes: int = 0
    total_compressed_bytes: int = 0
    compression_ratio: float = 0.0
    has_trained_dict: bool = (
        False  # Whether a trained dict exists for this continuation
    )


class CompressionStatsByContinuationResponse(BaseModel):
    """Response model for compression stats grouped by continuation."""

    items: list[CompressionStatsByContinuationItem]
    grand_total_responses: int
    grand_total_original: int
    grand_total_compressed: int
    overall_ratio: float


# Default training parameters
DEFAULT_DICT_SIZE = 112640  # 110KB
DEFAULT_SAMPLE_LIMIT = 100

# Large collection thresholds
LARGE_COLLECTION_THRESHOLD = 1024 * 1024 * 1024  # 1GB
LARGE_DICT_SIZE = 1024 * 1024  # 1MB
TARGET_SAMPLE_BYTES = 100 * 1024 * 1024  # 100MB


@router.post("/train-dict", response_model=TrainDictResponse)
async def train_dictionary(
    run_id: str,
    request: TrainDictRequest,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> TrainDictResponse:
    """Train a compression dictionary from stored responses.

    Samples responses for the specified continuation and trains a zstd
    dictionary that can significantly improve compression ratios.

    For large collections (>1GB uncompressed), automatically uses a larger
    dictionary (1MB) and samples ~100MB of data for training.

    Args:
        run_id: The run identifier.
        request: Training parameters.

    Returns:
        Training result with new dictionary ID.

    Raises:
        HTTPException: 400 if not enough responses to train.
    """
    from kent.driver.persistent_driver.compression import (
        train_compression_dict,
    )

    debugger = await get_debugger(run_id, manager, read_only=False)

    # Calculate smart defaults if not provided
    dict_size = request.dict_size
    sample_limit = request.sample_limit

    if dict_size is None or sample_limit is None:
        # Query collection stats for this continuation
        from kent.driver.persistent_driver.models import (
            Request as RequestModel,
        )

        async with debugger._session_factory() as session:
            result = await session.execute(
                select(
                    sa.func.count(),
                    sa.func.coalesce(
                        sa.func.sum(RequestModel.content_size_original), 0
                    ),
                ).where(
                    RequestModel.continuation == request.continuation,
                    RequestModel.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            row = result.first()
        response_count = row[0] or 0  # type: ignore[index]
        total_original_bytes = row[1] or 0  # type: ignore[index]

        if dict_size is None:
            if total_original_bytes > LARGE_COLLECTION_THRESHOLD:
                dict_size = LARGE_DICT_SIZE
            else:
                dict_size = DEFAULT_DICT_SIZE

        if sample_limit is None:
            if (
                total_original_bytes > LARGE_COLLECTION_THRESHOLD
                and response_count > 0
            ):
                # Calculate average doc size and target ~100MB of samples
                avg_doc_size = total_original_bytes // response_count
                if avg_doc_size > 0:
                    sample_limit = TARGET_SAMPLE_BYTES // avg_doc_size
                    sample_limit = max(1, sample_limit)  # At least 1 sample
                else:
                    sample_limit = DEFAULT_SAMPLE_LIMIT
            else:
                sample_limit = DEFAULT_SAMPLE_LIMIT

    try:
        dict_id = await train_compression_dict(
            debugger._session_factory,
            request.continuation,
            sample_limit=sample_limit,
            dict_size=dict_size,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    # Get sample count for response
    from kent.driver.persistent_driver.models import CompressionDict

    async with debugger._session_factory() as session:
        result = await session.execute(
            select(CompressionDict.sample_count).where(
                CompressionDict.id == dict_id
            )
        )
        sample_count = result.scalar() or 0

    return TrainDictResponse(
        dict_id=dict_id,
        continuation=request.continuation,
        sample_count=sample_count,  # type: ignore[arg-type]
        dict_size=dict_size,
        message=f"Trained dictionary {dict_id} from {sample_count} samples",
    )


@router.post("/recompress", response_model=RecompressResponse)
async def recompress_responses(
    run_id: str,
    request: RecompressRequest,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> RecompressResponse:
    """Recompress responses using the latest dictionary.

    After training a new dictionary, use this endpoint to recompress
    existing responses for better compression ratios.

    Args:
        run_id: The run identifier.
        request: Recompression parameters.

    Returns:
        Recompression statistics.

    Raises:
        HTTPException: 400 if no dictionary exists for continuation.
    """
    from kent.driver.persistent_driver.compression import (
        recompress_responses as do_recompress,
    )

    debugger = await get_debugger(run_id, manager, read_only=False)

    try:
        count, total_original, total_compressed = await do_recompress(
            debugger._session_factory,
            request.continuation,
            level=request.compression_level,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    ratio = total_original / total_compressed if total_compressed > 0 else 0

    return RecompressResponse(
        count=count,
        total_original_bytes=total_original,
        total_compressed_bytes=total_compressed,
        compression_ratio=round(ratio, 2),
        message=f"Recompressed {count} responses for '{request.continuation}'",
    )


@router.get("/stats", response_model=CompressionStatsResponse)
async def get_compression_stats(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> CompressionStatsResponse:
    """Get compression statistics for the run.

    Args:
        run_id: The run identifier.

    Returns:
        Compression statistics.
    """
    debugger = await get_debugger(run_id, manager, read_only=True)

    stats = await debugger.get_compression_stats()

    total = stats.get("total", 0)
    total_original = stats.get("total_original", 0)
    total_compressed = stats.get("total_compressed", 0)
    with_dict = stats.get("with_dict", 0)
    no_dict = stats.get("no_dict", 0)
    ratio = stats.get("compression_ratio", 0)

    return CompressionStatsResponse(
        total_responses=total,
        total_original_bytes=total_original,
        total_compressed_bytes=total_compressed,
        compression_ratio=round(ratio, 2),
        with_dict_count=with_dict,
        no_dict_count=no_dict,
    )


@router.get(
    "/stats-by-continuation",
    response_model=CompressionStatsByContinuationResponse,
)
async def get_compression_stats_by_continuation(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> CompressionStatsByContinuationResponse:
    """Get compression statistics grouped by continuation and dictionary version.

    Returns a breakdown of compression stats for each continuation, showing
    which dictionary is being used and the compression ratios achieved.

    Args:
        run_id: The run identifier.

    Returns:
        Compression statistics grouped by continuation.
    """
    from kent.driver.persistent_driver.models import (
        CompressionDict,
    )
    from kent.driver.persistent_driver.models import (
        Request as RequestModel,
    )

    debugger = await get_debugger(run_id, manager, read_only=True)

    async with debugger._session_factory() as session:
        # First, get set of continuations that have trained dictionaries
        result = await session.execute(
            select(CompressionDict.continuation).distinct()
        )
        continuations_with_dicts = {row[0] for row in result.all()}

        # Query stats grouped by continuation and dictionary
        result = await session.execute(
            select(  # type: ignore[call-overload]
                RequestModel.continuation,
                RequestModel.compression_dict_id,
                CompressionDict.version,
                sa.func.count().label("response_count"),
                sa.func.coalesce(
                    sa.func.sum(RequestModel.content_size_original), 0
                ).label("total_original"),
                sa.func.coalesce(
                    sa.func.sum(RequestModel.content_size_compressed), 0
                ).label("total_compressed"),
            )
            .where(
                RequestModel.response_status_code.isnot(None),  # type: ignore[union-attr]
            )
            .outerjoin(
                CompressionDict,
                RequestModel.compression_dict_id == CompressionDict.id,
            )
            .group_by(
                RequestModel.continuation,
                RequestModel.compression_dict_id,
            )
            .order_by(
                RequestModel.continuation,
                CompressionDict.version.desc().nulls_last(),  # type: ignore[attr-defined]
            )
        )
        rows = result.all()

    items: list[CompressionStatsByContinuationItem] = []
    grand_total_responses = 0
    grand_total_original = 0
    grand_total_compressed = 0

    for continuation, dict_id, version, count, total_orig, total_comp in rows:
        ratio = total_orig / total_comp if total_comp > 0 else 0.0
        items.append(
            CompressionStatsByContinuationItem(
                continuation=continuation,
                dict_id=dict_id,
                dict_version=version,
                response_count=count,
                total_original_bytes=total_orig,
                total_compressed_bytes=total_comp,
                compression_ratio=round(ratio, 2),
                has_trained_dict=continuation in continuations_with_dicts,
            )
        )
        grand_total_responses += count
        grand_total_original += total_orig
        grand_total_compressed += total_comp

    overall_ratio = (
        grand_total_original / grand_total_compressed
        if grand_total_compressed > 0
        else 0.0
    )

    return CompressionStatsByContinuationResponse(
        items=items,
        grand_total_responses=grand_total_responses,
        grand_total_original=grand_total_original,
        grand_total_compressed=grand_total_compressed,
        overall_ratio=round(overall_ratio, 2),
    )


@router.get("/dicts")
async def list_dictionaries(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> list[dict]:
    """List all compression dictionaries for the run.

    Args:
        run_id: The run identifier.

    Returns:
        List of dictionary metadata.
    """
    debugger = await get_debugger(run_id, manager, read_only=True)

    return await debugger.list_compression_dicts()
