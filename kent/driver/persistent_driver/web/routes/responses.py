"""REST API endpoints for viewing responses within a run.

This module provides endpoints for:
- Listing responses with filters
- Getting response details
- Getting decompressed response content
- Analyzing response output (continuation re-execution with XPath observation)
- Annotated HTML view with debug palette
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlmodel import select

from kent.driver.persistent_driver.web.app import (
    RunManager,
    get_run_manager,
)
from kent.driver.persistent_driver.web.routes._helpers import get_debugger

router = APIRouter(prefix="/api/runs/{run_id}/responses", tags=["responses"])


class ResponseResponse(BaseModel):
    """Response model for a single HTTP response record."""

    id: int
    status_code: int
    url: str
    content_size_original: int | None
    content_size_compressed: int | None
    compression_ratio: float | None
    continuation: str
    created_at: str | None
    compression_dict_id: int | None
    speculation_outcome: str | None = (
        None  # 'success', 'stopped', 'skipped', or None
    )


class ResponseListResponse(BaseModel):
    """Response model for listing responses."""

    items: list[ResponseResponse]
    total: int
    offset: int
    limit: int
    has_more: bool


class SpeculationSummaryResponse(BaseModel):
    """Response model for speculation outcome summary."""

    success: int = 0
    stopped: int = 0
    skipped: int = 0
    non_speculative: int = 0
    total: int = 0


# =============================================================================
# Response Output Analysis Models (for debug palette)
# =============================================================================


class SelectorInfo(BaseModel):
    """Information about an XPath/CSS selector query."""

    selector: str
    selector_type: str  # "xpath" or "css"
    description: str
    match_count: int
    expected_min: int
    expected_max: int | None
    sample_elements: list[str]
    element_id: str  # For highlighting
    status: str  # "pass" or "fail"
    children: list[SelectorInfo] = []
    parent_element_id: str | None = (
        None  # ID of parent query (for scoped highlights)
    )


class OutputYield(BaseModel):
    """A single item yielded by a continuation."""

    type: str  # "ParsedData", "Request", etc.
    data_type: str | None = None  # For ParsedData: the model class name
    preview: str | None = None  # For ParsedData: truncated string repr
    url: str | None = None  # For request types
    method: str | None = None  # For request types
    continuation: str | None = None  # For request types
    speculative_id: int | None = None  # For speculative requests
    expected_type: str | None = None  # For archive requests


class ResponseOutputResponse(BaseModel):
    """Response model for /output endpoint - continuation analysis."""

    request_id: int
    continuation: str
    is_html: bool
    selectors: list[SelectorInfo]
    yields: list[OutputYield]
    yield_summary: dict[str, int]  # e.g., {"ParsedData": 3, "Request": 2}
    error: str | None = None


@router.get("/speculation-summary", response_model=SpeculationSummaryResponse)
async def get_speculation_summary(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> SpeculationSummaryResponse:
    """Get summary of speculation outcomes for a run.

    Returns counts of:
    - success: Speculative requests that continued (2xx or callback approved)
    - stopped: Speculative requests that stopped (non-2xx, not approved)
    - skipped: Deduplicated speculative requests
    - non_speculative: Regular (non-speculative) requests
    """
    from kent.driver.persistent_driver.models import Request as RequestModel

    debugger = await get_debugger(run_id, manager, read_only=True)

    async with debugger._session_factory() as session:
        result = await session.execute(
            select(RequestModel.speculation_outcome, sa.func.count())
            .where(
                RequestModel.response_status_code.isnot(None),  # type: ignore[union-attr]
            )
            .group_by(RequestModel.speculation_outcome)
        )
        rows = result.all()

    summary = SpeculationSummaryResponse()
    for outcome, count in rows:
        if outcome == "success":
            summary.success = count
        elif outcome == "stopped":
            summary.stopped = count
        elif outcome == "skipped":
            summary.skipped = count
        elif outcome is None:
            summary.non_speculative = count
        summary.total += count

    return summary


@router.get("", response_model=ResponseListResponse)
async def list_responses(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_run_manager)],
    continuation: str | None = Query(
        None, description="Filter by continuation"
    ),
    request_id: int | None = Query(None, description="Filter by request ID"),
    speculation_outcome: str | None = Query(
        None,
        description="Filter by speculation outcome: 'success', 'stopped', 'skipped'",
    ),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=500, description="Pagination limit"),
) -> ResponseListResponse:
    """List responses for a run with optional filters.

    Args:
        run_id: The run identifier.
        continuation: Optional continuation name filter.
        request_id: Optional request ID filter.
        speculation_outcome: Optional speculation outcome filter.
        offset: Pagination offset.
        limit: Maximum number of results.

    Returns:
        Paginated list of responses.
    """
    debugger = await get_debugger(run_id, manager, read_only=True)

    page = await debugger.sql.list_responses(
        continuation=continuation,
        request_id=request_id,
        speculation_outcome=speculation_outcome,
        offset=offset,
        limit=limit,
    )

    items = [
        ResponseResponse(
            id=r.id,
            status_code=r.status_code,
            url=r.url,
            content_size_original=r.content_size_original,
            content_size_compressed=r.content_size_compressed,
            compression_ratio=r.compression_ratio,
            continuation=r.continuation,
            created_at=r.created_at,
            compression_dict_id=r.compression_dict_id,
            speculation_outcome=r.speculation_outcome,
        )
        for r in page.items
    ]

    return ResponseListResponse(
        items=items,
        total=page.total,
        offset=page.offset,
        limit=page.limit,
        has_more=page.has_more,
    )


@router.get("/{request_id}", response_model=ResponseResponse)
async def get_response(
    run_id: str,
    request_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ResponseResponse:
    """Get details for a specific response by request ID.

    Args:
        run_id: The run identifier.
        request_id: The request ID.

    Returns:
        Response details (excluding content).

    Raises:
        HTTPException: 404 if response not found.
    """
    debugger = await get_debugger(run_id, manager, read_only=True)

    record = await debugger.get_response(request_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response for request {request_id} not found in run '{run_id}'",
        )

    return ResponseResponse(
        id=record.id,
        status_code=record.status_code,
        url=record.url,
        content_size_original=record.content_size_original,
        content_size_compressed=record.content_size_compressed,
        compression_ratio=record.compression_ratio,
        continuation=record.continuation,
        created_at=record.created_at,
        compression_dict_id=record.compression_dict_id,
        speculation_outcome=record.speculation_outcome,
    )


@router.get("/{request_id}/content")
async def get_response_content(
    run_id: str,
    request_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> Response:
    """Get decompressed content for a response.

    Args:
        run_id: The run identifier.
        request_id: The request ID.

    Returns:
        Decompressed content as raw bytes.

    Raises:
        HTTPException: 404 if response not found.
        HTTPException: 500 if decompression fails.
    """
    content, headers_json = await _fetch_response_content(
        run_id, request_id, manager
    )

    # Try to get content-type from headers
    content_type = "application/octet-stream"
    if headers_json:
        try:
            headers = json.loads(headers_json)
            if isinstance(headers, dict):
                for key, value in headers.items():
                    if key.lower() == "content-type":
                        content_type = value
                        break
        except json.JSONDecodeError:
            pass

    return Response(content=content, media_type=content_type)


async def _fetch_response_content(
    run_id: str, request_id: int, manager: RunManager
) -> tuple[bytes, str | None]:
    """Load decompressed content + headers for a response, or raise an HTTPException.

    Returns:
        (content, headers_json) — content is non-empty bytes; headers_json may be None.

    Raises:
        HTTPException: 500 if decompression fails, 404 if the response is missing or empty.
    """
    debugger = await get_debugger(run_id, manager, read_only=True)

    try:
        result = await debugger.sql.get_response_content_with_headers(
            request_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to decompress content: {e}",
        ) from e

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response for request {request_id} not found in run '{run_id}'",
        )

    content, headers_json = result

    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response for request {request_id} has no content",
        )

    return content, headers_json


def _extract_content_type(headers_json: str | None) -> str:
    """Extract Content-Type from headers JSON."""
    if not headers_json:
        return "application/octet-stream"
    try:
        headers = json.loads(headers_json)
        if isinstance(headers, dict):
            for key, value in headers.items():
                if key.lower() == "content-type":
                    return value
    except json.JSONDecodeError:
        pass
    return "application/octet-stream"


def _is_html_content_type(content_type: str) -> bool:
    """Check if content type indicates HTML."""
    ct_lower = content_type.lower()
    return "text/html" in ct_lower or "application/xhtml" in ct_lower


def _selector_query_to_info(query: dict[str, Any]) -> SelectorInfo:
    """Convert SelectorObserver query dict to SelectorInfo model."""
    # Determine pass/fail status
    match_count = query.get("match_count", 0)
    expected_min = query.get("expected_min", 1)
    expected_max = query.get("expected_max")

    if match_count >= expected_min:
        if expected_max is None or match_count <= expected_max:
            status_val = "pass"
        else:
            status_val = "fail"
    else:
        status_val = "fail"

    return SelectorInfo(
        selector=query.get("selector", ""),
        selector_type=query.get("selector_type", "xpath"),
        description=query.get("description", ""),
        match_count=match_count,
        expected_min=expected_min,
        expected_max=expected_max,
        sample_elements=query.get("sample_elements", []),
        element_id=query.get("element_id", ""),
        status=status_val,
        children=[
            _selector_query_to_info(child)
            for child in query.get("children", [])
        ],
        parent_element_id=query.get("parent_element_id"),
    )


def _describe_yield_for_output(item: Any) -> OutputYield:
    """Create OutputYield from a yielded item."""
    from kent.data_types import (
        ParsedData,
        Request,
    )

    if isinstance(item, ParsedData):
        data = item.unwrap()
        data_str = str(data)
        return OutputYield(
            type="ParsedData",
            data_type=type(data).__name__,
            preview=data_str[:500] + "..."
            if len(data_str) > 500
            else data_str,
        )
    elif isinstance(item, Request):
        continuation = (
            item.continuation
            if isinstance(item.continuation, str)
            else item.continuation.__name__
        )
        if item.archive:
            return OutputYield(
                type="ArchiveRequest",
                url=item.request.url,
                method=item.request.method.value,
                continuation=continuation,
                expected_type=item.expected_type,
            )
        elif item.nonnavigating:
            return OutputYield(
                type="NonNavigatingRequest",
                url=item.request.url,
                method=item.request.method.value,
                continuation=continuation,
            )
        else:
            return OutputYield(
                type="NavigatingRequest",
                url=item.request.url,
                method=item.request.method.value,
                continuation=continuation,
            )
    elif item is None:
        return OutputYield(type="None")
    else:
        return OutputYield(
            type="Unknown",
            preview=str(item)[:200],
        )


async def _get_driver_for_run(run_id: str, manager: RunManager):
    """Get driver instance for a loaded run.

    Args:
        run_id: The run identifier.
        manager: The run manager.

    Returns:
        LocalDevDriver instance.

    Raises:
        HTTPException: 404 if run not found, 400 if not loaded.
    """
    run_info = await manager.get_run(run_id)
    if run_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    if run_info.driver is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run '{run_id}' is not loaded. Load it first.",
        )
    return run_info.driver


async def _resolve_scraper(run_id: str, manager: RunManager):
    """Resolve a scraper instance for a run.

    Prefers the loaded driver's scraper if available. Otherwise, resolves
    the scraper class from the registry using the run's metadata.

    Args:
        run_id: The run identifier.
        manager: The run manager.

    Returns:
        BaseScraper instance.

    Raises:
        HTTPException: 404 if run not found, 400 if scraper cannot be resolved.
    """
    from kent.driver.persistent_driver.web.app import (
        get_sql_manager_for_run,
    )
    from kent.driver.persistent_driver.web.scraper_registry import (
        get_registry,
    )

    # If run is loaded, use the driver's scraper directly
    run_info = await manager.get_run(run_id)
    if run_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    if run_info.driver is not None:
        return run_info.driver.scraper

    # Not loaded — resolve from registry
    sql_manager = await get_sql_manager_for_run(run_id, manager)
    run_metadata = await sql_manager.get_run_metadata()
    if run_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No metadata found for run '{run_id}'",
        )

    scraper_name = run_metadata.get("scraper_name")
    if not scraper_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run '{run_id}' has no scraper_name in metadata",
        )

    registry = get_registry()
    matching = registry.find_scrapers_by_name(scraper_name)

    if not matching:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Scraper '{scraper_name}' not found in registry",
        )

    scraper = registry.instantiate_scraper(matching[0].full_path)
    if scraper is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to instantiate scraper '{scraper_name}'",
        )
    return scraper


@router.get("/{request_id}/output", response_model=ResponseOutputResponse)
async def get_response_output(
    run_id: str,
    request_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> ResponseOutputResponse:
    """Analyze a response by re-running its continuation with XPath observation.

    This endpoint retrieves a stored response and re-runs the continuation
    method with a SelectorObserver active to capture all XPath/CSS queries.
    Returns structured data suitable for the debug palette UI.

    Works with both loaded and unloaded runs. For unloaded runs, the scraper
    class is resolved from the registry using the run's metadata.

    Args:
        run_id: The run identifier.
        request_id: The request ID of the response to analyze.

    Returns:
        Analysis results including selectors, yields, and any errors.

    Raises:
        HTTPException: 404 if run or response not found.
        HTTPException: 400 if scraper class cannot be resolved.
    """
    from kent.common.selector_observer import SelectorObserver
    from kent.data_types import (
        HttpMethod,
        HTTPRequestParams,
        Request,
    )
    from kent.data_types import (
        Response as ScraperResponse,
    )

    debugger = await get_debugger(run_id, manager, read_only=True)

    # Resolve the scraper instance: prefer loaded driver, fall back to registry
    scraper = await _resolve_scraper(run_id, manager)

    # Get response and request data - all in one table now
    from kent.driver.persistent_driver.models import (
        Request as RequestModel,
    )

    async with debugger._session_factory() as session:
        stmt = select(  # type: ignore[call-overload,misc]
            RequestModel.response_status_code,
            RequestModel.response_url,
            RequestModel.response_headers_json,
            RequestModel.continuation,
            RequestModel.method,
            RequestModel.url,
            RequestModel.accumulated_data_json,
            RequestModel.permanent_json,
        ).where(
            RequestModel.id == request_id,
            RequestModel.response_status_code.isnot(None),  # type: ignore[union-attr]
        )
        result = await session.execute(stmt)
        row = result.first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Response for request {request_id} not found",
        )

    (
        status_code,
        url,
        headers_json,
        continuation_name,
        method,
        request_url,
        accumulated_data_json,
        permanent_json,
    ) = row

    # Decompress content
    content = await debugger.get_response_content(request_id)
    if content is None:
        content = b""

    # Determine if this is HTML
    content_type = _extract_content_type(headers_json)
    is_html = _is_html_content_type(content_type)

    # Reconstruct Response object
    headers = json.loads(headers_json) if headers_json else {}
    accumulated_data = (
        json.loads(accumulated_data_json) if accumulated_data_json else {}
    )
    permanent = json.loads(permanent_json) if permanent_json else {}

    http_params = HTTPRequestParams(
        method=HttpMethod(method),
        url=request_url,
    )
    reconstructed_request = Request(
        request=http_params,
        continuation=continuation_name,
        current_location=request_url,
        accumulated_data=accumulated_data,
        permanent=permanent,
    )

    # Decode content to text
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="replace")

    response = ScraperResponse(
        status_code=status_code,
        url=url,
        content=content,
        text=text,
        headers=headers,
        request=reconstructed_request,
    )

    # Run continuation with observer
    yields: list[OutputYield] = []
    yield_summary: dict[str, int] = {}
    error: str | None = None
    selectors: list[SelectorInfo] = []

    with SelectorObserver() as observer:
        try:
            continuation_method = scraper.get_continuation(continuation_name)
            gen = continuation_method(response)

            for item in gen:
                yield_info = _describe_yield_for_output(item)
                yields.append(yield_info)

                # Update summary counts
                yield_summary[yield_info.type] = (
                    yield_summary.get(yield_info.type, 0) + 1
                )

        except Exception as e:
            import traceback

            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        # Convert observer queries to SelectorInfo list
        selectors = [_selector_query_to_info(q) for q in observer.json()]

    return ResponseOutputResponse(
        request_id=request_id,
        continuation=continuation_name,
        is_html=is_html,
        selectors=selectors,
        yields=yields,
        yield_summary=yield_summary,
        error=error,
    )


@router.get("/{request_id}/annotated")
async def get_annotated_response(
    run_id: str,
    request_id: int,
    manager: Annotated[RunManager, Depends(get_run_manager)],
) -> Response:
    """Get HTML response with injected debug palette.

    For HTML responses, this endpoint returns the content with JavaScript
    and CSS injected to display an interactive debug palette. The palette
    allows highlighting XPath/CSS selector matches and viewing continuation
    output.

    For non-HTML responses, redirects to the /content endpoint.

    Args:
        run_id: The run identifier.
        request_id: The request ID.

    Returns:
        HTML with injected debug palette, or redirect for non-HTML.

    Raises:
        HTTPException: 404 if response not found.
    """
    content, headers_json = await _fetch_response_content(
        run_id, request_id, manager
    )

    # Check if HTML
    content_type = _extract_content_type(headers_json)
    if not _is_html_content_type(content_type):
        # Redirect to raw content for non-HTML
        return RedirectResponse(
            url=f"/api/runs/{run_id}/responses/{request_id}/content",
            status_code=status.HTTP_302_FOUND,
        )

    # Decode HTML
    try:
        html = content.decode("utf-8")
    except UnicodeDecodeError:
        html = content.decode("utf-8", errors="replace")

    # Inject debug palette
    output_url = f"/api/runs/{run_id}/responses/{request_id}/output"
    injected_html = _inject_debug_palette(html, output_url)

    return Response(content=injected_html, media_type="text/html")


def _inject_debug_palette(html: str, output_url: str) -> str:
    """Inject debug palette assets into HTML.

    Args:
        html: The original HTML content.
        output_url: URL to fetch output data from.

    Returns:
        HTML with debug palette injected.
    """
    injection = f'''
<!-- Debug Palette Styles -->
<link rel="stylesheet" href="/static/css/debug_palette.css">

<!-- Debug Palette Container -->
<div id="debug-palette-root"></div>

<!-- Debug Palette Script -->
<script>
  window.DEBUG_PALETTE_CONFIG = {{
    outputUrl: "{output_url}"
  }};
</script>
<script src="/static/js/debug_palette.js"></script>
'''

    # Try to inject before </body>
    body_close_lower = html.lower().rfind("</body>")
    if body_close_lower != -1:
        # Find the actual position (case-preserved)
        return html[:body_close_lower] + injection + html[body_close_lower:]
    else:
        # No </body> tag, append to end
        return html + injection
