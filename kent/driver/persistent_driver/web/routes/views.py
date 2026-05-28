"""HTML view routes for the web dashboard.

This module provides routes for serving HTML templates:
- Index/dashboard page
- Run detail page
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["views"])

# Setup templates directory
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the main dashboard page.

    Args:
        request: The FastAPI request object.

    Returns:
        Rendered HTML dashboard.
    """
    return templates.TemplateResponse(request, "index.html")


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str) -> HTMLResponse:
    """Render the run detail page.

    Args:
        request: The FastAPI request object.
        run_id: The run identifier.

    Returns:
        Rendered HTML run detail page.
    """
    return templates.TemplateResponse(
        request, "run_detail.html", {"run_id": run_id}
    )
