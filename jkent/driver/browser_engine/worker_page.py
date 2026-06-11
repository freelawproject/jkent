"""WorkerPage - a Playwright page bound to a single worker.

Used by the unified driver's Playwright transport. Depends only on the shared
``database_engine.compression`` helper plus Playwright.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

from jkent.driver.database_engine.compression import compress

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)


class WorkerPage:
    """A Playwright page bound to a single worker, reused across requests.

    Encapsulates per-request state (incidental network requests) so that
    concurrent workers don't corrupt each other's data.
    """

    def __init__(self, page: Page, excluded_resource_types: set[str]):
        self.page = page
        self.incidental_requests: list[dict[str, Any]] = []
        self.current_parent_request_id: int | None = None
        self._excluded_resource_types = excluded_resource_types
        self._register_network_listeners()

    def _register_network_listeners(self) -> None:
        """Register network request/response listeners for incidental tracking."""

        incidentals = self.incidental_requests
        excluded = self._excluded_resource_types

        async def on_request(request: Any) -> None:
            incidental = {
                "resource_type": request.resource_type,
                "method": request.method,
                "url": request.url,
                "headers_json": json.dumps(dict(request.headers)),
                "body": None,
                "status_code": None,
                "response_headers_json": None,
                "content_compressed": None,
                "content_size_original": None,
                "content_size_compressed": None,
                "compression_dict_id": None,
                "started_at_ns": time.time_ns(),
                "completed_at_ns": None,
                "from_cache": None,
                "failure_reason": None,
            }
            incidentals.append(incidental)

        async def on_response(response: Any) -> None:
            request = response.request
            for incidental in incidentals:
                if (
                    incidental["url"] == request.url
                    and incidental["completed_at_ns"] is None
                ):
                    incidental["status_code"] = response.status
                    incidental["response_headers_json"] = json.dumps(
                        dict(response.headers)
                    )
                    incidental["completed_at_ns"] = time.time_ns()
                    incidental["from_cache"] = response.from_service_worker

                    if incidental["resource_type"] not in excluded:
                        try:
                            content = await response.body()
                            content_compressed = compress(content)
                            incidental["content_compressed"] = (
                                content_compressed
                            )
                            incidental["content_size_original"] = len(content)
                            incidental["content_size_compressed"] = len(
                                content_compressed
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to capture content for {request.url}: {e}"
                            )
                    break

        self.page.on("request", on_request)
        self.page.on("response", on_response)

    def clear_request_state(self) -> None:
        """Reset per-request state between navigations."""
        self.incidental_requests.clear()
        self.current_parent_request_id = None

    async def reset_for_reuse(self) -> None:
        """Lightweight cleanup between requests."""
        # Clear before navigation to discard stale events from the prior
        # page's in-flight sub-resources that may land during the goto.
        self.clear_request_state()
        await self.page.goto("about:blank", wait_until="commit")
        # Clear again to remove any events fired by the about:blank
        # navigation itself.
        self.clear_request_state()

    async def close(self) -> None:
        await self.page.close()
