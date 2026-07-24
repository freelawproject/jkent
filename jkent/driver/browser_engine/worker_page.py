"""WorkerPage - a Playwright page bound to a single worker.

Used by the unified driver's Playwright transport. Depends only on the shared
``database_engine.compression`` helper plus Playwright.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from jkent.driver.database_engine.compression import compress
from jkent.driver.unified_driver.transport import WorkerHandle

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)

# Upper bound on how long ``drain_captures`` waits for in-flight response-body
# reads to finish before it gives up on the stragglers. Capture is normally
# sub-millisecond once the response has arrived; this only guards against a
# body() read that never resolves.
_DRAIN_TIMEOUT_S = 10.0


class WorkerPage(WorkerHandle):
    """A Playwright page bound to a single worker, reused across requests.

    Encapsulates per-request state (incidental network requests) so that
    concurrent workers don't corrupt each other's data.

    Response bodies are read asynchronously off the Playwright ``response``
    event. Playwright fires listeners as detached tasks and never awaits them,
    so a naive listener leaves a window where an incidental row is present but
    its ``content_compressed`` is still ``None`` at persist time. Each capture
    is tracked as a task in :attr:`_pending_captures`; :meth:`drain_captures`
    awaits them so a caller that promotes an incidental's body (see the
    transport's ``incidental=`` request handling) is guaranteed the body is
    actually there.
    """

    def __init__(self, page: Page, excluded_resource_types: set[str]):
        self.page = page
        self.incidental_requests: list[dict[str, Any]] = []
        self._excluded_resource_types = excluded_resource_types
        self._pending_captures: set[asyncio.Task[None]] = set()
        self._register_network_listeners()

    def _register_network_listeners(self) -> None:
        """Register network request/response listeners for incidental tracking.

        Listeners are synchronous: ``on_request`` builds the incidental row
        inline, and ``on_response`` spawns a tracked task for the async
        body read so :meth:`drain_captures` can await it deterministically.
        """

        incidentals = self.incidental_requests

        def on_request(request: Any) -> None:
            # post_data_buffer is the raw request body (form/JSON/GraphQL);
            # None for GETs. Captured so incidental matching can disambiguate
            # multiple requests to the same URL by body (e.g. GraphQL ops).
            try:
                post_body = request.post_data_buffer
            except Exception:
                post_body = None
            incidental = {
                "resource_type": request.resource_type,
                "method": request.method,
                "url": request.url,
                "headers_json": json.dumps(dict(request.headers)),
                "body": post_body,
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

        def on_response(response: Any) -> None:
            task = asyncio.ensure_future(self._capture_response(response))
            self._pending_captures.add(task)
            task.add_done_callback(self._pending_captures.discard)

        self.page.on("request", on_request)
        self.page.on("response", on_response)

    async def _capture_response(self, response: Any) -> None:
        """Fill in status/headers/body for the matching incidental row."""
        request = response.request
        for incidental in self.incidental_requests:
            if (
                incidental["url"] == request.url
                and incidental["completed_at_ns"] is None
            ):
                incidental["status_code"] = response.status
                incidental["response_headers_json"] = json.dumps(
                    dict(response.headers)
                )
                incidental["completed_at_ns"] = time.time_ns()
                # Playwright exposes no HTTP-cache flag; service-worker
                # delivery is the only "served without hitting origin"
                # signal available, so that's what from_cache records.
                # A true disk/memory cache hit is NOT distinguished here.
                incidental["from_cache"] = response.from_service_worker

                if (
                    incidental["resource_type"]
                    not in self._excluded_resource_types
                ):
                    try:
                        content = await response.body()
                        content_compressed = compress(content)
                        incidental["content_compressed"] = content_compressed
                        incidental["content_size_original"] = len(content)
                        incidental["content_size_compressed"] = len(
                            content_compressed
                        )
                    except Exception as e:
                        logger.debug(
                            f"Failed to capture content for {request.url}: {e}"
                        )
                break

    async def drain_captures(self, timeout: float = _DRAIN_TIMEOUT_S) -> None:
        """Await outstanding response-body captures before a snapshot/persist.

        Makes incidental capture deterministic: after this returns, every
        response event seen so far has finished writing its body into the
        incidental row (or been abandoned on timeout). Stragglers still
        pending after ``timeout`` are cancelled rather than blocking forever.
        """
        pending = [t for t in self._pending_captures if not t.done()]
        if not pending:
            return
        _, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in still_pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    def clear_request_state(self) -> None:
        """Reset per-request state between navigations."""
        self.incidental_requests.clear()
        # Cancel any capture tasks that outlived their request so their late
        # writes can't land on the next request's (cleared) incidental list.
        for task in self._pending_captures:
            task.cancel()
        self._pending_captures.clear()

    async def reset_for_reuse(self) -> None:
        """Lightweight cleanup between requests."""
        # Clear before navigation to discard stale events from the prior
        # page's in-flight sub-resources that may land during the goto.
        self.clear_request_state()
        # The prior request may have left a navigation in flight (a timed-out
        # or abandoned goto keeps loading in the browser after Playwright
        # stops waiting); it would race the about:blank goto below
        # ("interrupted by another navigation" / NS_BINDING_ABORTED).
        # window.stop() aborts pending fetches and uncommitted navigations —
        # best-effort, since the execution context dies if that navigation
        # commits mid-call.
        with contextlib.suppress(Exception):
            await self.page.evaluate("window.stop()")
        await self.page.goto("about:blank", wait_until="commit")
        # Clear again to remove any events fired by the about:blank
        # navigation itself.
        self.clear_request_state()

    async def close(self) -> None:
        await self.page.close()
