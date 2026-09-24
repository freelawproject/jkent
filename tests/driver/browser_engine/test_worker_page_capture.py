"""``WorkerPage``'s incidental capture, driven through its Playwright events.

A fake page records the listeners ``WorkerPage`` registers, and the tests
fire ``request`` / ``response`` / ``requestfailed`` at them the way
Playwright does — so no browser is needed.

- A response is matched to the capture of *its* request, not to the first
  open capture with the same URL: two concurrent calls to one GraphQL
  endpoint must not swap bodies.
- A request that fails records Playwright's failure text, so a promotion
  can tell an aborted fetch from a response.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jkent.driver.browser_engine.worker_page import WorkerPage


class _Page:
    def __init__(self) -> None:
        self.listeners: dict[str, Callable[[Any], None]] = {}

    def on(self, event: str, listener: Callable[[Any], None]) -> None:
        self.listeners[event] = listener


@dataclass(eq=False)
class _Request:
    url: str
    post_data_buffer: bytes | None = None
    resource_type: str = "xhr"
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    failure: str | None = None


@dataclass
class _Response:
    request: _Request
    payload: bytes
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    from_service_worker: bool = False

    async def body(self) -> bytes:
        return self.payload


def _worker_page() -> tuple[WorkerPage, _Page]:
    page = _Page()
    return WorkerPage(page, frozenset()), page  # type: ignore[arg-type]


async def test_concurrent_calls_to_one_url_keep_their_own_bodies() -> None:
    worker_page, page = _worker_page()
    first = _Request("https://api.example/graphql", b'{"op":"cases"}')
    second = _Request("https://api.example/graphql", b'{"op":"parties"}')
    page.listeners["request"](first)
    page.listeners["request"](second)

    # The second call answers first.
    page.listeners["response"](_Response(second, b"parties"))
    page.listeners["response"](_Response(first, b"cases"))
    await worker_page.drain_captures()

    by_body = {c.body: c for c in worker_page.incidental_requests}
    assert by_body[b'{"op":"cases"}'].content_size_original == len(b"cases")
    assert by_body[b'{"op":"parties"}'].content_size_original == len(
        b"parties"
    )


async def test_a_failed_request_records_why() -> None:
    worker_page, page = _worker_page()
    aborted = _Request(
        "https://api.example/search", failure="net::ERR_ABORTED"
    )
    page.listeners["request"](aborted)

    page.listeners["requestfailed"](aborted)

    (capture,) = worker_page.incidental_requests
    assert capture.failure_reason == "net::ERR_ABORTED"
    assert capture.status_code is None
    assert capture.completed_at_ns is not None
