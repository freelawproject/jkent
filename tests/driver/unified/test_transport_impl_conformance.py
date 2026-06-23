"""Bind the reusable conformance suites to the real ``HttpxTransport``.

The reference fake in ``test_transport_conformance`` proves the suite is
self-consistent; this file proves the suite passes against the actual
``HttpxTransport`` — the Phase 0 "wire the real impls into conformance" step.
(``ReplayTransport`` conformance moved to jent with the replay driver.)

The transport holds no per-worker state, so its ``WorkerHandle`` is a no-op;
the suite's stability/freshness checks pin the get-or-create contract
(``acquire`` is stable per ``worker_id`` until ``release``) that it now honors.

To make ``resolve``/``resolve_archive`` actually resolve under the conformance
methods, the subject is paired with a coordinated backing: a live aiohttp
server answering ``200`` + a non-empty body for any path; ``make_queued``
points at it.

``HttpxTransport`` is also run through ``AsyncLifecycleConformance``. Its
resource-leak test is written against the reference fake, so the subclass
overrides it with a transport-specific "backing is released after aclose" check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from aiohttp import web

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import (
    HttpxTransport,
    QueuedRequest,
)
from tests.driver.unified.conftest import start_app, status_app
from tests.driver.unified.test_async_lifecycle_conformance import (
    AsyncLifecycleConformance,
)
from tests.driver.unified.test_transport_conformance import (
    ClassificationConformance,
    TransportConformance,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from jkent.data_types import BaseScraper, Response
    from jkent.driver.unified_driver import AsyncLifecycle


# --- HttpxTransport: live server ------------------------------------------


def _ok_app() -> web.Application:
    """Answer every request with 200 + a non-empty body (response & archive)."""

    async def handler(_request: web.Request) -> web.Response:
        return web.Response(status=200, body=b"<html>conformance</html>")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app


class TestHttpxTransportConformance(TransportConformance):
    """The conformance suite against a real ``HttpxTransport`` + server."""

    @pytest.fixture
    async def subject(self) -> AsyncIterator[HttpxTransport]:
        server = await start_app(_ok_app())
        self._base_url = server.base_url  # type: ignore
        transport = HttpxTransport()
        try:
            yield transport
        finally:
            # aclose is idempotent and safe on a never-opened transport, so
            # this releases the httpx client even if a test fails before its
            # own aclose() runs.
            await transport.aclose()
            await server.runner.cleanup()

    def make_queued(self, *, request_id: int = 1) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=f"{self._base_url}/r",  # type: ignore
                ),
                continuation="parse",
            ),
            request_id=request_id,
        )


# --- AsyncLifecycle conformance --------------------------------------------


class TestHttpxTransportLifecycle(AsyncLifecycleConformance):
    """``HttpxTransport`` honors the open -> use -> aclose lifecycle."""

    @pytest.fixture
    def subject(self) -> AsyncLifecycle:
        return HttpxTransport()

    def live_resources(self, subject: AsyncLifecycle) -> int:
        """The httpx client acquired in ``open`` and dropped by ``aclose``."""
        assert isinstance(subject, HttpxTransport)
        return 0 if subject._client is None else 1


# --- Classification conformance ---------------------------------------------


class TestHttpxTransportClassification(ClassificationConformance):
    """The classification contract against a real ``HttpxTransport`` + server.

    Each ``classify`` stands up a one-shot status server and resolves a single
    GET through a transport built for the given scraper. The deeper generative
    sweep (arbitrary override maps, the full code matrix) stays in
    ``test_httpx_classification``.
    """

    async def classify(
        self,
        scraper: type[BaseScraper[Any]],
        code: int,
        body: bytes,
        *,
        speculative: bool = False,
    ) -> Response:
        server = await start_app(status_app(code, body))
        transport = HttpxTransport(scraper=scraper)
        await transport.open()
        handle = await transport.acquire(0)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{server.base_url}/r"
            ),
            continuation="parse",
            is_speculative=speculative,
        )
        try:
            return await transport.resolve(
                handle, QueuedRequest(request=request, request_id=1)
            )
        finally:
            await transport.aclose()
            await server.runner.cleanup()
