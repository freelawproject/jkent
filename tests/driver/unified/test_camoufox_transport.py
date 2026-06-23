"""Tests for ``CamoufoxTransport`` (B6).

``CamoufoxTransport`` is a one-method subclass of ``PlaywrightTransport``: it
forces the camoufox engine. Everything else (lifecycle, resolve, crash
recovery, archive) is inherited and already proven for the parent in B1–B5, so
this module covers:

  - the engine-selection delta (browser-free): the subclass always builds a
    camoufox engine, where the parent defaults to playwright;
  - that it remains structurally a ``Transport`` and a ``PlaywrightTransport``;
  - that the inherited crash predicate recognizes camoufox's Firefox page-error
    crash (a ``Connection closed`` channel error);
  - the full ``TransportConformance`` over a REAL camoufox, gated on a launch
    probe so it skips cleanly where the camoufox/Firefox binary is absent.

Fidelity strategy mirrors B5: a real headless engine, not a stub. The archive
``resolve_archive`` path is inherited byte-for-byte from ``PlaywrightTransport``
(exercised by B4 + B5), so the two archive conformance methods are skipped here
rather than re-driving a camoufox download.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.browser_engine.engines import (
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.unified_driver import CamoufoxTransport, QueuedRequest
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle
from jkent.driver.unified_driver.transport import Transport, WorkerHandle
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.driver.unified.conftest import (
    single_page_app,
    start_app,
    status_app,
)
from tests.driver.unified.test_async_lifecycle_conformance import (
    AsyncLifecycleConformance,
)
from tests.driver.unified.test_playwright_transport import (
    _insert_request_row,
    _Scraper,
    _sql_manager,
)
from tests.driver.unified.test_transport_conformance import (
    ClassificationConformance,
    TransportConformance,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from jkent.data_types import BaseScraper, Response

# Camoufox can only launch one Firefox at a time, so co-locate every test in
# this module on a single xdist worker (honored under --dist loadgroup) — they
# never run concurrently with each other.
pytestmark = pytest.mark.xdist_group("camoufox")


# --- Browser-free unit checks --------------------------------------------


def test_build_engine_is_camoufox() -> None:
    """The subclass always builds a camoufox engine (no CFCAP_HANDLER needed)."""
    engine = CamoufoxTransport(_Scraper())._build_engine()
    assert isinstance(engine, CamoufoxEngine)


def test_parent_defaults_to_playwright_engine() -> None:
    """Contrast: a plain ``PlaywrightTransport`` builds a playwright engine."""
    engine = PlaywrightTransport(_Scraper())._build_engine()
    assert isinstance(engine, PlaywrightEngine)


def test_is_a_playwright_transport() -> None:
    """It inherits the whole Playwright transport surface."""
    transport = CamoufoxTransport(_Scraper())
    assert isinstance(transport, PlaywrightTransport)
    for method in (
        "open",
        "aclose",
        "acquire",
        "release",
        "resolve",
        "resolve_archive",
        "finish_archiving",
    ):
        assert callable(getattr(transport, method))


def test_should_restart_recognizes_camoufox_crash() -> None:
    """The inherited predicate flags camoufox's ``Connection closed`` crash."""
    transport = CamoufoxTransport(_Scraper())
    assert transport.should_restart(Exception("Connection closed")) is True
    assert transport.should_restart(ValueError("unrelated")) is False


# --- Real-camoufox conformance (skipped cleanly without the binary) -------


@pytest.fixture(scope="session")
def has_camoufox() -> bool:
    """Whether a camoufox engine can actually launch in this environment."""

    async def _launches() -> bool:
        transport = CamoufoxTransport(_Scraper(), headless=True)
        try:
            await transport.open()
            await transport.aclose()
        except Exception:
            return False
        return True

    return asyncio.run(_launches())


class TestCamoufoxTransportLifecycle(AsyncLifecycleConformance):
    """``CamoufoxTransport`` honors the open -> use -> aclose lifecycle."""

    @pytest.fixture
    async def subject(self, has_camoufox: bool):  # type: ignore[no-untyped-def]
        # Yield + aclose in teardown: the base suite's
        # ``test_open_awaits_to_none`` opens without closing, which for camoufox
        # would leave a Firefox holding the single-instance profile lock and
        # deadlock the next camoufox test. The teardown guarantees cleanup.
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        transport = CamoufoxTransport(_Scraper(), headless=True)
        try:
            yield transport
        finally:
            await transport.aclose()

    def live_resources(self, subject: AsyncLifecycle) -> int:
        """Engine, engine context-manager, context, and page handles."""
        assert isinstance(subject, CamoufoxTransport)
        return (
            (subject._engine is not None)
            + (subject._engine_cm is not None)
            + (subject._context is not None)
            + len(subject._handles)
        )


class TestCamoufoxTransportConformance(TransportConformance):
    """Run the shared ``Transport`` contract against a real camoufox engine."""

    @pytest.fixture
    async def subject(  # type: ignore[override]
        self,
        has_camoufox: bool,
        memory_session_factory: async_sessionmaker,
    ) -> AsyncIterator[CamoufoxTransport]:
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        html = "<html><body><p>camoufox conformance</p></body></html>"
        server = await start_app(single_page_app(html))
        self._url = f"{server.base_url}/page"  # type: ignore
        self._request_id = await _insert_request_row(  # type: ignore
            memory_session_factory, self._url, qc=1
        )
        transport = CamoufoxTransport(
            _Scraper(),
            headless=True,
            db=_sql_manager(memory_session_factory),
        )
        try:
            yield transport
        finally:
            # The conformance tests drive open()/aclose() themselves, but a
            # failure between them would skip their aclose() and leak the live
            # browser, deadlocking the profile lock for the next camoufox test.
            # aclose() is idempotent, so guarantee teardown here.
            await transport.aclose()
            await server.runner.cleanup()

    def make_queued(self, *, request_id: int | None = None) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=self._url,  # type: ignore
                ),
                continuation="parse",
            ),
            request_id=request_id
            if request_id is not None
            else self._request_id,  # type: ignore
        )

    async def test_resolve_archive_metadata_and_chunks(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """Archive path is inherited unchanged — covered by B4 + B5."""
        pytest.skip("resolve_archive identical to PlaywrightTransport (B4/B5)")

    async def test_finish_archiving_is_no_throw(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """Archive path is inherited unchanged — covered by B4 + B5."""
        pytest.skip(
            "finish_archiving identical to PlaywrightTransport (B4/B5)"
        )


class TestCamoufoxTransportClassification(ClassificationConformance):
    """Status classification over a real camoufox/firefox navigation.

    The classification logic is inherited unchanged from
    ``PlaywrightTransport``; what this binding pins is the engine half —
    firefox surfacing non-200 navigation statuses into ``resolve`` the same
    way chromium does.
    """

    # pyre-ignore[13]: bound by the autouse ``_bind_env`` fixture before
    # every test, which pyre can't see.
    _sf: async_sessionmaker

    @pytest.fixture(autouse=True)
    async def _bind_env(
        self,
        has_camoufox: bool,
        memory_session_factory: async_sessionmaker,
    ) -> None:
        if not has_camoufox:
            pytest.skip("no launchable camoufox engine in this environment")
        self._sf = memory_session_factory

    async def classify(
        self,
        scraper: type[BaseScraper[Any]],
        code: int,
        body: bytes,
        *,
        speculative: bool = False,
    ) -> Response:
        server = await start_app(status_app(code, body))
        request_id = await _insert_request_row(
            self._sf, f"{server.base_url}/page", qc=1
        )
        transport = CamoufoxTransport(
            scraper(), headless=True, db=_sql_manager(self._sf)
        )
        try:
            await transport.open()
            handle = await transport.acquire(0)
            request = Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=f"{server.base_url}/page"
                ),
                continuation="parse",
                is_speculative=speculative,
            )
            return await transport.resolve(
                handle, QueuedRequest(request=request, request_id=request_id)
            )
        finally:
            await transport.aclose()
            await server.runner.cleanup()
