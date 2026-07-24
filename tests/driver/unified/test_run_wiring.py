"""Tests for the four feature-wiring gaps closed in ``ScrapeRun``.

Covers run-level lifecycle events + callbacks, ssl/proxy threading into the
default ``HttpxTransport`` (and confirmation that FOLLOW_REDIRECTS is honored),
cookie load/save best-effort wiring against transports that support it (and
no-op against those that don't), and that ``aclose`` tolerates closing an
injected transport.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import RequestFailedHalt
from jkent.data_types import Response
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from tests.driver.unified.test_run import (
    SpyTransport,
    TrivialScraper,
    _make_run,
)

if TYPE_CHECKING:
    from pathlib import Path


# --- Gap 1: run-level lifecycle events + callbacks -----------------------


async def test_run_start_complete_callbacks_fire(tmp_path: Path) -> None:
    starts: list[str] = []
    completes: list[tuple[str, str, Exception | None]] = []

    async def on_run_start(name: str) -> None:
        starts.append(name)

    async def on_run_complete(
        name: str, status: str, error: Exception | None
    ) -> None:
        completes.append((name, status, error))

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        rate_limited=False,
        resume=False,
        on_run_start=on_run_start,
        on_run_complete=on_run_complete,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await run.run()
    finally:
        await run.aclose()

    assert starts == ["TrivialScraper"]
    assert completes == [("TrivialScraper", "completed", None)]


class _HaltTransport(SpyTransport):
    """A SpyTransport whose resolve raises a run-halting failure."""

    async def resolve(self, handle, queued, await_conditions=()) -> Any:
        raise RequestFailedHalt("halt")


async def test_run_complete_fires_with_error_status_on_exception(
    tmp_path: Path,
) -> None:
    completes: list[tuple[str, str, Exception | None]] = []

    async def on_run_complete(
        name: str, status: str, error: Exception | None
    ) -> None:
        completes.append((name, status, error))

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=_HaltTransport(),
        rate_limited=False,
        resume=False,
        on_run_complete=on_run_complete,
    )
    await run.open(setup_signal_handlers=False)
    # Seed one pending row so a worker resolves it and the transport halts;
    # RequestFailedHalt propagates worker -> _drain_workers -> out of run().
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, queue_counter, request_type, method,
                    url, continuation, current_location)
                VALUES ('pending', 5, 1, 'navigating', 'GET',
                    'http://127.0.0.1/x', 'parse', '')
                """
            )
        )
        await session.commit()

    try:
        with pytest.raises(RequestFailedHalt):
            await run.run()
        # The finally path still fired on_run_complete with status="error".
        assert len(completes) == 1
        name, status, error = completes[0]
        assert name == "TrivialScraper"
        assert status == "error"
        assert isinstance(error, RequestFailedHalt)
    finally:
        await run.aclose()


async def test_run_started_completed_progress_events(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_progress(event_type: str, data: dict[str, Any]) -> None:
        events.append((event_type, data))

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        rate_limited=False,
        resume=False,
        on_progress=on_progress,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await run.run()
    finally:
        await run.aclose()

    by_type = dict(events)
    assert by_type["run_started"] == {"scraper_name": "TrivialScraper"}
    assert by_type["run_completed"] == {
        "scraper_name": "TrivialScraper",
        "status": "completed",
        "error": None,
    }


# --- Gap 2: ssl/proxy threading + FOLLOW_REDIRECTS -----------------------


class _SslScraper(TrivialScraper):
    """A trivial scraper exposing a non-None SSL context."""

    _ctx = ssl.create_default_context()

    @classmethod
    def get_ssl_context(cls) -> ssl.SSLContext:
        return cls._ctx


async def test_default_transport_carries_ssl_and_proxy(tmp_path: Path) -> None:
    scraper = _SslScraper()
    run = ScrapeRun(
        scraper,
        tmp_path / "run.db",
        rate_limited=False,
        resume=False,
        proxy="http://proxy.example:3128",
    )
    await run.open(setup_signal_handlers=False)
    try:
        transport = run.transport
        assert isinstance(transport, HttpxTransport)
        assert transport._ssl_context is scraper.get_ssl_context()
        assert transport._proxy == "http://proxy.example:3128"
        # The opened client was built with them.
        assert transport._client is not None
        # FOLLOW_REDIRECTS derives from the scraper's requirements.
        assert transport._follow_redirects is False
    finally:
        await run.aclose()


# --- Gap 3: cookie load/save best-effort ---------------------------------


class _CookieSpyTransport(SpyTransport):
    """A SpyTransport that records cookie import/export round-trips."""

    def __init__(self, *, preset: str | None = None) -> None:
        super().__init__()
        self.imported: str | None = None
        self.to_export: str | None = preset

    async def import_cookies(self, cookies_json: str) -> None:
        self.imported = cookies_json

    async def export_cookies(self) -> str | None:
        return self.to_export


async def test_cookies_round_trip_through_db(tmp_path: Path) -> None:
    db_path = tmp_path / "run.db"

    # First run exports cookies on close; they land in the DB.
    saver = _CookieSpyTransport(preset='[{"name": "sid", "value": "abc"}]')
    run1 = _make_run(db_path, transport=saver)
    await run1.open(setup_signal_handlers=False)
    assert saver.imported is None  # nothing saved yet to import
    await run1.aclose()

    # Second run imports the persisted cookies on open.
    loader = _CookieSpyTransport()
    run2 = _make_run(db_path, transport=loader)
    await run2.open(setup_signal_handlers=False)
    try:
        assert loader.imported == '[{"name": "sid", "value": "abc"}]'
    finally:
        await run2.aclose()


async def test_http_transport_cookies_are_noop(tmp_path: Path) -> None:
    # A plain SpyTransport (no export/import attrs) is simply skipped:
    # open/close complete without error and nothing is persisted.
    transport = SpyTransport()
    assert not hasattr(transport, "export_cookies")
    assert not hasattr(transport, "import_cookies")
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open(setup_signal_handlers=False)
    await run.aclose()
    assert await run._db.get_browser_cookies() is None  # type: ignore[union-attr]


async def test_cookie_load_error_does_not_abort_open(tmp_path: Path) -> None:
    class _BadImport(_CookieSpyTransport):
        async def import_cookies(self, cookies_json: str) -> None:
            raise RuntimeError("boom")

    db_path = tmp_path / "run.db"
    # Seed cookies so import is attempted.
    seeder = _CookieSpyTransport(preset='[{"name": "x", "value": "1"}]')
    seed_run = _make_run(db_path, transport=seeder)
    await seed_run.open(setup_signal_handlers=False)
    await seed_run.aclose()

    run = _make_run(db_path, transport=_BadImport())
    await run.open(setup_signal_handlers=False)  # must not raise
    await run.aclose()


# --- Gap 4: aclose closes an injected transport --------------------------


async def test_aclose_closes_injected_transport(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open(setup_signal_handlers=False)
    assert transport.closed is False
    # aclose closes even an injected transport, exactly once, no error.
    await run.aclose()
    assert transport.closed is True


# --- Worker teardown on failure ------------------------------------------


class _BoomOrHangTransport(SpyTransport):
    """resolve halts on a /boom URL and hangs (cancellable) on anything else."""

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        url = queued.request.request.url
        if url.endswith("/boom"):
            raise RequestFailedHalt("boom")
        # A sibling request that is still in flight when the run halts; it must
        # be cancelled, not left running against a torn-down transport/DB.
        await asyncio.sleep(30)
        raise AssertionError("the hanging request should have been cancelled")


async def test_worker_failure_tears_down_in_flight_siblings(
    tmp_path: Path,
) -> None:
    # Two workers: one resolves /boom and raises a halting failure while the
    # other is mid-resolve. The halt propagates out of run(), and its finally
    # must cancel the surviving worker so none outlives the run.
    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=_BoomOrHangTransport(),
        rate_limited=False,
        resume=False,
        num_workers=2,
    )
    await run.open(setup_signal_handlers=False)
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, queue_counter, request_type, method,
                    url, continuation, current_location)
                VALUES
                    ('pending', 9, 1, 'navigating', 'GET',
                        'http://127.0.0.1/boom', 'parse', ''),
                    ('pending', 5, 2, 'navigating', 'GET',
                        'http://127.0.0.1/slow', 'parse', '')
                """
            )
        )
        await session.commit()

    try:
        with pytest.raises(RequestFailedHalt):
            await run.run()
        # No worker survives the run — the in-flight sibling was cancelled.
        assert run.active_worker_count == 0
    finally:
        await run.aclose()
