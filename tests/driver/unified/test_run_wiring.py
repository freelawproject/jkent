"""Tests for ``ScrapeRun``'s feature wiring.

Covers run-level lifecycle events + callbacks, ssl/proxy threading into the
default ``HttpxTransport`` (and confirmation that FOLLOW_REDIRECTS is honored),
cookie load/save best-effort wiring against transports that support it (and
no-op against those that don't), that ``aclose`` tolerates closing an
injected transport, and that a failing worker tears down in-flight siblings.
"""

from __future__ import annotations

import asyncio
import ssl
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from typing_extensions import override

from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    RequestFailedHalt,
)
from jkent.data_types import HttpMethod, Response
from jkent.driver.database_engine.enums import (
    RequestStatus,
    RequestType,
    RunStatus,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks
from tests.driver.unified.test_run import (
    SpyTransport,
    TrivialScraper,
    _make_run,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jkent.driver.handles import NoopHandle
    from jkent.driver.unified_driver.transport import (
        AwaitCondition,
        QueuedRequest,
    )


# --- Run-level lifecycle events + callbacks ------------------------------


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
        config=RunConfig(rate_limited=False),
        hooks=RunHooks(
            on_run_start=on_run_start, on_run_complete=on_run_complete
        ),
    )
    await run.open()
    try:
        await run.run()
    finally:
        await run.aclose()

    assert starts == ["TrivialScraper"]
    assert completes == [("TrivialScraper", "completed", None)]


class _HaltTransport(SpyTransport):
    """A SpyTransport whose resolve raises a run-halting failure."""

    @override
    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        raise RequestFailedHalt("halt")


async def _seed_one_pending(run: ScrapeRun) -> None:
    """Insert one pending navigating row for a worker to pick up."""
    sf = run._db.session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, request_type, method,
                    url, step, current_location)
                VALUES (:status, 5, :rtype, :method,
                    'http://127.0.0.1/x', 'parse', '')
                """
            ),
            {
                "status": RequestStatus.PENDING.code,
                "rtype": RequestType.NAVIGATING.code,
                "method": HttpMethod.GET.code,
            },
        )
        await session.commit()


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
        config=RunConfig(rate_limited=False),
        hooks=RunHooks(on_run_complete=on_run_complete),
    )
    await run.open()
    # Seed one pending row so a worker resolves it and the transport halts;
    # RequestFailedHalt propagates worker -> _drain_workers -> out of run().
    await _seed_one_pending(run)

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


async def test_an_error_after_a_stop_finalizes_as_error(
    tmp_path: Path,
) -> None:
    """A run that raises is ERROR even if the stop event was already set
    (the error budget sets it on its way out): INTERRUPTED would tell the
    operator a stopped-on-purpose run, with the error stored against it."""
    completes: list[tuple[str, str, Exception | None]] = []

    async def on_run_complete(
        name: str, status: str, error: Exception | None
    ) -> None:
        completes.append((name, status, error))

    class _StopThenHalt(SpyTransport):
        @override
        async def resolve(
            self,
            handle: NoopHandle,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            run.stop()
            raise RequestFailedHalt("halt")

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=_StopThenHalt(),
        config=RunConfig(rate_limited=False),
        hooks=RunHooks(on_run_complete=on_run_complete),
    )
    await run.open()
    await _seed_one_pending(run)
    try:
        with pytest.raises(RequestFailedHalt):
            await run.run()
        metadata = await run.db.get_run_metadata()
    finally:
        await run.aclose()

    assert [status for _, status, _ in completes] == ["error"]
    assert metadata is not None
    assert metadata.status is RunStatus.ERROR
    assert metadata.error_message == "halt"


async def test_an_error_budget_stop_finalizes_as_error(tmp_path: Path) -> None:
    """A budget stop is ERROR, naming the budget; Ctrl-C stays INTERRUPTED."""
    completes: list[tuple[str, str, Exception | None]] = []

    async def on_run_complete(
        name: str, status: str, error: Exception | None
    ) -> None:
        completes.append((name, status, error))

    class _Gone(SpyTransport):
        @override
        async def resolve(
            self,
            handle: NoopHandle,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            raise PersistentHTTPResponseException(404, "http://127.0.0.1/x")

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=_Gone(),
        config=RunConfig(rate_limited=False, max_persistent_errors=1),
        hooks=RunHooks(on_run_complete=on_run_complete),
    )
    await run.open()
    await _seed_one_pending(run)
    try:
        await run.run()
        metadata = await run.db.get_run_metadata()
    finally:
        await run.aclose()

    assert completes == [("TrivialScraper", "error", None)]
    assert metadata is not None
    assert metadata.status is RunStatus.ERROR
    assert metadata.error_message == (
        "Error budget exhausted: 1 persistent error(s) reached "
        "max_persistent_errors=1"
    )


async def test_run_started_completed_progress_events(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def on_progress(event_type: str, data: dict[str, Any]) -> None:
        events.append((event_type, data))

    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        config=RunConfig(rate_limited=False),
        hooks=RunHooks(on_progress=on_progress),
    )
    await run.open()
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


# --- ssl/proxy threading + FOLLOW_REDIRECTS ------------------------------


class _SslScraper(TrivialScraper):
    """A trivial scraper exposing a non-None SSL context."""

    _ctx = ssl.create_default_context()

    @classmethod
    @override
    def get_ssl_context(cls) -> ssl.SSLContext:
        return cls._ctx


async def test_default_transport_carries_the_scraper_ssl_context(
    tmp_path: Path,
) -> None:
    # The run's default transport is a plain HttpxTransport over the scraper.
    # Proxy/timeout are not run knobs any more — RunBootstrapper.build_transport
    # owns them (see test_bootstrap) — so there is nothing else to thread.
    scraper = _SslScraper()
    run = ScrapeRun(
        scraper,
        tmp_path / "run.db",
        config=RunConfig(rate_limited=False),
    )
    await run.open()
    try:
        transport = run.transport
        assert isinstance(transport, HttpxTransport)
        assert transport._ssl_context is scraper.get_ssl_context()
        assert transport._proxy is None
        # The opened client was built with it.
        assert transport._client is not None
        # FOLLOW_REDIRECTS derives from the scraper's requirements.
        assert transport._follow_redirects is False
    finally:
        await run.aclose()


# --- Cookie load/save best-effort ----------------------------------------


class _CookieSpyTransport(SpyTransport):
    """A SpyTransport that records cookie import/export round-trips."""

    def __init__(self, *, preset: str | None = None) -> None:
        super().__init__()
        self.imported: str | None = None
        self.to_export: str | None = preset

    @override
    async def import_cookies(self, cookies_json: str) -> None:
        self.imported = cookies_json

    @override
    async def export_cookies(self) -> str | None:
        return self.to_export


async def test_cookies_round_trip_through_db(tmp_path: Path) -> None:
    db_path = tmp_path / "run.db"

    # First run exports cookies on close; they land in the DB.
    saver = _CookieSpyTransport(preset='[{"name": "sid", "value": "abc"}]')
    run1 = _make_run(db_path, transport=saver)
    await run1.open()
    assert saver.imported is None  # nothing saved yet to import
    await run1.aclose()

    # Second run imports the persisted cookies on open.
    loader = _CookieSpyTransport()
    run2 = _make_run(db_path, transport=loader)
    await run2.open()
    try:
        assert loader.imported == '[{"name": "sid", "value": "abc"}]'
    finally:
        await run2.aclose()


async def test_transport_is_handed_the_run_db_before_it_opens(
    tmp_path: Path,
) -> None:
    """``bind_run_db`` reaches an injected transport, and before ``open``.

    A transport passed to ``ScrapeRun(transport=...)`` is built before the
    run's database exists, so this hook is its only route to one — and it has
    to land before ``open``, since a transport may use the handle while
    bringing itself up.
    """

    class _DbSpyTransport(SpyTransport):
        def __init__(self) -> None:
            super().__init__()
            self.bound: object = None
            self.bound_before_open = False

        @override
        def bind_run_db(self, db: object) -> None:
            self.bound = db

        @override
        async def open(self) -> None:
            self.bound_before_open = self.bound is not None
            await super().open()

    transport = _DbSpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    try:
        assert transport.bound is run.db
        assert transport.bound_before_open is True
    finally:
        await run.aclose()


async def test_http_transport_cookies_are_noop(tmp_path: Path) -> None:
    # A plain SpyTransport inherits the ABC's no-op cookie hooks: open/close
    # complete without error and nothing is persisted.
    transport = SpyTransport()
    assert await transport.export_cookies() is None
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    await run.aclose()
    async with SQLManager.open(tmp_path / "run.db") as db:
        assert await db.get_browser_cookies() is None


async def test_cookie_load_error_does_not_abort_open(tmp_path: Path) -> None:
    class _BadImport(_CookieSpyTransport):
        @override
        async def import_cookies(self, cookies_json: str) -> None:
            raise RuntimeError("boom")

    db_path = tmp_path / "run.db"
    # Seed cookies so import is attempted.
    seeder = _CookieSpyTransport(preset='[{"name": "x", "value": "1"}]')
    seed_run = _make_run(db_path, transport=seeder)
    await seed_run.open()
    await seed_run.aclose()

    run = _make_run(db_path, transport=_BadImport())
    await run.open()  # must not raise
    await run.aclose()


# --- aclose closes an injected transport ---------------------------------


async def test_aclose_closes_injected_transport(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    assert transport.closed is False
    # aclose closes even an injected transport, exactly once, no error.
    await run.aclose()
    assert transport.closed is True


# --- Worker teardown on failure ------------------------------------------


class _BoomOrHangTransport(SpyTransport):
    """resolve halts on a /boom URL and hangs (cancellable) on anything else."""

    @override
    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
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
        config=RunConfig(rate_limited=False, num_workers=2),
    )
    await run.open()
    sf = run._db.session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, request_type, method,
                    url, step, current_location)
                VALUES
                    (:status, 9, :rtype, :method,
                        'http://127.0.0.1/boom', 'parse', ''),
                    (:status, 5, :rtype, :method,
                        'http://127.0.0.1/slow', 'parse', '')
                """
            ),
            {
                "status": RequestStatus.PENDING.code,
                "rtype": RequestType.NAVIGATING.code,
                "method": HttpMethod.GET.code,
            },
        )
        await session.commit()

    try:
        with pytest.raises(RequestFailedHalt):
            await run.run()
        # No worker survives the run — the in-flight sibling was cancelled.
        assert run.active_worker_count == 0
    finally:
        await run.aclose()


def test_run_config_rejects_a_jitter_of_one_or_more() -> None:
    """``retry_jitter`` is a fraction of the delay; 1 or more is refused."""
    with pytest.raises(ValueError, match="retry_jitter"):
        RunConfig(retry_jitter=1.0)
