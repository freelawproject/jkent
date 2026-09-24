"""``RunBootstrapper`` — requirement-driven wiring for unified runs.

Covers transport selection, browser-profile auto-resolution from a fake
``JKENT_HOME``, STRICTLY_SERIAL capping, seed-params validation, and a
full HTTP run end-to-end through the bootstrapper (open → run → resume).
Browser transports are selected but never launched.
"""

from __future__ import annotations

import json
import logging
import signal
import sqlite3
import ssl
import subprocess
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from typing_extensions import override

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.browser_engine.engines import CamoufoxEngine
from jkent.driver.database_engine.enums import RunStatus
from jkent.driver.unified_driver import (
    HttpxTransport,
    PlaywrightTransport,
    RunBootstrapper,
    build_transport,
    resolve_browser_profile,
)
from jkent.driver.unified_driver.bootstrap import install_signal_handlers
from jkent.driver.unified_driver.bootstrap import logger as bootstrap_logger
from jkent.driver.unified_driver.wiring import RunConfig
from tests.driver.unified.conftest import HttpPageScraper
from tests.driver.unified.test_run import SpyTransport

# --- Scrapers --------------------------------------------------------------


def _scraper_with(*reqs: DriverRequirement) -> BaseScraper[dict[str, Any]]:
    class _Scraper(BaseScraper[dict[str, Any]]):
        driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

    return _Scraper()


class _NoRequestScraper(BaseScraper[dict[str, Any]]):
    """A scraper whose entry yields nothing: open() writes run metadata but
    no request rows are ever enqueued."""

    @entry(dict)
    def fetch_page(self, page_id: int) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={"body": response.text})


class _NoArgEntryScraper(BaseScraper[dict[str, Any]]):
    """A scraper with a no-arg entry (auto-seeded without seed_params) that
    yields nothing — so open() succeeds with no requests enqueued."""

    @entry(dict)
    def start(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={"body": response.text})


# --- Transport selection ----------------------------------------------------


class TestBuildTransport:
    def test_http_scraper_gets_httpx_with_ssl_proxy_timeout(self) -> None:
        # Proxy and timeout are transport configuration: they live on the
        # transport the bootstrapper builds, never on the run.
        class _SslScraper(BaseScraper[dict[str, Any]]):
            _ctx = ssl.create_default_context()

            @classmethod
            @override
            def get_ssl_context(cls) -> ssl.SSLContext:
                return cls._ctx

        scraper = _SslScraper()
        transport = build_transport(
            scraper, proxy="http://proxy.example:3128", timeout=12.5
        )
        assert isinstance(transport, HttpxTransport)
        assert transport._ssl_context is scraper.get_ssl_context()
        assert transport._proxy == "http://proxy.example:3128"
        assert transport._timeout == 12.5
        # FOLLOW_REDIRECTS derives from the scraper's requirements.
        assert transport._follow_redirects is False
        assert isinstance(
            build_transport(_scraper_with(DriverRequirement.H11_HEADER_FIXES)),
            HttpxTransport,
        )

    def test_browser_reqs_get_playwright(self) -> None:
        for req in (
            DriverRequirement.JS_EVAL,
            DriverRequirement.FF_ALIKE,
            DriverRequirement.CHROME_ALIKE,
            DriverRequirement.STRICTLY_SERIAL,
        ):
            transport = build_transport(_scraper_with(req))
            assert type(transport) is PlaywrightTransport, req

    def test_camoufox_reqs_get_camoufox_engine(self) -> None:
        # CFCAP and RCAP both demand the stealthy camoufox engine.
        for req in (
            DriverRequirement.CFCAP_HANDLER,
            DriverRequirement.RCAP_HANDLER,
        ):
            transport = build_transport(_scraper_with(req))
            assert type(transport) is PlaywrightTransport, req
            assert isinstance(transport._build_engine(), CamoufoxEngine), req


# --- Profile resolution -----------------------------------------------------


def _write_profile(home: Path, name: str, browser_type: str) -> Path:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": name,
                "browser_type": browser_type,
            }
        )
    )
    return profile_dir


class TestResolveBrowserProfile:
    def test_no_flavor_requirement_no_profile(self, tmp_path: Path) -> None:
        assert (
            resolve_browser_profile(
                _scraper_with(DriverRequirement.JS_EVAL), jkent_home=tmp_path
            )
            is None
        )

    def test_ff_alike_resolves_firefox(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "firefox", "firefox")
        profile = resolve_browser_profile(
            _scraper_with(DriverRequirement.FF_ALIKE), jkent_home=tmp_path
        )
        assert profile is not None
        assert profile.name == "firefox"

    def test_chrome_alike_resolves_chrome(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "chrome", "chromium")
        profile = resolve_browser_profile(
            _scraper_with(DriverRequirement.CHROME_ALIKE), jkent_home=tmp_path
        )
        assert profile is not None
        assert profile.name == "chrome"

    def test_cfcap_wins_over_flavors(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "camoufox", "firefox")
        _write_profile(tmp_path, "firefox", "firefox")
        profile = resolve_browser_profile(
            _scraper_with(
                DriverRequirement.CFCAP_HANDLER, DriverRequirement.FF_ALIKE
            ),
            jkent_home=tmp_path,
        )
        assert profile is not None
        assert profile.name == "camoufox"

    def test_rcap_wins_over_flavors(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "camoufox", "firefox")
        _write_profile(tmp_path, "firefox", "firefox")
        profile = resolve_browser_profile(
            _scraper_with(
                DriverRequirement.RCAP_HANDLER, DriverRequirement.FF_ALIKE
            ),
            jkent_home=tmp_path,
        )
        assert profile is not None
        assert profile.name == "camoufox"

    def test_missing_profile_warns_and_returns_none(
        self, tmp_path: Path
    ) -> None:
        # Unlike the CLI (hard error), unified engines run profile-less.
        assert (
            resolve_browser_profile(
                _scraper_with(DriverRequirement.FF_ALIKE), jkent_home=tmp_path
            )
            is None
        )

    def test_rejected_manifest_propagates(self, tmp_path: Path) -> None:
        """A profile that exists but is rejected stops the run.

        Running profile-less is right when there is no profile; when there
        is one the scraper asked for, launching without it drops the
        fingerprint it exists to carry.
        """
        profile_dir = _write_profile(tmp_path, "chrome", "chromium")
        manifest = profile_dir / "manifest.json"
        raw = json.loads(manifest.read_text())
        raw["protocol_params"] = {"assistantMode": True, "cdpPort": "auto"}
        manifest.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="protocol_params"):
            resolve_browser_profile(
                _scraper_with(DriverRequirement.CHROME_ALIKE),
                jkent_home=tmp_path,
            )


# --- Constructor validation -------------------------------------------------


class TestValidation:
    async def test_seed_params_rejected_on_existing_db(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "run.db"
        async with RunBootstrapper(
            HttpPageScraper(),
            db_path,
            setup_signal_handlers=False,
            config=RunConfig(
                seed_params=[{"fetch_page": {"page_id": 1}}],
                rate_limited=False,
            ),
        ):
            pass

        with pytest.raises(ValueError, match="recreate the run database"):
            async with RunBootstrapper(
                HttpPageScraper(),
                db_path,
                setup_signal_handlers=False,
                config=RunConfig(
                    seed_params=[{"fetch_page": {"page_id": 2}}],
                    rate_limited=False,
                ),
            ):
                pass

    async def test_seed_params_allowed_when_db_has_metadata_no_requests(
        self, tmp_path: Path
    ) -> None:
        # A fresh run that opened (writing run metadata) but enqueued no
        # requests — e.g. it died before seeding — must still be retryable
        # with the same seed_params. The guard gates on request rows, not on
        # metadata existence (which open() writes before any seeding).
        db_path = tmp_path / "run.db"
        async with RunBootstrapper(
            _NoRequestScraper(),
            db_path,
            setup_signal_handlers=False,
            config=RunConfig(
                seed_params=[{"fetch_page": {"page_id": 1}}],
                rate_limited=False,
            ),
        ):
            pass

        # Re-running with the *same* seed_params must not raise: the DB has
        # metadata but zero requests.
        async with RunBootstrapper(
            _NoRequestScraper(),
            db_path,
            setup_signal_handlers=False,
            config=RunConfig(
                seed_params=[{"fetch_page": {"page_id": 1}}],
                rate_limited=False,
            ),
        ):
            pass

    async def test_seed_params_allowed_on_an_empty_file(
        self, tmp_path: Path
    ) -> None:
        # A 0-byte file (a placeholder, say) holds no run, so seeding it is a
        # fresh run — the guard must not try to open it as an existing one.
        db_path = tmp_path / "run.db"
        db_path.touch()
        async with RunBootstrapper(
            HttpPageScraper(),
            db_path,
            setup_signal_handlers=False,
            config=RunConfig(
                seed_params=[{"fetch_page": {"page_id": 1}}],
                rate_limited=False,
            ),
        ):
            pass

    async def test_browser_transport_and_run_share_one_db_handle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The transport's DB handle has to exist before ScrapeRun does, so the
        # bootstrapper pre-inits one and passes it to both. Two managers on the
        # one file would carry two write locks — each serializing only its own
        # writers — and their connections would collide as independent SQLite
        # writers (transport incidentals vs. the run's dequeue).
        class _BrowserScraper(_NoArgEntryScraper):
            driver_requirements: ClassVar[list[DriverRequirement]] = [
                DriverRequirement.JS_EVAL
            ]

        captured: dict[str, Any] = {}

        def _fake_build_transport(
            _scraper: Any, **kwargs: Any
        ) -> SpyTransport:
            captured["db"] = kwargs["db"]
            return SpyTransport()

        monkeypatch.setattr(
            "jkent.driver.unified_driver.bootstrap.build_transport",
            _fake_build_transport,
        )
        bootstrapper = RunBootstrapper(
            _BrowserScraper(),
            tmp_path / "run.db",
            setup_signal_handlers=False,
            config=RunConfig(rate_limited=False),
        )
        run = await bootstrapper.bootstrap()
        try:
            assert run.db is captured["db"]
            assert run.db.lock is captured["db"].lock
        finally:
            await bootstrapper.aclose()
        # The bootstrapper owns the shared engine, so it disposed it.
        assert bootstrapper._db_engine is None

    async def test_a_failure_before_the_run_opens_disposes_the_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pre-initialized engine is released when setup fails."""

        class _BrowserScraper(_NoArgEntryScraper):
            driver_requirements: ClassVar[list[DriverRequirement]] = [
                DriverRequirement.JS_EVAL
            ]

        def _spy_transport(
            _scraper: BaseScraper[Any], **_kwargs: Any
        ) -> SpyTransport:
            return SpyTransport()

        monkeypatch.setattr(
            "jkent.driver.unified_driver.bootstrap.build_transport",
            _spy_transport,
        )
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        bootstrapper = RunBootstrapper(
            _BrowserScraper(),
            tmp_path / "run.db",
            storage_dir=blocker / "files",
            setup_signal_handlers=False,
            config=RunConfig(rate_limited=False),
        )
        with pytest.raises(OSError):
            await bootstrapper.bootstrap()
        assert bootstrapper._db_engine is None

    async def test_http_scraper_builds_its_own_db_handle(
        self, tmp_path: Path
    ) -> None:
        # No browser, no pre-init: ScrapeRun opens (and disposes) the only
        # handle on the file.
        bootstrapper = RunBootstrapper(
            _NoArgEntryScraper(),
            tmp_path / "run.db",
            setup_signal_handlers=False,
            config=RunConfig(rate_limited=False),
        )
        run = await bootstrapper.bootstrap()
        try:
            assert bootstrapper._db is None
            assert run._owns_engine is True
        finally:
            await bootstrapper.aclose()

    async def test_strictly_serial_caps_workers(self, tmp_path: Path) -> None:
        # The bootstrapper passes the worker count straight through;
        # ScrapeRun's constructor is the single STRICTLY_SERIAL enforcement
        # site. Pin that a serial scraper bootstrapped with 4 workers still
        # ends up serial.
        class _SerialScraper(HttpPageScraper):
            driver_requirements: ClassVar[list[DriverRequirement]] = [
                DriverRequirement.STRICTLY_SERIAL
            ]

        bootstrapper = RunBootstrapper(
            _SerialScraper(),
            tmp_path / "run.db",
            transport=HttpxTransport(),
            setup_signal_handlers=False,
            config=RunConfig(
                seed_params=[{"fetch_page": {"page_id": 1}}],
                num_workers=4,
                rate_limited=False,
            ),
        )
        run = await bootstrapper.bootstrap()
        try:
            assert run.config.num_workers == 1
        finally:
            await bootstrapper.aclose()

    async def test_second_bootstrap_is_refused(self, tmp_path: Path) -> None:
        bootstrapper = RunBootstrapper(
            _NoArgEntryScraper(),
            tmp_path / "run.db",
            transport=HttpxTransport(),
            setup_signal_handlers=False,
        )
        await bootstrapper.bootstrap()
        try:
            with pytest.raises(RuntimeError, match="already open"):
                await bootstrapper.bootstrap()
        finally:
            await bootstrapper.aclose()

    @pytest.mark.parametrize(
        "signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"]
    )
    async def test_signal_handlers_route_to_stop_and_are_restored(
        self, tmp_path: Path, signum: signal.Signals
    ) -> None:
        # Signals are process-wide state and belong to the entry point: the
        # bootstrapper installs them after the run opens and puts back the
        # handlers it found on aclose, so a bare ScrapeRun never touches them.
        def sentinel(_signum: int, _frame: Any) -> None:
            raise AssertionError("sentinel handler must not fire")

        before = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        for sig in before:
            signal.signal(sig, sentinel)
        bootstrapper = RunBootstrapper(
            _NoArgEntryScraper(),
            tmp_path / "run.db",
            transport=SpyTransport(),
            setup_signal_handlers=True,
            config=RunConfig(rate_limited=False),
        )
        try:
            run = await bootstrapper.bootstrap()
            handler = signal.getsignal(signum)
            assert handler is not sentinel
            assert callable(handler)
            handler(signum, None)
            assert run.stop_event.is_set()
            await bootstrapper.aclose()
            assert signal.getsignal(signal.SIGINT) is sentinel
            assert signal.getsignal(signal.SIGTERM) is sentinel
        finally:
            await bootstrapper.aclose()
            for sig, prior in before.items():
                signal.signal(sig, prior)


# --- End-to-end over HTTP ---------------------------------------------------


@pytest.fixture
async def page_server_url(serve_routes: Any) -> str:
    async def handle_page(request: web.Request) -> web.Response:
        return web.Response(text=f"page-{request.match_info['n']}")

    return await serve_routes({"/page/{n}": handle_page})


def _results(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT json_extract(data_json, '$.body') FROM results"
            ).fetchall()
        ]
    finally:
        conn.close()


async def test_http_run_end_to_end_with_resume(
    page_server_url: str, tmp_path: Path
) -> None:
    db_path = tmp_path / "run.db"

    def make_scraper() -> HttpPageScraper:
        scraper = HttpPageScraper()
        scraper.base = page_server_url
        return scraper

    async with RunBootstrapper(
        make_scraper(),
        db_path,
        setup_signal_handlers=False,
        config=RunConfig(
            seed_params=[{"fetch_page": {"page_id": 1}}], rate_limited=False
        ),
    ) as run:
        await run.run()
    assert sorted(_results(db_path)) == ["page-1"]

    # Resume takes no params and re-fetches nothing already done.
    async with RunBootstrapper(
        make_scraper(),
        db_path,
        setup_signal_handlers=False,
        config=RunConfig(rate_limited=False),
    ) as run:
        await run.run()
    assert sorted(_results(db_path)) == ["page-1"]


# --- Real signals, in a subprocess ------------------------------------------

#: Runs a slow 40-page HTTP scrape under a bootstrapper that owns the
#: signals, prints ``READY`` once the run starts, and reports how it ended.
_SIGNAL_CHILD = """
import asyncio, json, signal, sqlite3, sys
from pathlib import Path

from aiohttp import web

from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver import RunBootstrapper
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks
from tests.driver.unified.conftest import HttpPageScraper
from tests.servers import start_app

fired = []


def sentinel(signum, _frame):
    fired.append(signum)


for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, sentinel)


async def main(db_path):
    async def page(_request):
        await asyncio.sleep(0.2)
        return web.Response(text="page")

    app = web.Application()
    app.router.add_get("/page/{n}", page)
    server = await start_app(app)
    scraper = HttpPageScraper()
    scraper.base = server.base_url

    async def on_run_start(_name):
        print("READY", flush=True)

    try:
        async with RunBootstrapper(
            scraper,
            db_path,
            config=RunConfig(
                seed_params=[
                    {"fetch_page": {"page_id": n}} for n in range(40)
                ],
                rate_limited=False,
            ),
            hooks=RunHooks(on_run_start=on_run_start),
        ) as run:
            await run.run()
    finally:
        await server.aclose()
    async with SQLManager.open(db_path) as sql:
        status = (await sql.get_run_metadata()).status
    conn = sqlite3.connect(db_path)
    (results,) = conn.execute("SELECT COUNT(*) FROM results").fetchone()
    conn.close()
    print(json.dumps({
        "status": status.name,
        "results": results,
        "restored": all(
            signal.getsignal(s) is sentinel
            for s in (signal.SIGINT, signal.SIGTERM)
        ),
        "fired": fired,
    }), flush=True)


asyncio.run(main(Path(sys.argv[1])))
"""


@pytest.mark.parametrize(
    "signum", [signal.SIGINT, signal.SIGTERM], ids=["SIGINT", "SIGTERM"]
)
def test_signal_during_run_stops_it_interrupted(
    tmp_path: Path, signum: signal.Signals
) -> None:
    """A signal mid-run stops it as INTERRUPTED and restores the handlers."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _SIGNAL_CHILD, str(tmp_path / "run.db")],
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signum)
        out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, err
    report = json.loads(out.strip().splitlines()[-1])
    assert report["status"] == RunStatus.INTERRUPTED.name
    assert report["results"] < 40
    assert report["restored"] is True
    assert report["fired"] == []


def test_install_signal_handlers_off_main_thread_is_a_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off the main thread nothing is installed and ``None`` comes back."""
    calls: list[Any] = []
    real = signal.signal

    def spy(signum: int, handler: Any) -> Any:
        calls.append(signum)
        return real(signum, handler)

    monkeypatch.setattr(signal, "signal", spy)
    before = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    result: list[Any] = []
    worker = threading.Thread(
        target=lambda: result.append(install_signal_handlers(MagicMock()))
    )
    worker.start()
    worker.join()

    assert result == [None]
    assert calls == []
    assert {
        s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)
    } == before


def test_install_signal_handlers_off_main_thread_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A run whose SIGINT/SIGTERM will not stop it says so, rather than
    leaving the operator to find out at Ctrl-C."""
    worker = threading.Thread(
        target=lambda: install_signal_handlers(MagicMock())
    )
    with caplog.at_level(logging.WARNING, logger=bootstrap_logger.name):
        worker.start()
        worker.join()

    assert any(
        r.levelno == logging.WARNING and "main thread" in r.getMessage()
        for r in caplog.records
    )


def test_install_signal_handlers_refused_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Where the platform refuses a handler, the refusal is logged too."""
    real = signal.signal

    def refuse(signum: int, handler: Any) -> Any:
        if signum == signal.SIGTERM:
            raise ValueError("refused")
        return real(signum, handler)

    monkeypatch.setattr(signal, "signal", refuse)
    before = signal.getsignal(signal.SIGINT)
    with caplog.at_level(logging.WARNING, logger=bootstrap_logger.name):
        assert install_signal_handlers(MagicMock()) is None

    assert signal.getsignal(signal.SIGINT) == before
    assert any(
        r.levelno == logging.WARNING and "refused" in r.getMessage()
        for r in caplog.records
    )
