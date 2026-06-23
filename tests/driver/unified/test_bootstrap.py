"""``RunBootstrapper`` — requirement-driven wiring for unified runs.

Covers transport selection, browser-profile auto-resolution from a fake
``JKENT_HOME``, STRICTLY_SERIAL capping, seed/add-params validation, and a
full HTTP run end-to-end through the bootstrapper (open → run → resume with
``add_params``). Browser transports are selected but never launched.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from aiohttp import web

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.unified_driver import (
    CamoufoxTransport,
    HttpxTransport,
    PlaywrightTransport,
    RunBootstrapper,
    build_transport,
    resolve_browser_profile,
)
from tests.driver.unified.conftest import HttpPageScraper
from tests.driver.unified.test_run import SpyTransport

# --- Scrapers --------------------------------------------------------------


def _scraper_with(*reqs: DriverRequirement) -> BaseScraper:
    class _Scraper(BaseScraper[dict]):
        driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

    return _Scraper()


class _NoRequestScraper(BaseScraper[dict]):
    """A scraper whose entry yields nothing: open() writes run metadata but
    no request rows are ever enqueued."""

    @entry(dict)
    def fetch_page(self, page_id: int) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"body": response.text})


class _NoArgEntryScraper(BaseScraper[dict]):
    """A scraper with a no-arg entry (auto-seeded without seed_params) that
    yields nothing — so open() succeeds with no requests enqueued."""

    @entry(dict)
    def start(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"body": response.text})


# --- Transport selection ----------------------------------------------------


class TestBuildTransport:
    def test_http_scraper_gets_no_transport(self) -> None:
        # None → ScrapeRun builds its default HttpxTransport (which carries
        # the scraper's SSL context itself).
        assert build_transport(_scraper_with()) is None
        assert (
            build_transport(_scraper_with(DriverRequirement.H11_HEADER_FIXES))
            is None
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

    def test_camoufox_reqs_get_camoufox(self) -> None:
        # CFCAP, HCAP, and RCAP all demand the stealthy camoufox engine.
        for req in (
            DriverRequirement.CFCAP_HANDLER,
            DriverRequirement.HCAP_HANDLER,
            DriverRequirement.RCAP_HANDLER,
        ):
            transport = build_transport(_scraper_with(req))
            assert type(transport) is CamoufoxTransport, req


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

    def test_hcap_wins_over_flavors(self, tmp_path: Path) -> None:
        _write_profile(tmp_path, "camoufox", "firefox")
        _write_profile(tmp_path, "firefox", "firefox")
        profile = resolve_browser_profile(
            _scraper_with(
                DriverRequirement.HCAP_HANDLER, DriverRequirement.FF_ALIKE
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


# --- Constructor validation -------------------------------------------------


class TestValidation:
    def test_seed_and_add_params_are_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            RunBootstrapper(
                HttpPageScraper(),
                tmp_path / "run.db",
                seed_params=[{"fetch_page": {"page_id": 1}}],
                add_params=[{"fetch_page": {"page_id": 2}}],
            )

    def test_add_params_must_be_non_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RunBootstrapper(
                HttpPageScraper(), tmp_path / "run.db", add_params=[]
            )

    async def test_seed_params_rejected_on_existing_db(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "run.db"
        async with RunBootstrapper(
            HttpPageScraper(),
            db_path,
            seed_params=[{"fetch_page": {"page_id": 1}}],
            setup_signal_handlers=False,
            rate_limited=False,
        ):
            pass

        with pytest.raises(ValueError, match="add_params"):
            async with RunBootstrapper(
                HttpPageScraper(),
                db_path,
                seed_params=[{"fetch_page": {"page_id": 2}}],
                setup_signal_handlers=False,
                rate_limited=False,
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
            seed_params=[{"fetch_page": {"page_id": 1}}],
            setup_signal_handlers=False,
            rate_limited=False,
        ):
            pass

        # Re-running with the *same* seed_params must not raise: the DB has
        # metadata but zero requests.
        async with RunBootstrapper(
            _NoRequestScraper(),
            db_path,
            seed_params=[{"fetch_page": {"page_id": 1}}],
            setup_signal_handlers=False,
            rate_limited=False,
        ):
            pass

    async def test_open_phase_failure_tears_down_transport(
        self, tmp_path: Path
    ) -> None:
        # add_seed_params runs after open() brings the transport up; an
        # unknown entry makes it raise. bootstrap()'s except path must tear the
        # partially-opened run down (run.aclose) rather than leak the transport.
        transport = SpyTransport()
        bootstrapper = RunBootstrapper(
            _NoArgEntryScraper(),
            tmp_path / "run.db",
            add_params=[{"does_not_exist": {}}],
            transport=transport,
            setup_signal_handlers=False,
            rate_limited=False,
        )
        with pytest.raises(ValueError, match="Unknown entry"):
            await bootstrapper.bootstrap()
        assert transport.opened is True
        assert transport.closed is True

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
            seed_params=[{"fetch_page": {"page_id": 1}}],
            num_workers=4,
            # An explicit transport skips selection — STRICTLY_SERIAL would
            # otherwise pick a browser transport and launch one.
            transport=HttpxTransport(),
            setup_signal_handlers=False,
            rate_limited=False,
        )
        run = await bootstrapper.bootstrap()
        try:
            assert run.num_workers == 1
        finally:
            await bootstrapper.aclose()


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


async def test_http_run_end_to_end_with_resume_and_add_params(
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
        seed_params=[{"fetch_page": {"page_id": 1}}],
        setup_signal_handlers=False,
        rate_limited=False,
    ) as run:
        await run.run()
    assert sorted(_results(db_path)) == ["page-1"]

    # Resume with add_params: only the new page is fetched.
    async with RunBootstrapper(
        make_scraper(),
        db_path,
        add_params=[{"fetch_page": {"page_id": 2}}],
        setup_signal_handlers=False,
        rate_limited=False,
    ) as run:
        await run.run()
    assert sorted(_results(db_path)) == ["page-1", "page-2"]
