"""RunBootstrapper — the canonical entry point for unified scrape runs.

Consolidates the wiring that the CLI and ``scripts/run_unified.py`` used to
do by hand (REFACTOR_0.1.0.md Phase 0.7):

- **Transport selection** from ``scraper.driver_requirements``: no browser
  requirement → plain HTTP (``ScrapeRun`` builds its default
  ``HttpxTransport``); ``CFCAP_HANDLER`` → :class:`CamoufoxTransport`; any
  other browser requirement → :class:`PlaywrightTransport` (whose engine
  derives firefox/chromium from ``FF_ALIKE`` / ``CHROME_ALIKE``).
- **Browser-profile auto-resolution** from ``$JKENT_HOME/profiles/{name}``
  with the CLI's precedence (``CFCAP_HANDLER`` → camoufox, else ``FF_ALIKE``
  → firefox, else ``CHROME_ALIKE`` → chrome). Unlike the CLI, a missing
  profile directory is a warning, not an error — the unified engines run
  fine profile-less.
- **STRICTLY_SERIAL** worker capping.
- **DB pre-init** for browser transports, which need their own
  :class:`SQLManager` on the run's DB file (parent-tab staging +
  incidental writes).
- **Archive handler** defaulting (:class:`LocalAsyncStreamingArchiveHandler`
  over ``storage_dir``).
- **Seeding and resume**: ``seed_params`` is only valid on a fresh DB
  (mirroring the CLI's guard); ``add_params`` layers new invocations onto
  an existing run via :meth:`ScrapeRun.add_seed_params`.

Use as an async context manager::

    async with RunBootstrapper(scraper, db_path, storage_dir=store) as run:
        await run.run()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jkent.data_types import DriverRequirement
from jkent.driver.archive_handler import LocalAsyncStreamingArchiveHandler
from jkent.driver.browser_engine.browser_profile import load_browser_profile
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport.camoufox_transport import (
    CamoufoxTransport,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper
    from jkent.driver.browser_engine.browser_profile import BrowserProfile
    from jkent.driver.unified_driver.transport import Transport

logger = logging.getLogger(__name__)

#: Requirements that demand a live browser (mirrors the CLI's
#: ``needs_playwright`` set).
BROWSER_REQUIREMENTS = frozenset(
    {
        DriverRequirement.JS_EVAL,
        DriverRequirement.FF_ALIKE,
        DriverRequirement.CHROME_ALIKE,
        DriverRequirement.HCAP_HANDLER,
        DriverRequirement.RCAP_HANDLER,
        DriverRequirement.CFCAP_HANDLER,
        DriverRequirement.STRICTLY_SERIAL,
    }
)


def _jkent_home() -> Path:
    return Path(os.environ.get("JKENT_HOME", "~/.jkent")).expanduser()


def needs_browser(scraper: BaseScraper[Any]) -> bool:
    """Whether the scraper's requirements demand a browser transport."""
    reqs = getattr(scraper, "driver_requirements", [])
    return any(r in BROWSER_REQUIREMENTS for r in reqs)


def resolve_browser_profile(
    scraper: BaseScraper[Any],
    *,
    jkent_home: Path | None = None,
) -> BrowserProfile | None:
    """Auto-resolve a browser profile from ``$JKENT_HOME/profiles/{name}``.

    CLI-equivalent precedence: ``CFCAP_HANDLER`` always gets the camoufox
    profile (camoufox is the only engine that reliably passes CF managed
    challenges), else ``FF_ALIKE`` → firefox, else ``CHROME_ALIKE`` →
    chrome. A scraper with none of these (or a missing/invalid profile
    directory) runs profile-less, with a warning where the CLI errored.
    """
    reqs = getattr(scraper, "driver_requirements", [])
    profile_name: str | None = None
    if DriverRequirement.CFCAP_HANDLER in reqs:
        profile_name = "camoufox"
    elif DriverRequirement.FF_ALIKE in reqs:
        profile_name = "firefox"
    elif DriverRequirement.CHROME_ALIKE in reqs:
        profile_name = "chrome"
    if profile_name is None:
        return None

    profile_dir = (jkent_home or _jkent_home()) / "profiles" / profile_name
    if not profile_dir.exists():
        logger.warning(
            "Scraper prefers the %s browser profile but none exists at %s; "
            "running without a profile",
            profile_name,
            profile_dir,
        )
        return None
    try:
        return load_browser_profile(profile_dir)
    except Exception:
        logger.warning(
            "Failed to load browser profile at %s; running without",
            profile_dir,
            exc_info=True,
        )
        return None


def build_transport(
    scraper: BaseScraper[Any],
    *,
    headless: bool = True,
    proxy: str | None = None,
    browser_profile: BrowserProfile | None = None,
    db: SQLManager | None = None,
) -> Transport[Any] | None:
    """Select + build the transport for ``scraper``'s requirements.

    Returns ``None`` for pure-HTTP scrapers so :meth:`ScrapeRun.open` builds
    its default :class:`HttpxTransport` (which carries the scraper's SSL
    context and proxy itself).
    """
    if not needs_browser(scraper):
        return None
    cls = (
        CamoufoxTransport
        if DriverRequirement.CFCAP_HANDLER
        in getattr(scraper, "driver_requirements", [])
        else PlaywrightTransport
    )
    return cls(
        scraper,
        headless=headless,
        proxy=proxy,
        browser_profile=browser_profile,
        db=db,
    )


class RunBootstrapper:
    """Build, open, and (on exit) tear down a fully-wired :class:`ScrapeRun`.

    ``__aenter__`` returns the opened :class:`ScrapeRun`, ready for
    :meth:`ScrapeRun.run`. Anything not consumed here (callbacks, timeouts,
    ``rate_limited`` …) is forwarded to the :class:`ScrapeRun` constructor
    via ``**scrape_run_kwargs``.

    Args:
        scraper: The scraper instance to run.
        db_path: The run database (created if missing).
        storage_dir: Archive download directory; when set (and no explicit
            ``archive_handler`` is given) a
            :class:`LocalAsyncStreamingArchiveHandler` is built over it.
        seed_params: ``{entry: kwargs}`` invocations for ``initial_seed()``.
            Only valid on a fresh database — resuming with different params
            silently doing nothing is the CLI footgun this guard removes.
        add_params: Invocations to layer onto an *existing* run
            (``ScrapeRun.add_seed_params``). Mutually exclusive with
            ``seed_params``.
        resume: Restore the pending queue from the DB (default True).
        num_workers / max_workers: Worker pool bounds; capped to 1 for
            STRICTLY_SERIAL scrapers.
        headless / proxy / browser_profile / jkent_home: Browser transport
            configuration; ``browser_profile`` overrides auto-resolution
            from ``{jkent_home}/profiles``.
        transport: Explicit transport override; skips selection entirely
            (the caller then owns its DB wiring).
        archive_handler: Explicit archive handler override.
        setup_signal_handlers: Forwarded to :meth:`ScrapeRun.open`.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        storage_dir: Path | None = None,
        seed_params: list[dict[str, dict[str, Any]]] | None = None,
        add_params: list[dict[str, dict[str, Any]]] | None = None,
        resume: bool = True,
        num_workers: int = 1,
        max_workers: int = 10,
        headless: bool = True,
        proxy: str | None = None,
        browser_profile: BrowserProfile | None = None,
        jkent_home: Path | None = None,
        transport: Transport[Any] | None = None,
        archive_handler: Any | None = None,
        setup_signal_handlers: bool = True,
        **scrape_run_kwargs: Any,
    ) -> None:
        if seed_params is not None and add_params is not None:
            raise ValueError(
                "seed_params and add_params are mutually exclusive: "
                "seed_params starts a fresh run, add_params extends an "
                "existing one"
            )
        if add_params is not None and not add_params:
            raise ValueError("add_params must be a non-empty list")

        self.scraper = scraper
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.seed_params = seed_params
        self.add_params = add_params
        self.resume = resume
        self.num_workers = num_workers
        self.max_workers = max_workers
        self.headless = headless
        self.proxy = proxy
        self.browser_profile = browser_profile
        self.jkent_home = jkent_home
        self.archive_handler = archive_handler
        self.setup_signal_handlers = setup_signal_handlers
        self._scrape_run_kwargs = scrape_run_kwargs
        self._explicit_transport = transport
        self._run: ScrapeRun | None = None
        self._transport_engine: Any | None = None

    async def __aenter__(self) -> ScrapeRun:
        return await self.bootstrap()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def bootstrap(self) -> ScrapeRun:
        """Build the transport + ScrapeRun, open it, and apply add_params."""
        if self._run is not None:
            raise RuntimeError("RunBootstrapper is already open")

        if self.seed_params is not None:
            await self._reject_seed_params_on_existing_db()

        num_workers, max_workers = self.num_workers, self.max_workers
        reqs = getattr(self.scraper, "driver_requirements", [])
        if DriverRequirement.STRICTLY_SERIAL in reqs and (
            num_workers != 1 or max_workers != 1
        ):
            # A stateful session / postback chain must run one request at a
            # time in priority order (ScrapeRun enforces this too; capping
            # here keeps the choice visible to the caller).
            logger.warning(
                "STRICTLY_SERIAL scraper: capping workers to 1 "
                "(requested %d/%d)",
                num_workers,
                max_workers,
            )
            num_workers = max_workers = 1

        transport = self._explicit_transport
        if transport is None and needs_browser(self.scraper):
            # Browser transports need a DB handle on the same file ScrapeRun
            # uses (parent staging / incidentals). Pre-init the schema and
            # hand the transport its own SQLManager; ScrapeRun opens its own
            # handle on the same file.
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            engine, session_factory = await init_database(self.db_path)
            self._transport_engine = engine
            profile = self.browser_profile or resolve_browser_profile(
                self.scraper, jkent_home=self.jkent_home
            )
            transport = build_transport(
                self.scraper,
                headless=self.headless,
                proxy=self.proxy,
                browser_profile=profile,
                db=SQLManager(engine, session_factory),
            )

        archive_handler = self.archive_handler
        storage_dir = self.storage_dir
        if archive_handler is None and storage_dir is not None:
            storage_dir.mkdir(parents=True, exist_ok=True)
            archive_handler = LocalAsyncStreamingArchiveHandler(storage_dir)

        run = ScrapeRun(
            self.scraper,
            self.db_path,
            transport=transport,
            num_workers=num_workers,
            max_workers=max_workers,
            resume=self.resume,
            seed_params=self.seed_params,
            proxy=self.proxy,
            archive_handler=archive_handler,
            **self._scrape_run_kwargs,
        )
        try:
            await run.open(setup_signal_handlers=self.setup_signal_handlers)
            if self.add_params is not None:
                await run.add_seed_params(self.add_params)
        except BaseException:
            await self._dispose_engine()
            raise
        self._run = run
        return run

    async def aclose(self) -> None:
        """Close the run and dispose the transport's DB engine."""
        if self._run is not None:
            try:
                await self._run.aclose()
            finally:
                self._run = None
                await self._dispose_engine()
        else:
            await self._dispose_engine()

    async def _dispose_engine(self) -> None:
        if self._transport_engine is not None:
            await self._transport_engine.dispose()
            self._transport_engine = None

    async def _reject_seed_params_on_existing_db(self) -> None:
        """seed_params on an already-seeded DB is an error, not a no-op.

        ``ScrapeRun`` ignores ``seed_params`` when the queue already has
        requests, which silently discards the caller's intent. Mirror the
        CLI's guard: point them at ``add_params`` instead.
        """
        if not self.db_path.exists():
            return
        async with SQLManager.open(self.db_path) as sql:
            existing = await sql.get_run_metadata()
        if existing is not None:
            raise ValueError(
                f"Database {self.db_path} already has a run for scraper "
                f"'{existing.get('scraper_name')}'. seed_params is only "
                "valid on a fresh database; use add_params to add entries "
                "to an existing run."
            )
