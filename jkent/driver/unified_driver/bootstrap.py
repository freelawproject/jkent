"""RunBootstrapper — the canonical entry point for unified scrape runs.

Consolidates the wiring that the CLI and ``scripts/run_unified.py`` used to
do by hand (REFACTOR_0.1.0.md Phase 0.7):

- **Transport selection** from ``scraper.driver_requirements``: no browser
  requirement → plain HTTP (``ScrapeRun`` builds its default
  ``HttpxTransport``); otherwise
  :func:`~jkent.driver.unified_driver.requirements.select_browser` decides —
  camoufox → :class:`CamoufoxTransport`, anything else →
  :class:`PlaywrightTransport` (whose engine derives from the same
  function).
- **Browser-profile auto-resolution** from ``$JKENT_HOME/profiles/{name}``,
  named by the same ``select_browser`` choice. Unlike the CLI, a missing
  profile directory is a warning, not an error — the unified engines run
  fine profile-less.
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

import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jkent.driver.archive_handler import LocalAsyncStreamingArchiveHandler
from jkent.driver.browser_engine.browser_profile import load_browser_profile
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.requirements import (
    needs_browser,
    select_browser,
)
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


def _jkent_home() -> Path:
    return Path(os.environ.get("JKENT_HOME", "~/.jkent")).expanduser()


def resolve_browser_profile(
    scraper: BaseScraper[Any],
    *,
    jkent_home: Path | None = None,
) -> BrowserProfile | None:
    """Auto-resolve a browser profile from ``$JKENT_HOME/profiles/{name}``.

    The browser is chosen by :func:`select_browser` (the one
    selection-precedence site); this maps that choice to the profile
    directory name. A scraper with no browser preference (or a
    missing/invalid profile directory) runs profile-less, with a warning
    where the CLI errored.
    """
    choice = select_browser(scraper)
    if choice is None:
        return None
    profile_name = "chrome" if choice == "chromium" else choice

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
        if select_browser(scraper) == "camoufox"
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
        num_workers: Pinned worker-pool size — all spawned up front, never
            re-grown (:class:`ScrapeRun` caps it to 1 for STRICTLY_SERIAL
            scrapers).
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
            num_workers=self.num_workers,
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
            # open()/add_seed_params can fail after the transport (and its
            # browser process) is up and signal handlers are installed; tear
            # the partially-opened run down before disposing our own engine so
            # nothing is leaked. aclose is best-effort — never mask the original.
            with contextlib.suppress(Exception):
                await run.aclose()
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

        ``ScrapeRun`` ignores ``seed_params`` once the queue already has
        requests, which silently discards the caller's intent. Mirror that
        gate exactly — on request rows, not on run metadata, which ``open()``
        writes before any seeding so a fresh-but-failed run can still be
        retried with the same params — and point them at ``add_params``.
        """
        if not self.db_path.exists():
            return
        async with SQLManager.open(self.db_path) as sql:
            if not await sql.has_any_requests():
                return
            existing = await sql.get_run_metadata()
        scraper_name = existing.get("scraper_name") if existing else None
        raise ValueError(
            f"Database {self.db_path} already has a run for scraper "
            f"'{scraper_name}'. seed_params is only valid on a fresh "
            "database; use add_params to add entries to an existing run."
        )
