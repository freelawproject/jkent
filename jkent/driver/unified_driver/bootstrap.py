"""RunBootstrapper — the canonical entry point for unified scrape runs.

Consolidates the wiring the CLI and ``scripts/run_unified.py`` used to do by
hand:

- **Transport selection** from ``scraper.driver_requirements``
  (:func:`build_transport`): no browser requirement → :class:`HttpxTransport`
  carrying the scraper's SSL context plus the caller's proxy and timeout;
  otherwise :class:`PlaywrightTransport`, whose engine
  :class:`~jkent.driver.unified_driver.requirements.ResolvedRequirements`
  decides. Proxy and timeout are *transport* configuration and live here,
  not on the run — ``ScrapeRun`` never sees them.
- **Browser-profile auto-resolution** from ``$JKENT_HOME/profiles/{name}``,
  named by that same resolved choice. A missing profile directory is a
  warning, not an error — the unified engines run fine profile-less.
- **DB pre-init** for browser transports, which need a :class:`SQLManager`
  on the run's DB file (parent-tab staging + incidental writes) before
  ``ScrapeRun`` exists. The *same* manager is then handed to ``ScrapeRun``
  via ``db=``: one manager means one write lock and one connection pool
  over the file, so transport writes and run writes serialize against each
  other instead of colliding as two independent SQLite writers.
- **Archive handler** defaulting (:class:`LocalAsyncStreamingArchiveHandler`
  over ``storage_dir``).
- **Seeding and resume**: a run database is pinned to its seed set.
  ``config.seed_params`` is only valid on a database with no requests;
  resuming takes no params.
- **Signal handling**: SIGINT/SIGTERM → ``run.stop()`` while the run is
  open; the handlers found at install are restored on exit. Process-wide state belongs to the process entry
  point, which this is; hosts that own their signals (Prefect workers)
  pass ``setup_signal_handlers=False``.

Use as an async context manager::

    async with RunBootstrapper(scraper, db_path, storage_dir=store) as run:
        await run.run()
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jkent.driver.archive_handler import LocalAsyncStreamingArchiveHandler
from jkent.driver.browser_engine.browser_profile import load_browser_profile
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.requirements import (
    ResolvedRequirements,
    needs_browser,
)
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper
    from jkent.driver.archive_handler import AsyncStreamingArchiveHandler
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

    The browser — and with it the profile directory name — comes from
    :class:`ResolvedRequirements`, the one selection site. A scraper with no
    browser preference runs profile-less; so, with a warning, does one whose
    profile directory is missing.

    Raises:
        FileNotFoundError, ValueError: The profile directory exists but its
            manifest is missing or rejected (see :func:`load_browser_profile`).
    """
    profile_name = ResolvedRequirements.of(scraper).profile_name
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
    # A profile that exists but is rejected propagates: launching without it
    # would drop the fingerprint the scraper asked for.
    return load_browser_profile(profile_dir)


def build_transport(
    scraper: BaseScraper[Any],
    *,
    headless: bool = True,
    proxy: str | None = None,
    timeout: float | None = None,
    browser_profile: BrowserProfile | None = None,
    db: SQLManager | None = None,
) -> Transport[Any]:
    """Select + build the transport for ``scraper``'s requirements.

    Pure-HTTP scrapers get an :class:`HttpxTransport` carrying the scraper's
    SSL context, ``proxy``, and ``timeout``; browser scrapers get a
    :class:`PlaywrightTransport` with the same ``proxy`` and ``timeout``.
    On either, ``timeout`` is what a request waits when neither its own
    ``HTTPRequestParams.timeout`` nor its step's ``@step(timeout=)`` is set
    (``DEFAULT_TIMEOUT_S`` when this is unset too).
    """
    if not needs_browser(scraper):
        return HttpxTransport(
            scraper=scraper,
            ssl_context=scraper.get_ssl_context(),
            proxy=proxy,
            timeout=timeout,
        )
    return PlaywrightTransport(
        scraper,
        headless=headless,
        proxy=proxy,
        timeout=timeout,
        browser_profile=browser_profile,
        db=db,
    )


SignalHandlers = dict[signal.Signals, Any]


def install_signal_handlers(run: ScrapeRun) -> SignalHandlers | None:
    """Route SIGINT/SIGTERM to ``run.stop()`` via :func:`signal.signal`.

    Returns the handlers they replaced, for
    :func:`restore_signal_handlers`. A no-op off the main thread (Python
    only allows handlers there) and where the platform refuses — both
    log a warning, since a signal will then not stop the run gracefully,
    and return ``None`` so the caller knows not to restore.
    """
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            "Not on the main thread: SIGINT/SIGTERM handlers not installed, "
            "so a signal will not stop the run gracefully"
        )
        return None

    def handle_signal(signum: int, _frame: Any) -> None:
        logger.info(
            "Received %s, initiating graceful shutdown...",
            signal.Signals(signum).name,
        )
        run.stop()

    previous: SignalHandlers = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, handle_signal)
    except (ValueError, AttributeError, OSError) as exc:
        restore_signal_handlers(previous)
        logger.warning(
            "SIGINT/SIGTERM handlers not installed (%s), so a signal will "
            "not stop the run gracefully",
            exc,
        )
        return None
    return previous


def restore_signal_handlers(previous: SignalHandlers) -> None:
    """Reinstall the handlers :func:`install_signal_handlers` replaced.

    A handler that was not installed from Python (``signal.signal``
    reported ``None``) cannot be reinstalled; that signal gets
    ``SIG_DFL``.
    """
    with contextlib.suppress(ValueError, AttributeError, OSError):
        for signum, handler in previous.items():
            signal.signal(
                signum, signal.SIG_DFL if handler is None else handler
            )


class RunBootstrapper:
    """Build, open, and (on exit) tear down a fully-wired :class:`ScrapeRun`.

    ``__aenter__`` returns the opened :class:`ScrapeRun`, ready for
    :meth:`ScrapeRun.run`.

    Args:
        scraper: The scraper instance to run.
        db_path: The run database (created if missing).
        config: The run's knobs (worker count, ``seed_params``,
            breaker policy, error budget …); see :class:`RunConfig`.
            ``config.seed_params`` is only valid on a fresh database —
            resuming with different params silently doing nothing is the
            CLI footgun this guard removes.
        hooks: The host's callbacks; see :class:`RunHooks`.
        storage_dir: Archive download directory; when set (and no explicit
            ``archive_handler`` is given) a
            :class:`LocalAsyncStreamingArchiveHandler` is built over it.
        headless / proxy / timeout / browser_profile / jkent_home: Transport
            configuration, consumed by :func:`build_transport`;
            ``browser_profile`` overrides auto-resolution from
            ``{jkent_home}/profiles``.
        transport: Explicit transport override; skips selection entirely
            (the caller then owns its DB wiring and its proxy/timeout).
        archive_handler: Explicit archive handler override.
        setup_signal_handlers: Route SIGINT/SIGTERM to ``run.stop()`` while
            the run is open.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        config: RunConfig | None = None,
        hooks: RunHooks | None = None,
        storage_dir: Path | None = None,
        headless: bool = True,
        proxy: str | None = None,
        timeout: float | None = None,
        browser_profile: BrowserProfile | None = None,
        jkent_home: Path | None = None,
        transport: Transport[Any] | None = None,
        archive_handler: AsyncStreamingArchiveHandler | None = None,
        setup_signal_handlers: bool = True,
    ) -> None:
        self.config = config or RunConfig()
        self.hooks = hooks or RunHooks()

        self.scraper = scraper
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.headless = headless
        self.proxy = proxy
        self.timeout = timeout
        self.browser_profile = browser_profile
        self.jkent_home = jkent_home
        self.archive_handler = archive_handler
        self.setup_signal_handlers = setup_signal_handlers
        self._explicit_transport = transport
        self._run: ScrapeRun | None = None
        self._db_engine: Any | None = None
        self._db: SQLManager | None = None
        self._previous_signal_handlers: SignalHandlers | None = None

    async def __aenter__(self) -> ScrapeRun:
        return await self.bootstrap()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def bootstrap(self) -> ScrapeRun:
        """Build the transport + ScrapeRun and open it."""
        if self._run is not None:
            raise RuntimeError("RunBootstrapper is already open")

        if self.config.seed_params is not None:
            await self._reject_seed_params_on_existing_db()

        try:
            transport = self._explicit_transport
            if transport is None:
                transport = await self._select_transport()

            archive_handler = self.archive_handler
            storage_dir = self.storage_dir
            if archive_handler is None and storage_dir is not None:
                storage_dir.mkdir(parents=True, exist_ok=True)
                archive_handler = LocalAsyncStreamingArchiveHandler(
                    storage_dir
                )

            run = ScrapeRun(
                self.scraper,
                self.db_path,
                config=self.config,
                hooks=self.hooks,
                db=self._db,
                transport=transport,
                archive_handler=archive_handler,
            )
            await run.open()
        except BaseException:
            # open() unwinds its own partial state; the engine
            # _select_transport may have opened is ours.
            await self._dispose_engine()
            raise
        if self.setup_signal_handlers:
            self._previous_signal_handlers = install_signal_handlers(run)
        self._run = run
        return run

    async def _select_transport(self) -> Transport[Any]:
        """Pick and build the transport, pre-initializing the DB for browsers."""
        profile: BrowserProfile | None = None
        if needs_browser(self.scraper):
            # Browser transports need a DB handle on the same file ScrapeRun
            # uses (parent staging / incidentals), and they need it before
            # ScrapeRun is constructed. Pre-init the schema here and give the
            # transport and the run the *same* SQLManager — a second handle
            # would carry its own write lock and race this one.
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            engine, session_factory = await init_database(self.db_path)
            self._db_engine = engine
            self._db = SQLManager(engine, session_factory)
            profile = self.browser_profile or resolve_browser_profile(
                self.scraper, jkent_home=self.jkent_home
            )
        return build_transport(
            self.scraper,
            headless=self.headless,
            proxy=self.proxy,
            timeout=self.timeout,
            browser_profile=profile,
            db=self._db,
        )

    async def aclose(self) -> None:
        """Restore signals, close the run, and dispose the shared DB engine."""
        previous = self._previous_signal_handlers
        if previous is not None:
            self._previous_signal_handlers = None
            restore_signal_handlers(previous)
        if self._run is not None:
            try:
                await self._run.aclose()
            finally:
                self._run = None
                await self._dispose_engine()
        else:
            await self._dispose_engine()

    async def _dispose_engine(self) -> None:
        # ScrapeRun borrows this engine (constructed with db=), so disposing
        # it is ours to do — after run.aclose() has finished with it.
        if self._db_engine is not None:
            await self._db_engine.dispose()
            self._db_engine = None
            self._db = None

    async def _reject_seed_params_on_existing_db(self) -> None:
        """seed_params on an already-seeded DB is an error, not a no-op.

        ``ScrapeRun`` ignores ``seed_params`` once the queue already has
        requests, which silently discards the caller's intent. Mirror that
        gate exactly — on request rows, not on run metadata, which ``open()``
        writes before any seeding so a fresh-but-failed run can still be
        retried with the same params.
        """
        # A 0-byte file holds no run either (``SQLManager.open`` refuses it).
        if not self.db_path.exists() or self.db_path.stat().st_size == 0:
            return
        async with SQLManager.open(self.db_path) as sql:
            if not await sql.has_any_requests():
                return
            existing = await sql.get_run_metadata()
        scraper_name = existing.scraper_name if existing else None
        raise ValueError(
            f"Database {self.db_path} already has a run for scraper "
            f"'{scraper_name}'. seed_params is only valid on a fresh "
            "database; a run is pinned to its seed set, so recreate the run "
            "database to run different params."
        )
