"""Playwright driver implementation for JavaScript-heavy websites.

This driver extends LocalDevDriver to handle JavaScript-heavy court websites
using Playwright browser automation. It maintains step function purity by:

1. Rendering pages in a real browser
2. Serializing the rendered DOM to HTML
3. Parsing HTML with LXML and injecting as PageElement
4. Never passing live browser references to step functions

Key features:
- DOM snapshot model for step function purity
- Via handling for form submission and navigation replay
- Await list for explicit wait conditions before snapshot
- Autowait for automatic retry on element query failures
- Incidental requests tracking for browser-initiated network activity
- Rate limiting via pyrate_limiter
- Browser lifecycle management with context persistence
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from playwright.async_api import (
    Browser,
    BrowserContext,
    ElementHandle,
    Page,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from kent.common.decorators import get_step_metadata
from kent.common.exceptions import (
    HTMLStructuralAssumptionException,
    TransientException,
)
from kent.common.page_element import (
    ViaFormSubmit,
    ViaLink,
)
from kent.common.selector_observer import (
    SelectorObserver,
    SelectorQuery,
)
from kent.data_types import (
    ArchiveResponse,
    BaseRequest,
    BaseScraper,
    Response,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from kent.driver.interstitials import (
    INTERSTITIAL_HANDLERS,
    InterstitialHandler,
)
from kent.driver.persistent_driver._staging import StagedWrites
from kent.driver.persistent_driver.compression import (
    compress,
)
from kent.driver.persistent_driver.persistent_driver import (
    PersistentDriver,
)
from kent.driver.persistent_driver.sql_manager import (
    SQLManager,
)
from kent.driver.playwright_driver.browser_profile import BrowserProfile

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

logger = logging.getLogger(__name__)

ScraperReturnDatatype = TypeVar("ScraperReturnDatatype")


def _parse_proxy_for_playwright(proxy_url: str) -> dict[str, str]:
    """Convert a proxy URL into Playwright's ``proxy=`` dict.

    Playwright expects ``{"server": "<scheme>://<host>:<port>"}`` with
    credentials in separate ``username`` / ``password`` fields — not
    embedded in the URL.  Accepts any scheme Playwright supports
    (``http``, ``https``, ``socks4``, ``socks5``).
    """
    from urllib.parse import unquote, urlsplit

    parts = urlsplit(proxy_url)
    if not parts.scheme or not parts.hostname:
        raise ValueError(f"Invalid proxy URL: {proxy_url!r}")

    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port is not None:
        server += f":{parts.port}"

    result: dict[str, str] = {"server": server}
    if parts.username:
        result["username"] = unquote(parts.username)
    if parts.password:
        result["password"] = unquote(parts.password)
    return result


def _resolve_user_data_dir(
    scraper: BaseScraper[Any],
    profile_name: str,
) -> Path:
    """Determine the user_data_dir for a persistent browser context.

    Returns ``~/.cache/kent/<scraper_module>/<profile_name>/browser-data/``,
    creating the directory if needed.
    """
    scraper_module = scraper.__class__.__module__.replace(".", "_")
    cache_dir = (
        Path.home()
        / ".cache"
        / "kent"
        / scraper_module
        / profile_name
        / "browser-data"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


async def _launch_persistent(
    playwright: Any,
    scraper: BaseScraper[Any],
    profile: BrowserProfile,
    headless: bool,
    proxy: str | None = None,
) -> BrowserContext:
    """Launch a persistent browser context from a :class:`BrowserProfile`.

    Handles protocol param injection, user data dir resolution, and
    init script loading.
    """
    from kent.driver.playwright_driver.browser_profile import (
        inject_protocol_params,
    )

    browser_launcher = getattr(playwright, profile.browser_type)

    # Inject protocol params (e.g. assistantMode, cdpPort) if configured
    if profile.protocol_params:
        inject_protocol_params(
            browser_launcher._impl_obj, profile.protocol_params
        )

    user_data_dir = _resolve_user_data_dir(scraper, profile.name)

    # Merge launch + context options for persistent context
    persistent_kwargs: dict[str, Any] = {}
    persistent_kwargs.update(profile.launch_options)
    persistent_kwargs.update(profile.context_options)
    persistent_kwargs["headless"] = headless
    if profile.channel:
        persistent_kwargs["channel"] = profile.channel
    if proxy:
        persistent_kwargs["proxy"] = _parse_proxy_for_playwright(proxy)

    context = await browser_launcher.launch_persistent_context(
        str(user_data_dir),
        **persistent_kwargs,
    )

    # Add init scripts from profile
    for script_path in profile.init_scripts:
        js = script_path.read_text(encoding="utf-8")
        await context.add_init_script(js)

    return context


class WorkerPage:
    """A Playwright page bound to a single worker, reused across requests.

    Encapsulates per-request state (incidental network requests) so that
    concurrent workers don't corrupt each other's data.
    """

    def __init__(self, page: Page, excluded_resource_types: set[str]):
        self.page = page
        self.incidental_requests: list[dict[str, Any]] = []
        self.current_parent_request_id: int | None = None
        self._excluded_resource_types = excluded_resource_types
        self._register_network_listeners()

    def _register_network_listeners(self) -> None:
        """Register network request/response listeners for incidental tracking."""

        incidentals = self.incidental_requests
        excluded = self._excluded_resource_types

        async def on_request(request: Any) -> None:
            incidental = {
                "resource_type": request.resource_type,
                "method": request.method,
                "url": request.url,
                "headers_json": json.dumps(dict(request.headers)),
                "body": None,
                "status_code": None,
                "response_headers_json": None,
                "content_compressed": None,
                "content_size_original": None,
                "content_size_compressed": None,
                "compression_dict_id": None,
                "started_at_ns": time.time_ns(),
                "completed_at_ns": None,
                "from_cache": None,
                "failure_reason": None,
            }
            incidentals.append(incidental)

        async def on_response(response: Any) -> None:
            request = response.request
            for incidental in incidentals:
                if (
                    incidental["url"] == request.url
                    and incidental["completed_at_ns"] is None
                ):
                    incidental["status_code"] = response.status
                    incidental["response_headers_json"] = json.dumps(
                        dict(response.headers)
                    )
                    incidental["completed_at_ns"] = time.time_ns()
                    incidental["from_cache"] = response.from_service_worker

                    if incidental["resource_type"] not in excluded:
                        try:
                            content = await response.body()
                            content_compressed = compress(content)
                            incidental["content_compressed"] = (
                                content_compressed
                            )
                            incidental["content_size_original"] = len(content)
                            incidental["content_size_compressed"] = len(
                                content_compressed
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to capture content for {request.url}: {e}"
                            )
                    break

        self.page.on("request", on_request)
        self.page.on("response", on_response)

    def clear_request_state(self) -> None:
        """Reset per-request state between navigations."""
        self.incidental_requests.clear()
        self.current_parent_request_id = None

    async def reset_for_reuse(self) -> None:
        """Lightweight cleanup between requests."""
        # Clear before navigation to discard stale events from the prior
        # page's in-flight sub-resources that may land during the goto.
        self.clear_request_state()
        await self.page.goto("about:blank", wait_until="commit")
        # Clear again to remove any events fired by the about:blank
        # navigation itself.
        self.clear_request_state()

    async def close(self) -> None:
        await self.page.close()


class PlaywrightDriver(
    PersistentDriver[ScraperReturnDatatype], Generic[ScraperReturnDatatype]
):
    """Playwright-based driver for JavaScript-heavy court websites.

    Extends LocalDevDriver to use browser automation instead of HTTP requests.
    Maintains step function purity through DOM snapshotting.

    Args:
        scraper: The scraper instance to run.
        db: SQLManager for database operations.
        browser_context: Playwright browser context for navigations.
        storage_dir: Directory for downloaded files.
        num_workers: Number of initial concurrent workers (default: 1).
        max_workers: Maximum workers for dynamic scaling (default: 10).
        resume: If True, resume from existing queue state (default: True).
        max_backoff_time: Maximum total backoff time before marking failed (default: 3600.0).
        request_manager: AsyncRequestManager for handling HTTP requests.
        enable_monitor: If True (default), start the worker monitor for dynamic scaling.

    Example:
        async with PlaywrightDriver.open(scraper, db_path) as driver:
            driver.on_progress = lambda e: print(e.to_json())
            await driver.run()
    """

    def __init__(
        self,
        scraper: BaseScraper[ScraperReturnDatatype],
        db: SQLManager,
        browser_context: BrowserContext,
        storage_dir: Path | None = None,
        num_workers: int = 1,
        max_workers: int = 10,
        resume: bool = True,
        max_backoff_time: float = 3600.0,
        request_manager: Any | None = None,
        enable_monitor: bool = True,
        excluded_resource_types: set[str] | None = None,
        rates: list[Any] | None = None,
    ) -> None:
        """Initialize the Playwright driver.

        Note: Use PlaywrightDriver.open() for proper async initialization.

        Args:
            scraper: The scraper instance to run.
            db: SQLManager for database operations.
            browser_context: Playwright browser context for navigations.
            storage_dir: Directory for downloaded files.
            num_workers: Number of initial concurrent workers.
            max_workers: Maximum workers for dynamic scaling.
            resume: If True, resume from existing queue state.
            max_backoff_time: Maximum total backoff time before marking failed.
            request_manager: AsyncRequestManager for handling HTTP requests.
            enable_monitor: If True (default), start the worker monitor for dynamic scaling.
            excluded_resource_types: Resource types to exclude from content capture (default: {"image", "media", "font"}).
            rates: Optional list of pyrate_limiter Rate objects for rate limiting.
        """
        super().__init__(
            scraper=scraper,
            db=db,
            storage_dir=storage_dir,
            num_workers=num_workers,
            max_workers=max_workers,
            resume=resume,
            max_backoff_time=max_backoff_time,
            request_manager=request_manager,
            enable_monitor=enable_monitor,
            rates=rates,
        )

        self.browser_context = browser_context
        # Per-worker page registry: each worker owns a long-lived page
        self._worker_pages: dict[int, WorkerPage] = {}
        # Resource types to exclude from content capture
        self.excluded_resource_types = excluded_resource_types or {
            "image",
            "media",
            "font",
        }
        # Interstitial handlers resolved from scraper's driver_requirements
        self._interstitial_handlers: list[InterstitialHandler] = [
            INTERSTITIAL_HANDLERS[req]
            for req in getattr(scraper, "driver_requirements", [])
            if req in INTERSTITIAL_HANDLERS
        ]
        # Browser restart state — populated by open() for crash recovery
        self._playwright: Any | None = None
        self._browser_obj: Browser | None = None
        self._browser_launcher: Any | None = None
        self._launch_kwargs: dict[str, Any] = {}
        self._context_kwargs: dict[str, Any] = {}
        self._browser_profile: BrowserProfile | None = None
        self._browser_restart_lock = asyncio.Lock()

    async def _acquire_worker_page(self, worker_id: int) -> WorkerPage:
        """Get or create the reusable page for a worker."""
        wp = self._worker_pages.get(worker_id)
        if wp is not None and wp.page.is_closed():
            self._worker_pages.pop(worker_id)
            wp = None
        if wp is None:
            try:
                page = await self.browser_context.new_page()
            except PlaywrightError as e:
                if self._is_connection_dead(e):
                    async with self._browser_restart_lock:
                        # Double-check: another worker may have restarted already
                        try:
                            page = await self.browser_context.new_page()
                        except PlaywrightError:
                            await self._restart_browser_context()
                            page = await self.browser_context.new_page()
                else:
                    raise
            wp = WorkerPage(page, self.excluded_resource_types)
            self._worker_pages[worker_id] = wp
        return wp

    async def _release_worker_page(self, worker_id: int) -> None:
        """Close and remove the page when a worker exits."""
        wp = self._worker_pages.pop(worker_id, None)
        if wp:
            await wp.close()

    def _is_connection_dead(self, error: PlaywrightError) -> bool:
        """Check if a PlaywrightError indicates the browser connection died."""
        msg = str(error)
        return (
            "Connection closed" in msg
            or "Browser has been closed" in msg
            or "Target page, context or browser has been closed" in msg
        )

    async def _restart_browser_context(self) -> None:
        """Restart the browser and context after a crash.

        Called under ``_browser_restart_lock`` to prevent concurrent
        restarts from multiple workers.  Only the standard (non-persistent)
        launch path is supported; persistent contexts cannot be safely
        restarted.
        """
        if self._browser_launcher is None:
            raise TransientException(
                "Browser connection lost and restart is not available "
                "(persistent context or missing launch params)"
            )

        logger.warning("Browser connection lost — restarting browser")

        # Discard all worker pages (they reference the dead browser)
        self._worker_pages.clear()

        # Best-effort close of the old browser
        if self._browser_obj is not None:
            try:
                await self._browser_obj.close()
            except Exception:
                pass

        # Relaunch browser and context
        new_browser = await self._browser_launcher.launch(
            **self._launch_kwargs
        )
        self._browser_obj = new_browser
        self.browser_context = await new_browser.new_context(
            **self._context_kwargs
        )

        # Re-add init scripts from browser profile
        if self._browser_profile is not None:
            for script_path in self._browser_profile.init_scripts:
                js = script_path.read_text(encoding="utf-8")
                await self.browser_context.add_init_script(js)

        logger.info("Browser restarted successfully")

    async def _db_worker(self, worker_id: int) -> None:
        """Wrap parent _db_worker to clean up the worker's page on exit."""
        try:
            await super()._db_worker(worker_id)
        finally:
            await self._release_worker_page(worker_id)

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        scraper: BaseScraper[ScraperReturnDatatype],
        db_path: Path,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
        locale: str = "en-US",
        timezone_id: str = "America/New_York",
        browser_profile: BrowserProfile | None = None,
        proxy: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[PlaywrightDriver[ScraperReturnDatatype]]:
        """Open Playwright driver as async context manager.

        Ensures proper initialization and cleanup of browser and DB connections.

        Args:
            scraper: The scraper instance to run.
            db_path: Path to SQLite database file.
            browser_type: Browser type: "chromium", "firefox", or "webkit" (default: "chromium").
            headless: Run browser in headless mode (default: True).
            viewport: Browser viewport size {"width": 1280, "height": 720} (default: None = 1280x720).
            user_agent: Custom user agent string (default: None = browser default).
            locale: Browser locale (default: "en-US").
            timezone_id: Browser timezone (default: "America/New_York").
            browser_profile: Optional :class:`BrowserProfile` loaded from a
                profile directory.  When provided, overrides browser_type,
                channel, viewport, and launch strategy.
            proxy: Optional proxy URL for the browser (e.g.
                ``"socks5://user:pass@host:1080"``).  Forwarded to
                Playwright's ``launch(proxy=...)``.  If the supplied
                :class:`BrowserProfile` already configures a proxy, this
                kwarg overrides it.
            **kwargs: Additional arguments passed to __init__.

        Yields:
            Initialized PlaywrightDriver instance.

        Example:
            async with PlaywrightDriver.open(
                scraper,
                Path("run.db"),
                browser_type="chromium",
                headless=True,
            ) as driver:
                await driver.run()
        """
        # Extract driver-specific kwargs
        storage_dir = kwargs.pop("storage_dir", None)
        num_workers = kwargs.pop("num_workers", 1)
        max_workers = kwargs.pop("max_workers", 10)
        resume = kwargs.pop("resume", True)
        max_backoff_time = kwargs.pop("max_backoff_time", 3600.0)
        enable_monitor = kwargs.pop("enable_monitor", True)
        excluded_resource_types = kwargs.pop("excluded_resource_types", None)
        rates = kwargs.pop("rates", None)
        seed_params = kwargs.pop("seed_params", None)
        request_preps = kwargs.pop("request_preps", None)

        # Validate request_preps and build dispatch table.
        from kent.preps import build_provided_preps

        provided_preps = build_provided_preps(
            scraper, request_preps, allow_live_page_providers=True
        )

        # Default viewport
        if viewport is None:
            viewport = {"width": 1280, "height": 720}

        # Check if we're resuming and should load browser config from DB
        stored_browser_config = None
        if resume and db_path.exists():
            # Temporarily open DB to check for stored config
            async with SQLManager.open(db_path) as temp_db:
                run_metadata = await temp_db.get_run_metadata()
                if run_metadata and run_metadata.get("browser_config"):
                    stored_browser_config = run_metadata["browser_config"]

        # On resume, reload browser profile from stored path if available
        if stored_browser_config and stored_browser_config.get("profile_path"):
            from kent.driver.playwright_driver.browser_profile import (
                load_browser_profile,
            )

            stored_path = Path(stored_browser_config["profile_path"])
            if stored_path.is_dir():
                browser_profile = load_browser_profile(stored_path)
            else:
                logger.warning(
                    "Stored browser profile not found at %s, "
                    "falling back to standard launch",
                    stored_path,
                )
                browser_profile = None

        # Use stored config if resuming, otherwise create new config
        if stored_browser_config:
            browser_config = stored_browser_config
            # Extract values from stored config
            browser_type = browser_config.get("browser_type", browser_type)
            headless = browser_config.get("headless", headless)
            viewport = browser_config.get("viewport", viewport)
            user_agent = browser_config.get("user_agent", user_agent)
            locale = browser_config.get("locale", locale)
            timezone_id = browser_config.get("timezone_id", timezone_id)
        else:
            # Create new browser configuration for persistence
            browser_config = {
                "browser_type": browser_type,
                "headless": headless,
                "viewport": viewport,
                "user_agent": user_agent,
                "locale": locale,
                "timezone_id": timezone_id,
            }
            if browser_profile is not None:
                browser_config["profile_path"] = str(
                    browser_profile.profile_dir
                )
                browser_config["browser_type"] = browser_profile.browser_type

        _engine, db = await cls._init_db(
            scraper,
            db_path,
            num_workers=num_workers,
            max_backoff_time=max_backoff_time,
            resume=resume,
            seed_params=seed_params,
            browser_config=browser_config,
        )

        # Initialize Playwright
        playwright = await async_playwright().start()
        try:
            browser_obj: Browser | None = None
            browser_context: BrowserContext

            if (
                browser_profile is not None
                and browser_profile.persistent_context
            ):
                # === Persistent context path (for Cloudflare bypass, etc.) ===
                browser_context = await _launch_persistent(
                    playwright,
                    scraper,
                    browser_profile,
                    headless,
                    proxy=proxy,
                )
            else:
                # === Standard path (existing behavior) ===
                effective_type = (
                    browser_profile.browser_type
                    if browser_profile is not None
                    else browser_type
                )
                browser_launcher = getattr(playwright, effective_type)

                launch_kwargs: dict[str, Any] = {"headless": headless}
                if browser_profile is not None:
                    launch_kwargs.update(browser_profile.launch_options)
                    if browser_profile.channel:
                        launch_kwargs["channel"] = browser_profile.channel
                if proxy:
                    launch_kwargs["proxy"] = _parse_proxy_for_playwright(proxy)

                browser_obj = await browser_launcher.launch(**launch_kwargs)

                context_kwargs: dict[str, Any] = {
                    "viewport": viewport,
                    "locale": locale,
                    "timezone_id": timezone_id,
                    "accept_downloads": True,
                }
                if user_agent:
                    context_kwargs["user_agent"] = user_agent
                if browser_profile is not None:
                    context_kwargs.update(browser_profile.context_options)

                browser_context = await browser_obj.new_context(
                    **context_kwargs
                )

                # Add init scripts (works for non-persistent profiles too)
                if browser_profile is not None:
                    for script_path in browser_profile.init_scripts:
                        js = script_path.read_text(encoding="utf-8")
                        await browser_context.add_init_script(js)

            driver: PlaywrightDriver[ScraperReturnDatatype] | None = None
            try:
                # Create driver instance (no request manager needed for Playwright)
                effective_rates = rates or scraper.rate_limits
                driver = cls(
                    scraper=scraper,
                    db=db,
                    browser_context=browser_context,
                    storage_dir=storage_dir,
                    num_workers=num_workers,
                    max_workers=max_workers,
                    resume=resume,
                    max_backoff_time=max_backoff_time,
                    request_manager=None,
                    enable_monitor=enable_monitor,
                    excluded_resource_types=excluded_resource_types,
                    rates=effective_rates,
                )
                driver._provided_preps = provided_preps

                # Store restart params for crash recovery (standard path only;
                # persistent contexts leave these as None).
                if not (
                    browser_profile is not None
                    and browser_profile.persistent_context
                ):
                    driver._playwright = playwright
                    driver._browser_obj = browser_obj
                    driver._browser_launcher = browser_launcher  # type: ignore[possibly-undefined]
                    driver._launch_kwargs = launch_kwargs  # type: ignore[possibly-undefined]
                    driver._context_kwargs = context_kwargs  # type: ignore[possibly-undefined]
                    driver._browser_profile = browser_profile

                # Restore cookies on resume
                if resume:
                    try:
                        cookies_json = await db.get_browser_cookies()
                        if cookies_json:
                            cookies = json.loads(cookies_json)
                            await browser_context.add_cookies(cookies)
                            logger.info(
                                f"Restored {len(cookies)} browser cookies from DB"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to restore browser cookies: {e}"
                        )

                yield driver

                # driver.close() handles DB engine disposal
                await driver.close()

            finally:
                # After a browser restart, driver.browser_context and
                # driver._browser_obj may differ from the originals.
                # Close whichever is current.
                ctx_to_close = (
                    driver.browser_context
                    if driver is not None
                    else browser_context
                )
                try:
                    await ctx_to_close.close()
                except Exception:
                    pass
                current_browser = (
                    driver._browser_obj if driver is not None else browser_obj
                )
                if current_browser is not None:
                    try:
                        await current_browser.close()
                    except Exception:
                        pass

        finally:
            await playwright.stop()

    async def _setup_tab_with_parent_response(
        self,
        page: Page,
        parent_request_id: int,
    ) -> bool:
        """Load parent's cached response into a new tab via route interception.

        Queries the parent's stored response from DB, sets up a route handler
        to serve the cached HTML, navigates to the response URL (which the
        handler intercepts), then removes the route so future navigations
        hit the real server.

        Args:
            page: The Playwright page to set up.
            parent_request_id: DB ID of the parent request.

        Returns:
            True if route interception succeeded, False if parent has no
            stored response.
        """
        from kent.driver.persistent_driver.compression import decompress

        parent_data = await self.db.get_parent_response_for_tab(
            parent_request_id
        )
        if parent_data is None:
            return False

        (
            response_url,
            content_compressed,
            compression_dict_id,
            response_headers_json,
            response_status_code,
        ) = parent_data

        if not response_url or not content_compressed:
            return False

        # Decompress content
        dictionary = None
        if compression_dict_id is not None:
            dictionary = await self.db.get_compression_dict(
                compression_dict_id
            )
        body = decompress(content_compressed, dictionary=dictionary)

        # Parse response headers
        headers: dict[str, str] = {}
        if response_headers_json:
            headers = json.loads(response_headers_json)
        # Ensure content-type is set for HTML
        if "content-type" not in {k.lower() for k in headers}:
            headers["content-type"] = "text/html; charset=utf-8"

        status = response_status_code or 200

        # Set up route to intercept the response_url
        async def _intercept_handler(route):
            await route.fulfill(
                status=status,
                headers=headers,
                body=body,
            )

        await page.route(response_url, _intercept_handler)

        # Navigate to the response URL — interceptor serves cached HTML
        await page.goto(response_url, wait_until="domcontentloaded")

        # Remove route so future navigations (via clicks/form submits) hit
        # the real server
        await page.unroute(response_url, _intercept_handler)

        return True

    async def _process_regular_request(
        self,
        request_id: int,
        request: BaseRequest,
        continuation_name: str,
        parent_request_id: int | None = None,
        worker_id: int = 0,
        archive_decision: Any = None,
    ) -> None:
        """Process a request using the worker's reusable Playwright page.

        Each worker owns a long-lived page that is reset between requests
        via ``about:blank`` navigation.  Per-request network state lives on
        the :class:`WorkerPage`, eliminating shared-state races.

        Args:
            request_id: Database ID of the request.
            request: The request to process.
            continuation_name: The continuation method name to invoke after navigation.
            parent_request_id: Parent request ID for tab route interception.
            worker_id: Identifier of the calling worker.
            archive_decision: Pre-computed ArchiveDecision from the worker loop.
        """
        wp = await self._acquire_worker_page(worker_id)
        try:
            await wp.reset_for_reuse()
        except Exception as e:
            # about:blank navigation can fail if the prior page's JS
            # triggered a navigation.  Clear state anyway and proceed —
            # the upcoming real navigation will replace the page content.
            logger.debug(f"reset_for_reuse failed for worker {worker_id}: {e}")
            wp.clear_request_state()
        wp.current_parent_request_id = request_id
        page = wp.page

        try:
            # Archive requests: click triggers a download, not a navigation
            is_archive = getattr(request, "archive", False)

            # Skipped archives (download=False) need no browser interaction,
            # so handle them before the via/parent_request_id guard.
            if (
                is_archive
                and archive_decision is not None
                and not archive_decision.download
            ):
                response: Response = ArchiveResponse(
                    status_code=200,
                    url=request.request.url,
                    content=b"",
                    text="",
                    headers={},
                    request=request,
                    file_url=archive_decision.file_url,
                )

                await self._complete_request(
                    request_id, response, request, continuation_name
                )
                return

            if is_archive and parent_request_id and request.via is not None:
                dedup_key = (
                    request.deduplication_key
                    if isinstance(request.deduplication_key, str)
                    else None
                )
                expected_type = getattr(request, "expected_type", None)

                # archive_decision is pre-computed by _db_worker for all
                # archive requests; it should never be None here.
                assert archive_decision is not None, (
                    f"archive_decision not set for archive request {request_id}"
                )
                decision = archive_decision

                if not decision.download:
                    response = ArchiveResponse(
                        status_code=200,
                        url=request.request.url,
                        content=b"",
                        text="",
                        headers={},
                        request=request,
                        file_url=decision.file_url,
                    )
                else:
                    success = await self._setup_tab_with_parent_response(
                        page, parent_request_id
                    )
                    if not success:
                        raise TransientException(
                            f"Archive request {request_id}: parent has no stored response"
                        )

                    download = await self._execute_via_download(request, page)

                    download_path = await download.path()
                    if download_path is None:
                        raise TransientException(
                            f"Archive request {request_id}: download produced no file"
                        )
                    suggested = download.suggested_filename
                    ext = Path(suggested).suffix if suggested else ""

                    if hasattr(self.archive_handler, "save_stream"):
                        # Playwright already placed the file at a unique
                        # path; reuse its stem for the archive URL suffix.
                        unique_filename = f"{download_path.stem}{ext}"
                        archive_url = (
                            f"{request.request.url}/{unique_filename}"
                        )

                        async def _iter_chunks(
                            path: Path = download_path,
                        ) -> AsyncIterator[bytes]:
                            with path.open("rb") as src:
                                while True:
                                    chunk = src.read(64 * 1024)
                                    if not chunk:
                                        break
                                    yield chunk

                        file_url = await self.archive_handler.save_stream(
                            url=archive_url,
                            deduplication_key=dedup_key,
                            expected_type=expected_type,
                            hash_header_value=None,
                            chunks=_iter_chunks(),
                        )

                        response = ArchiveResponse(
                            status_code=200,
                            url=download.url or request.request.url,
                            content=b"",
                            text="",
                            headers={},
                            request=request,
                            file_url=file_url,
                        )
                    else:
                        import hashlib as _hashlib

                        file_content = download_path.read_bytes()
                        content_hash = _hashlib.sha256(
                            file_content
                        ).hexdigest()
                        unique_filename = f"{content_hash}{ext}"
                        archive_url = (
                            f"{request.request.url}/{unique_filename}"
                        )

                        file_url = await self.archive_handler.save(
                            url=archive_url,
                            deduplication_key=dedup_key,
                            expected_type=expected_type,
                            hash_header_value=None,
                            content=file_content,
                        )

                        response = ArchiveResponse(
                            status_code=200,
                            url=download.url or request.request.url,
                            content=file_content,
                            text="",
                            headers={},
                            request=request,
                            file_url=file_url,
                        )

                await self._complete_request(
                    request_id, response, request, continuation_name
                )

            elif is_archive:
                # Bare archive Request — no Via, so there is no click target
                # in any parent page to drive a Playwright download event.
                # Issue the fetch through the BrowserContext's APIRequestContext
                # (cookies + proxy come along for the ride), buffer the body in
                # memory, then hand it to the archive_handler the same way as
                # the via-download branch above. Used for follow-up downloads
                # whose URL is computed at runtime (e.g. an ASX redirect
                # resolved to its underlying media file).
                response = await self._fetch_bare_archive_request(
                    request, archive_decision
                )

                await self._complete_request(
                    request_id, response, request, continuation_name
                )

            else:
                # Non-archive: navigate and snapshot HTML

                # Propagate Request.headers to the upcoming navigation via
                # set_extra_http_headers. We always call it (with the
                # request's headers, or {} for none) so headers from a
                # previous request on this reused worker page never leak
                # forward.
                await page.set_extra_http_headers(
                    request.request.headers or {}
                )

                # Navigate: route-intercept parent's cached page then via,
                # or navigate directly
                nav_error: (
                    HTMLStructuralAssumptionException
                    | TransientException
                    | None
                ) = None
                if parent_request_id and request.via is not None:
                    success = await self._setup_tab_with_parent_response(
                        page, parent_request_id
                    )
                    if success:
                        # Parent page is loaded from cache; execute via navigation
                        try:
                            await self._execute_via_navigation(request, page)
                        except HTMLStructuralAssumptionException as e:
                            nav_error = e
                        except PlaywrightError as e:
                            if "NS_ERROR_ABORT" in str(e):
                                nav_error = TransientException(
                                    f"Navigation aborted: {e}"
                                )
                                nav_error.__cause__ = e
                            else:
                                raise
                    else:
                        # Parent has no stored response — fall back to direct URL
                        await page.goto(
                            request.request.url, wait_until="domcontentloaded"
                        )
                elif request.via is not None:
                    # Has via but no parent (shouldn't normally happen) — direct
                    await page.goto(
                        request.request.url, wait_until="domcontentloaded"
                    )
                else:
                    # Entry point or direct URL (no via)
                    await page.goto(
                        request.request.url, wait_until="domcontentloaded"
                    )

                # Process await_list if continuation has one (skip on nav error).
                # When interstitial handlers are configured, race them against
                # the scraper's await_list; if an interstitial wins, navigate
                # through it then re-process the scraper's own conditions.
                await_list_error: (
                    PlaywrightTimeoutError | TransientException | None
                ) = None
                if continuation_name and nav_error is None:
                    continuation = getattr(
                        self.scraper, continuation_name, None
                    )
                    if continuation:
                        metadata = get_step_metadata(continuation)
                        if metadata and metadata.await_list:
                            try:
                                if self._interstitial_handlers:
                                    winner = await self._race_await_lists(
                                        page, metadata.await_list
                                    )
                                    if winner is not None:
                                        await winner.navigate_through(page)
                                        await self._process_await_list(
                                            page, metadata.await_list
                                        )
                                else:
                                    await self._process_await_list(
                                        page, metadata.await_list
                                    )
                            except (
                                PlaywrightTimeoutError,
                                TransientException,
                            ) as e:
                                await_list_error = e

                # Snapshot DOM (always — even on timeout, for debugging)
                html_content = await page.content()

                # Create Response object
                response = Response(
                    status_code=200,
                    url=page.url,
                    content=html_content.encode("utf-8"),
                    text=html_content,
                    headers={"content-type": "text/html; charset=utf-8"},
                    request=request,
                )

                # Store response with DOM snapshot
                await self._store_response(
                    request_id=request_id,
                    response=response,
                    continuation=continuation_name,
                    speculation_outcome=None,
                )

                # Store incidental requests from this worker's page.
                # Snapshot the list to avoid iterating a live list that
                # on_response callbacks may still be appending to.
                for incidental in list(wp.incidental_requests):
                    await self.db.insert_incidental_request(
                        parent_request_id=request_id, **incidental
                    )

                # Re-raise navigation error after storing response
                if nav_error is not None:
                    raise nav_error

                # Re-raise await_list timeout after storing response
                if await_list_error is not None:
                    raise await_list_error

                # Run continuation (with autowait if configured) and mark completed.
                # Response is already stored above (before incidentals/error checks),
                # so we pass store_response=False.
                await self._complete_request(
                    request_id,
                    response,
                    request,
                    continuation_name,
                    page=page,
                    store_response=False,
                )

        except PlaywrightTimeoutError as e:
            # Timeout waiting for selector/load state
            logger.warning(f"Playwright timeout for request {request_id}: {e}")
            raise TransientException(f"Playwright timeout: {e}") from e

        except PlaywrightError as e:
            if self._is_connection_dead(e):
                # Browser process crashed — evict the dead worker page
                # so the next retry gets a fresh one after browser restart.
                self._worker_pages.pop(worker_id, None)
                raise TransientException(
                    f"Browser connection lost: {e}"
                ) from e
            if "NS_ERROR_ABORT" in str(e):
                logger.warning(
                    f"Navigation aborted for request {request_id}: {e}"
                )
                raise TransientException(f"Navigation aborted: {e}") from e
            raise

        except HTMLStructuralAssumptionException:
            # Structural failure - will be handled by autowait if enabled
            raise

        except Exception as e:
            logger.error(
                f"Error processing Playwright request {request_id}: {e}",
                exc_info=True,
            )
            raise

    async def _wait_for_required_element(
        self,
        page: Page,
        selector: str,
        selector_type: str,
        request_url: str,
    ) -> ElementHandle:
        """Wait for a required selector, raising structurally on miss/timeout.

        Both a None result and a PlaywrightTimeoutError are converted to
        HTMLStructuralAssumptionException so callers can rely on a non-None
        return.
        """
        label = selector_type.capitalize()
        try:
            element = await page.wait_for_selector(selector, timeout=5000)
            if not element:
                raise HTMLStructuralAssumptionException(
                    selector=selector,
                    selector_type=selector_type,
                    description=f"{label} selector not found: {selector}",
                    expected_min=1,
                    expected_max=1,
                    actual_count=0,
                    request_url=request_url,
                )
        except PlaywrightTimeoutError as e:
            raise HTMLStructuralAssumptionException(
                selector=selector,
                selector_type=selector_type,
                description=f"{label} selector timeout: {selector}",
                expected_min=1,
                expected_max=1,
                actual_count=0,
                request_url=request_url,
            ) from e
        return element

    async def _fill_form_fields(
        self,
        form_element: ElementHandle,
        field_data: dict[str, Any],
    ) -> None:
        """Populate form fields by name based on tag/type/visibility."""
        for field_name, field_value in field_data.items():
            field_selector = f'[name="{field_name}"]'
            field_element = await form_element.query_selector(field_selector)
            if not field_element:
                continue
            tag = await field_element.evaluate(
                "el => el.tagName.toLowerCase()"
            )
            input_type = await field_element.get_attribute("type")
            is_visible = await field_element.is_visible()
            str_value = str(field_value)

            if tag == "select":
                await field_element.select_option(value=str_value)
            elif input_type == "radio":
                radio = await form_element.query_selector(
                    f'[name="{field_name}"][value="{str_value}"]'
                )
                if radio:
                    await radio.evaluate("(el) => el.checked = true")
            elif input_type == "checkbox":
                # When multiple checkboxes share a name, the value distinguishes
                # them — find the matching one (mirroring the radio path).
                checkbox = await form_element.query_selector(
                    f'[name="{field_name}"][value="{str_value}"]'
                )
                if checkbox:
                    await checkbox.evaluate("(el) => el.checked = true")
                else:
                    await field_element.evaluate(
                        "(el, val) => el.checked = !!val", str_value
                    )
            elif input_type in ("hidden", "submit") or not is_visible:
                # Invisible-but-not-type=hidden covers e.g. 1x1px
                # Telerik RadDatePicker parent inputs; fill() requires
                # visibility, so assign .value directly.
                await field_element.evaluate(
                    "(el, val) => el.value = val", str_value
                )
            else:
                await field_element.fill(str_value)

    async def _fetch_bare_archive_request(
        self,
        request: BaseRequest,
        archive_decision: Any,
    ) -> ArchiveResponse:
        """Fetch an archive Request that has no Via via APIRequestContext.

        The standard archive path drives a click in the parent page's
        DOM and captures the resulting download event. That requires a
        ViaFormSubmit/ViaLink whose selector resolves against the parent
        page. For follow-up downloads whose URL is only known after the
        parent page has been parsed (e.g. an ASX stub that resolves to
        an ``http://`` recording on a different host), no such anchor
        exists — so we issue the fetch through the BrowserContext's
        :class:`APIRequestContext` instead. The browser's cookies and
        any configured proxy still apply, but the response body is
        delivered in-process rather than as a Playwright download,
        which sidesteps the
        ``Page.goto: Download is starting`` failure mode when the
        server returns ``Content-Disposition: attachment``.

        Buffers the body in memory and hands it to the
        :attr:`archive_handler` exactly like the via-download branch
        does, so dedup keys, expected-type suffixing, and on-disk
        layout all match. Suitable for files up to a few hundred MB;
        Playwright's :class:`APIResponse` has no streaming API.
        """
        params = request.request
        method = params.method.value.upper()
        api_kwargs: dict[str, Any] = {}
        if params.headers:
            api_kwargs["headers"] = dict(params.headers)
        if params.params:
            api_kwargs["params"] = params.params
        if params.data is not None:
            api_kwargs["data"] = params.data
        if params.json is not None:
            api_kwargs["data"] = params.json
        if params.timeout is not None:
            api_kwargs["timeout"] = (
                float(params.timeout) * 1000.0
                if not isinstance(params.timeout, tuple)
                else float(params.timeout[1]) * 1000.0
            )
        if not params.allow_redirects:
            api_kwargs["max_redirects"] = 0

        request_context = self.browser_context.request
        api_response = await request_context.fetch(
            params.url, method=method, **api_kwargs
        )
        try:
            content = await api_response.body()
        finally:
            await api_response.dispose()

        dedup_key = (
            request.deduplication_key
            if isinstance(request.deduplication_key, str)
            else None
        )
        expected_type = getattr(request, "expected_type", None)

        if hasattr(self.archive_handler, "save_stream"):

            async def _iter_chunks(
                payload: bytes = content,
            ) -> AsyncIterator[bytes]:
                # APIResponse.body() is not streamable; yield in
                # 64KB slices so save_stream's hashing/writer doesn't
                # have to special-case a single giant chunk.
                view = memoryview(payload)
                step = 64 * 1024
                for i in range(0, len(view), step):
                    yield bytes(view[i : i + step])

            file_url = await self.archive_handler.save_stream(
                url=api_response.url or params.url,
                deduplication_key=dedup_key,
                expected_type=expected_type,
                hash_header_value=None,
                chunks=_iter_chunks(),
            )
        else:
            file_url = await self.archive_handler.save(
                url=api_response.url or params.url,
                deduplication_key=dedup_key,
                expected_type=expected_type,
                hash_header_value=None,
                content=content,
            )

        return ArchiveResponse(
            status_code=api_response.status,
            url=api_response.url or params.url,
            content=content,
            text="",
            headers=dict(api_response.headers),
            request=request,
            file_url=file_url,
        )

    async def _execute_via_download(
        self, request: BaseRequest, page: Page
    ) -> Any:
        """Execute a via action that triggers a file download instead of navigation.

        Similar to _execute_via_navigation but wraps the click in
        page.expect_download() instead of page.expect_navigation().

        Returns:
            The Playwright Download object.
        """
        if isinstance(request.via, ViaLink):
            # Link download: click the link, expect a download
            link_via = request.via
            link_element = await self._wait_for_required_element(
                page, link_via.selector, "link", request.request.url
            )

            async with page.expect_download() as download_info:
                await link_element.click()
            return await download_info.value

        elif isinstance(request.via, ViaFormSubmit):
            form_via = request.via

            form_element = await self._wait_for_required_element(
                page, form_via.form_selector, "form", request.request.url
            )

            await self._fill_form_fields(form_element, form_via.field_data)

            # Click submit button, expecting a download instead of navigation
            if form_via.submit_selector:
                submit_element = await form_element.query_selector(
                    form_via.submit_selector
                )
                if not submit_element:
                    raise HTMLStructuralAssumptionException(
                        selector=form_via.submit_selector,
                        selector_type="submit",
                        description=f"Submit selector not found: {form_via.submit_selector}",
                        expected_min=1,
                        expected_max=1,
                        actual_count=0,
                        request_url=request.request.url,
                    )
                async with page.expect_download() as download_info:
                    await submit_element.click()
                return await download_info.value
            else:
                # Fallback: click first submit element
                submit_element = await form_element.query_selector(
                    'button[type="submit"], input[type="submit"]'
                )
                if not submit_element:
                    raise HTMLStructuralAssumptionException(
                        selector=form_via.form_selector,
                        selector_type="form",
                        description="No submit button found in form for download",
                        expected_min=1,
                        expected_max=1,
                        actual_count=0,
                        request_url=request.request.url,
                    )
                async with page.expect_download() as download_info:
                    await submit_element.click()
                return await download_info.value

        else:
            raise ValueError(
                f"Archive download requires ViaLink or ViaFormSubmit, got {type(request.via)}"
            )

    async def _execute_via_navigation(
        self, request: BaseRequest, page: Page
    ) -> None:
        """Execute browser navigation based on via field.

        Args:
            request: The request with via field (ViaFormSubmit or ViaLink).
            page: The Playwright page to navigate on.

        Raises:
            HTMLStructuralAssumptionException: If selector doesn't match in live DOM.
        """

        if isinstance(request.via, ViaFormSubmit):
            # Form submission
            form_via = request.via

            # --- Phase 1: locate form and elements (structural) ---
            form_element = await self._wait_for_required_element(
                page, form_via.form_selector, "form", request.request.url
            )

            await self._fill_form_fields(form_element, form_via.field_data)

            # --- Phase 2: submit and navigate (transient on timeout) ---
            if form_via.submit_selector:
                submit_element = await form_element.query_selector(
                    form_via.submit_selector
                )
                if not submit_element:
                    raise HTMLStructuralAssumptionException(
                        selector=form_via.submit_selector,
                        selector_type="submit",
                        description=f"Submit selector not found: {form_via.submit_selector}",
                        expected_min=1,
                        expected_max=1,
                        actual_count=0,
                        request_url=request.request.url,
                    )
                # Wait for navigation after click
                async with page.expect_navigation():
                    await submit_element.click()
            elif "__EVENTTARGET" in form_via.field_data:
                # ASP.NET __doPostBack-style submission: submit the
                # form programmatically via JS.  This avoids clicking
                # a named submit button, which would cause ASP.NET
                # to handle the button-click event instead of the
                # __EVENTTARGET postback event.
                async with page.expect_navigation():
                    await form_element.evaluate("(form) => form.submit()")
            else:
                # Click first submit-type element
                submit_element = await form_element.query_selector(
                    'button[type="submit"], input[type="submit"]'
                )
                if not submit_element:
                    raise HTMLStructuralAssumptionException(
                        selector=form_via.form_selector,
                        selector_type="form",
                        description="No submit button found in form",
                        expected_min=1,
                        expected_max=1,
                        actual_count=0,
                        request_url=request.request.url,
                    )
                async with page.expect_navigation():
                    await submit_element.click()

            # Navigation timeouts from Phase 2 are NOT caught here;
            # they propagate to _process_regular_request where
            # PlaywrightTimeoutError is handled as a TransientException
            # and retried.

        elif isinstance(request.via, ViaLink):
            # Link navigation
            link_via = request.via

            # --- Phase 1: locate link element (structural) ---
            link_element = await self._wait_for_required_element(
                page, link_via.selector, "link", request.request.url
            )

            # --- Phase 2: click and navigate (transient on timeout) ---
            # Navigation timeouts propagate to _process_regular_request
            # where PlaywrightTimeoutError becomes TransientException.
            async with page.expect_navigation():
                await link_element.click()

        else:
            # Direct URL navigation (no via)
            await page.goto(request.request.url, wait_until="domcontentloaded")

    async def _process_await_list(
        self, page: Page, await_list: list[Any]
    ) -> None:
        """Process await_list wait conditions before taking DOM snapshot.

        Args:
            page: The Playwright page to wait on.
            await_list: List of wait condition objects.

        Raises:
            TransientException: If a wait condition times out.
        """
        for condition in await_list:
            try:
                if isinstance(condition, WaitForSelector):
                    await page.wait_for_selector(
                        condition.selector,
                        state=condition.state,  # type: ignore[arg-type]
                        timeout=condition.timeout,
                    )

                elif isinstance(condition, WaitForLoadState):
                    await page.wait_for_load_state(
                        condition.state,  # type: ignore[arg-type]
                        timeout=condition.timeout,
                    )

                elif isinstance(condition, WaitForURL):
                    await page.wait_for_url(
                        condition.url, timeout=condition.timeout
                    )

                elif isinstance(condition, WaitForTimeout):
                    await asyncio.sleep(condition.timeout / 1000.0)

                else:
                    logger.warning(
                        f"Unknown wait condition type: {type(condition)}"
                    )

            except PlaywrightTimeoutError as e:
                raise TransientException(
                    f"Wait condition timeout: {condition}"
                ) from e

    async def _race_await_lists(
        self,
        page: Page,
        scraper_await_list: list[Any],
    ) -> InterstitialHandler | None:
        """Race scraper waitlist against interstitial handler waitlists.

        Each group's conditions are awaited sequentially (conjunction).
        The first group to fully resolve wins; losing tasks are cancelled.

        Returns:
            The winning ``InterstitialHandler``, or ``None`` if the
            scraper's own await_list completed first.
        """

        async def _run_group(conditions: list[Any]) -> None:
            for condition in conditions:
                if isinstance(condition, WaitForSelector):
                    await page.wait_for_selector(
                        condition.selector,
                        state=condition.state,  # type: ignore[arg-type]
                        timeout=condition.timeout,
                    )
                elif isinstance(condition, WaitForLoadState):
                    await page.wait_for_load_state(
                        condition.state,  # type: ignore[arg-type]
                        timeout=condition.timeout,
                    )
                elif isinstance(condition, WaitForURL):
                    await page.wait_for_url(
                        condition.url, timeout=condition.timeout
                    )
                elif isinstance(condition, WaitForTimeout):
                    await asyncio.sleep(condition.timeout / 1000.0)

        scraper_task = asyncio.create_task(
            _run_group(scraper_await_list), name="scraper"
        )
        handler_tasks: dict[asyncio.Task[None], InterstitialHandler] = {}
        for handler in self._interstitial_handlers:
            task = asyncio.create_task(
                _run_group(handler.waitlist()),
                name=type(handler).__name__,
            )
            handler_tasks[task] = handler

        all_tasks = {scraper_task} | set(handler_tasks.keys())
        try:
            done, _pending = await asyncio.wait(
                all_tasks, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in all_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)

        winner = next(iter(done))
        winner.result()  # re-raise if the winner raised

        if winner is scraper_task:
            return None
        return handler_tasks[winner]

    async def _process_generator_with_autowait(
        self,
        continuation: Callable,
        response: Response,
        parent_request: BaseRequest,
        request_id: int,
        auto_await_timeout: int,
        page: Page | None = None,
        staged: StagedWrites | None = None,
    ) -> None:
        """Process generator with autowait retry logic.

        Args:
            continuation: The step function to invoke.
            response: The response to pass to the step function.
            parent_request: The parent request.
            request_id: The database request ID.
            auto_await_timeout: Timeout in milliseconds for autowait retries.
            page: The Playwright page for re-snapshot (optional for non-Playwright).
            staged: Buffer to collect deferred writes; reset on each retry so
                only the successful attempt's yields land at flush time.
        """
        assert page is not None, "Page must be provided for autowait"
        assert staged is not None, "StagedWrites buffer must be provided"
        start_time = time.time()
        timeout_seconds = auto_await_timeout / 1000.0

        while True:
            try:
                # Try to process the generator. Reset the staged buffer so a
                # failed prior attempt's partial yields are discarded.
                staged.results.clear()
                staged.estimates.clear()
                staged.requests.clear()
                staged.seen_dedup_keys.clear()
                gen = continuation(response)
                await self._process_generator_with_storage(
                    gen,
                    response,
                    parent_request,
                    continuation.__name__,
                    request_id,
                    staged,
                )
                # Success - exit retry loop
                break

            except HTMLStructuralAssumptionException as e:
                # Check if we've exhausted timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    logger.warning(
                        f"Autowait timeout exhausted ({auto_await_timeout}ms) for request {request_id}"
                    )
                    raise

                # Check if selector is Playwright-compatible
                if not self._is_playwright_compatible_selector(
                    e.selector, e.selector_type
                ):
                    logger.debug(
                        f"Selector not Playwright-compatible, skipping autowait: {e.selector}"
                    )
                    raise

                # Get observer from last execution (if available)
                metadata = get_step_metadata(continuation)
                observer = metadata.observer if metadata else None

                # Compose absolute selector
                if observer and e.selector:
                    absolute_selector = self._compose_absolute_selector(
                        e.selector, observer
                    )
                else:
                    absolute_selector = e.selector

                logger.info(
                    f"Autowait: waiting for selector {absolute_selector}"
                )

                # Wait for selector in live browser
                remaining_timeout = int(
                    (timeout_seconds - elapsed) * 1000
                )  # Convert to ms
                try:
                    await page.wait_for_selector(
                        absolute_selector, timeout=remaining_timeout
                    )
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"Autowait failed: selector {absolute_selector} not found within timeout"
                    )
                    raise e from None  # Re-raise original exception

                # Re-snapshot DOM
                html_content = await page.content()
                response = Response(
                    status_code=response.status_code,
                    url=page.url,
                    content=html_content.encode("utf-8"),
                    text=html_content,
                    headers=response.headers,
                    request=response.request,
                )

                # Update stored response
                await self._store_response(
                    request_id=request_id,
                    response=response,
                    continuation=continuation.__name__,
                    speculation_outcome=None,
                )

                # Retry step function with fresh DOM
                logger.info(
                    "Autowait: retrying step function with fresh DOM snapshot"
                )

    def _is_playwright_compatible_selector(
        self, selector: str, selector_type: str | None
    ) -> bool:
        """Check if selector is compatible with Playwright wait_for_selector.

        Args:
            selector: The XPath or CSS selector.
            selector_type: The type of selector (xpath, css, etc).

        Returns:
            True if compatible, False otherwise.
        """
        # Check for non-element XPath nodes
        if selector.endswith("/text()") or selector.endswith("/@"):
            return False

        # Check for EXSLT extensions
        if any(
            prefix in selector
            for prefix in ["re:", "str:", "math:", "set:", "dyn:"]
        ):
            return False

        # Check for XPath variables
        return "$" not in selector

    def _compose_absolute_selector(
        self, selector: str, observer: SelectorObserver
    ) -> str:
        """Compose absolute selector from relative selector and observer.

        Args:
            selector: The relative selector that failed.
            observer: The SelectorObserver with query tree.

        Returns:
            Absolute selector composed from parent chain.
        """
        # If already absolute, return as-is
        if selector.startswith("//") or selector.startswith("/"):
            return selector

        # Find the query in the observer's tree that matches this selector
        query = self._find_query_by_selector(observer.queries, selector)
        if not query:
            # No matching query found, return selector as-is
            return selector

        # Build path by walking up the parent chain
        path_parts: list[str] = []
        current: SelectorQuery | None = query
        while current:
            path_parts.append(current.selector)
            current = current.parent

        # Reverse to get root-to-leaf order
        path_parts.reverse()

        # Compose absolute XPath by joining parts
        # If the selector is relative (starts with .), we need to compose properly
        if path_parts and path_parts[0].startswith("//"):
            # First part is already absolute, join rest
            result = path_parts[0]
            for part in path_parts[1:]:
                if part.startswith(".//"):
                    # Descendant: replace .// with //
                    result = result + "//" + part[3:]
                elif part.startswith("./"):
                    # Child: replace ./ with /
                    result = result + "/" + part[2:]
                elif part.startswith("."):
                    # Self or relative
                    result = result + "/" + part[1:]
                else:
                    # Shouldn't happen but handle gracefully
                    result = result + "/" + part
            return result
        else:
            # Fallback: join with /
            return "/".join(path_parts)

    def _find_query_by_selector(
        self, queries: list, selector: str
    ) -> SelectorQuery | None:
        """Find a SelectorQuery in the tree matching the given selector.

        Args:
            queries: List of SelectorQuery objects to search.
            selector: The selector string to find.

        Returns:
            The matching SelectorQuery, or None if not found.
        """
        for query in queries:
            if query.selector == selector:
                return query
            # Recursively search children
            found = self._find_query_by_selector(query.children, selector)
            if found:
                return found
        return None

    async def close(self) -> None:
        """Close driver, save cookies, and cleanup resources."""
        # Close any remaining worker pages
        for wp in self._worker_pages.values():
            await wp.close()
        self._worker_pages.clear()

        # Save browser cookies for resume
        try:
            cookies = await self.browser_context.cookies()
            if cookies:
                await self.db.save_browser_cookies(json.dumps(cookies))
        except Exception as e:
            logger.warning(f"Failed to save browser cookies: {e}")

        # Call parent close to persist state
        await super().close()
