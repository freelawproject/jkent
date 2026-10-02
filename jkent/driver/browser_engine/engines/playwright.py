"""Playwright engine: wraps the standard ``async_playwright()`` lifecycle.

Supports both the persistent-context launch path (used by FF_ALIKE /
CHROME_ALIKE profiles with cached user data) and the standard
``launch() + new_context()`` path (used by tests + scrapers without
a profile).  The standard path supports ``restart_context()``; the
persistent path does not (matches today's behaviour).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from playwright.async_api import async_playwright
from typing_extensions import override

from jkent.common.exceptions import TransientException, TransientKind
from jkent.driver.browser_engine.engines.base import (
    BrowserEngine,
    close_quietly,
    parse_proxy_for_playwright,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from playwright.async_api import Browser, BrowserContext, Playwright

    from jkent.data_types import BaseScraper
    from jkent.driver.browser_engine.browser_profile import BrowserProfile


class PlaywrightEngine(BrowserEngine):
    """Engine wrapping ``async_playwright().start()`` lifecycle."""

    def __init__(
        self,
        scraper: BaseScraper[Any],
        browser_profile: BrowserProfile | None = None,
        browser_type: str = "chromium",
        headless: bool = True,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
        locale: str = "en-US",
        timezone_id: str = "America/New_York",
        proxy: str | None = None,
        user_data_root: Path | None = None,
    ) -> None:
        super().__init__(scraper, browser_profile, user_data_root)
        self._browser_type = browser_type
        self._headless = headless
        self._viewport = viewport
        self._user_agent = user_agent
        self._locale = locale
        self._timezone_id = timezone_id
        self._proxy = proxy
        # Runtime state, set inside acquire().
        self._playwright: Playwright | None = None
        self._browser_obj: Browser | None = None
        self._browser_context: BrowserContext | None = None

    @property
    def _persistent(self) -> bool:
        return (
            self._browser_profile is not None
            and self._browser_profile.persistent_context
        )

    @asynccontextmanager
    @override
    async def _session(self) -> AsyncIterator[None]:
        self._playwright = await async_playwright().start()
        try:
            yield
        finally:
            await close_quietly("playwright", self._playwright.stop)
            self._playwright = None

    async def _open_context(self) -> BrowserContext:
        if self._persistent:
            self._browser_context = await self._launch_persistent()
        else:
            self._browser_context = await self._launch_standard()
        return self._browser_context

    async def _close_context(self) -> None:
        # After a restart these are the rebuilt objects, not the originals.
        if self._browser_context is not None:
            await close_quietly("browser context", self._browser_context.close)
        if self._browser_obj is not None:
            await close_quietly("browser", self._browser_obj.close)
        self._browser_context = None
        self._browser_obj = None

    @override
    def _check_restartable(self) -> None:
        super()._check_restartable()
        if self._persistent:
            raise TransientException(
                "Browser connection lost and restart is not available "
                "(persistent context)",
                kind=TransientKind.BROWSER_CRASH,
            )

    async def _launch_persistent(self) -> BrowserContext:
        assert self._browser_profile is not None
        profile = self._browser_profile
        browser_launcher = getattr(self._playwright, profile.browser_type)

        user_data_dir = self._user_data_dir(profile.name)
        user_data_dir.mkdir(parents=True, exist_ok=True)

        persistent_kwargs: dict[str, Any] = {}
        persistent_kwargs.update(profile.launch_options)
        persistent_kwargs.update(profile.context_options)
        persistent_kwargs["headless"] = self._headless
        if profile.channel:
            persistent_kwargs["channel"] = profile.channel
        if self._proxy:
            persistent_kwargs["proxy"] = parse_proxy_for_playwright(
                self._proxy
            )

        context: BrowserContext = (
            await browser_launcher.launch_persistent_context(
                str(user_data_dir),
                **persistent_kwargs,
            )
        )
        return context

    async def _launch_standard(self) -> BrowserContext:
        profile = self._browser_profile
        effective_type = (
            profile.browser_type if profile is not None else self._browser_type
        )
        browser_launcher = getattr(self._playwright, effective_type)

        launch_kwargs: dict[str, Any] = {"headless": self._headless}
        if profile is not None:
            launch_kwargs.update(profile.launch_options)
            if profile.channel:
                launch_kwargs["channel"] = profile.channel
        if self._proxy:
            launch_kwargs["proxy"] = parse_proxy_for_playwright(self._proxy)

        browser: Browser = await browser_launcher.launch(**launch_kwargs)
        self._browser_obj = browser

        context_kwargs: dict[str, Any] = {
            "viewport": self._viewport,
            "locale": self._locale,
            "timezone_id": self._timezone_id,
            "accept_downloads": True,
        }
        if self._user_agent:
            context_kwargs["user_agent"] = self._user_agent
        if profile is not None:
            context_kwargs.update(profile.context_options)

        return await browser.new_context(**context_kwargs)
