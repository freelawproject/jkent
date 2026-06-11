"""Playwright engine: wraps the standard ``async_playwright()`` lifecycle.

Supports both the persistent-context launch path (used by FF_ALIKE /
CHROME_ALIKE profiles with cached user data) and the standard
``launch() + new_context()`` path (used by tests + scrapers without
a profile).  The standard path supports ``restart_context()``; the
persistent path does not (matches today's behaviour).
"""

from __future__ import annotations

import logging
import socket
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from playwright.async_api import (
    Browser,
    BrowserContext,
    async_playwright,
)

from jkent.common.exceptions import TransientException
from jkent.driver.browser_engine.engines.base import (
    BrowserEngine,
    parse_proxy_for_playwright,
    resolve_user_data_dir,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from jkent.data_types import BaseScraper
    from jkent.driver.browser_engine.browser_profile import BrowserProfile

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _inject_protocol_params(
    browser_type_impl: Any,
    protocol_params: dict[str, Any],
) -> None:
    """Monkey-patch a BrowserType to inject protocol params.

    Wraps ``browser_type_impl._channel.send_return_as_dict`` so that
    ``launchPersistentContext`` calls receive the extra params.  The
    special value ``cdpPort="auto"`` is replaced with an actual free
    TCP port number.
    """
    original_send = browser_type_impl._channel.send_return_as_dict

    async def _send_with_params(
        method, timeout_calc, params=None, is_internal=False, title=None
    ):
        if method == "launchPersistentContext" and params is not None:
            for key, value in protocol_params.items():
                if key == "cdpPort" and value == "auto":
                    params[key] = _find_free_port()
                else:
                    params[key] = value
        return await original_send(
            method, timeout_calc, params, is_internal, title
        )

    browser_type_impl._channel.send_return_as_dict = _send_with_params


class PlaywrightEngine(BrowserEngine):
    """Engine wrapping ``async_playwright().start()`` lifecycle."""

    engine_name: ClassVar[str] = "playwright"

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
    ) -> None:
        self._scraper = scraper
        self._browser_profile = browser_profile
        self._browser_type = browser_type
        self._headless = headless
        self._viewport = viewport
        self._user_agent = user_agent
        self._locale = locale
        self._timezone_id = timezone_id
        self._proxy = proxy
        # Runtime state — set during acquire(), consumed by restart_context().
        self._playwright: Any | None = None
        self._browser_obj: Browser | None = None
        self._browser_context: BrowserContext | None = None
        self._browser_launcher: Any | None = None
        self._launch_kwargs: dict[str, Any] = {}
        self._context_kwargs: dict[str, Any] = {}

    @property
    def supports_restart(self) -> bool:
        """Restart only works for the standard non-persistent path."""
        return not (
            self._browser_profile is not None
            and self._browser_profile.persistent_context
        )

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[BrowserContext]:
        playwright = await async_playwright().start()
        self._playwright = playwright
        try:
            if (
                self._browser_profile is not None
                and self._browser_profile.persistent_context
            ):
                self._browser_context = await self._launch_persistent()
            else:
                self._browser_context = await self._launch_standard()
            yield self._browser_context
        finally:
            # After a restart, _browser_context / _browser_obj may differ
            # from the originals — close whichever is current.
            if self._browser_context is not None:
                try:
                    await self._browser_context.close()
                except Exception:
                    pass
            if self._browser_obj is not None:
                try:
                    await self._browser_obj.close()
                except Exception:
                    pass
            try:
                await playwright.stop()
            except Exception:
                pass
            self._playwright = None
            self._browser_obj = None
            self._browser_context = None

    async def _launch_persistent(self) -> BrowserContext:
        assert self._browser_profile is not None
        profile = self._browser_profile
        browser_launcher = getattr(self._playwright, profile.browser_type)

        if profile.protocol_params:
            _inject_protocol_params(
                browser_launcher._impl_obj, profile.protocol_params
            )

        user_data_dir = resolve_user_data_dir(self._scraper, profile.name)

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

        context = await browser_launcher.launch_persistent_context(
            str(user_data_dir),
            **persistent_kwargs,
        )

        for script_path in profile.init_scripts:
            js = script_path.read_text(encoding="utf-8")
            await context.add_init_script(js)

        return context

    async def _launch_standard(self) -> BrowserContext:
        effective_type = (
            self._browser_profile.browser_type
            if self._browser_profile is not None
            else self._browser_type
        )
        browser_launcher = getattr(self._playwright, effective_type)
        self._browser_launcher = browser_launcher

        launch_kwargs: dict[str, Any] = {"headless": self._headless}
        if self._browser_profile is not None:
            launch_kwargs.update(self._browser_profile.launch_options)
            if self._browser_profile.channel:  # type: ignore
                launch_kwargs["channel"] = self._browser_profile.channel
        if self._proxy:
            launch_kwargs["proxy"] = parse_proxy_for_playwright(self._proxy)
        self._launch_kwargs = launch_kwargs

        browser_obj = await browser_launcher.launch(**launch_kwargs)
        self._browser_obj = browser_obj

        context_kwargs: dict[str, Any] = {
            "viewport": self._viewport,
            "locale": self._locale,
            "timezone_id": self._timezone_id,
            "accept_downloads": True,
        }
        if self._user_agent:
            context_kwargs["user_agent"] = self._user_agent
        if self._browser_profile is not None:
            context_kwargs.update(self._browser_profile.context_options)
        self._context_kwargs = context_kwargs

        browser_context = await browser_obj.new_context(**context_kwargs)

        if self._browser_profile is not None:
            for script_path in self._browser_profile.init_scripts:  # type: ignore
                js = script_path.read_text(encoding="utf-8")
                await browser_context.add_init_script(js)

        return browser_context

    async def restart_context(self) -> BrowserContext:
        """Restart the browser and rebuild the context after a crash.

        Raises ``TransientException`` for persistent-context profiles,
        which cannot be safely restarted.
        """
        if not self.supports_restart:
            raise TransientException(
                "Browser connection lost and restart is not available "
                "(persistent context)"
            )
        assert self._browser_launcher is not None
        logger.warning("Browser connection lost — restarting browser")

        if self._browser_obj is not None:
            try:
                await self._browser_obj.close()
            except Exception:
                pass

        new_browser = await self._browser_launcher.launch(
            **self._launch_kwargs
        )
        self._browser_obj = new_browser
        new_context = await new_browser.new_context(**self._context_kwargs)
        self._browser_context = new_context

        if self._browser_profile is not None:
            for script_path in self._browser_profile.init_scripts:  # type: ignore
                js = script_path.read_text(encoding="utf-8")
                await new_context.add_init_script(js)

        logger.info("Browser restarted successfully")
        return new_context
