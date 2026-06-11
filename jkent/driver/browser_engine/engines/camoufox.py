"""Camoufox engine: wraps ``AsyncCamoufox`` for CF-bypass-grade stealth.

Camoufox is a custom Firefox build with anti-detection patches baked
into the binary (closed shadow-root handling, navigator/plugin shape,
WebGL/canvas/audio fingerprints, Marionette protocol cleanup, etc.).
It's driven via Playwright's Firefox API, so the yielded
``BrowserContext`` quacks like every other Playwright context.

Cloudflare's managed-challenge orchestrator passes against camoufox
where it stalls against patchright/playwright — see ``try_camoufox.py``
at the repo root for the proof-of-concept that drove this engine.
"""

from __future__ import annotations

import contextlib
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar, cast

from camoufox.async_api import AsyncCamoufox

from jkent.common.exceptions import TransientException
from jkent.driver.browser_engine.engines.base import (
    BrowserEngine,
    apply_init_scripts,
    parse_proxy_for_playwright,
    resolve_user_data_dir,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from playwright.async_api import BrowserContext

    from jkent.data_types import BaseScraper
    from jkent.driver.browser_engine.browser_profile import BrowserProfile

logger = logging.getLogger(__name__)


class CamoufoxEngine(BrowserEngine):
    """Engine wrapping ``camoufox.async_api.AsyncCamoufox``.

    Camoufox is Firefox-only and always launches in a persistent-context
    style.  As a consequence:

    - ``browser_profile.browser_type`` / ``channel`` are ignored (warn-log).
    - ``protocol_params`` is ignored (warn-log) — camoufox doesn't use
      Chromium's ``assistantMode`` / ``cdpPort`` hack.

    Profile-level ``camoufox_options`` (humanize, geoip, os, screen,
    fonts, block_images, block_webrtc, …) flow through to ``AsyncCamoufox``
    as kwargs.

    Restart is supported by tearing down the current ``AsyncCamoufox``
    context manager and entering a fresh one with the same kwargs —
    necessary because the Playwright driver's Node.js process
    occasionally crashes on Firefox page-error events (Playwright bug
    in ``pageError.location.url`` handling), and the persistent
    profile carries any ``cf_clearance`` cookies forward on restart.
    """

    engine_name: ClassVar[str] = "camoufox"

    def __init__(
        self,
        scraper: BaseScraper[Any],
        browser_profile: BrowserProfile | None = None,
        headless: bool = True,
        locale: str = "en-US",
        proxy: str | None = None,
        humanize: bool = True,
    ) -> None:
        self._scraper = scraper
        self._browser_profile = browser_profile
        self._headless = headless
        self._locale = locale
        self._proxy = proxy
        self._humanize = humanize
        self._browser_context: BrowserContext | None = None
        self._cm: Any | None = None  # AsyncCamoufox context manager handle
        self._launch_kwargs: dict[str, Any] = {}  # captured for restart

    @property
    def supports_restart(self) -> bool:
        return True

    def _build_launch_kwargs(self) -> dict[str, Any]:
        profile = self._browser_profile
        kwargs: dict[str, Any] = {
            "headless": self._headless,
            "humanize": self._humanize,
            "locale": self._locale,
            "persistent_context": True,
            "no_viewport": True,  # Fingerprinting-important
            # Force new-window/new-tab requests into the current tab so a
            # target=_blank link (or a scripted window.open) can't spawn an
            # orphan tab the driver never closes — the transport reuses one
            # page per worker and has no popup lifecycle. open_newwindow=1
            # routes to the current tab; restriction=0 applies that to
            # everything, including window.open with window features.
            "firefox_user_prefs": {
                "browser.link.open_newwindow": 1,
                "browser.link.open_newwindow.restriction": 0,
                # Firefox renders PDFs inline via its built-in pdf.js viewer,
                # which is a navigation, not a download — so an archive click
                # on a .PDF link never fires a `download` event and
                # `expect_download` times out even though the fetch succeeds.
                # Disable the viewer so PDFs download like any other file
                # (DOCX/DOC already do, having no inline viewer). Playwright's
                # accept_downloads then intercepts and saves the download.
                "pdfjs.disabled": True,
                "browser.download.open_pdf_attachments_inline": False,
            },
        }
        if profile is not None:
            # camoufox_options take precedence over our defaults so a
            # profile can override e.g. humanize=False.  Note: we
            # deliberately don't forward context_options.timezone_id
            # because AsyncCamoufox passes kwargs straight through to
            # Playwright's launch_persistent_context, which doesn't
            # accept a ``timezone`` kwarg — and ``geoip=True`` (set in
            # the camoufox profile) derives a matching timezone from
            # the IP automatically.
            # Merge firefox_user_prefs rather than let a profile's dict clobber
            # our tab-normalization prefs; the profile still wins per-key.
            profile_prefs = profile.camoufox_options.get("firefox_user_prefs")
            kwargs.update(profile.camoufox_options)
            if profile_prefs is not None:
                kwargs["firefox_user_prefs"] = {
                    **{
                        "browser.link.open_newwindow": 1,
                        "browser.link.open_newwindow.restriction": 0,
                        "pdfjs.disabled": True,
                        "browser.download.open_pdf_attachments_inline": False,
                    },
                    **profile_prefs,
                }
            user_data_dir = resolve_user_data_dir(self._scraper, profile.name)
        else:
            # Anonymous run — still need a user_data_dir for persistence.
            user_data_dir = resolve_user_data_dir(self._scraper, "camoufox")
        kwargs["user_data_dir"] = str(user_data_dir)
        if self._proxy:
            kwargs["proxy"] = parse_proxy_for_playwright(self._proxy)
        return kwargs

    async def _enter_new_context(self) -> BrowserContext:
        """Open a fresh ``AsyncCamoufox`` and return its context.

        Replays the captured ``_launch_kwargs`` and re-applies the
        profile's init scripts.  Used by both ``acquire()`` and
        ``restart_context()``.
        """
        self._cm = AsyncCamoufox(**self._launch_kwargs)
        # AsyncCamoufox is typed as yielding Browser | BrowserContext, but
        # persistent_context=True always yields a BrowserContext.
        browser_context = cast("BrowserContext", await self._cm.__aenter__())
        self._browser_context = browser_context
        await apply_init_scripts(browser_context, self._browser_profile)
        return browser_context

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[BrowserContext]:
        profile = self._browser_profile
        if profile is not None:
            if profile.protocol_params:
                logger.warning(
                    "CamoufoxEngine ignoring protocol_params %r — "
                    "Chromium-specific, no analogue in camoufox",
                    list(profile.protocol_params),
                )
            if profile.channel:
                logger.warning(
                    "CamoufoxEngine ignoring channel %r — camoufox bundles "
                    "its own Firefox binary",
                    profile.channel,
                )

        self._launch_kwargs = self._build_launch_kwargs()
        browser_context = await self._enter_new_context()
        try:
            yield browser_context
        finally:
            if self._cm is not None:
                with contextlib.suppress(Exception):
                    await self._cm.__aexit__(None, None, None)
            self._cm = None
            self._browser_context = None

    async def restart_context(self) -> BrowserContext:
        if not self._launch_kwargs:
            raise TransientException(
                "CamoufoxEngine.restart_context called before acquire()"
            )
        logger.warning(
            "Camoufox/Firefox process died — restarting from cached "
            "user_data_dir"
        )
        # Best-effort tear-down of the dead AsyncCamoufox.  The Node.js
        # driver process may already be gone; __aexit__ then raises,
        # which we swallow.
        if self._cm is not None:
            with contextlib.suppress(Exception):
                await self._cm.__aexit__(None, None, None)
            self._cm = None
        new_context = await self._enter_new_context()
        logger.info("Camoufox restarted successfully")
        return new_context
