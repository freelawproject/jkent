"""Camoufox transport — a Playwright transport over a camoufox engine.

Camoufox is the engine that reliably passes Cloudflare's CAPTCHA, hCaptcha,
and reCAPTCHA, so ``CFCAP_HANDLER``, ``HCAP_HANDLER``, and ``RCAP_HANDLER``
scrapers run on it. It is a ``PlaywrightTransport`` in every
respect except the engine: a persistent-context Firefox that carries
``cf_clearance`` cookies forward across restarts. Its Firefox process
occasionally crashes on page-error events, surfacing as a ``Connection closed``
channel error — already recognized by the inherited ``should_restart``
predicate, so crash recovery (B3) needs no override. Launch + restart deltas
live in ``CamoufoxEngine``; this transport just selects it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jkent.driver.browser_engine.engines import CamoufoxEngine
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)

if TYPE_CHECKING:
    from jkent.driver.browser_engine.engines import BrowserEngine


class CamoufoxTransport(PlaywrightTransport):
    """A :class:`PlaywrightTransport` that always launches a camoufox engine.

    Unlike the parent (which picks camoufox only for ``CFCAP_HANDLER`` /
    ``HCAP_HANDLER`` / ``RCAP_HANDLER`` scrapers), choosing this class *is*
    the choice of engine — it always uses
    camoufox. Everything else (lifecycle, resolve, recovery, archive) is
    inherited unchanged.
    """

    def _build_engine(self) -> BrowserEngine:
        """Always a camoufox engine, regardless of the scraper's requirements."""
        return CamoufoxEngine(
            scraper=self._scraper,
            browser_profile=self._browser_profile,
            headless=self._headless,
            locale=self._locale,
            proxy=self._proxy,
            # Mouse humanization can stall clicks indefinitely; disable it so
            # via-navigation/download clicks don't hang.
            humanize=False,
        )
