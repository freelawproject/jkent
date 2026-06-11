"""Requirement-classification policy: whether a ``DriverRequirement`` set
demands a browser, and which browser it selects (:func:`select_browser`).

Pure policy derived from :class:`~jkent.data_types.DriverRequirement` — a leaf
module with no driver dependencies, so both the run bootstrapper (transport
and browser-profile selection) and the Playwright transport (engine
selection) can import it without a cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import DriverRequirement

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper

#: The browser a scraper's requirements select, or ``None`` for "no
#: preference" (a browser may still be needed — e.g. ``JS_EVAL`` — in which
#: case the engine default applies).
BrowserChoice = Literal["camoufox", "firefox", "chromium"]

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

#: Requirements that demand the camoufox engine. Camoufox is the stealthy
#: Firefox build that reliably passes Cloudflare, hCaptcha, *and* reCAPTCHA
#: challenges; ``CFCAP_HANDLER``, ``HCAP_HANDLER``, and ``RCAP_HANDLER``
#: scrapers all run on it (transport selection, engine build, and
#: browser-profile resolution all key off this).
CAMOUFOX_REQUIREMENTS = frozenset(
    {
        DriverRequirement.CFCAP_HANDLER,
        DriverRequirement.HCAP_HANDLER,
        DriverRequirement.RCAP_HANDLER,
    }
)


def needs_browser(scraper: BaseScraper[Any]) -> bool:
    """Whether the scraper's requirements demand a browser transport."""
    reqs = getattr(scraper, "driver_requirements", [])
    return any(r in BROWSER_REQUIREMENTS for r in reqs)


def select_browser(scraper: BaseScraper[Any]) -> BrowserChoice | None:
    """The single browser-selection precedence, from ``driver_requirements``.

    A camoufox requirement (``CFCAP_HANDLER``, ``HCAP_HANDLER``, or
    ``RCAP_HANDLER``) always wins — camoufox is the only engine that reliably
    passes CF managed challenges, hCaptcha, and reCAPTCHA — else ``FF_ALIKE``
    → firefox, else ``CHROME_ALIKE`` → chromium, else ``None`` (no
    preference). Every site that picks a transport class, browser engine, or
    browser profile derives it from this one answer.

    Raises :class:`ScraperConfigError` if the scraper declares both
    ``FF_ALIKE`` and ``CHROME_ALIKE``.
    """
    reqs = getattr(scraper, "driver_requirements", [])
    if (
        DriverRequirement.FF_ALIKE in reqs
        and DriverRequirement.CHROME_ALIKE in reqs
    ):
        raise ScraperConfigError(
            f"Scraper '{type(scraper).__name__}' declares both FF_ALIKE and "
            "CHROME_ALIKE driver requirements. These are mutually exclusive."
        )
    if any(r in CAMOUFOX_REQUIREMENTS for r in reqs):
        return "camoufox"
    if DriverRequirement.FF_ALIKE in reqs:
        return "firefox"
    if DriverRequirement.CHROME_ALIKE in reqs:
        return "chromium"
    return None
