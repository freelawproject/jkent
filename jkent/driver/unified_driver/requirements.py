"""Requirement-classification policy: the one table that turns a scraper's
``driver_requirements`` into every driver-selection answer.

:data:`REQUIREMENTS` holds one row per :class:`~jkent.data_types.DriverRequirement`
— does it need a browser, which browser does it select, does it contribute an
interstitial handler — and :data:`BROWSERS` holds one row per browser (profile
directory name, engine key). :class:`ResolvedRequirements` reads both and is
the single answer every caller consumes: transport selection, browser-profile
resolution, engine construction, and interstitial-handler wiring.

Adding a requirement is one row in :data:`REQUIREMENTS`; the table is checked
against the enum at import, so a new member with no row fails loudly instead
of silently classifying as "no browser needed".

Pure policy derived from :class:`~jkent.data_types.DriverRequirement` — a leaf
module with no driver dependencies, so both the run bootstrapper (transport
and browser-profile selection) and the Playwright transport (engine and
interstitial selection) can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import DriverRequirement

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from jkent.data_types import BaseScraper

#: The browser a scraper's requirements select, or ``None`` for "no
#: preference" (a browser may still be needed — e.g. ``JS_EVAL`` — in which
#: case the engine default applies).
BrowserChoice = Literal["camoufox", "firefox", "chromium"]


@dataclass(frozen=True)
class BrowserSpec:
    """What a :data:`BrowserChoice` implies downstream.

    Attributes:
        profile_name: Directory under ``$JKENT_HOME/profiles`` holding this
            browser's profile manifest (browser user data lives next to the
            run database, in ``<run db>.browser-data``).
        engine: Key the Playwright transport's engine table is indexed by.
            Camoufox is its own engine; the two plain Playwright flavors
            share one and are told apart by ``browser_type``.
    """

    profile_name: str
    engine: Literal["camoufox", "playwright"]


#: One row per browser. Camoufox is the stealthy Firefox build that reliably
#: passes Cloudflare, hCaptcha, *and* reCAPTCHA challenges — hence its place
#: at the top of :data:`BROWSER_PRECEDENCE`.
BROWSERS: dict[BrowserChoice, BrowserSpec] = {
    "camoufox": BrowserSpec(profile_name="camoufox", engine="camoufox"),
    "firefox": BrowserSpec(profile_name="firefox", engine="playwright"),
    "chromium": BrowserSpec(profile_name="chrome", engine="playwright"),
}

#: Browser-selection precedence, strongest first. A scraper whose
#: requirements name several browsers gets the earliest one in this order.
BROWSER_PRECEDENCE: tuple[BrowserChoice, ...] = (
    "camoufox",
    "firefox",
    "chromium",
)

#: The browser a scraper gets when it needs one but names none.
DEFAULT_BROWSER: BrowserChoice = "chromium"


@dataclass(frozen=True)
class RequirementSpec:
    """What one :class:`DriverRequirement` implies for driver selection.

    Attributes:
        needs_browser: The requirement cannot be served by plain HTTP.
        browser: The browser this requirement selects, or ``None`` for "any"
            — resolved against :data:`BROWSER_PRECEDENCE` when a scraper
            names more than one.
        interstitial: The requirement contributes an interstitial handler.
            :mod:`~jkent.driver.unified_driver.interstitials` registers a
            factory per such requirement and checks the two agree at import.
        exclusive_with: Requirements that cannot be declared alongside this
            one. Checked symmetrically, so only one side needs the entry.
    """

    needs_browser: bool = False
    browser: BrowserChoice | None = None
    interstitial: bool = False
    exclusive_with: frozenset[DriverRequirement] = frozenset()

    def __post_init__(self) -> None:
        if self.browser is not None and not self.needs_browser:
            raise ValueError(
                f"RequirementSpec selects browser {self.browser!r} but is "
                "not marked needs_browser"
            )
        if self.interstitial and not self.needs_browser:
            raise ValueError(
                "RequirementSpec contributes an interstitial handler but is "
                "not marked needs_browser: an interstitial handler only runs "
                "on a live browser page"
            )


#: The selection table: one row per :class:`DriverRequirement`, checked for
#: completeness against the enum at import (:func:`_verify_table`).
REQUIREMENTS: dict[DriverRequirement, RequirementSpec] = {
    DriverRequirement.JS_EVAL: RequirementSpec(needs_browser=True),
    DriverRequirement.FF_ALIKE: RequirementSpec(
        needs_browser=True,
        browser="firefox",
        exclusive_with=frozenset({DriverRequirement.CHROME_ALIKE}),
    ),
    DriverRequirement.CHROME_ALIKE: RequirementSpec(
        needs_browser=True, browser="chromium"
    ),
    # Camoufox is a Firefox build, so the handlers that select it cannot be
    # declared with CHROME_ALIKE; precedence would drop that one silently.
    DriverRequirement.RCAP_HANDLER: RequirementSpec(
        needs_browser=True,
        browser="camoufox",
        interstitial=True,
        exclusive_with=frozenset({DriverRequirement.CHROME_ALIKE}),
    ),
    DriverRequirement.CFCAP_HANDLER: RequirementSpec(
        needs_browser=True,
        browser="camoufox",
        interstitial=True,
        exclusive_with=frozenset({DriverRequirement.CHROME_ALIKE}),
    ),
    DriverRequirement.H11_HEADER_FIXES: RequirementSpec(),
    DriverRequirement.FOLLOW_REDIRECTS: RequirementSpec(),
    DriverRequirement.STRICTLY_SERIAL: RequirementSpec(needs_browser=True),
}


def _verify_table() -> None:
    """Every requirement has a row, and every row names a known browser."""
    missing = set(DriverRequirement) - set(REQUIREMENTS)
    if missing:
        names = ", ".join(sorted(r.name for r in missing))
        raise RuntimeError(
            f"DriverRequirement members with no REQUIREMENTS row: {names}. "
            "Add one — an absent row silently means 'plain HTTP is fine'."
        )
    for req, spec in REQUIREMENTS.items():
        if spec.browser is not None and spec.browser not in BROWSERS:
            raise RuntimeError(
                f"{req.name} selects unknown browser {spec.browser!r}"
            )
    for browser in BROWSERS:
        if browser not in BROWSER_PRECEDENCE:
            raise RuntimeError(
                f"Browser {browser!r} is missing from BROWSER_PRECEDENCE"
            )


_verify_table()

#: Requirements that contribute an interstitial handler. Derived, never
#: hand-written — :mod:`.interstitials` checks its factory table against it.
INTERSTITIAL_REQUIREMENTS: frozenset[DriverRequirement] = frozenset(
    req for req, spec in REQUIREMENTS.items() if spec.interstitial
)


@dataclass(frozen=True)
class ResolvedRequirements:
    """A scraper's requirement set, resolved into driver-selection answers.

    Built once per scraper by :meth:`of`, which validates the set (the
    mutually-exclusive pairs the enum docstring calls undefined behaviour are
    an error here, not a coin flip) and derives every downstream answer, so
    no caller re-derives one from the raw list.

    Attributes:
        requirements: The declared set, deduplicated.
        needs_browser: Whether a browser transport is required.
        browser: The selected browser, or ``None`` for no preference.
        profile_name: Profile directory name for :attr:`browser`, or ``None``.
        engine: Engine key for :attr:`browser`, or ``None``.
        interstitials: Requirements contributing an interstitial handler, in
            declaration order.
    """

    requirements: frozenset[DriverRequirement]
    needs_browser: bool
    browser: BrowserChoice | None
    profile_name: str | None
    engine: Literal["camoufox", "playwright"] | None
    interstitials: tuple[DriverRequirement, ...]

    @classmethod
    def of(cls, scraper: BaseScraper[Any] | None) -> ResolvedRequirements:
        """Resolve a scraper's declared requirements.

        Raises:
            ScraperConfigError: The set contains a mutually-exclusive pair
                (e.g. ``FF_ALIKE`` with ``CHROME_ALIKE``), or a requirement
                with no :data:`REQUIREMENTS` row.
        """
        declared: Iterable[DriverRequirement] = (
            getattr(scraper, "driver_requirements", None) or []
        )
        return cls.from_requirements(
            declared, source=type(scraper).__name__ if scraper else "scraper"
        )

    @classmethod
    def from_requirements(
        cls,
        declared: Iterable[DriverRequirement],
        *,
        source: str = "scraper",
    ) -> ResolvedRequirements:
        """Resolve a bare requirement iterable (the ``of`` core)."""
        ordered = list(dict.fromkeys(declared))
        unknown = [r for r in ordered if r not in REQUIREMENTS]
        if unknown:
            raise ScraperConfigError(
                f"'{source}' declares driver requirements with no selection "
                f"policy: {', '.join(str(r) for r in unknown)}"
            )
        declared_set = frozenset(ordered)
        for req in ordered:
            clash = REQUIREMENTS[req].exclusive_with & declared_set
            for other in clash:
                raise ScraperConfigError(
                    f"Scraper '{source}' declares both {req.name} and "
                    f"{other.name} driver requirements. These are mutually "
                    "exclusive."
                )

        specs = [REQUIREMENTS[r] for r in ordered]
        wanted: set[BrowserChoice] = {
            s.browser for s in specs if s.browser is not None
        }
        browser: BrowserChoice | None = next(
            (b for b in BROWSER_PRECEDENCE if b in wanted),
            None,
        )
        spec = BROWSERS[browser] if browser is not None else None
        return cls(
            requirements=declared_set,
            needs_browser=any(s.needs_browser for s in specs),
            browser=browser,
            profile_name=spec.profile_name if spec else None,
            engine=spec.engine if spec else None,
            interstitials=tuple(
                r for r in ordered if REQUIREMENTS[r].interstitial
            ),
        )

    def __contains__(self, req: object) -> bool:
        return req in self.requirements

    def __iter__(self) -> Iterator[DriverRequirement]:
        return iter(self.requirements)

    def engine_for(
        self, browser_type: str | None
    ) -> tuple[Literal["camoufox", "playwright"], str]:
        """The ``(engine, browser_type)`` pair to build for this scraper.

        ``browser_type`` is the transport's explicit constructor override; it
        wins over the requirement-selected browser, except that asking for
        camoufox by either route selects the camoufox engine. A scraper that
        needs a browser but names none gets :data:`DEFAULT_BROWSER`.
        """
        if self.engine == "camoufox" or browser_type == "camoufox":
            return "camoufox", "camoufox"
        return "playwright", browser_type or self.browser or DEFAULT_BROWSER


def needs_browser(scraper: BaseScraper[Any]) -> bool:
    """Whether the scraper's requirements demand a browser transport."""
    return ResolvedRequirements.of(scraper).needs_browser
