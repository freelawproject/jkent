"""The requirement selection table and ``ResolvedRequirements``.

The table in ``requirements.py`` replaced three hand-maintained frozensets
(``BROWSER_REQUIREMENTS`` / ``CAMOUFOX_REQUIREMENTS`` / ``INTERSTITIAL_HANDLERS``)
that had to be edited together. Forgetting the ``BROWSER_REQUIREMENTS`` entry
for a new ``*_HANDLER`` failed *silently*: the scraper got an ``HttpxTransport``
and snapshotted the challenge page as content — the exact mode
``CloudflareHandler``'s docstring says must not happen.

These tests pin the two guards that make that impossible now (a missing row is
an import-time error, and a handler requirement that doesn't demand a browser
is unconstructable) plus the derivations every caller reads.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import BaseScraper, DriverRequirement
from jkent.driver.unified_driver.interstitials import (
    CloudflareHandler,
    interstitial_for,
)
from jkent.driver.unified_driver.requirements import (
    BROWSER_PRECEDENCE,
    BROWSERS,
    INTERSTITIAL_REQUIREMENTS,
    REQUIREMENTS,
    RequirementSpec,
    ResolvedRequirements,
    needs_browser,
)


def _scraper_with(*reqs: DriverRequirement) -> BaseScraper[Any]:
    class _Scraper(BaseScraper[dict[str, Any]]):
        driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

    return _Scraper()


class TestTableCompleteness:
    def test_every_selected_browser_is_known(self) -> None:
        selected = {
            s.browser for s in REQUIREMENTS.values() if s.browser is not None
        }
        assert selected <= set(BROWSERS)
        assert set(BROWSERS) == set(BROWSER_PRECEDENCE)

    def test_every_handler_requirement_demands_a_browser(self) -> None:
        """An interstitial handler only ever runs on a live browser page."""
        for req in INTERSTITIAL_REQUIREMENTS:
            assert REQUIREMENTS[req].needs_browser, req.name

    def test_interstitial_row_without_browser_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="live browser page"):
            RequirementSpec(interstitial=True)

    def test_browser_row_without_needs_browser_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not marked needs_browser"):
            RequirementSpec(browser="firefox")

    def test_factory_for_unregistered_requirement_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not declared interstitial"):
            interstitial_for(DriverRequirement.JS_EVAL)(CloudflareHandler)

    def test_duplicate_factory_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="already has an interstitial"):
            interstitial_for(DriverRequirement.CFCAP_HANDLER)(
                CloudflareHandler
            )


class TestResolution:
    def test_no_requirements_needs_no_browser(self) -> None:
        resolved = ResolvedRequirements.of(_scraper_with())
        assert resolved.needs_browser is False
        assert resolved.browser is None
        assert resolved.profile_name is None
        assert resolved.engine is None
        assert resolved.interstitials == ()

    def test_js_eval_needs_a_browser_without_selecting_one(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.JS_EVAL)
        )
        assert resolved.needs_browser is True
        assert resolved.browser is None
        assert resolved.engine_for(None) == ("playwright", "chromium")

    @pytest.mark.parametrize(
        ("requirement", "browser", "profile"),
        [
            (DriverRequirement.FF_ALIKE, "firefox", "firefox"),
            (DriverRequirement.CHROME_ALIKE, "chromium", "chrome"),
            (DriverRequirement.CFCAP_HANDLER, "camoufox", "camoufox"),
            (DriverRequirement.RCAP_HANDLER, "camoufox", "camoufox"),
        ],
    )
    def test_browser_and_profile(
        self,
        requirement: DriverRequirement,
        browser: str,
        profile: str,
    ) -> None:
        resolved = ResolvedRequirements.of(_scraper_with(requirement))
        assert resolved.browser == browser
        assert resolved.profile_name == profile

    @pytest.mark.parametrize(
        "handler",
        [DriverRequirement.RCAP_HANDLER, DriverRequirement.CFCAP_HANDLER],
    )
    def test_chrome_alike_clashes_with_camoufox_handlers(
        self, handler: DriverRequirement
    ) -> None:
        """Camoufox is a Firefox build: it cannot honor CHROME_ALIKE.

        Precedence used to pick camoufox and drop CHROME_ALIKE silently.
        """
        for declared in (
            (DriverRequirement.CHROME_ALIKE, handler),
            (handler, DriverRequirement.CHROME_ALIKE),
        ):
            with pytest.raises(ScraperConfigError, match="mutually exclusive"):
                ResolvedRequirements.of(_scraper_with(*declared))

    def test_precedence_is_order_independent(self) -> None:
        """Declaration order must not change the selected browser."""
        forward = ResolvedRequirements.of(
            _scraper_with(
                DriverRequirement.RCAP_HANDLER, DriverRequirement.FF_ALIKE
            )
        )
        reverse = ResolvedRequirements.of(
            _scraper_with(
                DriverRequirement.FF_ALIKE, DriverRequirement.RCAP_HANDLER
            )
        )
        assert forward.browser == reverse.browser == "camoufox"

    def test_interstitials_follow_declaration_order(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(
                DriverRequirement.CFCAP_HANDLER,
                DriverRequirement.JS_EVAL,
                DriverRequirement.RCAP_HANDLER,
            )
        )
        assert resolved.interstitials == (
            DriverRequirement.CFCAP_HANDLER,
            DriverRequirement.RCAP_HANDLER,
        )

    def test_mutually_exclusive_flavors_raise(self) -> None:
        with pytest.raises(ScraperConfigError, match="mutually exclusive"):
            ResolvedRequirements.of(
                _scraper_with(
                    DriverRequirement.FF_ALIKE, DriverRequirement.CHROME_ALIKE
                )
            )

    def test_exclusion_is_symmetric(self) -> None:
        """Only FF_ALIKE carries the row; CHROME_ALIKE must still clash."""
        assert (
            DriverRequirement.CHROME_ALIKE
            in REQUIREMENTS[DriverRequirement.FF_ALIKE].exclusive_with
        )
        assert not REQUIREMENTS[DriverRequirement.CHROME_ALIKE].exclusive_with
        with pytest.raises(ScraperConfigError):
            ResolvedRequirements.of(
                _scraper_with(
                    DriverRequirement.CHROME_ALIKE, DriverRequirement.FF_ALIKE
                )
            )

    def test_duplicates_are_tolerated(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.JS_EVAL, DriverRequirement.JS_EVAL)
        )
        assert resolved.requirements == {DriverRequirement.JS_EVAL}

    def test_membership_reads_like_the_raw_list(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.STRICTLY_SERIAL)
        )
        assert DriverRequirement.STRICTLY_SERIAL in resolved
        assert DriverRequirement.JS_EVAL not in resolved


class TestEngineFor:
    def test_explicit_camoufox_overrides_no_preference(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.JS_EVAL)
        )
        assert resolved.engine_for("camoufox") == ("camoufox", "camoufox")

    def test_selected_camoufox_ignores_an_explicit_flavor(self) -> None:
        """A captcha scraper cannot be talked out of camoufox."""
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.CFCAP_HANDLER)
        )
        assert resolved.engine_for("firefox") == ("camoufox", "camoufox")

    def test_explicit_flavor_wins_over_the_selected_one(self) -> None:
        resolved = ResolvedRequirements.of(
            _scraper_with(DriverRequirement.CHROME_ALIKE)
        )
        assert resolved.engine_for("firefox") == ("playwright", "firefox")


class TestLegacyReaders:
    """``needs_browser`` — what ``bootstrap`` asks to pick a transport."""

    @pytest.mark.parametrize("requirement", list(DriverRequirement))
    def test_needs_browser_matches_the_table(
        self, requirement: DriverRequirement
    ) -> None:
        scraper = _scraper_with(requirement)
        assert (
            needs_browser(scraper) == REQUIREMENTS[requirement].needs_browser
        )
