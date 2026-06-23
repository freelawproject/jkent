"""Requirement-driven engine selection in ``PlaywrightTransport``.

Mirrors the old driver/CLI behavior (REFACTOR_0.1.0.md Phase 0.3):
``CFCAP_HANDLER``, ``HCAP_HANDLER``, and ``RCAP_HANDLER`` select the camoufox
engine; otherwise ``FF_ALIKE`` maps to firefox and ``CHROME_ALIKE`` (or no flavor requirement) to
chromium. An explicit ``browser_type`` constructor arg wins over the
derivation, and declaring both flavors is a configuration error.

Engines are only constructed, never launched — no browser needed.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import BaseScraper, DriverRequirement
from jkent.driver.browser_engine.engines import (
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)


def _scraper_with(*reqs: DriverRequirement) -> BaseScraper:
    class _Scraper(BaseScraper[dict]):
        driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

    return _Scraper()


def _built_engine(scraper: BaseScraper, **kwargs: object):
    return PlaywrightTransport(scraper, **kwargs)._build_engine()  # type: ignore[arg-type]


class TestEngineSelection:
    def test_cfcap_selects_camoufox(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.CFCAP_HANDLER))
        assert isinstance(engine, CamoufoxEngine)

    def test_hcap_selects_camoufox(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.HCAP_HANDLER))
        assert isinstance(engine, CamoufoxEngine)

    def test_rcap_selects_camoufox(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.RCAP_HANDLER))
        assert isinstance(engine, CamoufoxEngine)

    def test_cfcap_wins_over_ff_alike(self) -> None:
        engine = _built_engine(
            _scraper_with(
                DriverRequirement.CFCAP_HANDLER, DriverRequirement.FF_ALIKE
            )
        )
        assert isinstance(engine, CamoufoxEngine)

    def test_ff_alike_selects_firefox(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.FF_ALIKE))
        assert isinstance(engine, PlaywrightEngine)
        assert engine._browser_type == "firefox"

    def test_chrome_alike_selects_chromium(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.CHROME_ALIKE))
        assert isinstance(engine, PlaywrightEngine)
        assert engine._browser_type == "chromium"

    def test_no_flavor_requirement_defaults_to_chromium(self) -> None:
        engine = _built_engine(_scraper_with(DriverRequirement.JS_EVAL))
        assert isinstance(engine, PlaywrightEngine)
        assert engine._browser_type == "chromium"

    def test_explicit_browser_type_wins_over_derivation(self) -> None:
        engine = _built_engine(
            _scraper_with(DriverRequirement.FF_ALIKE), browser_type="webkit"
        )
        assert isinstance(engine, PlaywrightEngine)
        assert engine._browser_type == "webkit"

    def test_both_flavors_is_a_config_error(self) -> None:
        with pytest.raises(ScraperConfigError, match="mutually exclusive"):
            _built_engine(
                _scraper_with(
                    DriverRequirement.FF_ALIKE, DriverRequirement.CHROME_ALIKE
                )
            )
