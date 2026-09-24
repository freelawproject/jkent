"""``PlaywrightTransport._build_engine`` maps a choice to a constructor.

The choice itself is ``ResolvedRequirements.engine_for``, pinned for every
requirement combination in ``test_requirements_table.py``. What is left here
is the transport's own half: ``_ENGINE_BUILDERS`` has an entry for each answer
``engine_for`` can give, and the browser type it returns — including one the
constructor was handed rather than derived — reaches the engine.

Engines are only constructed, never launched — no browser needed.
"""

from __future__ import annotations

from typing import Any, ClassVar

from jkent.data_types import BaseScraper, DriverRequirement
from jkent.driver.browser_engine.engines import (
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)


def _scraper_with(*reqs: DriverRequirement) -> BaseScraper[dict[str, Any]]:
    class _Scraper(BaseScraper[dict[str, Any]]):
        driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

    return _Scraper()


def _built_engine(scraper: BaseScraper[dict[str, Any]], **kwargs: object):
    return PlaywrightTransport(scraper, **kwargs)._build_engine()  # type: ignore[arg-type]


def test_camoufox_choice_builds_a_camoufox_engine() -> None:
    engine = _built_engine(_scraper_with(DriverRequirement.CFCAP_HANDLER))
    assert isinstance(engine, CamoufoxEngine)


def test_playwright_choice_builds_an_engine_on_the_chosen_browser() -> None:
    """The constructor's ``browser_type`` travels through to the engine."""
    engine = _built_engine(
        _scraper_with(DriverRequirement.FF_ALIKE), browser_type="webkit"
    )
    assert isinstance(engine, PlaywrightEngine)
    assert engine._browser_type == "webkit"
