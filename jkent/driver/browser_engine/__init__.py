"""Shared browser-engine layer for the unified driver's browser transports.

- ``engines`` -- the ``BrowserEngine`` interface and the ``CamoufoxEngine`` /
  ``PlaywrightEngine`` implementations (engine launch + lifecycle).
- ``browser_profile`` -- the ``BrowserProfile`` config.
- ``worker_page`` -- ``WorkerPage``, a Playwright page bound to one worker.

Consumers may import the submodules directly or the names re-exported here.
"""

from jkent.driver.browser_engine.browser_profile import BrowserProfile
from jkent.driver.browser_engine.engines import (
    BrowserEngine,
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.browser_engine.worker_page import WorkerPage

__all__ = [
    "BrowserEngine",
    "BrowserProfile",
    "CamoufoxEngine",
    "PlaywrightEngine",
    "WorkerPage",
]
