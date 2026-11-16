"""Browser engines for the Playwright driver.

Each engine encapsulates the engine-specific launch + lifecycle
behaviour behind a ``BrowserEngine`` interface.  The driver receives
a ``BrowserContext`` from the engine's ``acquire()`` and is otherwise
engine-agnostic.
"""

from jkent.driver.browser_engine.engines.base import BrowserEngine
from jkent.driver.browser_engine.engines.camoufox import CamoufoxEngine
from jkent.driver.browser_engine.engines.playwright import PlaywrightEngine

__all__ = ["BrowserEngine", "CamoufoxEngine", "PlaywrightEngine"]
