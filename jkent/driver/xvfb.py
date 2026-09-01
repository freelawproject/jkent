"""Per-browser Xvfb displays, so OS-level input can target one browser.

``CloudflareHandler`` clears a challenge by posting a real X11 pointer event
(``xdotool``), which is the only input Turnstile has accepted since 2026-08-09.
X has **one pointer per display** and, with no window manager under Xvfb, every
window is placed at 0,0 — so windows sharing a display overlap exactly and only
the topmost receives pointer input. Measured, not assumed — see ``cfhandler.md``.

This module gives each **browser** its own display, which isolates browsers from
each other and from anything else on the host's display (the container's shared
``:99``, other tooling, a second engine in the same process).

⚠️ **It does not make concurrent workers safe on its own.** The transport runs a
single browser and leases each worker its own page, and every Playwright page is
a separate OS window inside that one browser — 3 pages produce 3 X windows, all
at 0,0 and all the same size. Those windows share this display, so
``CloudflareHandler`` still has to ``bring_to_front()`` the page it means to
click and hold a lock across raise → locate → click. Per-display isolation is
about browsers; ``bring_to_front`` + the lock is about pages.

Why not camoufox's own ``headless="virtual"``:

* It spawns Xvfb with ``-screen 0 1x1x24``. On a 1x1 screen the pointer is
  clamped to 0,0 — ``xdotool mousemove 400 300`` then reports ``x:0 y:0`` — so
  no click can be placed.
* It kills the display from ``browser.close``, which breaks
  :meth:`CamoufoxEngine.restart_context`: a rolling restart closes the browser,
  the display dies with it, and the replacement has nowhere to map a window.
* Passing ``virtual_display=`` mutates ``os.environ["DISPLAY"]`` in this process
  (camoufox does ``env = environ`` then assigns into it), so with several
  browsers the last launch wins and nothing can tell which display is whose.

Hence: we own the Xvfb lifecycle, pass ``env=`` explicitly, and keep the
display→browser mapping in :data:`_displays` rather than in the environment.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from shutil import which
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

logger = logging.getLogger(__name__)

# Matches the container's screen. Large enough for camoufox's randomised window
# sizes (~1500x900 observed) with room to spare, and an unremarkable resolution
# — the display's dimensions are visible to the page as ``screen.width/height``.
DEFAULT_GEOMETRY = "1920x1080x24"

# Display numbers to hunt through. Starts well above :0 (a developer's real
# session) and above :99 (the container entrypoint's shared display), so an
# allocation never collides with either.
_SEARCH_START = 101
_SEARCH_END = 400

# Which display each browser context is on. Keyed weakly so a closed context
# does not pin its entry, and deliberately *not* stored in ``os.environ``:
# workers are asyncio tasks in one process, so the environment cannot describe
# more than one browser.
_displays: WeakKeyDictionary[BrowserContext, str] = WeakKeyDictionary()


def register(context: BrowserContext, display: str) -> None:
    """Record that ``context``'s browser is on ``display``."""
    _displays[context] = display


def display_for(context: BrowserContext | None) -> str | None:
    """The display ``context``'s browser is on, or ``None`` if unknown.

    ``None`` means "no private display was allocated for this browser" — the
    caller should fall back to ``$DISPLAY`` *and* to serialising its input,
    because on a shared display it cannot know its clicks are unambiguous.
    """
    if context is None:
        return None
    try:
        return _displays.get(context)
    except TypeError:
        # Some test doubles are unhashable; treat as "no private display".
        return None


def supported() -> bool:
    """Whether a private display can be allocated here."""
    return sys.platform.startswith("linux") and which("Xvfb") is not None


class XvfbDisplay:
    """One Xvfb process, owned by whoever started it.

    Lifetime is deliberately **wider than the browser's**: the engine starts
    this in ``acquire()`` and stops it when the context manager exits, so a
    rolling browser restart in between re-maps its window onto the same
    still-running display.
    """

    def __init__(self, geometry: str = DEFAULT_GEOMETRY) -> None:
        self._geometry = geometry
        self._proc: subprocess.Popen[bytes] | None = None
        self._display: str | None = None

    @property
    def display(self) -> str | None:
        return self._display

    def start(self) -> str:
        """Spawn Xvfb on a free display number and return it (e.g. ``":137"``).

        Blocking: spawns a process and waits for the server to accept
        connections. Call it off the event loop (``asyncio.to_thread``).

        Racy by nature — another process can take a display number between our
        check and our spawn — so a failed number is skipped rather than fatal.
        """
        if not supported():
            raise RuntimeError(
                "Xvfb is unavailable (needs Linux + Xvfb on PATH)"
            )
        for number in range(_SEARCH_START, _SEARCH_END):
            display = f":{number}"
            # Xvfb refuses to start on a number already locked; skipping the
            # obvious ones first keeps the common case to a single spawn.
            if Path(f"/tmp/.X{number}-lock").exists():  # noqa: S108 — X's own path
                continue
            proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                [
                    "Xvfb",
                    display,
                    "-screen",
                    "0",
                    self._geometry,
                    "-nolisten",
                    "tcp",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if self._wait_ready(proc, display):
                self._proc = proc
                self._display = display
                logger.info(
                    "Allocated virtual display %s (%s) for a camoufox browser",
                    display,
                    self._geometry,
                )
                return display
            proc.kill()
            proc.wait(timeout=5)
        raise RuntimeError(
            f"No free X display in :{_SEARCH_START}-:{_SEARCH_END}"
        )

    def _wait_ready(
        self,
        proc: subprocess.Popen[bytes],
        display: str,
        timeout_s: float = 10.0,
    ) -> bool:
        """Whether the server came up and answers queries."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False  # died — number was probably taken
            probe = subprocess.run(  # noqa: S603 — fixed argv, no shell
                ["xdpyinfo", "-display", display],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        """Terminate the server. Safe to call twice, and never raises."""
        proc, display = self._proc, self._display
        self._proc = None
        self._display = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 — teardown must not mask a real error
            logger.debug("Xvfb %s teardown failed", display, exc_info=True)
        else:
            logger.debug("Stopped virtual display %s", display)


def env_for(display: str) -> dict[str, str]:
    """A child environment pointed at ``display``.

    A copy — never a mutated ``os.environ``, which would redirect every later
    launch in this process and is the bug camoufox's ``virtual_display`` has.
    """
    return {**os.environ, "DISPLAY": display}
