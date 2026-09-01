"""Per-browser virtual displays: allocation, the registry, and restart survival.

The behaviour that matters here is lifetime. The display must outlive the
browser process, because ``CamoufoxEngine.restart_context`` replaces the browser
(and its ``BrowserContext``) mid-run: if the display died with the browser — as
camoufox's own ``headless="virtual"`` arranges, killing it from
``browser.close`` — the replacement would have nowhere to map a window, and
OS-level clicking would silently stop working after the first rolling restart.
"""

from __future__ import annotations

import gc
import os
import sys
import weakref
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import unified_driver first. jkent.driver.browser_engine's __init__ imports
# worker_page, which imports unified_driver.transport, which imports worker_page
# back — a pre-existing cycle that only resolves when unified_driver is entered
# first, which is what the rest of the suite happens to do. Without this line the
# file collects fine in a full run and fails when run on its own.
import jkent.driver.unified_driver  # noqa: F401
from jkent.driver import xvfb
from jkent.driver.browser_engine.engines.camoufox import CamoufoxEngine


class TestRegistry:
    def test_unregistered_context_has_no_display(self) -> None:
        assert xvfb.display_for(MagicMock()) is None

    def test_none_context_has_no_display(self) -> None:
        assert xvfb.display_for(None) is None

    def test_registered_context_reports_its_display(self) -> None:
        context = MagicMock()
        xvfb.register(context, ":142")
        assert xvfb.display_for(context) == ":142"

    def test_registry_does_not_keep_contexts_alive(self) -> None:
        """Weakly keyed: a finished run must not pin its context forever.

        Workers churn browsers for the life of a scrape, so a strong map here
        would be an unbounded leak.
        """
        context = MagicMock()
        ref = weakref.ref(context)
        xvfb.register(context, ":143")
        del context
        gc.collect()
        assert ref() is None

    def test_env_for_does_not_mutate_the_process(self) -> None:
        """The bug camoufox's virtual_display= has: it assigns into os.environ.

        With several browsers per process the last launch would win and nothing
        could tell which display belonged to which browser.
        """
        before = os.environ.get("DISPLAY")
        env = xvfb.env_for(":144")
        assert env["DISPLAY"] == ":144"
        assert os.environ.get("DISPLAY") == before


class TestSupported:
    def test_unsupported_off_linux(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        assert xvfb.supported() is False

    def test_unsupported_without_xvfb(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("jkent.driver.xvfb.which", lambda _: None)
        assert xvfb.supported() is False

    def test_supported_with_linux_and_xvfb(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(
            "jkent.driver.xvfb.which", lambda _: "/usr/bin/Xvfb"
        )
        assert xvfb.supported() is True


class TestWantsVirtualDisplay:
    @staticmethod
    def _engine(**kw: Any) -> CamoufoxEngine:
        return CamoufoxEngine(scraper=MagicMock(), **kw)

    def test_explicit_true_wins_over_everything(self, monkeypatch) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "0")
        assert self._engine(virtual_display=True)._wants_virtual_display()

    def test_explicit_false_wins_over_everything(self, monkeypatch) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "1")
        assert not self._engine(virtual_display=False)._wants_virtual_display()

    def test_env_opt_out(self, monkeypatch) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "0")
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert not self._engine(headless=False)._wants_virtual_display()

    def test_auto_off_when_headless(self, monkeypatch) -> None:
        """Headless browsers map no window, so there is nothing to click."""
        monkeypatch.delenv("JKENT_VIRTUAL_DISPLAY", raising=False)
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert not self._engine(headless=True)._wants_virtual_display()

    def test_auto_on_when_headed_and_supported(self, monkeypatch) -> None:
        monkeypatch.delenv("JKENT_VIRTUAL_DISPLAY", raising=False)
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert self._engine(headless=False)._wants_virtual_display()

    def test_auto_off_when_unsupported(self, monkeypatch) -> None:
        monkeypatch.delenv("JKENT_VIRTUAL_DISPLAY", raising=False)
        monkeypatch.setattr(xvfb, "supported", lambda: False)
        assert not self._engine(headless=False)._wants_virtual_display()


class _FakeDisplay:
    """Stand-in for XvfbDisplay that records its lifecycle."""

    def __init__(self, geometry: str = "") -> None:
        self.display: str | None = None
        self.started = 0
        self.stopped = 0

    def start(self) -> str:
        self.started += 1
        self.display = ":150"
        return self.display

    def stop(self) -> None:
        self.stopped += 1
        self.display = None


@pytest.mark.asyncio
class TestEngineDisplayLifecycle:
    """Where the display is created and destroyed, relative to the browser."""

    @staticmethod
    def _engine_with(fake: _FakeDisplay) -> CamoufoxEngine:
        engine = CamoufoxEngine(
            scraper=MagicMock(), headless=False, virtual_display=True
        )
        engine._display = None
        return engine

    async def test_display_injected_as_env_and_registered(
        self, monkeypatch, tmp_path
    ) -> None:
        fake = _FakeDisplay()
        monkeypatch.setattr(xvfb, "XvfbDisplay", lambda *a, **k: fake)
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.resolve_user_data_dir",
            lambda *a: tmp_path,
        )
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.apply_init_scripts",
            AsyncMock(),
        )
        context = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=context)
        cm.__aexit__ = AsyncMock(return_value=None)

        engine = self._engine_with(fake)
        with patch(
            "jkent.driver.browser_engine.engines.camoufox.AsyncCamoufox",
            return_value=cm,
        ):
            async with engine.acquire() as ctx:
                assert ctx is context
                # the browser was pointed at the private display...
                assert engine._launch_kwargs["env"]["DISPLAY"] == ":150"
                # ...and the handler can find it from the page's context
                assert xvfb.display_for(context) == ":150"
                assert fake.started == 1
                assert fake.stopped == 0
        assert fake.stopped == 1

    async def test_display_survives_a_rolling_restart(
        self, monkeypatch, tmp_path
    ) -> None:
        """The restart question: same display, and the NEW context registered.

        A restart yields a fresh BrowserContext object; if only the original
        were registered, every OS click after the first restart would fall back
        to the shared display and start clicking at the wrong window.
        """
        fake = _FakeDisplay()
        monkeypatch.setattr(xvfb, "XvfbDisplay", lambda *a, **k: fake)
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.resolve_user_data_dir",
            lambda *a: tmp_path,
        )
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.apply_init_scripts",
            AsyncMock(),
        )
        first, second = MagicMock(name="ctx1"), MagicMock(name="ctx2")
        contexts = [first, second]

        def _new_cm(**kwargs: Any) -> MagicMock:
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(side_effect=lambda: contexts.pop(0))
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        engine = self._engine_with(fake)
        with patch(
            "jkent.driver.browser_engine.engines.camoufox.AsyncCamoufox",
            side_effect=_new_cm,
        ):
            async with engine.acquire() as ctx:
                assert ctx is first
                restarted = await engine.restart_context()

                assert restarted is second
                # the Xvfb was never restarted underneath the browser
                assert (fake.started, fake.stopped) == (1, 0)
                # both the replayed launch and the new context agree on it
                assert engine._launch_kwargs["env"]["DISPLAY"] == ":150"
                assert xvfb.display_for(second) == ":150"
        assert fake.stopped == 1

    async def test_failure_to_allocate_degrades_instead_of_raising(
        self, monkeypatch, tmp_path
    ) -> None:
        """A run without a display still runs — it just cannot OS-click."""

        class _Boom:
            display = None

            def start(self) -> str:
                raise RuntimeError("no free display")

            def stop(self) -> None:  # pragma: no cover — never reached
                raise AssertionError("stop() on a display that never started")

        monkeypatch.setattr(xvfb, "XvfbDisplay", lambda *a, **k: _Boom())
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.resolve_user_data_dir",
            lambda *a: tmp_path,
        )
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.camoufox.apply_init_scripts",
            AsyncMock(),
        )
        context = MagicMock()
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=context)
        cm.__aexit__ = AsyncMock(return_value=None)

        engine = CamoufoxEngine(
            scraper=MagicMock(), headless=False, virtual_display=True
        )
        with patch(
            "jkent.driver.browser_engine.engines.camoufox.AsyncCamoufox",
            return_value=cm,
        ):
            async with engine.acquire() as ctx:
                assert ctx is context
                assert "env" not in engine._launch_kwargs
                assert xvfb.display_for(context) is None
