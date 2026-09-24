"""Per-browser virtual displays: allocation, the registry, and restart survival.

The behaviour that matters here is lifetime. The display must outlive the
browser process, because ``CamoufoxEngine.restart_context`` replaces the browser
(and its ``BrowserContext``) mid-run: if the display died with the browser — as
camoufox's own ``headless="virtual"`` arranges, killing it from
``browser.close`` — the replacement would have nowhere to map a window, and
OS-level clicking would silently stop working after the first rolling restart.
"""

from __future__ import annotations

import asyncio
import gc
import os
import subprocess
import sys
import threading
import weakref
from collections.abc import Callable
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typing_extensions import override

from jkent.driver import xvfb
from jkent.driver.browser_engine.engines.camoufox import CamoufoxEngine

_T = TypeVar("_T")


def _always(value: _T) -> Callable[..., _T]:
    """A stand-in callable that ignores its arguments and returns ``value``."""

    def make(*_a: Any, **_k: Any) -> _T:
        return value

    return make


class TestRegistry:
    def test_unregistered_context_has_no_display(self) -> None:
        assert xvfb.display_for(MagicMock()) is None

    def test_none_context_has_no_display(self) -> None:
        assert xvfb.display_for(None) is None

    def test_a_non_context_is_an_error_not_no_display(self) -> None:
        """A real BrowserContext is weakly hashable; anything else is a bug.

        Reporting "no private display" for it would silently send OS clicks to
        the shared ``$DISPLAY``.
        """
        with pytest.raises(TypeError):
            xvfb.display_for([])  # type: ignore[arg-type]

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
    def test_unsupported_off_linux(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "darwin")
        assert xvfb.supported() is False

    def test_unsupported_without_xvfb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("jkent.driver.xvfb.which", _always(None))
        assert xvfb.supported() is False

    def test_supported_with_linux_and_xvfb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "platform", "linux")

        def which(name: str) -> str:
            return f"/usr/bin/{name}"

        monkeypatch.setattr("jkent.driver.xvfb.which", which)
        assert xvfb.supported() is True

    def test_unsupported_without_xdpyinfo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Readiness is probed with xdpyinfo, a separate package."""
        monkeypatch.setattr(sys, "platform", "linux")

        def which(name: str) -> str | None:
            return "/usr/bin/Xvfb" if name == "Xvfb" else None

        monkeypatch.setattr("jkent.driver.xvfb.which", which)
        assert xvfb.supported() is False


def test_a_readiness_probe_that_raises_kills_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xvfb, "supported", lambda: True)
    monkeypatch.setattr(xvfb.Path, "exists", _always(False))
    proc = MagicMock()
    monkeypatch.setattr(xvfb.subprocess, "Popen", _always(proc))

    def probe_missing(*_a: Any, **_k: Any) -> Any:
        raise FileNotFoundError("xdpyinfo")

    monkeypatch.setattr(xvfb.XvfbDisplay, "_wait_ready", probe_missing)
    with pytest.raises(FileNotFoundError):
        xvfb.XvfbDisplay().start()
    proc.kill.assert_called_once()


def test_a_second_start_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overwriting the running server's handle would orphan it."""
    monkeypatch.setattr(xvfb, "supported", lambda: True)
    monkeypatch.setattr(xvfb.Path, "exists", _always(False))
    spawned: list[MagicMock] = []

    def popen(*_a: Any, **_k: Any) -> MagicMock:
        spawned.append(MagicMock())
        return spawned[-1]

    monkeypatch.setattr(xvfb.subprocess, "Popen", popen)
    monkeypatch.setattr(xvfb.XvfbDisplay, "_wait_ready", _always(True))
    display = xvfb.XvfbDisplay()
    display.start()
    with pytest.raises(RuntimeError, match="already started"):
        display.start()
    assert len(spawned) == 1


class TestWantsVirtualDisplay:
    @staticmethod
    def _engine(**kw: Any) -> CamoufoxEngine:
        return CamoufoxEngine(scraper=MagicMock(), **kw)

    def test_explicit_true_wins_over_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "0")
        assert self._engine(virtual_display=True)._wants_virtual_display()

    def test_explicit_false_wins_over_everything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "1")
        assert not self._engine(virtual_display=False)._wants_virtual_display()

    def test_env_opt_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JKENT_VIRTUAL_DISPLAY", "0")
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert not self._engine(headless=False)._wants_virtual_display()

    def test_auto_off_when_headless(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Headless browsers map no window, so there is nothing to click."""
        monkeypatch.delenv("JKENT_VIRTUAL_DISPLAY", raising=False)
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert not self._engine(headless=True)._wants_virtual_display()

    def test_auto_on_when_headed_and_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("JKENT_VIRTUAL_DISPLAY", raising=False)
        monkeypatch.setattr(xvfb, "supported", lambda: True)
        assert self._engine(headless=False)._wants_virtual_display()

    def test_auto_off_when_unsupported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDisplay()
        monkeypatch.setattr(xvfb, "XvfbDisplay", _always(fake))
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.base.apply_init_scripts",
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

    async def test_a_bad_launch_config_still_stops_the_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A proxy URL that won't parse must not orphan an X server."""
        fake = _FakeDisplay()
        monkeypatch.setattr(xvfb, "XvfbDisplay", _always(fake))
        engine = self._engine_with(fake)

        def bad_kwargs() -> dict[str, Any]:
            raise ValueError("unparseable proxy")

        monkeypatch.setattr(engine, "_build_launch_kwargs", bad_kwargs)
        with pytest.raises(ValueError, match="unparseable proxy"):
            async with engine.acquire():
                pass
        assert (fake.started, fake.stopped) == (1, 1)

    async def test_display_survives_a_rolling_restart(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The restart question: same display, and the NEW context registered.

        A restart yields a fresh BrowserContext object; if only the original
        were registered, every OS click after the first restart would fall back
        to the shared display and start clicking at the wrong window.
        """
        fake = _FakeDisplay()
        monkeypatch.setattr(xvfb, "XvfbDisplay", _always(fake))
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.base.apply_init_scripts",
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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run without a display still runs — it just cannot OS-click."""

        class _Boom:
            display: str | None = None
            stopped = 0

            def start(self) -> str:
                raise RuntimeError("no free display")

            def stop(self) -> None:
                # A start() that fails part-way may leave a server behind;
                # stop() is idempotent, so the engine always calls it.
                self.stopped += 1

        boom = _Boom()
        monkeypatch.setattr(xvfb, "XvfbDisplay", _always(boom))
        monkeypatch.setattr(
            "jkent.driver.browser_engine.engines.base.apply_init_scripts",
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
        assert boom.stopped == 1

    async def test_cancel_during_start_still_stops_the_display(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling acquire mid-start must not orphan the X server.

        The thread running start() cannot be interrupted, so it finishes after
        the task is gone; the server it brought up still has to be stopped.
        """
        entered, release = threading.Event(), threading.Event()

        class _Slow(_FakeDisplay):
            @override
            def start(self) -> str:
                entered.set()
                release.wait(timeout=5)
                return super().start()

        fake = _Slow()
        monkeypatch.setattr(xvfb, "XvfbDisplay", _always(fake))
        engine = self._engine_with(fake)

        async def _enter() -> None:
            async with engine.acquire():
                pass  # pragma: no cover — cancelled before the body

        task = asyncio.create_task(_enter())
        await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(200):
            if fake.stopped:
                break
            await asyncio.sleep(0.01)
        assert (fake.started, fake.stopped) == (1, 1)


@pytest.mark.skipif(
    not xvfb.supported(), reason="needs Linux + Xvfb + xdpyinfo"
)
def test_real_display_starts_answers_and_stops() -> None:
    """A real server comes up on the returned display and is gone after stop."""
    display = xvfb.XvfbDisplay()
    name = display.start()
    proc = display._proc
    assert proc is not None
    try:
        assert name.startswith(":")
        allocated: str | None = display.display
        assert allocated == name
        probe = subprocess.run(
            ["xdpyinfo", "-display", name],
            capture_output=True,
            check=False,
        )
        assert probe.returncode == 0
        assert b"1920x1080" in probe.stdout
    finally:
        display.stop()
    assert proc.poll() is not None
    assert display.display is None
    display.stop()  # idempotent
