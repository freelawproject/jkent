"""What each engine hands its launcher: profile options, channel, proxy, dirs.

The Playwright driver and ``AsyncCamoufox`` are replaced by recorders, so no
browser is launched; the run's browser-data root is ``tmp_path``, so the
persistent user-data dir lands there.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from jkent.common.exceptions import TransientException, TransientKind
from jkent.data_types import BaseScraper
from jkent.driver.browser_engine.browser_profile import BrowserProfile
from jkent.driver.browser_engine.engines.camoufox import CamoufoxEngine
from jkent.driver.browser_engine.engines.playwright import PlaywrightEngine

_PROXY = "http://us%40er:pw@proxy.example:3128"
_PROXY_DICT = {
    "server": "http://proxy.example:3128",
    "username": "us@er",
    "password": "pw",
}


class _Scraper(BaseScraper[dict[str, Any]]):
    pass


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """The run's browser-data root (``<run db>.browser-data``)."""
    return tmp_path / "run.db.browser-data"


def _profile(**overrides: Any) -> BrowserProfile:
    fields: dict[str, Any] = {
        "profile_dir": Path("/nonexistent"),
        "schema_version": 1,
        "name": "ff",
        "description": "",
        "browser_type": "firefox",
        "channel": None,
        "persistent_context": False,
    }
    fields.update(overrides)
    return BrowserProfile(**fields)


class _Browser:
    def __init__(self) -> None:
        self.context_kwargs: dict[str, Any] | None = None

    async def new_context(self, **kwargs: Any) -> str:
        self.context_kwargs = kwargs
        return "context"


class _Launcher:
    def __init__(self) -> None:
        self.browser = _Browser()
        self.launch_kwargs: dict[str, Any] | None = None
        self.persistent: tuple[str, dict[str, Any]] | None = None

    async def launch(self, **kwargs: Any) -> _Browser:
        self.launch_kwargs = kwargs
        return self.browser

    async def launch_persistent_context(
        self, user_data_dir: str, **kwargs: Any
    ) -> _PersistentContext:
        self.persistent = (user_data_dir, kwargs)
        return _PersistentContext()


class _PersistentContext:
    async def close(self) -> None:
        return None


class _FakePlaywright:
    def __init__(self) -> None:
        self.chromium = _Launcher()
        self.firefox = _Launcher()
        self.webkit = _Launcher()


def _playwright_engine(
    **kwargs: Any,
) -> tuple[PlaywrightEngine, _FakePlaywright]:
    engine = PlaywrightEngine(_Scraper(), **kwargs)
    fake = _FakePlaywright()
    engine._playwright = fake  # type: ignore[assignment]
    return engine, fake


class TestPlaywrightStandardLaunch:
    async def test_no_profile_uses_constructor_settings(self) -> None:
        engine, fake = _playwright_engine(
            browser_type="webkit",
            headless=False,
            viewport={"width": 800, "height": 600},
            user_agent="UA/1",
            locale="fr-FR",
            timezone_id="Europe/Paris",
        )
        assert await engine._open_context() == "context"
        launcher = fake.webkit
        assert launcher.launch_kwargs == {"headless": False}
        assert launcher.browser.context_kwargs == {
            "viewport": {"width": 800, "height": 600},
            "locale": "fr-FR",
            "timezone_id": "Europe/Paris",
            "accept_downloads": True,
            "user_agent": "UA/1",
        }
        assert engine._browser_obj is launcher.browser

    async def test_user_agent_is_omitted_when_unset(self) -> None:
        engine, fake = _playwright_engine()
        await engine._open_context()
        assert "user_agent" not in fake.chromium.browser.context_kwargs  # type: ignore[operator]

    async def test_profile_options_channel_and_proxy_reach_the_launcher(
        self,
    ) -> None:
        profile = _profile(
            channel="chrome",
            launch_options={"slow_mo": 5, "headless": True},
            context_options={"locale": "de-DE", "color_scheme": "dark"},
        )
        engine, fake = _playwright_engine(
            browser_profile=profile,
            browser_type="chromium",
            headless=False,
            proxy=_PROXY,
        )
        await engine._open_context()
        launcher = fake.firefox  # the profile's browser_type wins
        assert fake.chromium.launch_kwargs is None
        assert launcher.launch_kwargs == {
            "headless": True,  # profile launch_options override
            "slow_mo": 5,
            "channel": "chrome",
            "proxy": _PROXY_DICT,
        }
        context_kwargs = launcher.browser.context_kwargs
        assert context_kwargs is not None
        assert context_kwargs["locale"] == "de-DE"
        assert context_kwargs["color_scheme"] == "dark"


class TestPlaywrightPersistentLaunch:
    async def test_kwargs_and_user_data_dir(self, root: Path) -> None:
        profile = _profile(
            name="ff-alike",
            persistent_context=True,
            channel="firefox-beta",
            launch_options={"slow_mo": 5},
            context_options={"locale": "de-DE"},
        )
        engine, fake = _playwright_engine(
            browser_profile=profile,
            headless=False,
            proxy=_PROXY,
            user_data_root=root,
        )
        assert isinstance(await engine._open_context(), _PersistentContext)
        assert fake.firefox.launch_kwargs is None
        assert fake.firefox.persistent is not None
        user_data_dir, kwargs = fake.firefox.persistent
        assert user_data_dir == str(root / "ff-alike")
        assert (root / "ff-alike").is_dir()
        assert kwargs == {
            "slow_mo": 5,
            "locale": "de-DE",
            "headless": False,
            "channel": "firefox-beta",
            "proxy": _PROXY_DICT,
        }

    async def test_no_channel_or_proxy_leaves_them_out(
        self, root: Path
    ) -> None:
        engine, fake = _playwright_engine(
            browser_profile=_profile(persistent_context=True),
            user_data_root=root,
        )
        await engine._open_context()
        assert fake.firefox.persistent is not None
        _, kwargs = fake.firefox.persistent
        assert kwargs == {"headless": True}

    def test_persistent_engine_refuses_restart(self) -> None:
        engine, _ = _playwright_engine(
            browser_profile=_profile(persistent_context=True)
        )
        engine._acquired = True
        with pytest.raises(TransientException, match="persistent") as exc:
            engine._check_restartable()
        assert exc.value.kind is TransientKind.BROWSER_CRASH

    def test_standard_engine_permits_restart(self) -> None:
        engine, _ = _playwright_engine(browser_profile=_profile())
        engine._acquired = True
        engine._check_restartable()


class TestPerRunUserDataDir:
    async def test_without_a_root_a_scratch_dir_lives_as_long_as_acquire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, fake = _playwright_engine(
            browser_profile=_profile(persistent_context=True)
        )
        monkeypatch.setattr(engine, "_session", nullcontext)
        async with engine.acquire():
            assert fake.firefox.persistent is not None
            user_data_dir = Path(fake.firefox.persistent[0])
            assert user_data_dir.is_dir()
        assert not user_data_dir.parent.exists()

    def test_two_engines_without_a_root_never_share_one(self) -> None:
        dirs = [
            CamoufoxEngine(_Scraper())._build_launch_kwargs()["user_data_dir"]
            for _ in range(2)
        ]
        assert dirs[0] != dirs[1]


class TestCamoufoxLaunchKwargs:
    def test_anonymous_run_gets_a_camoufox_user_data_dir(
        self, root: Path
    ) -> None:
        kwargs = CamoufoxEngine(
            _Scraper(), user_data_root=root
        )._build_launch_kwargs()
        assert kwargs["user_data_dir"] == str(root / "camoufox")
        # Created at launch, not while the kwargs are built: a launch that
        # never happens leaves nothing behind.
        assert not root.exists()
        assert kwargs["persistent_context"] is True
        assert "proxy" not in kwargs

    def test_profile_names_the_dir_and_options_override(
        self, root: Path
    ) -> None:
        profile = _profile(
            name="cf",
            camoufox_options={
                "humanize": False,
                "firefox_user_prefs": {"pdfjs.disabled": False, "x.y": 1},
            },
        )
        kwargs = CamoufoxEngine(
            _Scraper(),
            browser_profile=profile,
            proxy=_PROXY,
            user_data_root=root,
        )._build_launch_kwargs()
        assert kwargs["user_data_dir"] == str(root / "cf")
        assert kwargs["humanize"] is False
        assert kwargs["proxy"] == _PROXY_DICT
        prefs = kwargs["firefox_user_prefs"]
        assert prefs["pdfjs.disabled"] is False  # profile wins per-key
        assert prefs["x.y"] == 1
        assert prefs["browser.link.open_newwindow"] == 1  # defaults kept
