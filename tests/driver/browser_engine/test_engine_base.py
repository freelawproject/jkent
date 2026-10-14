"""Shared engine helpers: proxy parsing, init-script injection, restart guard,
dead-connection recognition.

No browser is launched; contexts are fakes that record what they are handed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jkent.common.exceptions import TransientException
from jkent.driver.browser_engine.browser_profile import BrowserProfile
from jkent.driver.browser_engine.engines.base import (
    DEAD_CONNECTION_MESSAGES,
    BrowserEngine,
    apply_init_scripts,
    parse_proxy_for_playwright,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://proxy.example", {"server": "http://proxy.example"}),
        (
            "http://proxy.example:8080",
            {"server": "http://proxy.example:8080"},
        ),
        (
            "https://proxy.example:443",
            {"server": "https://proxy.example:443"},
        ),
        ("socks5://10.0.0.1", {"server": "socks5://10.0.0.1"}),
        (
            "socks5://10.0.0.1:1080",
            {"server": "socks5://10.0.0.1:1080"},
        ),
        (
            "http://alice:s3cret@proxy.example:3128",
            {
                "server": "http://proxy.example:3128",
                "username": "alice",
                "password": "s3cret",
            },
        ),
        (
            "http://us%40er:p%3Ass%2Fw@proxy.example:3128",
            {
                "server": "http://proxy.example:3128",
                "username": "us@er",
                "password": "p:ss/w",
            },
        ),
        (
            "http://alice@proxy.example",
            {"server": "http://proxy.example", "username": "alice"},
        ),
    ],
)
def test_proxy_url_becomes_playwright_dict(
    url: str, expected: dict[str, str]
) -> None:
    """Credentials leave the server URL and come back percent-decoded."""
    assert parse_proxy_for_playwright(url) == expected


@pytest.mark.parametrize(
    "url", ["proxy.example:8080", "//proxy.example:8080", "http://", ""]
)
def test_proxy_url_without_scheme_or_host_is_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="Invalid proxy URL"):
        parse_proxy_for_playwright(url)


class _RecordingContext:
    def __init__(self) -> None:
        self.scripts: list[str] = []

    async def add_init_script(self, script: str) -> None:
        self.scripts.append(script)


def _profile(**overrides: object) -> BrowserProfile:
    fields: dict[str, object] = {
        "profile_dir": Path("/nonexistent"),
        "schema_version": 1,
        "name": "p",
        "description": "",
        "browser_type": "chromium",
        "channel": None,
        "persistent_context": False,
    }
    fields.update(overrides)
    return BrowserProfile(**fields)  # type: ignore[arg-type]


async def test_no_profile_injects_nothing() -> None:
    context = _RecordingContext()
    await apply_init_scripts(context, None)  # type: ignore[arg-type]
    assert context.scripts == []


async def test_each_init_script_is_injected_in_order(tmp_path: Path) -> None:
    paths = []
    for name, body in [("b.js", "second()"), ("a.js", "first()")]:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    context = _RecordingContext()
    await apply_init_scripts(context, _profile(init_scripts=paths))  # type: ignore[arg-type]
    assert context.scripts == ["second()", "first()"]


class _NullEngine(BrowserEngine):
    async def _open_context(self):
        return MagicMock()

    async def _close_context(self) -> None:
        return None


async def test_restart_outside_acquire_is_a_programming_error() -> None:
    """Not a crash: nothing died, the caller has no browser to restart."""
    engine = _NullEngine(scraper=MagicMock())
    with pytest.raises(RuntimeError, match="outside acquire") as exc:
        await engine.restart_context()
    assert not isinstance(exc.value, TransientException)


async def test_restart_inside_acquire_is_allowed() -> None:
    engine = _NullEngine(scraper=MagicMock())
    async with engine.acquire():
        assert await engine.restart_context() is not None


@pytest.mark.parametrize("message", DEAD_CONNECTION_MESSAGES)
def test_should_restart_recognizes_each_dead_message(message: str) -> None:
    """Every dead-resource message, embedded in Playwright's text, is death."""
    assert BrowserEngine.should_restart(Exception(f"... {message} ..."))


def test_should_restart_rejects_other_errors() -> None:
    assert not BrowserEngine.should_restart(Exception("some other error"))


def test_content_process_crashes_count_as_dead() -> None:
    """A crashed page is NOT ``is_closed()``, so without these ``acquire``
    hands the wreck back and the damage only surfaces as a generic
    ``reset_for_reuse`` warning that reads like a navigation race — which is
    what made a suspected camoufox tab crash indistinguishable from an
    about:blank collision.
    """
    assert {"Page crashed", "Target crashed"} <= set(DEAD_CONNECTION_MESSAGES)
