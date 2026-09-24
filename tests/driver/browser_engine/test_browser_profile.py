"""``load_browser_profile`` — manifest validation.

A profile's ``name`` and its ``init_scripts`` both become filesystem paths
(``resolve_user_data_dir`` and ``add_init_script``), so both must stay inside
the directory they are meant for.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jkent.driver.browser_engine.browser_profile import load_browser_profile


def _write(profile_dir: Path, **manifest: Any) -> Path:
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "name": "p", **manifest})
    )
    return profile_dir


def test_valid_manifest_resolves_its_scripts(tmp_path: Path) -> None:
    profile_dir = _write(tmp_path / "p", init_scripts=["js/a.js"])
    (profile_dir / "js").mkdir()
    (profile_dir / "js" / "a.js").write_text("")

    profile = load_browser_profile(profile_dir)

    assert profile.name == "p"
    assert profile.browser_type == "chromium"
    assert profile.init_scripts == [(profile_dir / "js" / "a.js").resolve()]


@pytest.mark.parametrize(
    "name", ["../../shared", "ff/alike", "..", ".", "a\\b", "/abs"]
)
def test_name_must_be_one_path_component(tmp_path: Path, name: str) -> None:
    """``name`` is a directory under the scraper's cache: it cannot climb
    out of it or add levels to it."""
    with pytest.raises(ValueError, match="name"):
        load_browser_profile(_write(tmp_path / "p", name=name))


@pytest.mark.parametrize("script", ["../outside.js", "/etc/passwd"])
def test_script_outside_the_profile_is_rejected(
    tmp_path: Path, script: str
) -> None:
    (tmp_path / "outside.js").write_text("")
    with pytest.raises(ValueError, match="escapes"):
        load_browser_profile(_write(tmp_path / "p", init_scripts=[script]))


def test_script_symlinked_outside_the_profile_is_rejected(
    tmp_path: Path,
) -> None:
    (tmp_path / "outside.js").write_text("")
    profile_dir = _write(tmp_path / "p", init_scripts=["link.js"])
    (profile_dir / "link.js").symlink_to(tmp_path / "outside.js")
    with pytest.raises(ValueError, match="escapes"):
        load_browser_profile(profile_dir)


def test_sibling_directory_sharing_a_prefix_is_outside(
    tmp_path: Path,
) -> None:
    """``/x/p2/a.js`` is not inside ``/x/p`` just because the text starts
    with it."""
    (tmp_path / "p2").mkdir()
    (tmp_path / "p2" / "a.js").write_text("")
    with pytest.raises(ValueError, match="escapes"):
        load_browser_profile(
            _write(tmp_path / "p", init_scripts=["../p2/a.js"])
        )


@pytest.mark.parametrize(
    ("manifest", "match"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"name": ""}, "name"),
        ({"name": 3}, "name"),
        ({"browser_type": "netscape"}, "browser_type"),
        ({"init_scripts": "a.js"}, "init_scripts"),
        ({"init_scripts": [3]}, "string"),
        ({"protocol_params": {"cdpPort": "auto"}}, "protocol_params"),
    ],
)
def test_invalid_manifest_is_rejected(
    tmp_path: Path, manifest: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_browser_profile(_write(tmp_path / "p", **manifest))


def test_missing_script_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Init script"):
        load_browser_profile(_write(tmp_path / "p", init_scripts=["a.js"]))


def test_directory_without_manifest_is_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        load_browser_profile(tmp_path)
