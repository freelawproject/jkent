"""Declarative browser profile for Playwright driver.

A browser profile is a directory containing a ``manifest.json`` and optional
JavaScript init scripts.  It configures how the Playwright driver launches the
browser — enabling persistent contexts, protocol-level params, and init
scripts that run before page JavaScript.

Security:
    - ``manifest.json`` is parsed with ``json.loads()`` — no code execution.
    - Init scripts run in the browser page context via Playwright's
      ``add_init_script()`` — they cannot access Python or the filesystem.
    - Protocol params are JSON primitives injected into Playwright protocol
      messages and validated server-side by Playwright's protocol validator.
    - Script paths are resolved and checked against the profile directory
      to prevent path traversal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_BROWSER_TYPES = frozenset({"chromium", "firefox", "webkit"})


@dataclass(frozen=True)
class BrowserProfile:
    """Immutable browser configuration loaded from a profile directory.

    Attributes:
        profile_dir: Resolved absolute path to the profile directory.
        schema_version: Manifest schema version (must be 1).
        name: Human-readable profile name.
        description: Optional description.
        browser_type: Playwright browser type.
        channel: Optional browser channel (e.g. ``"chrome"``).
        persistent_context: If True, use ``launch_persistent_context()``.
        launch_options: Options passed to ``launch()`` or
            ``launch_persistent_context()``.
        context_options: Options passed to ``new_context()`` (non-persistent)
            or merged into ``launch_persistent_context()`` (persistent).
        protocol_params: Params injected into the Playwright protocol message
            (e.g. ``assistantMode``, ``cdpPort``).  PlaywrightEngine only.
        camoufox_options: Camoufox-specific kwargs (e.g. ``humanize``,
            ``geoip``, ``os``, ``screen``, ``fonts``, ``block_images``,
            ``block_webrtc``).  CamoufoxEngine only.
        init_scripts: Resolved absolute paths to JS files loaded via
            ``context.add_init_script()``.
    """

    profile_dir: Path
    schema_version: int
    name: str
    description: str
    browser_type: str
    channel: str | None
    persistent_context: bool
    launch_options: dict[str, Any] = field(default_factory=dict)
    context_options: dict[str, Any] = field(default_factory=dict)
    protocol_params: dict[str, Any] = field(default_factory=dict)
    camoufox_options: dict[str, Any] = field(default_factory=dict)
    init_scripts: list[Path] = field(default_factory=list)


def load_browser_profile(profile_path: Path) -> BrowserProfile:
    """Load and validate a browser profile from a directory.

    Args:
        profile_path: Path to profile directory containing ``manifest.json``.

    Returns:
        Validated :class:`BrowserProfile` instance.

    Raises:
        FileNotFoundError: If the directory or ``manifest.json`` doesn't exist.
        ValueError: If the manifest is invalid or contains path traversal.
    """
    profile_dir = profile_path.resolve()
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile directory not found: {profile_path}")

    manifest_path = profile_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found in profile directory: {profile_dir}"
        )

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    # --- Validate required fields ---
    schema_version = raw.get("schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported schema_version: {schema_version} (expected 1)"
        )

    name = raw.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("manifest.json must have a non-empty 'name' string")

    browser_type = raw.get("browser_type", "chromium")
    if browser_type not in VALID_BROWSER_TYPES:
        raise ValueError(
            f"Invalid browser_type: {browser_type!r} "
            f"(expected one of {sorted(VALID_BROWSER_TYPES)})"
        )

    # --- Validate and resolve init script paths ---
    init_script_rels = raw.get("init_scripts", [])
    if not isinstance(init_script_rels, list):
        raise ValueError("init_scripts must be a list of relative paths")

    init_scripts: list[Path] = []
    for rel in init_script_rels:
        init_scripts.append(_validate_script_path(profile_dir, rel))

    return BrowserProfile(
        profile_dir=profile_dir,
        schema_version=schema_version,
        name=name,
        description=raw.get("description", ""),
        browser_type=browser_type,
        channel=raw.get("channel"),
        persistent_context=bool(raw.get("persistent_context", False)),
        launch_options=raw.get("launch_options", {}),
        context_options=raw.get("context_options", {}),
        protocol_params=raw.get("protocol_params", {}),
        camoufox_options=raw.get("camoufox_options", {}),
        init_scripts=init_scripts,
    )


def _validate_script_path(profile_dir: Path, script_rel: str) -> Path:
    """Validate that a script path stays inside the profile directory.

    Args:
        profile_dir: Resolved profile directory.
        script_rel: Relative path from the manifest.

    Returns:
        Resolved absolute path to the script file.

    Raises:
        ValueError: If the path escapes the profile directory.
        FileNotFoundError: If the script file doesn't exist.
    """
    if not isinstance(script_rel, str):
        raise ValueError(
            f"Init script path must be a string, got {type(script_rel)}"
        )

    resolved = (profile_dir / script_rel).resolve()
    if (
        not str(resolved).startswith(str(profile_dir) + "/")
        and resolved != profile_dir
    ):
        raise ValueError(
            f"Init script path escapes profile directory: {script_rel}"
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"Init script not found: {resolved}")

    return resolved
