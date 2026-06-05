#!/usr/bin/env python
"""Cut a kent release.

Verifies repo state, bumps the version in pyproject.toml, converts the
"## Coming up" section of CHANGES.md into a versioned `## X.Y.Z - DATE`
section, prepends a fresh empty "## Coming up" template above it, then
opens a `release/{version}` PR. The PyPI workflow tags + publishes once
the PR is merged.

Usage:
    uv run python scripts/release.py 0.11.0
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGES = REPO_ROOT / "CHANGES.md"

COMING_UP_TEMPLATE = """## Coming up

The following changes are not yet released, but are code complete:

Features:
-

Changes:
-

Fixes:
-
"""

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-.+][\w.-]+)?$")


def run(*args: str, capture: bool = True) -> str:
    result = subprocess.run(
        args, check=True, capture_output=capture, text=True
    )
    return (result.stdout or "").strip()


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def verify_git_state() -> None:
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        die(f"must be on main (currently on {branch!r})")
    if run("git", "status", "--porcelain"):
        die("working tree has uncommitted changes")
    run("git", "fetch", "origin", "main")
    local = run("git", "rev-parse", "main")
    remote = run("git", "rev-parse", "origin/main")
    if local != remote:
        die(
            f"local main ({local[:8]}) is not in sync with "
            f"origin/main ({remote[:8]})"
        )


def current_version() -> str:
    m = re.search(
        r'^version = "([^"]+)"',
        PYPROJECT.read_text(),
        re.MULTILINE,
    )
    if not m:
        die("could not find version in pyproject.toml")
    return m.group(1)


def update_pyproject(new_version: str) -> None:
    text = PYPROJECT.read_text()
    new = re.sub(
        r'^version = "[^"]+"',
        f'version = "{new_version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    PYPROJECT.write_text(new)


def split_sections(text: str) -> tuple[str, list[str]]:
    """Split CHANGES.md into (preamble, [section, section, ...]).

    Each section starts with a `## ` heading and runs to the next one
    (or end of file).
    """
    starts = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    if not starts:
        die("CHANGES.md has no '## ' sections")
    preamble = text[: starts[0]]
    sections = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        sections.append(text[start:end])
    return preamble, sections


def update_changes(new_version: str) -> None:
    text = CHANGES.read_text()
    preamble, sections = split_sections(text)

    if not sections[0].startswith("## Coming up"):
        die("first '## ' section in CHANGES.md must be '## Coming up'")
    coming_up = sections[0]

    bullets = [
        line
        for line in coming_up.splitlines()
        if line.startswith("- ") and line.strip() != "-"
    ]
    if not bullets:
        die("'## Coming up' section has no bullets; nothing to release")

    feat_m = re.search(r"^Features:", coming_up, re.MULTILINE)
    if not feat_m:
        die("'## Coming up' section missing 'Features:' subsection")
    release_body = coming_up[feat_m.start() :].rstrip()

    today = datetime.date.today().isoformat()
    versioned = f"## {new_version} - {today}\n\n{release_body}\n"

    fresh_coming_up = COMING_UP_TEMPLATE.rstrip() + "\n"
    rest = "".join(sections[1:])

    new_text = (
        preamble
        + fresh_coming_up
        + "\n"
        + versioned
        + "\n"
        + rest
    )
    CHANGES.write_text(new_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a kent release.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("version", help="new version, e.g. 0.11.0")
    args = parser.parse_args()

    if not VERSION_RE.match(args.version):
        die(f"invalid version: {args.version!r}")

    verify_git_state()

    old = current_version()
    new = args.version
    if old == new:
        die(f"version unchanged ({old})")
    print(f"Bumping {old} -> {new}")

    branch = f"release/{new}"
    run("git", "checkout", "-b", branch)
    update_pyproject(new)
    update_changes(new)
    run("uv", "sync")

    run(
        "git",
        "add",
        str(PYPROJECT.relative_to(REPO_ROOT)),
        str(CHANGES.relative_to(REPO_ROOT)),
        "uv.lock",
    )
    run(
        "git",
        "commit",
        "-m",
        f"release(v{new}) bump version and finalize changelog",
    )
    run("git", "push", "-u", "origin", branch)

    body = (
        f"Bumps kent from {old} to {new}.\n\n"
        "See `CHANGES.md` for what's in this release. Merging this PR "
        "will trigger the PyPI workflow to tag and publish the new "
        "version."
    )
    url = run(
        "gh",
        "pr",
        "create",
        "--title",
        f"release(v{new}) version bump",
        "--body",
        body,
        "--base",
        "main",
        "--head",
        branch,
    )
    print(url)
    if url.startswith("http"):
        subprocess.run(["open", url], check=False)


if __name__ == "__main__":
    main()
