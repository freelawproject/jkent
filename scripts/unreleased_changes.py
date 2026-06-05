#!/usr/bin/env python
"""Generate the "## Coming up" section of CHANGES.md from git history.

Collects commits between the latest `v*.*.*` tag and HEAD (or the full
history if no release tag exists), groups them by commitizen type, and
prints (or applies) a fresh "## Coming up" section.

Defaults to printing the generated section to stdout for preview. Pass
`--apply` to replace the existing "## Coming up" section in CHANGES.md
in place.

Commit subjects are parsed as commitizen / conventional commits:

    type(scope)[!]?[:]? description

Mapping:
    feat                            -> Features
    fix                             -> Fixes
    refactor, perf, breaking (`!`)  -> Changes

Types like ci, docs, doc, test, style, build, chore are skipped
silently. Unknown types are reported to stderr.

Usage:
    uv run python scripts/unreleased_changes.py            # preview to stdout
    uv run python scripts/unreleased_changes.py --apply    # rewrite CHANGES.md
    uv run python scripts/unreleased_changes.py --since v0.10.0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGES = REPO_ROOT / "CHANGES.md"

COMMIT_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r":?\s+"
    r"(?P<msg>.+)$"
)

TYPE_TO_SECTION = {
    "feat": "Features",
    "fix": "Fixes",
    "refactor": "Changes",
    "perf": "Changes",
    "breaking": "Changes",
}

IGNORED_TYPES = {
    "ci",
    "docs",
    "doc",
    "test",
    "tests",
    "style",
    "build",
    "chore",
}

SECTION_ORDER = ["Features", "Changes", "Fixes"]


def run(*args: str) -> str:
    result = subprocess.run(
        args, check=True, capture_output=True, text=True
    )
    return (result.stdout or "").strip()


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def latest_release_tag() -> str | None:
    out = run("git", "tag", "--list", "v*.*.*", "--sort=-v:refname")
    return out.splitlines()[0] if out else None


def commits_since(rev: str | None) -> list[str]:
    rng = f"{rev}..HEAD" if rev else "HEAD"
    out = run("git", "log", "--no-merges", "--pretty=format:%s", rng)
    return [line for line in out.splitlines() if line.strip()]


def group_commits(
    subjects: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    groups: dict[str, list[str]] = {s: [] for s in SECTION_ORDER}
    unknown: list[str] = []
    for subj in subjects:
        m = COMMIT_RE.match(subj)
        if not m:
            unknown.append(subj)
            continue
        ctype = m.group("type")
        breaking = m.group("breaking") == "!"
        section = TYPE_TO_SECTION.get(ctype)
        if breaking:
            section = "Changes"
        if not section:
            if ctype not in IGNORED_TYPES:
                unknown.append(subj)
            continue
        scope = m.group("scope")
        msg = m.group("msg").strip()
        if msg:
            msg = msg[0].upper() + msg[1:]
        prefix = "**Breaking**: " if breaking else ""
        bullet = (
            f"- `{scope}`: {prefix}{msg}"
            if scope
            else f"- {prefix}{msg}"
        )
        groups[section].append(bullet)
    return groups, unknown


def render(groups: dict[str, list[str]]) -> str:
    lines = [
        "## Coming up",
        "",
        "The following changes are not yet released, but are code complete:",
        "",
    ]
    for i, section in enumerate(SECTION_ORDER):
        lines.append(f"{section}:")
        lines.extend(groups[section] if groups[section] else ["-"])
        if i < len(SECTION_ORDER) - 1:
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def apply_to_changes(new_section: str) -> None:
    """Replace the existing '## Coming up' section in CHANGES.md."""
    text = CHANGES.read_text()
    starts = [m.start() for m in re.finditer(r"^## ", text, re.MULTILINE)]
    if not starts or not text[starts[0] :].startswith("## Coming up"):
        die("CHANGES.md does not start with a '## Coming up' section")
    preamble = text[: starts[0]]
    next_start = starts[1] if len(starts) > 1 else len(text)
    rest = text[next_start:]
    CHANGES.write_text(preamble + new_section + "\n" + rest)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the '## Coming up' section of CHANGES.md from "
            "git history."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--since",
        help=(
            "git rev to use as the lower bound "
            "(default: latest v*.*.* tag)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "rewrite the '## Coming up' section of CHANGES.md "
            "(default: print to stdout)"
        ),
    )
    args = parser.parse_args()

    since = args.since or latest_release_tag()
    if since:
        print(f"Collecting commits since {since}", file=sys.stderr)
    else:
        print("No v*.*.* tag found; using full history", file=sys.stderr)

    subjects = commits_since(since)
    if not subjects:
        print("No commits found in range; nothing to do", file=sys.stderr)
        sys.exit(1)

    groups, unknown = group_commits(subjects)
    content = render(groups)

    if args.apply:
        apply_to_changes(content)
        total = sum(len(v) for v in groups.values())
        rel = CHANGES.relative_to(REPO_ROOT)
        print(
            f"Wrote {total} bullets into '## Coming up' in {rel}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(content)

    if unknown:
        print(
            f"\nSkipped {len(unknown)} commits with unrecognized type:",
            file=sys.stderr,
        )
        for subj in unknown:
            print(f"  {subj}", file=sys.stderr)


if __name__ == "__main__":
    main()
