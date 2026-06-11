"""Errors raised by the replay transport."""

from __future__ import annotations

from pathlib import Path


class ReplayScraperMismatchError(Exception):
    """A source DB was produced by a different scraper class than the one
    being replayed.

    Raised at driver startup, before any work is dispatched. The error names
    each offending DB and the scraper recorded in its ``run_metadata``.
    """

    def __init__(
        self,
        *,
        expected: str,
        mismatches: list[tuple[Path, str | None]],
    ) -> None:
        self.expected = expected
        self.mismatches = mismatches
        lines = [
            f"Source DB(s) produced by a different scraper than expected "
            f"({expected!r}):"
        ]
        for path, found in mismatches:
            lines.append(f"  {path}: scraper_name={found!r}")
        super().__init__("\n".join(lines))
