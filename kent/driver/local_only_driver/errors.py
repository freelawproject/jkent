"""Errors raised by :mod:`kent.driver.local_only_driver`."""

from __future__ import annotations

from pathlib import Path


class LocalOnlyMiss(Exception):
    """A yielded request has no fulfillable match in the source index.

    Raised by :meth:`LocalOnlyDriver.resolve_request` when the configured
    miss policy is ``raise``. Aborts the replay run.

    Attributes:
        dedup_key: The deduplication key that wasn't found, or ``None`` if
            the request opted out of dedup checking.
        url: URL of the yielded request, for the operator's benefit.
    """

    def __init__(self, *, dedup_key: str | None, url: str) -> None:
        self.dedup_key = dedup_key
        self.url = url
        if dedup_key is None:
            msg = (
                f"LocalOnly miss: {url} (request opted out of deduplication; "
                "no key available for source lookup)"
            )
        else:
            msg = (
                f"LocalOnly miss: {url} (dedup_key={dedup_key[:16]}… "
                "not found in source index)"
            )
        super().__init__(msg)


class LocalOnlyScraperMismatchError(Exception):
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
