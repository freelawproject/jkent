"""Header merging shared by the request model and every transport."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["merge_headers"]


def merge_headers(
    defaults: Mapping[str, str], explicit: Mapping[str, str] | None
) -> dict[str, str]:
    """``defaults`` under ``explicit``, one value per header name.

    Header names are case-insensitive, within either side as well as across
    them: each name keeps its last value and that value's spelling, and an
    explicit header replaces a default of the same name. Exactly one value
    per name goes on the wire. The transports layer a scraper's
    ``default_headers`` under a request's headers with this, and
    :class:`~jkent.common.request.Request` layers ``permanent`` headers —
    parent under child, then under the request's own — with it too.
    """
    merged: dict[str, tuple[str, str]] = {}
    for side in (defaults, explicit or {}):
        for name, value in side.items():
            merged[name.lower()] = (name, value)
    return dict(merged.values())
