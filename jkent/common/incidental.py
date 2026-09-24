"""Incidental-request matching: promoting a captured browser sub-request.

A browser navigation fires sub-requests (XHR, fetch) the scraper may want as
a response of its own. :class:`IncidentalMatch` is the allowlist spec a
:class:`~jkent.data_types.Request` carries in ``incidental=`` to pick one
(:class:`Singular`) or every (:class:`Multiple`) matching capture.

A leaf: stdlib only.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse


def _json_deep_contains(actual: object, expected: object) -> bool:
    """True if ``expected`` is a structural subset of ``actual``.

    Dicts match when every expected key is present and its value deep-contains;
    lists match when every expected item is deep-contained by some actual item;
    scalars match on equality. Extra keys/items in ``actual`` are ignored — this
    is an allowlist ("match what I name, ignore the rest"), so volatile fields
    the caller doesn't mention (tokens, nonces) never break the match.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            k in actual and _json_deep_contains(actual[k], v)
            for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_json_deep_contains(a, e) for a in actual) for e in expected
        )
    return actual == expected


@dataclass(frozen=True)
class IncidentalMatch:
    """Allowlist spec for selecting a captured incidental request to promote.

    Attached to a :class:`Request` via ``incidental=`` (as one of the
    :class:`Singular` / :class:`Multiple` subclasses, which fix the match
    cardinality). The request does not navigate: the Playwright transport
    matches this spec against the sub-requests its *parent* navigation captured
    and promotes the matching response into this request's :class:`Response`.

    Matching is an **allowlist** — only the fields you set are checked, and
    everything unspecified (the dozens of browser-added headers, cache-busting
    params, volatile JWTs) is ignored. That is what lets a spec survive tokens
    it can't predict: you don't redact them, you simply never mention them.

    Attributes:
        url: fnmatch-style glob (case-sensitive, ``*`` spans ``/``) tested
            against the full captured URL, e.g. ``*getcasedetaildata*``.
        method: HTTP method, matched case-insensitively.
        resource_type: the browser's request-initiation kind, matched
            case-insensitively (e.g. ``fetch``/``xhr``/``document``). Use it to
            pick the JSON ``fetch`` when a page navigation and a client-side
            fetch hit the same URL (the document response would otherwise
            collide with the fetch response).
        query_contains: subset of query params that must be present with the
            given value (parsed from the captured URL).
        header_contains: subset of request headers that must be present with
            the given value; header names are matched case-insensitively.
        body_contains: for a captured request body — a ``str`` is a substring
            test against the decoded body; a ``dict`` is a deep-contains against
            the body parsed as JSON, falling back to form-encoded params.
    """

    url: str | None = None
    method: str | None = None
    resource_type: str | None = None
    query_contains: Mapping[str, str] | None = None
    header_contains: Mapping[str, str] | None = None
    body_contains: Mapping[str, Any] | str | None = None

    def __post_init__(self) -> None:
        if not any(
            (
                self.url,
                self.method,
                self.resource_type,
                self.query_contains,
                self.header_contains,
                self.body_contains,
            )
        ):
            raise ValueError(
                "IncidentalMatch requires at least one match field (url/method/"
                "resource_type/query_contains/header_contains/body_contains)."
            )

    def matches(
        self,
        *,
        url: str,
        method: str | None,
        headers: Mapping[str, str],
        body: bytes | None,
        resource_type: str | None = None,
    ) -> bool:
        """True if a captured incidental request satisfies every set field."""
        if self.url is not None and not fnmatch.fnmatchcase(url, self.url):
            return False
        if (
            self.method is not None
            and (method or "").upper() != self.method.upper()
        ):
            return False
        if (
            self.resource_type is not None
            and (resource_type or "").lower() != self.resource_type.lower()
        ):
            return False
        if self.query_contains and not self._query_matches(url):
            return False
        if self.header_contains and not self._headers_match(headers):
            return False
        return self.body_contains is None or self._body_matches(body)

    def _query_matches(self, url: str) -> bool:
        assert self.query_contains is not None
        params = parse_qs(urlparse(url).query)
        for key, value in self.query_contains.items():
            if value not in params.get(key, []):
                return False
        return True

    def _headers_match(self, headers: Mapping[str, str]) -> bool:
        assert self.header_contains is not None
        lowered = {k.lower(): v for k, v in headers.items()}
        for key, value in self.header_contains.items():
            if lowered.get(key.lower()) != value:
                return False
        return True

    def _body_matches(self, body: bytes | None) -> bool:
        contains = self.body_contains
        assert contains is not None
        if body is None:
            return False
        text = body.decode("utf-8", errors="replace")
        if isinstance(contains, str):
            return contains in text
        # dict: JSON deep-contains, then form-encoded subset as a fallback.
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and _json_deep_contains(parsed, dict(contains)):
            return True
        form = parse_qs(text)
        return all(
            value in form.get(key, []) for key, value in contains.items()
        )


@dataclass(frozen=True)
class Singular(IncidentalMatch):
    """Promote exactly one captured incidental.

    Zero matches or more than one match raises
    :class:`~jkent.common.exceptions.IncidentalRequestAssumptionException` —
    the spec is meant to pin down a single sub-request, so ambiguity is a
    structural assumption violation to fix, not a silent pick.
    """


@dataclass(frozen=True)
class Multiple(IncidentalMatch):
    """Promote every captured incidental that matches (one child each).

    The transport enqueues one promoted request per match and lets the
    continuation disambiguate by inspecting each response. Zero matches raises
    :class:`~jkent.common.exceptions.IncidentalRequestAssumptionException`.
    """
