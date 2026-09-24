"""Base class for page-level parsers.

A ``JKentParser`` is a Callable that takes a ``PageElement`` and returns
a list of ``DeferredValidation[T]`` — partial values for the eventual
``ParsedData`` payload of type T. Steps construct a parser, call it on
the page they received, and merge the resulting raw_data into their own
emission. The same parser can be exercised offline against saved HTML
via the ``from_response`` / ``from_string`` / ``from_file`` classmethods,
which take the production parse path (``decorators._parse_html``) so an
offline fixture is parsed exactly as the live page was.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from jkent.common.decorators import _parse_html
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.lxml_page_element import LxmlPageElement
from jkent.common.request import HttpMethod, HTTPRequestParams, Request
from jkent.common.response import Response

T = TypeVar("T", bound=BaseModel)


class JKentParser(ABC, Generic[T]):
    """Callable that extracts ``ParsedData`` fields from a page.

    Subclasses implement ``__call__(page)``, returning one
    ``DeferredValidation[T]`` per logical record extractable from the
    page (single-record pages return a one-element list; row-based
    pages return one entry per row).
    """

    @abstractmethod
    def __call__(
        self, page: LxmlPageElement
    ) -> list[DeferredValidation[T]]: ...

    @classmethod
    def from_response(cls, response: Response) -> list[DeferredValidation[T]]:
        """Run the parser on a response exactly as a ``@step`` would.

        The production parse path: bytes are decoded by the document's own
        declaration, else the ``Content-Type`` charset, before lxml sees
        them, and a parse failure surfaces as ``ScraperAssumptionException``.
        ``from_string`` / ``from_file`` are conveniences over this.
        """
        return cls()(_parse_html(response))

    @classmethod
    def from_string(
        cls,
        html: str | bytes,
        url: str = "",
        *,
        headers: Mapping[str, str] | None = None,
    ) -> list[DeferredValidation[T]]:
        """Parse an HTML string/bytes and run the parser on it.

        Args:
            html: Raw HTML markup. Bytes are what the scraper sees at run
                time and are preferred: the page's declared encoding (a
                ``<meta charset>``, else ``headers``' ``Content-Type``) is
                honoured. A ``str`` is taken as already decoded.
            url: Base URL for resolving relative links. Optional.
            headers: The response headers the page was served with — the
                ``Content-Type`` charset matters for a page that declares
                its encoding nowhere else.
        """
        response = _offline_response(html, url, headers)
        if isinstance(html, str):
            return cls()(_parse_html(response, text=html))
        return cls.from_response(response)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        url: str = "",
        *,
        headers: Mapping[str, str] | None = None,
    ) -> list[DeferredValidation[T]]:
        """Read an HTML file from disk and run the parser on it.

        Reads as bytes so the declared encoding is honoured; ``headers``
        supplies the ``Content-Type`` the page was served with.
        """
        return cls.from_string(
            Path(path).read_bytes(), url=url, headers=headers
        )


#: The step an offline parse pretends to run under — a ``Request`` must name
#: one, and no scraper is in play to dispatch it.
_OFFLINE_STEP = "offline"


def _offline_response(
    html: str | bytes, url: str, headers: Mapping[str, str] | None
) -> Response:
    """A ``Response`` standing in for the one a transport would have built."""
    return Response(
        status_code=200,
        headers=dict(headers or {}),
        content=html.encode("utf-8") if isinstance(html, str) else html,
        url=url,
        request=Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=url),
            step=_OFFLINE_STEP,
        ),
    )
