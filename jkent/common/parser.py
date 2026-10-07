"""Base class for page-level parsers.

A ``JKentParser`` is a Callable that takes a ``PageElement`` and returns
a list of ``DeferredValidation[T]`` — partial values for the eventual
``ParsedData`` payload of type T. Steps construct a parser, call it on
the page they received, and merge the resulting raw_data into their own
emission. The same parser can be exercised offline against saved HTML
via ``from_response`` / ``from_string`` / ``from_file``, which take the
production parse path (``decorators._parse_html``) so an offline fixture
is parsed exactly as the live page was. Call them on the instance a step
would build (``CaseDetailParser(court="x").from_file(...)``); calling them
on the class is shorthand for a no-argument instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Concatenate, Generic, ParamSpec, TypeVar

from pydantic import BaseModel

from jkent.common.decorators import _apply_encoding, _parse_html
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.page_element import PageElement
from jkent.common.request import HttpMethod, HTTPRequestParams, Request
from jkent.common.response import Response

T = TypeVar("T", bound=BaseModel)
_Parser = TypeVar("_Parser", bound="JKentParser[Any]")
_P = ParamSpec("_P")
_R = TypeVar("_R")


class _offline_entry(Generic[_Parser, _P, _R]):
    """Bind to the parser instance it is read from, or to ``owner()``.

    A parser configured through its constructor must be exercised offline
    as the instance a step builds, not a default-constructed stand-in, so
    these entry points are instance methods; reading one off the class
    keeps the ``Parser.from_string(...)`` shorthand for parsers that take
    no arguments.
    """

    def __init__(self, func: Callable[Concatenate[_Parser, _P], _R]) -> None:
        self._func = func
        self.__doc__ = func.__doc__

    def __get__(
        self, instance: _Parser | None, owner: type[_Parser]
    ) -> Callable[_P, _R]:
        parser = owner() if instance is None else instance
        func = self._func

        def bound(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(parser, *args, **kwargs)

        return bound


class JKentParser(ABC, Generic[T]):
    """Callable that extracts ``ParsedData`` fields from a page.

    Subclasses implement ``__call__(page)``, returning one
    ``DeferredValidation[T]`` per logical record extractable from the
    page (single-record pages return a one-element list; row-based
    pages return one entry per row).
    """

    @abstractmethod
    def __call__(self, page: PageElement) -> list[DeferredValidation[T]]: ...

    @_offline_entry
    def from_response(self, response: Response) -> list[DeferredValidation[T]]:
        """Run the parser on ``response.text`` exactly as a ``@step`` would.

        A parse failure surfaces as ``ScraperAssumptionException``.
        ``from_string`` / ``from_file`` are conveniences over this.
        """
        return self(_parse_html(response))

    @_offline_entry
    def from_string(
        self,
        html: str | bytes,
        url: str = "",
        *,
        encoding: str = "utf-8",
    ) -> list[DeferredValidation[T]]:
        """Parse an HTML string/bytes and run the parser on it.

        Args:
            html: Raw HTML markup. A ``str`` is taken as already decoded.
            url: Base URL for resolving relative links. Optional.
            encoding: Charset that ``bytes`` are strictly decoded with.
        """
        return self.from_response(_offline_response(html, url, encoding))

    @_offline_entry
    def from_file(
        self,
        path: str | Path,
        url: str = "",
        *,
        encoding: str = "utf-8",
    ) -> list[DeferredValidation[T]]:
        """Read an HTML file from disk, decode it with ``encoding``, and run
        the parser on it."""
        return self.from_string(
            Path(path).read_bytes(), url=url, encoding=encoding
        )


#: The step an offline parse pretends to run under — a ``Request`` must name
#: one, and no scraper is in play to dispatch it.
_OFFLINE_STEP = "offline"


def _offline_response(html: str | bytes, url: str, encoding: str) -> Response:
    """A ``Response`` standing in for the one a transport would have built."""
    response = Response(
        status_code=200,
        headers={},
        content=html.encode("utf-8") if isinstance(html, str) else html,
        text=html if isinstance(html, str) else "",
        url=url,
        request=Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=url),
            step=_OFFLINE_STEP,
        ),
    )
    if isinstance(html, bytes):
        _apply_encoding(response, encoding)
    return response
