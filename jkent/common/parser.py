"""Base class for page-level parsers.

A ``JKentParser`` is a Callable that takes a ``PageElement`` and returns
a list of ``DeferredValidation[T]`` — partial values for the eventual
``ParsedData`` payload of type T. Steps construct a parser, call it on
the page they received, and merge the resulting raw_data into their own
emission. The same parser can be exercised offline against saved HTML
via the ``from_string`` / ``from_file`` classmethods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from lxml import html as lxml_html
from pydantic import BaseModel

from jkent.common.deferred_validation import DeferredValidation
from jkent.common.lxml_page_element import LxmlPageElement

if TYPE_CHECKING:
    from jkent.common.page_element import PageElement

T = TypeVar("T", bound=BaseModel)


class JKentParser(ABC, Generic[T]):
    """Callable that extracts ``ParsedData`` fields from a page.

    Subclasses implement ``__call__(page)``, returning one
    ``DeferredValidation[T]`` per logical record extractable from the
    page (single-record pages return a one-element list; row-based
    pages return one entry per row).
    """

    @abstractmethod
    def __call__(self, page: PageElement) -> list[DeferredValidation[T]]: ...

    @classmethod
    def from_string(
        cls, html: str | bytes, url: str = ""
    ) -> list[DeferredValidation[T]]:
        """Parse an HTML string/bytes and run the parser on it.

        Args:
            html: Raw HTML markup. Bytes are preferred so lxml can
                detect the page's declared encoding from a ``<meta>``
                tag; strings are accepted for convenience.
            url: Base URL for resolving relative links. Optional.
        """
        element = lxml_html.fromstring(html)
        page = LxmlPageElement(element, url)
        return cls()(page)  # type: ignore[arg-type]

    @classmethod
    def from_file(
        cls, path: str | Path, url: str = ""
    ) -> list[DeferredValidation[T]]:
        """Read an HTML file from disk and run the parser on it.

        Reads as bytes so lxml can detect declared encoding.
        """
        return cls.from_string(Path(path).read_bytes(), url=url)
