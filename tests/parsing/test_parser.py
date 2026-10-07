"""Tests for ``JKentParser`` (jkent.common.parser).

The offline-parsing entry point for scraper authors: ``from_string`` /
``from_file`` build an ``PageElement`` and run the parser on it.
Public SDK surface with no in-repo production consumers, so it gets
direct coverage here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import ScraperAssumptionException
from jkent.common.parser import JKentParser
from jkent.data_types import Response, Selector

if TYPE_CHECKING:
    from pathlib import Path

    from jkent.common.page_element import PageElement


class CaseTitle(BaseModel):
    title: str


class TitleParser(JKentParser[CaseTitle]):
    """One DeferredValidation per ``<h2>`` on the page."""

    def __call__(
        self, page: PageElement
    ) -> list[DeferredValidation[CaseTitle]]:
        return [
            DeferredValidation(CaseTitle, title=element.text_content())
            for element in page.query(
                Selector.XPath("//h2"), "case titles", min_count=0
            )
        ]


_HTML = """
<html><body>
  <h2>Ant v. Bee</h2>
  <h2>Cricket v. Dragonfly</h2>
</body></html>
"""


def test_from_string_runs_the_parser() -> None:
    results = TitleParser.from_string(_HTML)
    assert [r.confirm().title for r in results] == [
        "Ant v. Bee",
        "Cricket v. Dragonfly",
    ]
    assert all(r.model_name == "CaseTitle" for r in results)


def test_from_string_bytes_decode_with_the_given_encoding() -> None:
    html = (
        "<html><body><h2>S\xe9ance v. Apparition</h2></body></html>"
    ).encode("iso-8859-1")
    results = TitleParser.from_string(html, encoding="iso-8859-1")
    assert [r.confirm().title for r in results] == ["S\xe9ance v. Apparition"]


def test_from_string_bytes_that_do_not_fit_the_encoding_raise() -> None:
    html = "<h2>S\xe9ance</h2>".encode("iso-8859-1")
    with pytest.raises(ScraperAssumptionException):
        TitleParser.from_string(html)


def test_from_file_reads_bytes(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_bytes(_HTML.encode())
    results = TitleParser.from_file(path)
    assert [r.confirm().title for r in results] == [
        "Ant v. Bee",
        "Cricket v. Dragonfly",
    ]

    # str paths are accepted too.
    assert len(TitleParser.from_file(str(path))) == 2


# --- Offline parsing takes the production parse path -----------------------

_UTF8 = "<html><body><h2>Acción v. Reacción</h2></body></html>".encode()


def _titles(results: list[DeferredValidation[CaseTitle]]) -> list[str]:
    return [r.confirm().title for r in results]


def _response(content: bytes) -> Response:
    return Response(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=content,
        text=content.decode("utf-8"),
        url="http://example.test/list",
        request=None,  # type: ignore[arg-type]
    )


def test_from_response_is_the_production_entry_point() -> None:
    assert _titles(TitleParser.from_response(_response(_UTF8))) == [
        "Acción v. Reacción"
    ]


def test_from_string_wraps_parse_failures_like_production() -> None:
    with pytest.raises(ScraperAssumptionException):
        TitleParser.from_string(b"")


# --- Offline entry points run the parser a step would build -----------------


class PrefixedTitleParser(TitleParser):
    """Configured through its constructor, as many site parsers are."""

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix

    def __call__(
        self, page: PageElement
    ) -> list[DeferredValidation[CaseTitle]]:
        return [
            DeferredValidation(
                CaseTitle, title=self.prefix + r.confirm().title
            )
            for r in super().__call__(page)
        ]


def test_entry_points_on_an_instance_use_its_configuration(
    tmp_path: Path,
) -> None:
    parser = PrefixedTitleParser(prefix="No. ")
    path = tmp_path / "page.html"
    path.write_bytes(_UTF8)

    expected = ["No. Acción v. Reacción"]
    assert _titles(parser.from_string(_UTF8)) == expected
    assert _titles(parser.from_file(path)) == expected
    assert _titles(parser.from_response(_response(_UTF8))) == expected


def test_entry_points_on_the_class_use_a_default_instance() -> None:
    assert _titles(PrefixedTitleParser.from_string(_HTML)) == [
        "Ant v. Bee",
        "Cricket v. Dragonfly",
    ]


def test_class_entry_point_needs_a_no_argument_constructor() -> None:
    class NeedsPrefix(TitleParser):
        def __init__(self, prefix: str) -> None:
            self.prefix = prefix

    with pytest.raises(TypeError, match="prefix"):
        NeedsPrefix.from_string(_HTML)
    assert len(NeedsPrefix("x").from_string(_HTML)) == 2
