"""Tests for ``JKentParser`` (jkent.common.parser).

The offline-parsing entry point for scraper authors: ``from_string`` /
``from_file`` build an ``LxmlPageElement`` and run the parser on it.
Public SDK surface with no in-repo production consumers, so it gets
direct coverage here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from jkent.common.deferred_validation import DeferredValidation
from jkent.common.parser import JKentParser
from jkent.data_types import Selector

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


def test_from_string_bytes_honors_declared_encoding() -> None:
    """Bytes input lets lxml read the page's declared (non-UTF-8) charset."""
    html = (
        '<html><head><meta charset="iso-8859-1"></head>'
        "<body><h2>S\xe9ance v. Apparition</h2></body></html>"
    ).encode("iso-8859-1")
    results = TitleParser.from_string(html)
    assert [r.confirm().title for r in results] == ["S\xe9ance v. Apparition"]


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
