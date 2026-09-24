"""Tests for ``JKentParser`` (jkent.common.parser).

The offline-parsing entry point for scraper authors: ``from_string`` /
``from_file`` build an ``LxmlPageElement`` and run the parser on it.
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


# --- Offline parsing takes the production parse path -----------------------
#
# Production parses through ``jkent.common.decorators._parse_html``: a page that
# declares its charset only in the HTTP ``Content-Type`` (no BOM, no meta) is
# decoded from that header before lxml sees it, because libxml2's HTML4
# default is ISO-8859-1 and would turn ``Acción`` into ``AcciÃ³n``. An offline
# entry point that hands bytes straight to lxml exercises a different parser
# than the one the scraper runs under — the exact bug class that helper exists
# to prevent.

_HEADER_ONLY_UTF8 = (
    "<html><body><h2>Acción v. Reacción</h2></body></html>".encode()
)


def _titles(results: list[DeferredValidation[CaseTitle]]) -> list[str]:
    return [r.confirm().title for r in results]


def test_from_string_honors_the_header_charset() -> None:
    results = TitleParser.from_string(
        _HEADER_ONLY_UTF8,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )
    assert _titles(results) == ["Acción v. Reacción"]


def test_from_file_honors_the_header_charset(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_bytes(_HEADER_ONLY_UTF8)
    results = TitleParser.from_file(
        path, headers={"Content-Type": "text/html; charset=utf-8"}
    )
    assert _titles(results) == ["Acción v. Reacción"]


def test_from_response_is_the_production_entry_point() -> None:
    response = Response(
        status_code=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
        content=_HEADER_ONLY_UTF8,
        url="http://example.test/list",
        request=None,  # type: ignore[arg-type]
    )
    assert _titles(TitleParser.from_response(response)) == [
        "Acción v. Reacción"
    ]


def test_from_string_wraps_parse_failures_like_production() -> None:
    with pytest.raises(ScraperAssumptionException):
        TitleParser.from_string(b"")
