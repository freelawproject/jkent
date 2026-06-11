"""The ``@step(preprocess=...)`` document-repair hook.

A ``preprocess`` callable receives the decoded response text and returns
repaired text; the repaired document feeds the ``text``, ``lxml_tree``, and
``page`` injections, so a scraper can fix malformed HTML (e.g. an unclosed
``<style>`` that swallows the rest of the document) without hand-building a
page + observer (what New York's Court-PASS scraper had to do).
"""

from __future__ import annotations

import re
from collections.abc import Generator
from typing import Any

import pytest

from jkent.common.decorators import step
from jkent.common.exceptions import ScraperAssumptionException
from jkent.data_types import ParsedData, Response, XPath

# An unclosed <style pdffontname=...> makes lxml treat everything after it
# as CSS text, so the case-name div disappears from the parsed DOM.
_BROKEN_HTML = (
    "<html><body>"
    '<style pdffontname="F1">'
    '<div id="case-name">Smith v. Jones</div>'
    "</body></html>"
)

_STYLE_RE = re.compile(r"<style pdffontname=[^>]*>")


def _repair(text: str) -> str:
    return _STYLE_RE.sub("", text)


def _explode(text: str) -> str:
    raise ZeroDivisionError(f"no repair for {len(text)} chars")


def _response(html: str = _BROKEN_HTML) -> Response:
    return Response(
        status_code=200,
        headers={"content-type": "text/html"},
        content=html.encode("utf-8"),
        text=html,
        url="http://example.test/detail",
        request=None,  # type: ignore[arg-type]
    )


class _Host:
    """Bare method holder — @step only needs ``self`` to pass through."""

    @step(preprocess=_repair)
    def parse_with_repair(
        self, page: Any, text: str
    ) -> Generator[ParsedData, None, None]:
        name = page.query_strings(
            XPath("//div[@id='case-name']/text()"), "case name"
        )
        yield ParsedData(data={"name": name[0], "text": text})

    @step
    def parse_without_repair(
        self, page: Any
    ) -> Generator[ParsedData, None, None]:
        found = page._element.xpath("//div[@id='case-name']/text()")
        yield ParsedData(data={"found": found})

    @step(preprocess=_repair)
    def parse_tree(self, lxml_tree: Any) -> Generator[ParsedData, None, None]:
        found = lxml_tree._element.xpath("//div[@id='case-name']/text()")
        yield ParsedData(data={"found": found})

    @step(preprocess=_explode)
    def parse_exploding(self, text: str) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={})  # pragma: no cover - preprocess raises


def test_preprocess_feeds_page_and_text() -> None:
    results = list(_Host().parse_with_repair(_response()))
    assert results[0].data["name"] == "Smith v. Jones"
    # The text injection sees the repaired document too.
    assert "<style" not in results[0].data["text"]
    assert "Smith v. Jones" in results[0].data["text"]


def test_without_preprocess_the_breakage_is_real() -> None:
    # Pin the premise: unrepaired, the unclosed <style> swallows the div.
    results = list(_Host().parse_without_repair(_response()))
    assert results[0].data["found"] == []


def test_preprocess_feeds_lxml_tree() -> None:
    results = list(_Host().parse_tree(_response()))
    assert results[0].data["found"] == ["Smith v. Jones"]


def test_preprocess_failure_wraps_in_assumption_taxonomy() -> None:
    with pytest.raises(ScraperAssumptionException, match="preprocess"):
        list(_Host().parse_exploding(_response()))
