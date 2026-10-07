"""``@step(encoding=)`` and the text every injection reads.

Without ``encoding`` the transport's ``response.text`` is used as is. With
it, ``response.text`` is replaced by ``content`` strictly decoded as UTF-8,
else with that charset — for sites whose bytes don't match the charset they declare — and
``text``, ``json_content``, ``lxml_tree`` and ``page`` all read the result.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from typing import Any

import pytest

from jkent.common.decorators import step
from jkent.common.exceptions import ScraperAssumptionException
from jkent.data_types import BaseScraper, PageElement, ParsedData, Response

# "Acción" — the ó is two bytes in UTF-8 (\xc3\xb3), one in cp1252 (\xf3).
_ACCENTED = "Acción"
_UTF8 = _ACCENTED.encode("utf-8")
_CP1252 = _ACCENTED.encode("windows-1252")
_XML_DECLARATION = b'<?xml version="1.0" encoding="utf-8"?>'


def _body(word: bytes, prolog: bytes = b"") -> bytes:
    return prolog + b"<html><body><p id='w'>" + word + b"</p></body></html>"


def _response(content: bytes, text: str) -> Response:
    return Response(
        status_code=200,
        headers={},
        content=content,
        text=text,
        url="http://example.test/detail",
        request=None,  # type: ignore[arg-type]
    )


_WORD = re.compile(r"<p id='w'>(.*?)</p>")


def _word_in(text: str) -> str:
    match = _WORD.search(text)
    assert match is not None, text
    return match.group(1)


_METHODS = ("parse_page", "parse_tree", "parse_text", "parse_repaired")


def _host(encoding: str | None) -> Any:
    """Steps reading each text path, all declared with *encoding*."""

    class Host(BaseScraper[dict[str, Any]]):
        @step(encoding=encoding)
        def parse_page(
            self, page: PageElement
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": page.text_content().strip()})

        @step(encoding=encoding)
        def parse_tree(
            self, lxml_tree: PageElement
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": lxml_tree.text_content().strip()})

        @step(encoding=encoding)
        def parse_text(
            self, text: str
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": _word_in(text)})

        @step(encoding=encoding, preprocess=lambda document: document)
        def parse_repaired(
            self, page: PageElement
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": page.text_content().strip()})

        @step(encoding=encoding)
        def parse_json(
            self, json_content: dict[str, str]
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data=json_content)

    return Host()


def _words(response: Response, encoding: str | None) -> dict[str, str]:
    host = _host(encoding)
    return {
        method: list(getattr(host, method)(response=response))[0].data["word"]
        for method in _METHODS
    }


def test_without_encoding_every_path_reads_the_transport_text() -> None:
    # The transport's text wins even where it disagrees with the bytes.
    response = _response(_body(_CP1252), _body(_UTF8).decode("utf-8"))
    assert _words(response, None) == dict.fromkeys(_METHODS, _ACCENTED)


def test_encoding_overrides_a_misdeclared_charset() -> None:
    # A server that says utf-8 over cp1252 bytes: the transport's text is
    # garbled, the step's encoding repairs it for every path.
    content = _body(_CP1252)
    response = _response(content, content.decode("utf-8", errors="replace"))
    assert _words(response, "windows-1252") == dict.fromkeys(
        _METHODS, _ACCENTED
    )
    assert _word_in(response.text) == _ACCENTED


def test_utf8_wins_over_encoding_when_the_bytes_are_utf8() -> None:
    # A step that sees both UTF-8 and cp1252 pages: the UTF-8 ones must not
    # be read as cp1252 (which would give "AcciÃ³n").
    content = _body(_UTF8)
    response = _response(content, "")
    assert _words(response, "windows-1252") == dict.fromkeys(
        _METHODS, _ACCENTED
    )


def test_encoding_overrides_json_content() -> None:
    content = b'{"word": "' + _CP1252 + b'"}'
    response = _response(content, content.decode("utf-8", errors="replace"))
    [parsed] = list(_host("windows-1252").parse_json(response=response))
    assert parsed.data == {"word": _ACCENTED}


def test_bytes_that_do_not_fit_the_encoding_are_a_scraper_error() -> None:
    content = _body(_CP1252)
    response = _response(content, content.decode("utf-8", errors="replace"))
    with pytest.raises(ScraperAssumptionException, match="@step encoding"):
        _words(response, "utf-8")


def test_unknown_encoding_is_a_scraper_error() -> None:
    content = _body(_CP1252)
    response = _response(content, content.decode("utf-8", errors="replace"))
    with pytest.raises(ScraperAssumptionException, match="@step encoding"):
        _words(response, "definitely-not-a-codec")


@pytest.mark.parametrize(
    "prolog",
    [
        _XML_DECLARATION,
        b"\n  " + _XML_DECLARATION,
        b"\xef\xbb\xbf" + _XML_DECLARATION,
    ],
    ids=["bare", "leading whitespace", "bom"],
)
def test_xml_declaration_is_stripped_before_lxml(prolog: bytes) -> None:
    # lxml rejects a ``str`` that carries an encoding declaration.
    content = _body(_UTF8, prolog)
    response = _response(content, content.decode("utf-8"))
    assert _words(response, None) == dict.fromkeys(_METHODS, _ACCENTED)
    assert _words(response, "utf-8") == dict.fromkeys(_METHODS, _ACCENTED)


def test_empty_body_is_a_scraper_error() -> None:
    with pytest.raises(ScraperAssumptionException):
        list(_host(None).parse_page(response=_response(b"", "")))
