"""One charset rule for ``response.text``, ``text``, ``lxml_tree`` and ``page``.

Every row of the table is one set of bytes, one ``Content-Type``, and one
``@step(encoding=)``; every path that turns those bytes into text must agree
on the word they spell. The rule (:func:`jkent.common.response.decode_text`):
the document's own declaration, then the header charset, each decoded
strictly and skipped on failure, then the step encoding with replacement.
"""

from __future__ import annotations

import re
from collections.abc import Generator
from typing import Any

import pytest

from jkent.common.decorators import step
from jkent.common.exceptions import ScraperAssumptionException
from jkent.data_types import BaseScraper, ParsedData, Response

# "Acción" — the ó is two bytes in UTF-8 (\xc3\xb3), one in cp1252 (\xf3).
_ACCENTED = "Acción"
_UTF8 = _ACCENTED.encode("utf-8")
_CP1252 = _ACCENTED.encode("windows-1252")

_META_1252 = b'<meta charset="windows-1252">'
_META_EQUIV_1252 = b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">'
_META_UTF16 = b'<meta charset="utf-16">'
_XML_UTF8 = b'<?xml version="1.0" encoding="utf-8"?>'
_XML_1252 = b'<?xml version="1.0" encoding="windows-1252"?>'
_BOM = b"\xef\xbb\xbf"


def _body(word: bytes, head: bytes = b"", prolog: bytes = b"") -> bytes:
    return (
        prolog
        + b"<html><head>"
        + head
        + b"</head><body><p id='w'>"
        + word
        + b"</p></body></html>"
    )


def _response(
    content: bytes, content_type: str | None, text: str = ""
) -> Response:
    return Response(
        status_code=200,
        headers={} if content_type is None else {"Content-Type": content_type},
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


def _host(encoding: str) -> Any:
    """Steps reading each text path, all declared with *encoding*."""

    class Host(BaseScraper[dict[str, Any]]):
        @step(encoding=encoding)
        def parse_page(
            self, page: Any
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": page.text_content().strip()})

        @step(encoding=encoding)
        def parse_tree(
            self, lxml_tree: Any
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": lxml_tree.text_content().strip()})

        @step(encoding=encoding)
        def parse_text(
            self, text: str
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": _word_in(text)})

        @step(encoding=encoding, preprocess=lambda document: document)
        def parse_repaired(
            self, page: Any
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"word": page.text_content().strip()})

    return Host()


def _words(response: Response, encoding: str) -> dict[str, str]:
    host = _host(encoding)
    return {
        method: list(getattr(host, method)(response=response))[0].data["word"]
        for method in (
            "parse_page",
            "parse_tree",
            "parse_text",
            "parse_repaired",
        )
    }


# (content, Content-Type, @step encoding, the word every path must read)
_TABLE = {
    "header only": (
        _body(_UTF8),
        "text/html; charset=utf-8",
        "utf-8",
        _ACCENTED,
    ),
    "header quoted": (
        _body(_UTF8),
        'text/html; charset="UTF-8"',
        "utf-8",
        _ACCENTED,
    ),
    "header with params": (
        _body(_UTF8),
        "text/html;charset=utf-8; foo=bar",
        "utf-8",
        _ACCENTED,
    ),
    "header cp1252": (
        _body(_CP1252),
        "text/html; charset=windows-1252",
        "utf-8",
        _ACCENTED,
    ),
    "nothing declared": (_body(_UTF8), None, "utf-8", _ACCENTED),
    "no header charset": (_body(_UTF8), "text/html", "utf-8", _ACCENTED),
    "unknown header codec": (
        _body(_UTF8),
        "text/html; charset=definitely-not-a-codec",
        "utf-8",
        _ACCENTED,
    ),
    "meta beats header": (
        _body(_CP1252, _META_1252),
        "text/html; charset=utf-8",
        "utf-8",
        _ACCENTED,
    ),
    "meta http-equiv beats header": (
        _body(_CP1252, _META_EQUIV_1252),
        "text/html; charset=utf-8",
        "utf-8",
        _ACCENTED,
    ),
    "bom beats header": (
        _BOM + _body(_UTF8),
        "text/html; charset=windows-1252",
        "utf-8",
        _ACCENTED,
    ),
    "xml declaration utf-8": (
        _body(_UTF8, prolog=_XML_UTF8),
        "text/html; charset=utf-8",
        "utf-8",
        _ACCENTED,
    ),
    "xml declaration cp1252": (
        _body(_CP1252, prolog=_XML_1252),
        None,
        "utf-8",
        _ACCENTED,
    ),
    "meta utf-16 on ascii-compatible bytes": (
        _body(_UTF8, _META_UTF16),
        None,
        "utf-8",
        _ACCENTED,
    ),
    "step encoding when nothing declares": (
        _body(_CP1252),
        None,
        "windows-1252",
        _ACCENTED,
    ),
    "step encoding when the header misdescribes": (
        _body(_CP1252),
        "text/html; charset=utf-8",
        "windows-1252",
        _ACCENTED,
    ),
    "header misdescribes, default step encoding": (
        _body(_CP1252),
        "text/html; charset=utf-8",
        "utf-8",
        "Acci�n",
    ),
}


@pytest.mark.parametrize(
    ("content", "content_type", "encoding", "expected"),
    list(_TABLE.values()),
    ids=list(_TABLE),
)
def test_every_text_path_reads_the_same_word(
    content: bytes, content_type: str | None, encoding: str, expected: str
) -> None:
    response = _response(content, content_type)
    assert _words(response, encoding) == dict.fromkeys(
        ("parse_page", "parse_tree", "parse_text", "parse_repaired"), expected
    )
    if encoding == "utf-8":
        assert _word_in(response.text) == expected


def test_supplied_text_is_what_every_path_reads() -> None:
    """A transport's already-decoded text wins over re-decoding the bytes.

    A browser's DOM serialization is UTF-8 but keeps the page's original
    ``<meta charset>``; decoding those bytes by the meta would garble them.
    """
    content = _body(_UTF8, _META_1252)
    response = _response(
        content, "text/html; charset=utf-8", text=content.decode("utf-8")
    )
    assert set(_words(response, "utf-8").values()) == {_ACCENTED}


def test_unknown_step_encoding_is_a_scraper_error() -> None:
    with pytest.raises(ScraperAssumptionException, match="Unknown @step"):
        _words(_response(_body(_CP1252), None), "definitely-not-a-codec")


def test_empty_body_is_a_scraper_error() -> None:
    with pytest.raises(ScraperAssumptionException):
        list(_host("utf-8").parse_page(response=_response(b"", None)))
