"""Encoding resolution for ``lxml_tree``/``page`` injection.

lxml sniffs *bytes*: BOM, XML declaration, meta charset. A server that declares
its charset only in the HTTP ``Content-Type`` header and emits no in-document
declaration therefore fell through to libxml2's HTML4 default of ISO-8859-1,
and every non-ASCII character came back mojibake — ``Acción`` parsed as
``AcciÃ³n`` — on a page whose bytes *and* header were both correct
(appellatecases.courtinfo.ca.gov is one such site, on all nine of its courts).

So the header charset is now consulted, but only when the document declares
nothing itself. That deliberately inverts WHATWG precedence, which ranks the
HTTP charset above a meta tag: a server sending a blanket
``charset=iso-8859-1`` over a page that correctly declares UTF-8 in a meta tag
is the more common misconfiguration, and spec order would corrupt it.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from jkent.common.decorators import step
from jkent.common.exceptions import ScraperAssumptionException
from jkent.data_types import ParsedData, Response

# "Acción" — the ó is two bytes in UTF-8 (\xc3\xb3), one in cp1252 (\xf3).
_ACCENTED = "Acción"
_UTF8 = _ACCENTED.encode("utf-8")
_CP1252 = _ACCENTED.encode("windows-1252")
# What ISO-8859-1 makes of the UTF-8 bytes.
_MOJIBAKE = _UTF8.decode("latin-1")

_META_1252 = b'<meta charset="windows-1252">'
_META_EQUIV_1252 = b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">'


def _body(word: bytes, head: bytes = b"") -> bytes:
    return (
        b"<html><head>"
        + head
        + b"</head><body><p id='w'>"
        + word
        + b"</p></body></html>"
    )


def _response(content: bytes, content_type: str | None) -> Response:
    return Response(
        status_code=200,
        headers={} if content_type is None else {"Content-Type": content_type},
        content=content,
        # Deliberately blank: these injections read ``content``, and a
        # pre-decoded ``text`` would hide which path ran.
        text="",
        url="http://example.test/detail",
        request=None,  # type: ignore[arg-type]
    )


class _Host:
    """Bare method holder — @step only needs ``self`` to pass through."""

    @step
    def parse_page(self, page: Any) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"word": page.text_content().strip()})

    @step
    def parse_tree(self, lxml_tree: Any) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"word": lxml_tree.text_content().strip()})


def _word(response: Response, method: str = "parse_page") -> str:
    results = list(getattr(_Host(), method)(response=response))
    return results[0].data["word"]


@pytest.mark.parametrize("method", ["parse_page", "parse_tree"])
@pytest.mark.parametrize(
    "content_type",
    [
        "text/html; charset=utf-8",
        "text/html; charset=UTF-8",
        'text/html; charset="utf-8"',
        "text/html;charset=utf-8",
        "text/html; charset=utf-8; foo=bar",
    ],
)
def test_header_charset_used_when_document_declares_none(
    content_type: str, method: str
) -> None:
    """The bug: charset stated only in the header, in its various spellings."""
    assert _word(_response(_body(_UTF8), content_type), method) == _ACCENTED


def test_header_charset_handles_non_utf8() -> None:
    """Not UTF-8-specific — a cp1252 header decodes cp1252 bytes."""
    got = _word(_response(_body(_CP1252), "text/html; charset=windows-1252"))
    assert got == _ACCENTED


@pytest.mark.parametrize("head", [_META_1252, _META_EQUIV_1252])
def test_document_declaration_beats_the_header(head: bytes) -> None:
    """A meta charset wins, even against a header that contradicts it."""
    response = _response(_body(_CP1252, head), "text/html; charset=utf-8")
    assert _word(response) == _ACCENTED


def test_bom_beats_the_header() -> None:
    """A UTF-8 BOM is a document declaration too."""
    response = _response(
        b"\xef\xbb\xbf" + _body(_UTF8), "text/html; charset=windows-1252"
    )
    assert _word(response) == _ACCENTED


def test_xml_declaration_defers_to_lxml_which_assumes_utf8() -> None:
    """An XML declaration counts as "declared", but lxml ignores what it says.

    libxml2's HTML parser does not honour an XML declaration's ``encoding=``;
    its mere presence switches the default from ISO-8859-1 to UTF-8. So a
    document declaring cp1252 this way is still read as UTF-8 and its cp1252
    byte comes back as U+FFFD -- unchanged from before the header fix, and the
    reason such documents are left to lxml rather than handed the header
    charset: the bet is that an XHTML-ish page is UTF-8, which beats trusting a
    possibly-stale ``Content-Type``.
    """
    response = _response(
        b'<?xml version="1.0" encoding="windows-1252"?>' + _body(_CP1252),
        "text/html; charset=utf-8",
    )
    assert _word(response) == "Acci�n"


def test_xml_declaration_with_matching_utf8_bytes() -> None:
    """The common XHTML shape: declaration present, bytes actually UTF-8."""
    response = _response(
        b'<?xml version="1.0" encoding="utf-8"?>' + _body(_UTF8),
        "text/html; charset=utf-8",
    )
    assert _word(response) == _ACCENTED


@pytest.mark.parametrize(
    "content_type",
    [
        None,
        "text/html",
        "text/html; charset=definitely-not-a-codec",
    ],
)
def test_falls_back_to_lxml_sniffing(content_type: str | None) -> None:
    """No usable header charset leaves the long-standing byte path in place."""
    assert _word(_response(_body(_UTF8), content_type)) == _MOJIBAKE


def test_header_that_misdescribes_the_bytes_does_not_raise() -> None:
    """A header claiming utf-8 over cp1252 bytes falls back instead of failing."""
    assert _word(_response(_body(_CP1252), "text/html; charset=utf-8")) == (
        _CP1252.decode("latin-1")
    )


def test_empty_body_is_unchanged() -> None:
    """An empty document raises the same way it always has."""
    with pytest.raises(ScraperAssumptionException):
        _word(_response(b"", "text/html; charset=utf-8"))
