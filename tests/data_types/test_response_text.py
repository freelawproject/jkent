"""``Response.text`` has one derivation.

``text`` is derived from ``content`` + ``headers`` when a transport does not
supply it, with the precedence the ``page`` injection already uses: the
document's own declaration (BOM, XML declaration, ``<meta charset>``), then
the ``Content-Type`` charset, then UTF-8 — never raising — so ``text`` and
``page`` agree about the same bytes.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from jkent.common.decorators import _get_text
from jkent.common.response import decode_text, utf8_document
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.unified_driver import QueuedRequest
from tests.httpx_mock import mocked_httpx_transport

_CAFE_LATIN1 = "café".encode("latin-1")
# Curly quotes: single bytes in cp1252, not valid UTF-8.
_QUOTED_CP1252 = "“quoted”".encode("cp1252")
_META_1252 = b'<meta charset="windows-1252">'


def _response(
    content: bytes, headers: dict[str, str] | None = None, **kw: Any
) -> Response:
    return Response(
        status_code=200,
        headers=headers or {},
        content=content,
        url="http://example.test/",
        request=None,  # type: ignore[arg-type]
        **kw,
    )


class TestDerivation:
    def test_header_charset_decodes_content(self) -> None:
        resp = _response(
            _CAFE_LATIN1, {"Content-Type": "text/html; charset=iso-8859-1"}
        )
        assert resp.text == "café"

    def test_document_declaration_beats_the_header(self) -> None:
        # The server's blanket utf-8 header is wrong; the page says cp1252.
        resp = _response(
            b"<html><head>"
            + _META_1252
            + b"</head><body>"
            + _QUOTED_CP1252
            + b"</body></html>",
            {"Content-Type": "text/html; charset=utf-8"},
        )
        assert "“quoted”" in resp.text

    def test_undeclared_falls_back_to_utf8_without_raising(self) -> None:
        resp = _response(b"ok \xff\xfe end")
        assert resp.text.startswith("ok ")
        assert "�" in resp.text

    def test_unknown_declared_codec_falls_through(self) -> None:
        resp = _response(b'<meta charset="no-such-codec">' + "café".encode())
        assert "café" in resp.text

    def test_empty_content_is_empty_text(self) -> None:
        assert _response(b"").text == ""

    def test_explicit_text_is_kept_verbatim(self) -> None:
        assert _response(b"bytes", text="given").text == "given"


class TestStepInjection:
    """The ``text`` a step receives is the same derivation."""

    def test_text_injection_honors_the_header_charset(self) -> None:
        resp = _response(
            _CAFE_LATIN1, {"Content-Type": "text/html; charset=iso-8859-1"}
        )
        assert _get_text(resp) == "café"

    def test_step_encoding_is_the_fallback_when_nothing_declares(self) -> None:
        assert _get_text(_response(_CAFE_LATIN1), "latin-1") == "café"
        # ...and does not override a declaration.
        resp = _response(
            "café".encode(),
            {"Content-Type": "text/html; charset=utf-8"},
        )
        assert _get_text(resp, "latin-1") == "café"


class TestHttpxAgreesWithPage:
    """What a scraper reads from ``response.text`` matches what ``page`` parses."""

    async def test_httpx_text_uses_the_document_declaration(self) -> None:
        body = (
            b"<html><head>"
            + _META_1252
            + b"</head><body><p id='q'>"
            + _QUOTED_CP1252
            + b"</p></body></html>"
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=body,
                headers={"content-type": "text/html; charset=utf-8"},
            )

        transport = mocked_httpx_transport(handler)
        try:
            handle = await transport.acquire(0)
            request = Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url="http://test/page"
                ),
                step="parse_page",
            )
            resp = await transport.resolve(
                handle, QueuedRequest(request=request, request_id=1)
            )
        finally:
            await transport.aclose()
        assert "“quoted”" in resp.text


class TestUtf8Document:
    """A serialized DOM encoded to UTF-8 has to say so itself.

    ``decode_text`` ranks the document's declaration above the header, so
    UTF-8 bytes still carrying the page's ``<meta charset="windows-1252">``
    re-decode as cp1252 wherever they are read back.
    """

    @pytest.mark.parametrize(
        "declaration",
        [
            '<meta charset="windows-1252">',
            "<meta charset=windows-1252>",
            '<meta http-equiv="Content-Type" '
            'content="text/html; charset=iso-8859-1">',
            '<?xml version="1.0" encoding="iso-8859-1"?>',
        ],
    )
    def test_rewrites_a_non_utf8_declaration(self, declaration: str) -> None:
        text = f"<html><head>{declaration}</head><body>café</body></html>"
        raw = utf8_document(text)
        assert decode_text(raw, {}) == raw.decode("utf-8")
        assert "café" in decode_text(raw, {})

    @pytest.mark.parametrize(
        "text",
        [
            "<html><body>café</body></html>",
            '<html><head><meta charset="utf-8"></head><body>café</body></html>',
        ],
    )
    def test_leaves_utf8_and_undeclared_documents_alone(
        self, text: str
    ) -> None:
        assert utf8_document(text) == text.encode("utf-8")

    def test_only_the_declaration_is_rewritten(self) -> None:
        text = (
            '<html><head><meta charset="windows-1252"></head>'
            "<body><p>charset=windows-1252</p></body></html>"
        )
        assert utf8_document(text).decode("utf-8") == text.replace(
            'charset="windows-1252"', 'charset="utf-8"', 1
        )
