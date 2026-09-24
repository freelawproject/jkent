"""What a transport hands back: :class:`Response` and its archive variants.

Below :mod:`jkent.common.request` in the layering — a request resolves
against a response at runtime (``isinstance``), while a response only
*names* its request — so this module imports the request type for type
checking alone.

This is also where response bytes become text. :func:`decode_text` is the
one derivation, with one precedence: the document's own declaration (a
BOM, an XML declaration, a ``<meta charset>``), then the HTTP
``Content-Type`` charset, then a fallback. ``response.text`` and every
``@step`` text injection — ``text``, ``lxml_tree``, ``page`` — read it, so
they agree about the same bytes.
"""

from __future__ import annotations

import codecs
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jkent.common.request import Request
    from jkent.common.selector_observer import SelectorObserver

__all__ = [
    "ArchiveDecision",
    "ArchiveResponse",
    "Response",
    "declared_charset",
    "decode_text",
    "header_charset",
    "utf8_document",
]

_HEADER_CHARSET = re.compile(
    r"""charset\s*=\s*["']?([\w.:+-]+)""", re.IGNORECASE
)
# The document speaking for itself: an XML declaration's ``encoding=`` or a
# ``<meta charset=...>`` / ``<meta http-equiv=Content-Type content="...;
# charset=...">``. Only the value is captured.
_DOC_CHARSET = re.compile(
    rb"""<\?xml[^>]*\bencoding\s*=\s*["']([\w.:+-]+)"""
    rb"""|<meta[^>]*\bcharset\s*=\s*["']?([\w.:+-]+)""",
    re.IGNORECASE,
)
# Browsers pre-scan roughly the first kilobyte for a meta charset; a wider
# window costs nothing and tolerates a long <head>.
_SNIFF_LIMIT = 4096


def header_charset(headers: Mapping[str, str]) -> str | None:
    """The charset the ``Content-Type`` header declares, if any."""
    for name, value in headers.items():
        if name.lower() == "content-type":
            match = _HEADER_CHARSET.search(value or "")
            return match.group(1) if match else None
    return None


def declared_charset(raw: bytes) -> str | None:
    """The charset the document itself declares, if any.

    A BOM wins outright (``utf-8-sig`` / ``utf-16`` consume it); otherwise
    the first :data:`_SNIFF_LIMIT` bytes are scanned for an XML declaration
    or a meta charset. A declared UTF-16 is read as UTF-8, as browsers do:
    a document whose declaration can be found by an ASCII scan is not
    UTF-16, whatever it says.
    """
    if raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    match = _DOC_CHARSET.search(raw[:_SNIFF_LIMIT])
    if match is None:
        return None
    value = (match.group(1) or match.group(2)).decode("ascii", errors="ignore")
    if value.lower().replace("-", "").startswith("utf16"):
        return "utf-8"
    return value or None


def utf8_document(text: str) -> bytes:
    """Encode an already-decoded document as UTF-8 that says it is UTF-8.

    A browser's DOM serialization keeps the page's own declaration, so
    encoding it to UTF-8 leaves bytes that claim, say, windows-1252; and
    since :func:`decode_text` trusts the document over the header, every
    later read of those bytes would re-decode them wrongly. The declaration
    :func:`declared_charset` would read is rewritten to ``utf-8``, as a
    browser's "save page" does; nothing else in the document changes.
    """
    raw = text.encode("utf-8")
    match = _DOC_CHARSET.search(raw[:_SNIFF_LIMIT])
    if match is None:
        return raw
    group = 1 if match.group(1) is not None else 2
    if match.group(group).lower() in (b"utf-8", b"utf8"):
        return raw
    start, end = match.span(group)
    return raw[:start] + b"utf-8" + raw[end:]


def decode_text(
    content: bytes, headers: Mapping[str, str], fallback: str = "utf-8"
) -> str:
    """Decode ``content`` by document declaration, then header, then ``fallback``.

    The declared and header charsets are each tried strictly and skipped
    when Python does not know the codec or the bytes do not decode; only
    ``fallback`` decodes with replacement, so undecodable bytes become
    U+FFFD there and nowhere earlier. The document outranks the header, the
    reverse of WHATWG order: a server sending a blanket
    ``charset=iso-8859-1`` over a page that correctly declares UTF-8 is the
    more common misconfiguration.

    Raises:
        LookupError: ``fallback`` names a codec Python does not know.
    """
    if not content:
        return ""
    for charset in (declared_charset(content), header_charset(headers)):
        if charset is None:
            continue
        try:
            return content.decode(charset)
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode(fallback, errors="replace")


@dataclass
class Response:
    """HTTP response from fetching a page.

    Modeled after httpx.Response to provide a familiar interface.
    The driver creates Response objects and passes them to scraper
    step methods.

    Attributes:
        status_code: HTTP status code (200, 404, etc.).
        headers: Response headers.
        content: Raw response bytes.
        url: Final URL after any redirects.
        request: The Request that triggered this response.
        text: Decoded response text. Derived from ``content`` and
            ``headers`` by :func:`decode_text` when not supplied; a caller
            that already holds the decoded document passes it, and it is
            what every ``@step`` text injection reads. (A browser's DOM
            serialization is not passed: it is encoded by
            :func:`utf8_document` and derived like any other body.)
        observer: SelectorObserver recorded while a @step with ``page``
            injection executed against this response. Set by the step
            wrapper; per-execution by construction (the driver owns one
            Response per execution), so drivers read autowait/debug
            telemetry here. None until a page-injecting step runs.
    """

    status_code: int
    headers: dict[str, str]
    content: bytes
    url: str
    request: Request
    text: str = ""
    observer: SelectorObserver | None = None
    _text_supplied: bool = field(
        default=False, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        self._text_supplied = bool(self.text)
        if not self.text and self.content:
            self.text = self.decode()

    def decode(self, fallback: str = "utf-8") -> str:
        """The response as text, with ``fallback`` when nothing declares a charset.

        Supplied :attr:`text` is returned as is. Otherwise the same
        derivation as :attr:`text`; the step wrapper calls it with the
        ``@step`` ``encoding`` so that knob can name a site's real charset
        when neither the document nor the server does.
        """
        if self._text_supplied:
            return self.text
        return decode_text(self.content, self.headers, fallback)


@dataclass
class ArchiveResponse(Response):
    """HTTP response for an archived file.

    Extends Response with a ``file_url``  The @step machinery injects this
    value into steps as the ``local_filepath`` parameter;
    This lets scrapers include the file location in their ParsedData.

    Attributes:
        file_url: path where the downloaded file was stored.
            Injected into steps as ``local_filepath``.
        file_size: Bytes written to ``file_url``, or None when this fetch
            wrote nothing (an existing file was reused).
        content_hash: SHA-256 hex digest of those bytes, None likewise.
    """

    file_url: str = ""
    file_size: int | None = None
    content_hash: str | None = None


@dataclass
class ArchiveDecision:
    """An archive handler's answer to whether a file should be downloaded.

    Returned by
    :meth:`~jkent.driver.archive_handler.AsyncStreamingArchiveHandler.should_download`.

    Attributes:
        download: If True, the driver should proceed with downloading.
        file_url: When download=False, the location of the existing file.
            When download=True, may be empty (``save_stream`` determines the
            final path).
    """

    download: bool
    file_url: str = ""
