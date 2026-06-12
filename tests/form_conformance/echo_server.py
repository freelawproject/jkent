"""The echo + oracle-form aiohttp app and the canonical-submission helpers.

Two routes:

* ``GET /oracle`` — serves the *current* oracle form HTML (the rig sets
  ``app["oracle_html"]`` right before driving the oracle browser). Only the
  oracle uses this; the transports stage the rendered form from the DB
  (Playwright) or parse it locally (HTTP).
* ``ANY /echo`` — the form ``action``. It parses the submission in *received
  order* (``parse_qsl`` over the raw GET query string or the urlencoded POST
  body) and echoes it back inside ``<pre id="echo">…</pre>`` as JSON. Embedding
  the result in a stable DOM element means the same extractor works for an HTTP
  response body and for a browser-rendered snapshot alike — no fighting a
  browser's JSON viewer.
"""

from __future__ import annotations

import json
from html import escape
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

from aiohttp import web
from lxml import html as lxml_html

if TYPE_CHECKING:
    from collections.abc import Sequence

# A canonical submission: the method plus the ordered (name, value) pairs the
# server received.
Canonical = tuple[str, tuple[tuple[str, str], ...]]

# A mutable holder for the current oracle form HTML. The rig swaps the HTML per
# example by mutating the holder's contents rather than the (started) app
# mapping, which aiohttp forbids. Typed AppKey keeps aiohttp from warning.
ORACLE: web.AppKey[dict[str, str]] = web.AppKey("oracle", dict)


async def _oracle(request: web.Request) -> web.Response:
    body = (
        request.app[ORACLE].get("html") or "<html><body>no form</body></html>"
    )
    return web.Response(text=body, content_type="text/html")


async def _echo(request: web.Request) -> web.Response:
    if request.method == "GET":
        pairs = parse_qsl(request.query_string, keep_blank_values=True)
    else:
        raw = await request.text()
        pairs = parse_qsl(raw, keep_blank_values=True)
    payload = json.dumps({"method": request.method, "pairs": pairs})
    return web.Response(
        text=(
            "<!doctype html><html><body>"
            f'<pre id="echo">{escape(payload)}</pre>'
            "</body></html>"
        ),
        content_type="text/html",
    )


def create_app() -> web.Application:
    """Build the echo + oracle-form application."""
    app = web.Application()
    app[ORACLE] = {"html": ""}
    app.router.add_get("/oracle", _oracle)
    app.router.add_route("*", "/echo", _echo)
    return app


def extract_echo(html_text: str) -> Canonical:
    """Pull the canonical submission out of an echoed page.

    Works on a raw HTTP response body and on a browser DOM snapshot — both
    carry the ``#echo`` element; ``lxml`` un-escapes its text for us.

    Raises:
        ValueError: if the page carries no ``#echo`` element (e.g. the request
            never reached ``/echo`` — itself a useful signal).
    """
    doc = lxml_html.fromstring(html_text)
    nodes = doc.xpath('//*[@id="echo"]')
    if not nodes:
        raise ValueError(
            f"no #echo element in echoed page: {html_text[:300]!r}"
        )
    payload = json.loads(nodes[0].text_content())
    pairs = tuple((str(k), str(v)) for k, v in payload["pairs"])
    return (str(payload["method"]), pairs)


def format_canonical(label: str, canonical: Canonical | str) -> str:
    """One-line-per-pair rendering of a canonical submission for diffs.

    Tolerates an error-marker string (a transport that raised) so the diff
    message stays readable instead of itself blowing up.
    """
    if isinstance(canonical, str):
        return f"  {label}: {canonical}"
    method, pairs = canonical
    lines = [f"  {label}: {method}"]
    lines.extend(f"    {name!r} = {value!r}" for name, value in pairs)
    if not pairs:
        lines.append("    <no fields>")
    return "\n".join(lines)


def diff_message(
    case_repr: str,
    oracle: Canonical,
    results: Sequence[tuple[str, Canonical | str]],
) -> str:
    """Build a readable assertion message comparing transports to the oracle."""
    blocks = [
        "transport submission diverged from the browser oracle",
        case_repr,
        format_canonical("oracle (browser)", oracle),
    ]
    blocks.extend(format_canonical(name, c) for name, c in results)
    return "\n".join(blocks)
