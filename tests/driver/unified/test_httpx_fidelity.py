"""Fidelity rig for ``HttpxTransport`` (the v1 HTTP backend).

Hypothesis generates request/response entries; each example stands up a live
aiohttp *matching server* and points an ``HttpxTransport`` at it. The server
returns the stored response only if the incoming request matches what the
``Request`` specified — so this rig checks two things at once:

- **Request fidelity** (server-enforced): the transport actually puts the
  specified method, URL (path + query), headers, cookies, and body on the
  wire. The server signals a mismatch via an ``x-match: no`` response header
  plus a diagnostic body (it always answers 2xx, so the transport's
  status-code classification never turns a mismatch into an exception).
- **Response fidelity**: the returned ``Response`` carries the server's status
  code, raw content bytes, request URL, and custom response header.

Match contract (what the server asserts):
  - method            — exact (case-insensitive)
  - path + query      — exact (query compared as a dict)
  - request headers   — every specified header present with its value
                        (case-insensitive name; extra client headers ignored)
  - cookies           — every specified cookie present with its value
  - body              — exact bytes

Deliberately NOT asserted (transport/client noise): httpx-added headers
(Host, User-Agent, Accept, Accept-Encoding, Connection, Content-Length,
Cookie), and the full set of response headers (only a custom marker is
checked, since the server/client add their own).

v1 simplifications:
  - Query params are carried in the URL (the persistent driver bakes params
    into the URL and does not re-send ``HTTPRequestParams.params``).
  - Generated bodies are raw ``bytes`` (``data``), avoiding json/form
    serialization ambiguity. ``HTTPRequestParams.json`` *is* re-sent by the
    port (the queue carries it as its own column); a dedicated test below
    covers it directly.
  - Statuses are 200/201/202 (bodied 2xx); methods are GET/POST/PUT/PATCH.
"""

from __future__ import annotations

import asyncio
import json
import string
from dataclasses import dataclass
from urllib.parse import urlencode

import pytest
from aiohttp import web
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.data_types import HttpMethod, HTTPRequestParams, Request
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest

# --- strategies ----------------------------------------------------------

_KEY = string.ascii_lowercase + string.digits
_VAL = string.ascii_letters + string.digits
_key = st.text(alphabet=_KEY, min_size=1, max_size=6)
_val = st.text(alphabet=_VAL, min_size=1, max_size=6)
_token = st.text(alphabet=_KEY, min_size=1, max_size=6)
_methods = st.sampled_from(
    [HttpMethod.GET, HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH]
)
_BODY_METHODS = {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH}


@dataclass
class _Entry:
    method: HttpMethod
    query: dict[str, str]
    headers: dict[str, str]
    cookies: dict[str, str]
    body: bytes
    status: int
    content: bytes
    marker: str


@st.composite
def _entry(draw: st.DrawFn) -> _Entry:
    method = draw(_methods)
    body = draw(st.binary(max_size=200)) if method in _BODY_METHODS else b""
    return _Entry(
        method=method,
        query=draw(st.dictionaries(_key, _val, max_size=3)),
        headers=draw(
            st.dictionaries(_key.map(lambda k: f"x-gen-{k}"), _val, max_size=3)
        ),
        cookies=draw(
            st.dictionaries(_key.map(lambda k: f"ck{k}"), _val, max_size=3)
        ),
        body=body,
        status=draw(st.sampled_from([200, 201, 202])),
        content=draw(st.binary(max_size=200)),
        marker=draw(_token),
    )


# --- matching server -----------------------------------------------------


def _build_app(entries: list[_Entry]) -> web.Application:
    """An app that returns each entry's response iff the request matches it."""

    async def handler(request: web.Request) -> web.Response:
        entry = entries[int(request.match_info["idx"])]
        body = await request.read()
        problems: list[str] = []
        if request.method.upper() != entry.method.value.upper():
            problems.append(f"method {request.method}!={entry.method.value}")
        if dict(request.query) != entry.query:
            problems.append(f"query {dict(request.query)}!={entry.query}")
        for name, value in entry.headers.items():
            if request.headers.get(name) != value:
                problems.append(f"header {name!r}")
        for name, value in entry.cookies.items():
            if request.cookies.get(name) != value:
                problems.append(f"cookie {name!r}")
        if body != entry.body:
            problems.append(f"body {body!r}!={entry.body!r}")

        if problems:
            return web.Response(
                status=200,
                body=("; ".join(problems)).encode(),
                headers={"x-match": "no"},
            )
        return web.Response(
            status=entry.status,
            body=entry.content,
            headers={"x-match": "yes", "x-resp-marker": entry.marker},
        )

    app = web.Application()
    app.router.add_route("*", "/r{idx}", handler)
    return app


@dataclass
class _Server:
    runner: web.AppRunner
    base_url: str


async def _start_server(entries: list[_Entry]) -> _Server:
    runner = web.AppRunner(_build_app(entries))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][0], runner.addresses[0][1]
    return _Server(runner=runner, base_url=f"http://{host}:{port}")


def _request_for(base_url: str, idx: int, entry: _Entry) -> Request:
    url = f"{base_url}/r{idx}"
    if entry.query:
        url = f"{url}?{urlencode(entry.query)}"
    return Request(
        request=HTTPRequestParams(
            method=entry.method,
            url=url,
            headers=dict(entry.headers) or None,
            cookies=dict(entry.cookies) or None,
            data=entry.body or None,
        ),
        continuation="parse",
    )


# --- properties ----------------------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(entries=st.lists(_entry(), min_size=1, max_size=5))
def test_request_and_response_fidelity(entries: list[_Entry]) -> None:
    async def run() -> None:
        server = await _start_server(entries)
        transport = HttpxTransport()
        await transport.open()
        handle = await transport.acquire(0)
        try:
            for i, entry in enumerate(entries):
                request = _request_for(server.base_url, i, entry)
                resp = await transport.resolve(
                    handle, QueuedRequest(request=request, request_id=1)
                )
                # Request fidelity: the server matched what we sent.
                assert resp.headers.get("x-match") == "yes", (
                    f"server rejected the request: {resp.text}"
                )
                # Response fidelity.
                assert resp.status_code == entry.status
                assert resp.content == entry.content
                assert resp.url == request.request.url
                assert resp.headers.get("x-resp-marker") == entry.marker
        finally:
            await transport.aclose()
            await server.runner.cleanup()

    asyncio.run(run())


# --- the matcher is not vacuous ------------------------------------------


async def test_server_rejects_a_missing_header() -> None:
    entry = _Entry(
        method=HttpMethod.GET,
        query={},
        headers={"x-gen-need": "v"},
        cookies={},
        body=b"",
        status=200,
        content=b"ok",
        marker="m",
    )
    server = await _start_server([entry])
    transport = HttpxTransport()
    await transport.open()
    handle = await transport.acquire(0)
    try:
        # A request WITHOUT the required header must be rejected.
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{server.base_url}/r0"
            ),
            continuation="parse",
        )
        resp = await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
        assert resp.headers.get("x-match") == "no"
        assert "header" in resp.text
    finally:
        await transport.aclose()
        await server.runner.cleanup()


async def test_server_rejects_a_wrong_body() -> None:
    entry = _Entry(
        method=HttpMethod.POST,
        query={},
        headers={},
        cookies={},
        body=b"expected",
        status=200,
        content=b"ok",
        marker="m",
    )
    server = await _start_server([entry])
    transport = HttpxTransport()
    await transport.open()
    handle = await transport.acquire(0)
    try:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url=f"{server.base_url}/r0",
                data=b"different",
            ),
            continuation="parse",
        )
        resp = await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
        assert resp.headers.get("x-match") == "no"
        assert "body" in resp.text
    finally:
        await transport.aclose()
        await server.runner.cleanup()


async def test_json_body_is_sent_on_the_wire() -> None:
    """``HTTPRequestParams.json`` is serialized and POSTed by the transport.

    Regression: the transport dropped ``json`` entirely (the queue carries it
    as its own column, not folded into ``data``), so a JSON-body POST went out
    empty. The matching server records exactly what arrived.
    """
    received: dict[str, object] = {}

    async def handler(request: web.Request) -> web.Response:
        received["body"] = await request.read()
        received["content_type"] = request.headers.get("Content-Type")
        return web.Response(status=200, body=b"ok")

    app = web.Application()
    app.router.add_route("*", "/r0", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][0], runner.addresses[0][1]
    base_url = f"http://{host}:{port}"

    payload = {"courtName": "STATE OF ARKANSAS SUPREME COURT", "page": 1}
    transport = HttpxTransport()
    await transport.open()
    handle = await transport.acquire(0)
    try:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url=f"{base_url}/r0",
                json=payload,
            ),
            continuation="parse",
        )
        resp = await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
        assert resp.status_code == 200
        # Compare parsed payloads rather than raw bytes — httpx picks its own
        # JSON separators, but the decoded object must round-trip exactly.
        assert json.loads(received["body"]) == payload  # type: ignore[arg-type]
        assert received["content_type"] == "application/json"
    finally:
        await transport.aclose()
        await runner.cleanup()
