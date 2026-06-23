"""Generative status-code classification sweep for ``HttpxTransport``.

``resolve`` consults the scraper's classifier — ``is_transient_error`` /
``is_persistent_error``, which merge the framework defaults with the scraper's
``HTTP_CODE_TYPES`` override mapping — via the shared
``Transport.classify_and_raise``, and:

  - returns a ``Response`` for successful codes,
  - raises ``HTTPResponseAssumptionException`` for transient codes,
  - raises ``PersistentHTTPResponseException`` for persistent codes — which
    includes any code absent from the active map (the classifier's
    unlisted-is-persistent fallback).

The cross-transport contract (fixed representative codes, dynamic
content-based overrides, speculative narrowing) lives in
``ClassificationConformance`` (``test_transport_conformance``), bound to every
transport. What stays here is the *generative* half — the full code matrix and
arbitrary set-overrides, hypothesis-driven with the scraper's own classifier
as the oracle — which only the cheap HTTP transport can afford to sweep (a
fresh server + transport per example is too slow for a browser).

Out of scope: archive download. Header-based dynamic overrides aren't
exercised (server/client add their own headers, so the oracle stays
header-independent).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Literal, cast

import pytest
from aiohttp import web
from hypothesis import given, settings
from hypothesis import strategies as st

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
)
from jkent.data_types import (
    BaseScraper,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from tests.driver.unified.conftest import start_app

_SUCCESS = [200, 201, 202, 203]
_TRANSIENT = [408, 425, 429, 502, 503, 504]
_PERSISTENT = [400, 401, 403, 404, 409, 422, 500, 501]
# In no default bucket (redirects, nginx/Cloudflare nonstandards): persistent
# via the classifier's fallback unless an override map rescues them.
_UNLISTED = [302, 499, 520, 522]
_ALL = _SUCCESS + _TRANSIENT + _PERSISTENT + _UNLISTED

Outcome = Literal["response", "transient", "persistent"]


def _expected(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> Outcome:
    """The scraper's own verdict — the oracle ``_classify_and_raise`` follows."""
    if scraper.is_transient_error(code, None, body):
        return "transient"
    if scraper.is_persistent_error(code, None, body):
        return "persistent"
    return "response"


def _make_scraper(
    code_types: Mapping[int, HTTPCodeType],
) -> type[BaseScraper[Any]]:
    """A BaseScraper subclass with the given status-code override map."""
    cls = type(
        "_OverrideScraper",
        (BaseScraper,),
        {"HTTP_CODE_TYPES": dict(code_types)},
    )
    return cast("type[BaseScraper[Any]]", cls)


def _status_app(status: int, body: bytes) -> web.Application:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(status=status, body=body)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    return app


async def _resolve_status(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> Any:
    """Serve ``code``+``body`` and resolve one GET through HttpxTransport(scraper)."""
    server = await start_app(_status_app(code, body))
    transport = HttpxTransport(scraper=scraper)
    await transport.open()
    handle = await transport.acquire(0)
    try:
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{server.base_url}/r"
            ),
            continuation="parse",
        )
        return await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
    finally:
        await transport.aclose()
        await server.runner.cleanup()


def _assert_outcome(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> None:
    outcome = _expected(scraper, code, body)

    async def run() -> None:
        if outcome == "response":
            resp = await _resolve_status(scraper, code, body)
            assert resp.status_code == code
            assert resp.content == body
        elif outcome == "transient":
            with pytest.raises(HTTPResponseAssumptionException):
                await _resolve_status(scraper, code, body)
        else:
            with pytest.raises(PersistentHTTPResponseException):
                await _resolve_status(scraper, code, body)

    asyncio.run(run())


# --- default classification ----------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(code=st.sampled_from(_ALL))
def test_default_classification(code: int) -> None:
    _assert_outcome(BaseScraper, code, b"<html>body</html>")


# --- arbitrary set overrides ----------------------------------------------


@pytest.mark.generative
@settings(deadline=None)
@given(
    code=st.sampled_from(_ALL),
    # A mapping holds one type per code, so the override is disjoint by
    # construction — a code can never land in two buckets.
    code_types=st.dictionaries(
        keys=st.sampled_from(_ALL),
        values=st.sampled_from(list(HTTPCodeType)),
        max_size=5,
    ),
)
def test_override_classification(
    code: int, code_types: dict[int, HTTPCodeType]
) -> None:
    scraper = _make_scraper(code_types)
    _assert_outcome(scraper, code, b"x")
