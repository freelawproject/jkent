"""``HttpxTransport`` keeps one client pool per rate-limit lane.

The default lane with default verification is the main client; every other
(lane, verify) pair gets its own lazily-created pool, so a lane's connections
never queue behind another's. Replaces the old single bypass pool: the
"none" lane is just one more lane.

The pools share one cookie jar: a session is the site's, not a lane's.
"""

from __future__ import annotations

import pytest
from aiohttp import web

from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import QueuedRequest
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from tests.servers import start_app


def _request(*, rate_limit: str | None = None, verify: bool | str = True):
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/", verify=verify
        ),
        step="parse",
        rate_limit=rate_limit,
    )


async def test_default_lane_uses_the_main_client() -> None:
    transport = HttpxTransport()
    await transport.open()
    try:
        client, _ = transport._prepare(_request())
        assert client is transport._client
        client, _ = transport._prepare(_request(rate_limit=DEFAULT_RATE_LIMIT))
        assert client is transport._client
    finally:
        await transport.aclose()


async def test_each_lane_gets_its_own_pool_reused_across_requests() -> None:
    transport = HttpxTransport()
    await transport.open()
    try:
        none_a, _ = transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
        none_b, _ = transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
        downloads, _ = transport._prepare(_request(rate_limit="downloads"))
        assert none_a is none_b
        assert none_a is not transport._client
        assert downloads is not none_a
        assert downloads is not transport._client
    finally:
        await transport.aclose()


async def test_lane_and_verify_combine() -> None:
    transport = HttpxTransport()
    await transport.open()
    try:
        default_unverified, _ = transport._prepare(_request(verify=False))
        none_unverified, _ = transport._prepare(
            _request(rate_limit=NO_RATE_LIMIT, verify=False)
        )
        none_verified, _ = transport._prepare(
            _request(rate_limit=NO_RATE_LIMIT)
        )
        assert (
            len(
                {
                    id(default_unverified),
                    id(none_unverified),
                    id(none_verified),
                    id(transport._client),
                }
            )
            == 4
        )
    finally:
        await transport.aclose()


async def test_aclose_closes_every_lane_pool() -> None:
    transport = HttpxTransport()
    await transport.open()
    transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    transport._prepare(_request(rate_limit="downloads", verify=False))
    await transport.aclose()
    assert transport._client is None
    assert transport._alt_clients == {}


@pytest.mark.parametrize("closed", [False, True], ids=["unopened", "closed"])
async def test_a_lane_pool_is_not_created_outside_open(closed: bool) -> None:
    """Before ``open()`` or after ``aclose()`` a lane request raises like a
    default-lane one, rather than creating a pool nothing will close."""
    transport = HttpxTransport()
    if closed:
        await transport.open()
        await transport.aclose()
    with pytest.raises(RuntimeError, match="open"):
        transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    assert transport._alt_clients == {}


async def _session_site() -> tuple[object, list[str | None]]:
    """``/login`` sets a session cookie; ``/file`` records the Cookie header."""
    seen: list[str | None] = []

    async def login(_request: web.Request) -> web.Response:
        response = web.Response(text="ok")
        response.set_cookie("session", "s3cret")
        return response

    async def file(request: web.Request) -> web.Response:
        seen.append(request.headers.get("Cookie"))
        return web.Response(body=b"%PDF")

    app = web.Application()
    app.router.add_get("/login", login)
    app.router.add_get("/file", file)
    return await start_app(app), seen


def _at(
    url: str,
    *,
    rate_limit: str | None = None,
    cookies: dict[str, str] | None = None,
) -> QueuedRequest:
    return QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=url, cookies=cookies
            ),
            step="parse",
            rate_limit=rate_limit,
        ),
        request_id=1,
    )


async def test_a_session_cookie_reaches_every_lane() -> None:
    server, seen = await _session_site()
    transport = HttpxTransport()
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url  # type: ignore[attr-defined]
        await transport.resolve(handle, _at(f"{base}/login"))
        await transport.resolve(
            handle, _at(f"{base}/file", rate_limit="downloads")
        )
        assert seen == ["session=s3cret"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()  # type: ignore[attr-defined]


async def test_request_cookies_add_to_the_session_not_replace_it() -> None:
    server, seen = await _session_site()
    transport = HttpxTransport()
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url  # type: ignore[attr-defined]
        await transport.resolve(handle, _at(f"{base}/login"))
        await transport.resolve(
            handle, _at(f"{base}/file", cookies={"page": "2"})
        )
        assert seen == ["session=s3cret; page=2"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["Cookie", "cookie", "COOKIE"])
async def test_request_cookies_extend_an_explicit_cookie_header(
    name: str,
) -> None:
    """An explicit ``Cookie`` header (any case) is extended by the request's
    cookies in one header, and the session jar stays out of it."""
    server, seen = await _session_site()
    transport = HttpxTransport()
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url  # type: ignore[attr-defined]
        await transport.resolve(handle, _at(f"{base}/login"))
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{base}/file",
                headers={name: "mine=1"},
                cookies={"page": "2"},
            ),
            step="parse",
        )
        await transport.resolve(
            handle, QueuedRequest(request=request, request_id=1)
        )
        assert seen == ["mine=1; page=2"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()  # type: ignore[attr-defined]
