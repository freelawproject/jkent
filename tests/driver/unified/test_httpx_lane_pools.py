"""``HttpxTransport`` keeps one client pool per rate-limit lane.

The default lane with default verification is the main client; every other
(lane, verify) pair gets its own lazily-created pool, so a lane's connections
never queue behind another's. Replaces the old single bypass pool: the
"none" lane is just one more lane.

The pools share one cookie jar: a session is the site's, not a lane's.
"""

from __future__ import annotations

import pathlib
import ssl
from typing import TYPE_CHECKING, Any, ClassVar

import httpx2
import pytest
from aiohttp import web

from jkent.common.exceptions import TransientException, TransientKind
from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ProxySetting,
    Request,
)
from jkent.driver.unified_driver import QueuedRequest
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from tests.httpx_mock import mocked_httpx_transport
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from collections.abc import Mapping


def _request(
    *, rate_limit: str | None = None, verify: bool | str = True
) -> Request:
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
        client, _, _ = await transport._prepare(_request())
        assert client is transport._client
        client, _, _ = await transport._prepare(
            _request(rate_limit=DEFAULT_RATE_LIMIT)
        )
        assert client is transport._client
    finally:
        await transport.aclose()


async def test_each_lane_gets_its_own_pool_reused_across_requests() -> None:
    transport = HttpxTransport()
    await transport.open()
    try:
        none_a, _, _ = await transport._prepare(
            _request(rate_limit=NO_RATE_LIMIT)
        )
        none_b, _, _ = await transport._prepare(
            _request(rate_limit=NO_RATE_LIMIT)
        )
        downloads, _, _ = await transport._prepare(
            _request(rate_limit="downloads")
        )
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
        default_unverified, _, _ = await transport._prepare(
            _request(verify=False)
        )
        none_unverified, _, _ = await transport._prepare(
            _request(rate_limit=NO_RATE_LIMIT, verify=False)
        )
        none_verified, _, _ = await transport._prepare(
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
    await transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    await transport._prepare(_request(rate_limit="downloads", verify=False))
    await transport.aclose()
    assert transport._client is None
    assert transport._alt_clients == {}


async def test_aclose_closes_every_lane_pool_when_one_close_fails() -> None:
    transport = HttpxTransport()
    await transport.open()
    await transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    await transport._prepare(_request(rate_limit="downloads", verify=False))
    alts = list(transport._alt_clients.values())
    main = transport._client
    assert main is not None

    async def fail() -> None:
        raise RuntimeError("close failed")

    main.aclose = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="close failed"):
        await transport.aclose()
    assert all(client.is_closed for client in alts)


async def test_a_request_mid_close_creates_no_pool() -> None:
    """A request that arrives while ``aclose`` awaits a pool's close raises
    rather than creating a lane pool the close has already passed."""
    transport = HttpxTransport()
    await transport.open()
    await transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    lane_pool = transport._alt_clients[(NO_RATE_LIMIT, True)]
    main = transport._client
    assert main is not None
    close_main = main.aclose
    late: list[BaseException] = []

    async def aclose_with_a_late_request() -> None:
        try:
            await transport._prepare(_request(rate_limit="downloads"))
        except TransientException as exc:
            late.append(exc)
        await close_main()

    main.aclose = aclose_with_a_late_request  # type: ignore[method-assign]
    await transport.aclose()

    assert len(late) == 1
    assert transport._alt_clients == {}
    assert main.is_closed
    assert lane_pool.is_closed


@pytest.mark.parametrize(
    ("closed", "error"),
    [(False, RuntimeError), (True, TransientException)],
    ids=["unopened", "closed"],
)
async def test_a_lane_pool_is_not_created_outside_open(
    closed: bool, error: type[Exception]
) -> None:
    """Before ``open()`` or after ``aclose()`` a lane request raises like a
    default-lane one, rather than creating a pool nothing will close."""
    transport = HttpxTransport()
    if closed:
        await transport.open()
        await transport.aclose()
    with pytest.raises(error):
        await transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
    assert transport._alt_clients == {}


async def _session_site() -> tuple[StartedServer, list[str | None]]:
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
        base = server.base_url
        await transport.resolve(handle, _at(f"{base}/login"))
        await transport.resolve(
            handle, _at(f"{base}/file", rate_limit="downloads")
        )
        assert seen == ["session=s3cret"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()


async def test_request_cookies_add_to_the_session_not_replace_it() -> None:
    server, seen = await _session_site()
    transport = HttpxTransport()
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url
        await transport.resolve(handle, _at(f"{base}/login"))
        await transport.resolve(
            handle, _at(f"{base}/file", cookies={"page": "2"})
        )
        assert seen == ["session=s3cret; page=2"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()


@pytest.mark.parametrize("name", ["Cookie", "cookie", "COOKIE"])
async def test_an_explicit_cookie_header_joins_the_request_cookies(
    name: str,
) -> None:
    """An explicit ``Cookie`` header (any case) is the request's own, like
    its ``cookies``: both go over the session, in one header."""
    server, seen = await _session_site()
    transport = HttpxTransport()
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url
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
        assert seen == ["session=s3cret; mine=1; page=2"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()


class _CookieDefaultScraper(BaseScraper[Any]):
    default_headers: ClassVar[Mapping[str, str]] = {"Cookie": "consent=1"}


async def test_a_default_cookie_header_does_not_hide_the_session() -> None:
    """A ``Cookie`` in ``default_headers`` would stop httpx2's jar adding
    its own; it is folded into the request's cookies instead."""
    server, seen = await _session_site()
    transport = HttpxTransport(scraper=_CookieDefaultScraper)
    await transport.open()
    try:
        handle = await transport.acquire(0)
        base = server.base_url
        await transport.resolve(handle, _at(f"{base}/login"))
        await transport.resolve(
            handle, _at(f"{base}/file", cookies={"consent": "2"})
        )
        assert seen == ["session=s3cret; consent=2"]
    finally:
        await transport.aclose()
        await server.runner.cleanup()


async def test_lanes_share_one_ssl_context_per_verify_value() -> None:
    """CA material loads once per ``verify`` value, not once per pool."""
    transport = HttpxTransport()
    await transport.open()
    try:
        await transport._prepare(_request(rate_limit=NO_RATE_LIMIT))
        await transport._prepare(_request(rate_limit="downloads"))
        assert list(transport._ssl_contexts) == [True]
    finally:
        await transport.aclose()


async def _proxy_site() -> tuple[StartedServer, list[str]]:
    """A forward proxy that answers every request itself, recording the
    absolute-form target it was asked for."""
    seen: list[str] = []

    async def proxy(request: web.Request) -> web.Response:
        seen.append(str(request.url))
        return web.Response(text="via proxy")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", proxy)
    return await start_app(app), seen


class _ProxiedScraper(BaseScraper[Any]):
    proxy: ClassVar[str | ProxySetting | None] = "http://scraper-proxy:3128"


class _DirectScraper(BaseScraper[Any]):
    proxy: ClassVar[str | ProxySetting | None] = ProxySetting.DIRECT


@pytest.mark.parametrize(
    ("scraper", "proxy", "expected"),
    [
        (BaseScraper, None, (None, True)),
        (BaseScraper, ProxySetting.DIRECT, (None, False)),
        (
            BaseScraper,
            "http://run-proxy:3128",
            ("http://run-proxy:3128", False),
        ),
        (_ProxiedScraper, None, ("http://scraper-proxy:3128", False)),
        (_DirectScraper, None, (None, False)),
        (
            _ProxiedScraper,
            "http://run-proxy:3128",
            ("http://run-proxy:3128", False),
        ),
        (_ProxiedScraper, ProxySetting.ENVIRONMENT, (None, True)),
    ],
    ids=[
        "environment by default",
        "run: direct",
        "run: proxy url",
        "scraper: proxy url",
        "scraper: direct",
        "run over scraper",
        "run: environment over scraper",
    ],
)
def test_the_proxy_is_the_first_choice_made(
    scraper: type[BaseScraper[Any]],
    proxy: str | ProxySetting | None,
    expected: tuple[str | None, bool],
) -> None:
    """The run's choice, else the scraper's, else the environment's."""
    kwargs = HttpxTransport(scraper=scraper, proxy=proxy)._client_kwargs(True)
    assert (kwargs["proxy"], kwargs["trust_env"]) == expected


@pytest.mark.parametrize("proxy", [None, ProxySetting.DIRECT])
async def test_the_environment_proxy_applies_unless_overridden(
    monkeypatch: pytest.MonkeyPatch, proxy: ProxySetting | None
) -> None:
    """By default an ``HTTP_PROXY`` in the environment carries the request;
    ``ProxySetting.DIRECT`` goes straight to the site instead."""
    proxy_server, proxied = await _proxy_site()
    site, _ = await _session_site()
    monkeypatch.setenv("HTTP_PROXY", proxy_server.base_url)
    monkeypatch.setenv("NO_PROXY", "")
    transport = HttpxTransport(proxy=proxy)
    await transport.open()
    try:
        handle = await transport.acquire(0)
        response = await transport.resolve(
            handle, _at(f"{site.base_url}/login")
        )
    finally:
        await transport.aclose()
        await site.runner.cleanup()
        await proxy_server.runner.cleanup()

    if proxy is None:
        assert response.text == "via proxy"
        assert proxied == [f"{site.base_url}/login"]
    else:
        assert response.text == "ok"
        assert proxied == []


def test_ssl_cert_file_sets_the_default_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """``SSL_CERT_FILE`` replaces the OS trust store for default
    verification, as a proxy set in the environment usually needs."""
    certifi = pytest.importorskip("certifi")
    bundle = pathlib.Path(certifi.where()).read_text()
    end = "-----END CERTIFICATE-----"
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text(bundle[: bundle.index(end) + len(end)] + "\n")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_file))

    context = HttpxTransport()._ssl_context_for(True)

    assert len(context.get_ca_certs()) == 1


class _SSLScraper(BaseScraper[Any]):
    ssl_context: ClassVar[ssl.SSLContext | None] = ssl.create_default_context()


def test_the_scraper_ssl_context_is_the_default_verification() -> None:
    assert (
        HttpxTransport(scraper=_SSLScraper)._ssl_context_for(True)
        is _SSLScraper.ssl_context
    )


def test_the_run_ssl_context_wins_over_the_scraper() -> None:
    run_context = ssl.create_default_context()
    transport = HttpxTransport(scraper=_SSLScraper, ssl_context=run_context)
    assert transport._ssl_context_for(True) is run_context


async def test_the_mocked_transport_shares_the_session_jar() -> None:
    """``mocked_httpx_transport``'s client stores cookies in the jar every
    lane pool reads, as ``open``'s client does."""

    def login(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"Set-Cookie": "session=s3cret"})

    transport = mocked_httpx_transport(login)
    try:
        handle = await transport.acquire(0)
        await transport.resolve(handle, _at("http://test/login"))
        assert {c.name: c.value for c in transport._cookie_jar} == {
            "session": "s3cret"
        }
    finally:
        await transport.aclose()


async def test_a_request_whose_pool_closes_under_it_is_transient() -> None:
    """``aclose`` landing between a request taking its pool and sending on
    it — a file body read off the loop, a redirect hop — is a NETWORK
    transient, so the request is retried rather than failed outright."""
    transport = mocked_httpx_transport(lambda _: httpx2.Response(200))
    client, request, own = await transport._prepare(_request())
    await transport.aclose()

    with pytest.raises(TransientException) as excinfo:
        await transport._send(client, request, own, stream=False)

    assert excinfo.value.kind is TransientKind.NETWORK
