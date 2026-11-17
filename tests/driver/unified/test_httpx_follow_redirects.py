"""FOLLOW_REDIRECTS through ``HttpxTransport``.

The unified port of the old request-manager behavior
(tests/unorganized/test_follow_redirects.py): a scraper opts into httpx2
redirect-following by declaring ``DriverRequirement.FOLLOW_REDIRECTS``;
without it, redirect responses come back unfollowed. Covers both the
resolve and the streaming (archive) paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import httpx2
import pytest
from aiohttp import web

from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    TransientException,
    TransientKind,
)
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver import HttpxTransport, QueuedRequest
from tests.httpx_mock import mocked_httpx_transport
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping


class _RedirectingScraper(BaseScraper[dict[str, Any]]):
    driver_requirements: ClassVar[list[DriverRequirement]] = [
        DriverRequirement.FOLLOW_REDIRECTS,
    ]


class _PlainScraper(BaseScraper[dict[str, Any]]):
    pass


class _ManualRedirectScraper(BaseScraper[dict[str, Any]]):
    """No FOLLOW_REDIRECTS: consumes redirect responses itself.

    302 is in no default bucket, so the classifier's unlisted-is-persistent
    fallback would fail it fast; a scraper that reads redirects manually
    must claim the code as successful to receive the response.
    """

    HTTP_CODE_TYPES: ClassVar[Mapping[int, HTTPCodeType]] = {
        302: HTTPCodeType.SUCCESSFUL
    }


def _redirect_handler(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == "/start":
        return httpx2.Response(302, headers={"Location": "/dest"})
    if request.url.path == "/dest":
        return httpx2.Response(200, text="final")
    return httpx2.Response(404)


def _make_request(url: str) -> Request:
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        step="parse_page",
    )


@pytest.fixture
async def redirect_server() -> AsyncIterator[StartedServer]:
    """``/a`` redirects to ``/b``, which answers 200."""

    async def a(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/b")

    async def b(_request: web.Request) -> web.Response:
        return web.Response(text="final")

    app = web.Application()
    app.router.add_get("/a", a)
    app.router.add_get("/b", b)
    started = await start_app(app)
    try:
        yield started
    finally:
        await started.aclose()


class TestHttpxTransportFollowRedirects:
    async def test_resolve_follows_redirect(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _RedirectingScraper
        )
        try:
            handle = await transport.acquire(0)
            response = await transport.resolve(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            assert response.status_code == 200
            assert response.text == "final"
        finally:
            await transport.aclose()

    async def test_resolve_does_not_follow_when_off(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _ManualRedirectScraper
        )
        try:
            handle = await transport.acquire(0)
            response = await transport.resolve(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            assert response.status_code == 302
        finally:
            await transport.aclose()

    async def test_archive_stream_follows_redirect(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _RedirectingScraper
        )
        try:
            handle = await transport.acquire(0)
            stream = await transport.resolve_archive(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            try:
                body = b"".join([chunk async for chunk in stream])
                assert stream.status_code == 200
                assert body == b"final"
            finally:
                await transport.finish_archiving(stream)
        finally:
            await transport.aclose()

    async def test_archive_stream_does_not_follow_when_off(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _ManualRedirectScraper
        )
        try:
            handle = await transport.acquire(0)
            stream = await transport.resolve_archive(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            try:
                assert stream.status_code == 302
            finally:
                await transport.finish_archiving(stream)
        finally:
            await transport.aclose()

    @pytest.mark.parametrize(
        "archive", [False, True], ids=["resolve", "archive"]
    )
    async def test_url_is_the_final_url_after_redirects(
        self, redirect_server: StartedServer, archive: bool
    ) -> None:
        transport = HttpxTransport(scraper=_RedirectingScraper)
        await transport.open()
        try:
            handle = await transport.acquire(0)
            queued = QueuedRequest(
                request=_make_request(f"{redirect_server.base_url}/a"),
                request_id=1,
            )
            if archive:
                stream = await transport.resolve_archive(handle, queued)
                try:
                    assert stream.status_code == 200
                    assert stream.url == f"{redirect_server.base_url}/b"
                finally:
                    await transport.finish_archiving(stream)
            else:
                response = await transport.resolve(handle, queued)
                assert response.status_code == 200
                assert response.url == f"{redirect_server.base_url}/b"
        finally:
            await transport.aclose()


def _redirect_to_missing(request: httpx2.Request) -> httpx2.Response:
    if request.url.path == "/start":
        return httpx2.Response(302, headers={"Location": "/gone"})
    return httpx2.Response(404)


class TestHttpxTransportRedirectUrl:
    """``url`` is where the exchange landed, success or failure alike."""

    async def test_a_followed_response_carries_the_final_url(self) -> None:
        transport = mocked_httpx_transport(
            _redirect_handler, _RedirectingScraper
        )
        try:
            handle = await transport.acquire(0)
            response = await transport.resolve(
                handle,
                QueuedRequest(
                    request=_make_request("http://test/start"), request_id=1
                ),
            )
            assert response.url == "http://test/dest"
        finally:
            await transport.aclose()

    @pytest.mark.parametrize("path", ["resolve", "archive"])
    async def test_a_failure_after_a_redirect_names_the_final_url(
        self, path: str
    ) -> None:
        transport = mocked_httpx_transport(
            _redirect_to_missing, _RedirectingScraper
        )
        queued = QueuedRequest(
            request=_make_request("http://test/start"), request_id=1
        )
        try:
            handle = await transport.acquire(0)
            with pytest.raises(PersistentHTTPResponseException) as excinfo:
                if path == "resolve":
                    await transport.resolve(handle, queued)
                else:
                    await transport.resolve_archive(handle, queued)
        finally:
            await transport.aclose()

        assert excinfo.value.url == "http://test/gone"


def _cookie_redirect_handler(
    seen: list[str | None],
) -> Callable[[httpx2.Request], httpx2.Response]:
    """``/start`` sets a session cookie and redirects to ``/dest``, which
    records the Cookie header it got."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/start":
            return httpx2.Response(
                302,
                headers={"Location": "/dest", "Set-Cookie": "session=s1"},
            )
        seen.append(request.headers.get("Cookie"))
        return httpx2.Response(200, text="final")

    return handler


class TestHttpxTransportRedirectCookies:
    """A request's own cookies ride every hop, over the session's."""

    @pytest.mark.parametrize("path", ["resolve", "archive"])
    async def test_request_cookies_survive_a_redirect(self, path: str) -> None:
        seen: list[str | None] = []
        transport = mocked_httpx_transport(
            _cookie_redirect_handler(seen), _RedirectingScraper
        )
        queued = QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url="http://test/start",
                    headers={"Cookie": "mine=1"},
                    cookies={"page": "2"},
                ),
                step="parse_page",
            ),
            request_id=1,
        )
        try:
            handle = await transport.acquire(0)
            if path == "resolve":
                await transport.resolve(handle, queued)
            else:
                stream = await transport.resolve_archive(handle, queued)
                await transport.finish_archiving(stream)
        finally:
            await transport.aclose()

        assert seen == ["session=s1; mine=1; page=2"]

    @pytest.mark.parametrize(
        "set_cookie",
        ["sid=new; Path=/other", "sid=new; Domain=test.example"],
        ids=["other path", "domain cookie"],
    )
    async def test_a_redirect_keeps_a_request_cookie_it_sets_apart(
        self, set_cookie: str
    ) -> None:
        """A ``Set-Cookie`` for another path or for the domain is a separate
        cookie in a browser's jar, so the request's own cookie of that name
        — host-only, path ``/`` — still goes on later hops."""
        seen: list[tuple[str, str | None]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append((request.url.path, request.headers.get("Cookie")))
            if request.url.path == "/login":
                return httpx2.Response(
                    302,
                    headers={"Location": "/home", "Set-Cookie": set_cookie},
                )
            return httpx2.Response(200, text="home")

        transport = mocked_httpx_transport(handler, _RedirectingScraper)
        queued = QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url="http://test.example/login",
                    cookies={"sid": "old"},
                ),
                step="parse_page",
            ),
            request_id=1,
        )
        try:
            await transport.resolve(await transport.acquire(0), queued)
        finally:
            await transport.aclose()

        assert seen == [("/login", "sid=old"), ("/home", "sid=old")]

    @pytest.mark.parametrize(
        ("set_cookie", "expected"),
        [
            ("sid=new; Path=/", "sid=new"),
            ("sid=new", "sid=new"),
            ("sid=; Max-Age=0; Path=/", None),
        ],
        ids=["reset", "reset at default path", "deleted"],
    )
    @pytest.mark.parametrize("own", ["cookies", "header"])
    async def test_a_redirect_supersedes_a_request_cookie_it_sets(
        self, set_cookie: str, expected: str | None, own: str
    ) -> None:
        """A ``Set-Cookie`` on a same-host redirect wins over the request's
        own cookie of that name on later hops, as it would in a browser's
        jar: a login redirect that rotates the session lands with the new
        one, and one that deletes it lands with none."""
        seen: list[tuple[str, str | None]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append((request.url.path, request.headers.get("Cookie")))
            if request.url.path == "/login":
                return httpx2.Response(
                    302,
                    headers={"Location": "/home", "Set-Cookie": set_cookie},
                )
            return httpx2.Response(200, text="home")

        transport = mocked_httpx_transport(handler, _RedirectingScraper)
        queued = QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url="http://test/login",
                    headers={"Cookie": "sid=old"} if own == "header" else {},
                    cookies={"sid": "old"} if own == "cookies" else None,
                ),
                step="parse_page",
            ),
            request_id=1,
        )
        try:
            await transport.resolve(await transport.acquire(0), queued)
        finally:
            await transport.aclose()

        assert seen == [("/login", "sid=old"), ("/home", expected)]

    async def test_another_hosts_set_cookie_keeps_the_request_cookie(
        self,
    ) -> None:
        """A cookie set by a redirect from another host is that host's: it
        does not unseat the request's own cookie when the chain returns."""
        seen: list[tuple[str, str | None]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            seen.append((request.url.host, request.headers.get("Cookie")))
            if request.url.path == "/start":
                return httpx2.Response(
                    302, headers={"Location": "http://sso.test/hop"}
                )
            if request.url.host == "sso.test":
                return httpx2.Response(
                    302,
                    headers={
                        "Location": "http://test/dest",
                        "Set-Cookie": "sid=sso",
                    },
                )
            return httpx2.Response(200, text="final")

        transport = mocked_httpx_transport(handler, _RedirectingScraper)
        queued = QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url="http://test/start",
                    cookies={"sid": "mine"},
                ),
                step="parse_page",
            ),
            request_id=1,
        )
        try:
            await transport.resolve(await transport.acquire(0), queued)
        finally:
            await transport.aclose()

        assert seen == [
            ("test", "sid=mine"),
            ("sso.test", None),
            ("test", "sid=mine"),
        ]


def _redirect_to(location: str) -> Callable[[httpx2.Request], httpx2.Response]:
    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(302, headers={"Location": location})

    return handler


@pytest.mark.parametrize(
    "location", ["ftp://test/file", "mailto:clerk@test", "javascript:void(0)"]
)
@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_a_redirect_to_an_unfetchable_url_is_site_degraded(
    location: str, path: str
) -> None:
    """The site chose the Location, so it is the site's fault and
    retryable — not a ``ScraperConfigError``."""
    transport = mocked_httpx_transport(
        _redirect_to(location), _RedirectingScraper
    )
    queued = QueuedRequest(
        request=_make_request("http://test/start"), request_id=1
    )
    try:
        handle = await transport.acquire(0)
        with pytest.raises(TransientException) as excinfo:
            if path == "resolve":
                await transport.resolve(handle, queued)
            else:
                await transport.resolve_archive(handle, queued)
    finally:
        await transport.aclose()

    assert excinfo.value.kind is TransientKind.SITE_DEGRADED


def _cross_host_cookie_handler(
    seen: dict[str, str | None],
) -> Callable[[httpx2.Request], httpx2.Response]:
    """``test/start`` redirects off-site to ``other/dest``, which redirects
    back to ``test/back``; each records the Cookie header it got."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen[f"{request.url.host}{request.url.path}"] = request.headers.get(
            "Cookie"
        )
        if request.url.path == "/start":
            return httpx2.Response(
                302, headers={"Location": "http://other/dest"}
            )
        if request.url.path == "/dest":
            return httpx2.Response(
                302, headers={"Location": "http://test/back"}
            )
        return httpx2.Response(200, text="final")

    return handler


@pytest.mark.parametrize("path", ["resolve", "archive"])
async def test_request_cookies_stay_on_their_host_across_redirects(
    path: str,
) -> None:
    """A request's own cookies are its host's: a hop off-site goes without
    them, and a hop back to the host carries them again."""
    seen: dict[str, str | None] = {}
    transport = mocked_httpx_transport(
        _cross_host_cookie_handler(seen), _RedirectingScraper
    )
    queued = QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://test/start",
                headers={"Cookie": "mine=1"},
                cookies={"page": "2"},
            ),
            step="parse_page",
        ),
        request_id=1,
    )
    try:
        handle = await transport.acquire(0)
        if path == "resolve":
            await transport.resolve(handle, queued)
        else:
            stream = await transport.resolve_archive(handle, queued)
            await transport.finish_archiving(stream)
    finally:
        await transport.aclose()

    assert seen == {
        "test/start": "mine=1; page=2",
        "other/dest": None,
        "test/back": "mine=1; page=2",
    }


@pytest.mark.parametrize("status", [302, 303])
async def test_a_post_redirected_to_get_drops_its_body_headers(
    status: int,
) -> None:
    """As a browser does: the GET has no body, so no Content-Type."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        if request.url.path == "/form":
            return httpx2.Response(status, headers={"Location": "/results"})
        return httpx2.Response(200, text="final")

    transport = mocked_httpx_transport(handler, _RedirectingScraper)
    queued = QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url="http://test/form",
                data={"a": "1"},
                headers={"Content-Language": "en"},
            ),
            step="parse_page",
        ),
        request_id=1,
    )
    try:
        handle = await transport.acquire(0)
        await transport.resolve(handle, queued)
    finally:
        await transport.aclose()

    post, get = seen
    assert post.headers["Content-Type"] == "application/x-www-form-urlencoded"
    assert get.method == "GET"
    assert "Content-Type" not in get.headers
    assert "Content-Language" not in get.headers


def _scheme_hop_handler(
    seen: dict[str, str | None], location: str
) -> Callable[[httpx2.Request], httpx2.Response]:
    """``/start`` redirects to ``location``; each hop records its Cookie."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen[f"{request.url.scheme}:{request.url.path}"] = request.headers.get(
            "Cookie"
        )
        if request.url.path == "/start":
            return httpx2.Response(302, headers={"Location": location})
        return httpx2.Response(200, text="final")

    return handler


@pytest.mark.parametrize(
    ("start", "location", "expected"),
    [
        (
            "https://test/start",
            "http://test/dest",
            {"https:/start": "page=2", "http:/dest": None},
        ),
        (
            "http://test/start",
            "https://test/dest",
            {"http:/start": "page=2", "https:/dest": "page=2"},
        ),
        (
            "http://test/start",
            "http://test/dest",
            {"http:/start": "page=2", "http:/dest": "page=2"},
        ),
    ],
    ids=["https to http", "http to https", "http to http"],
)
async def test_request_cookies_never_downgrade_to_plain_http(
    start: str, location: str, expected: dict[str, str | None]
) -> None:
    """After an https first hop, a request's own cookies go over https only;
    a request that started on plain http carries them as it always did."""
    seen: dict[str, str | None] = {}
    transport = mocked_httpx_transport(
        _scheme_hop_handler(seen, location), _RedirectingScraper
    )
    queued = QueuedRequest(
        request=Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=start, cookies={"page": "2"}
            ),
            step="parse_page",
        ),
        request_id=1,
    )
    try:
        handle = await transport.acquire(0)
        await transport.resolve(handle, queued)
    finally:
        await transport.aclose()

    assert seen == expected


async def test_a_redirect_body_is_not_decoded() -> None:
    """A hop's body is discarded unread by the scraper, so one labeled gzip
    that is not gzip does not fail the chain, as it would not in a
    browser."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/a":
            return httpx2.Response(
                302,
                headers={"Location": "/b", "Content-Encoding": "gzip"},
                stream=httpx2.ByteStream(b"Moved"),
            )
        return httpx2.Response(200, text="final")

    transport = mocked_httpx_transport(handler, _RedirectingScraper)
    try:
        response = await transport.resolve(
            await transport.acquire(0),
            QueuedRequest(
                request=_make_request("http://test/a"), request_id=1
            ),
        )
    finally:
        await transport.aclose()

    assert response.status_code == 200
    assert response.text == "final"
