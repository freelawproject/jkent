"""HTTP transport — a self-contained async httpx client.

The wire behavior is fixed and explicit: ``resolve`` sends ``method``,
``url`` (verbatim — query baked in), the scraper's ``default_headers``
overlaid with explicitly-set ``headers`` (per-request wins per name,
case-insensitively), cookies merged into a ``Cookie`` header, the body
(``data`` bytes as content / dict as form), and a ``json`` payload
(serialized by httpx as a JSON body). ``Response.url`` is the *request* URL.
``params`` is not re-sent — the queue folds it into the url upstream — but
``json`` is carried through as its own column and re-sent here, so a
request's JSON body is preserved end-to-end.

Redirect-following is per-scraper: ``DriverRequirement.FOLLOW_REDIRECTS``
opts the whole transport into ``follow_redirects=True`` on every request
(resolve and stream paths alike).

There is deliberately no transport-level rate limiting: the unified driver
rate-limits in the worker via its own
:class:`~jkent.driver.unified_driver.rate_limiter.RateLimiter`. Each
rate-limit *lane* other than the default does get its own client pool, so a
lane's connections never queue behind another's.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import ssl
from collections.abc import Iterable, Iterator
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlencode

import h11._events as _h11_events
import h11._headers as _h11_headers
import httpx
from typing_extensions import override

from jkent.common.exceptions import (
    RequestTimeoutException,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.common.rate_limits import DEFAULT_RATE_LIMIT
from jkent.common.request import DEFAULT_TIMEOUT_S
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    Request,
    RequestData,
    Response,
    TimeoutType,
    ViaFormSubmit,
)

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper


from jkent.common.headers import merge_headers
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    NoopHandle,
    StatelessTransport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from jkent.driver.unified_driver.transport import (
        AwaitCondition,
        QueuedRequest,
    )

logger = logging.getLogger(__name__)


# Opt-in leniency patches for h11's strict header validation.
# The patch is gated on a :class:`ContextVar` so it only loosens behavior for
# the scraper that asked for it. Other scrapers running concurrently in the
# same process see vanilla h11.
# Pinned to a specific h11 version in ``pyproject.toml`` because the patch
# depends on h11 internals (``h11._headers.normalize_and_validate``).

_lenient_te: contextvars.ContextVar[bool] = contextvars.ContextVar[bool](
    "jkent_h11_lenient_te", default=False
)

_orig_normalize_and_validate = _h11_headers.normalize_and_validate


def _dedupe_transfer_encoding(
    headers: Iterable[tuple[Any, Any]],
) -> list[tuple[Any, Any]]:
    """``headers`` with identical ``Transfer-Encoding`` repeats dropped.

    Distinct values (``gzip`` then ``chunked``) all pass through, so h11
    reports the framing the server really sent.
    """
    seen: set[bytes] = set()
    out: list[tuple[Any, Any]] = []
    for name, value in headers:
        key = (
            name.lower()
            if isinstance(name, bytes)
            else name.lower().encode("ascii")
        )
        if key == b"transfer-encoding":
            coding = (
                (
                    value
                    if isinstance(value, bytes)
                    else value.encode("latin-1")
                )
                .strip()
                .lower()
            )
            if coding in seen:
                continue
            seen.add(coding)
        out.append((name, value))
    return out


def _patched_normalize_and_validate(
    headers: Any, _parsed: bool = False
) -> Any:
    # Only loosen for parsed (response) headers. Outbound requests stay
    # strict so we don't mask request-smuggling shapes we generate ourselves.
    if _parsed and _lenient_te.get():
        headers = _dedupe_transfer_encoding(headers)
    return _orig_normalize_and_validate(headers, _parsed=_parsed)


def install() -> None:
    # h11._events imports normalize_and_validate by name at module load time
    # (`from ._headers import normalize_and_validate`), so the response-parsing
    # path resolves the symbol via _events' module globals and never touches
    # _headers.normalize_and_validate. Patch both bindings.
    if _h11_headers.normalize_and_validate is _patched_normalize_and_validate:
        return
    _h11_headers.normalize_and_validate = _patched_normalize_and_validate
    _h11_events.normalize_and_validate = _patched_normalize_and_validate


@contextlib.contextmanager
def lenient_te() -> Iterator[None]:
    token = _lenient_te.set(True)
    try:
        yield
    finally:
        _lenient_te.reset(token)


def lenient_te_for(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
) -> contextlib.AbstractContextManager[None]:
    """Context manager that enables lenient TE iff the scraper opts in.

    The run enters it around its ``run()`` body so child tasks (workers,
    monitors) inherit the contextvar via :pep:`asyncio.Task` snapshotting.
    """
    enabled = DriverRequirement.H11_HEADER_FIXES in getattr(
        scraper, "driver_requirements", []
    )
    return lenient_te() if enabled else contextlib.nullcontext()


install()


def _httpx_timeout(timeout: TimeoutType) -> Any:
    """Translate jkent's TimeoutType to httpx's per-request timeout.

    When ``timeout`` is ``None`` we return ``USE_CLIENT_DEFAULT`` so the
    client-level timeout is preserved; passing ``None`` directly would
    instead disable the timeout for this request.
    """
    if timeout is None:
        return httpx.USE_CLIENT_DEFAULT
    if isinstance(timeout, tuple):
        connect, read = timeout
        return httpx.Timeout(read, connect=connect)
    return timeout


def _timeout_seconds_for_error(
    timeout: TimeoutType, client_timeout: float
) -> float:
    """The effective timeout, in seconds, for RequestTimeoutException.

    A request's own timeout (the read element of a ``(connect, read)``
    tuple), else ``client_timeout`` — the client-level timeout a request
    without one runs under.
    """
    if isinstance(timeout, int | float):
        return float(timeout)
    if isinstance(timeout, tuple):
        return float(timeout[1])
    return client_timeout


@contextlib.contextmanager
def _translate(
    url: str, timeout: TimeoutType, client_timeout: float
) -> Iterator[None]:
    """Re-raise httpx transport failures as the driver's exceptions.

    A timeout becomes :class:`RequestTimeoutException`, reporting the
    effective timeout (``timeout``, else ``client_timeout``); an unsupported
    scheme is the scraper's URL, not the network, so it becomes
    :class:`ScraperConfigError` and is not retried; any other
    ``httpx.TransportError`` (connection reset, protocol error, DNS failure)
    becomes :class:`TransientException`.
    """
    try:
        yield
    except httpx.UnsupportedProtocol as exc:
        raise ScraperConfigError(
            f"{type(exc).__name__} from {url}: {exc}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RequestTimeoutException(
            url=url,
            timeout_seconds=_timeout_seconds_for_error(
                timeout, client_timeout
            ),
        ) from exc
    except httpx.TransportError as exc:
        raise TransientException(
            f"{type(exc).__name__} from {url}: {exc}",
            url=url,
            kind=TransientKind.NETWORK,
        ) from exc


def _wants_follow_redirects(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
) -> bool:
    return DriverRequirement.FOLLOW_REDIRECTS in getattr(
        scraper, "driver_requirements", []
    )


def _merge_cookies_into_headers(
    cookies: dict[str, str] | Any | None,
    headers: dict[str, Any],
    *,
    session: str | None = None,
) -> None:
    """Merge per-request cookies into a Cookie header.

    httpx deprecated the per-request ``cookies`` kwarg, so they travel as a
    ``Cookie`` header — and any ``Cookie`` header stops the client's jar
    from adding its own. ``session`` is the jar's header for this URL,
    included here so request cookies add to the session rather than replace
    it (a request cookie wins over a jar cookie of the same name). An
    explicit ``Cookie`` header is the author's and is extended, not merged
    with the jar.
    """
    if not cookies:
        return
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())
    for k in headers:
        if k.lower() == "cookie":
            headers[k] = f"{headers[k]}; {cookie_str}"
            return
    if session:
        own = set(cookies)
        kept = [
            pair
            for pair in session.split("; ")
            if pair.split("=", 1)[0] not in own
        ]
        cookie_str = "; ".join([*kept, cookie_str])
    headers["Cookie"] = cookie_str


_FORM_CONTENT_TYPE = "application/x-www-form-urlencoded"


def _request_content_params(
    request_data: RequestData,
) -> tuple[bytes | None, dict[str, Any] | None, str | None]:
    """Split ``HTTPRequestParams.data`` into httpx ``content``/``data`` kwargs.

    The third element is a ``Content-Type`` to send when the request sets
    none. A pair list is form-encoded here, in the order given with its
    repeats — httpx's ``data=`` takes only a mapping, which would regroup
    ``a=1&b=2&a=3`` by name. A file-like body is read. Anything else raises
    rather than going out as an empty body.
    """
    if request_data is None:
        return None, None, None
    if isinstance(request_data, bytes):
        return request_data, None, None
    if isinstance(request_data, dict):
        return None, cast(dict[str, Any], request_data), None
    if isinstance(request_data, (list, tuple)):
        return urlencode(request_data).encode(), None, _FORM_CONTENT_TYPE
    if hasattr(request_data, "read"):
        return request_data.read(), None, None
    raise TypeError(
        f"HTTPRequestParams.data of type {type(request_data).__name__} "
        "cannot be sent"
    )


class _AsyncStreamingResponse:
    """Async streaming wrapper around an open :class:`httpx.Response`.

    ``url`` is the response's final URL, after any followed redirects.
    """

    def __init__(
        self,
        http_response: httpx.Response,
        *,
        headers: dict[str, Any],
        timeout: TimeoutType,
        client_timeout: float,
    ) -> None:
        self._response = http_response
        self.status_code = http_response.status_code
        self.headers = headers
        self.url = str(http_response.url)
        self._timeout = timeout
        self._client_timeout = client_timeout

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        # The body is consumed here, after the streaming context manager has
        # suspended at its ``yield`` — so a read timeout mid-download surfaces
        # in this loop, not in _stream_request, and is translated here too.
        with _translate(self.url, self._timeout, self._client_timeout):
            async for chunk in self._response.aiter_bytes(
                chunk_size=chunk_size
            ):
                yield chunk


class _HttpArchiveStream(ArchiveStream):
    """An ``ArchiveStream`` backed by an open httpx streaming response.

    The streaming context stays open until :meth:`HttpxTransport.finish_archiving`
    closes it, so the body must be consumed before then.
    """

    def __init__(self, cm: Any, streaming: _AsyncStreamingResponse) -> None:
        super().__init__(
            status_code=streaming.status_code,
            headers=streaming.headers,
            url=streaming.url,
        )
        self._cm = cm
        self._streaming = streaming
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._streaming.aiter_bytes()

    @override
    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cm.__aexit__(None, None, None)


class HttpxTransport(StatelessTransport):
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over httpx."""

    def __init__(
        self,
        *,
        timeout: float | None = None,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        proxy: str | None = None,
    ) -> None:
        super().__init__()
        # The client-level timeout: what a request without its own gets.
        self._timeout = DEFAULT_TIMEOUT_S if timeout is None else timeout
        self._scraper: type[BaseScraper[Any]] | BaseScraper[Any] = (
            scraper if scraper is not None else BaseScraper
        )
        self._ssl_context = ssl_context
        self._proxy = proxy
        self._follow_redirects = _wants_follow_redirects(self._scraper)
        self._default_headers = dict(
            getattr(self._scraper, "default_headers", None) or {}
        )
        self._client: httpx.AsyncClient | None = None
        # One pool per (lane, verify) pair other than the default lane with
        # default verification, which is _client. A tuple key: lane names
        # are scraper-chosen strings and verify may be a path, so a joined
        # string could alias two pairs onto one pool.
        self._alt_clients: dict[tuple[str, bool | str], httpx.AsyncClient] = {}
        # Every pool shares this jar (httpx adopts a CookieJar instance
        # rather than copying it): a session is the site's, not a lane's.
        self._cookie_jar = CookieJar()

    @property
    @override
    def timeout(self) -> float:
        """Seconds a request without its own ``timeout`` waits."""
        return self._timeout

    async def open(self) -> None:
        self._client = self._new_client(True)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        for client in self._alt_clients.values():
            await client.aclose()
        self._alt_clients.clear()

    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        """Fetch ``queued.request`` over HTTP (await conditions ignored)."""
        return await self._resolve_request(queued.request)

    async def resolve_archive(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
    ) -> ArchiveStream:
        """Open a streaming download of ``queued.request`` and return its stream.

        The caller (worker) has already decided to download via the archive
        handler; this just opens the stream.
        """
        cm = self._stream_request(queued.request)
        streaming = await cm.__aenter__()
        return _HttpArchiveStream(cm, streaming)

    @override
    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Close ``stream``, and with it the streaming connection behind it."""
        await stream.aclose()

    # --- Client pool ------------------------------------------------------

    def _new_client(self, verify: bool | str) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with our timeout/proxy and the right verify.

        ``verify=True`` means "use the configured default": the supplied SSL
        context if any, otherwise httpx's own default verification. An explicit
        bool/path ``verify`` overrides the context. This is the single place any
        client is constructed, so connection options stay consistent across the
        default and every per-lane / per-verify pool.
        """
        verify_arg: bool | ssl.SSLContext
        if isinstance(verify, str):
            # httpx deprecated str ``verify``; build the context it would
            # have: a directory is a ``capath``, anything else a ``cafile``.
            verify_arg = (
                ssl.create_default_context(capath=verify)
                if os.path.isdir(verify)
                else ssl.create_default_context(cafile=verify)
            )
        elif verify is True and self._ssl_context is not None:
            verify_arg = self._ssl_context
        else:
            verify_arg = verify
        return httpx.AsyncClient(
            verify=verify_arg,
            timeout=self._timeout,
            proxy=self._proxy,
            cookies=self._cookie_jar,
        )

    def _client_for(
        self, verify: bool | str, lane: str = DEFAULT_RATE_LIMIT
    ) -> httpx.AsyncClient:
        """The client pool for a rate-limit lane and ``verify`` value.

        The default lane with default verification is the main client;
        every other combination gets its own lazily-created pool — only
        while open, so a late request cannot create one ``aclose`` missed.
        """
        main = self._require_client()
        if lane == DEFAULT_RATE_LIMIT and verify is True:
            return main
        key = (lane, verify)
        if key not in self._alt_clients:
            self._alt_clients[key] = self._new_client(verify)
        return self._alt_clients[key]

    def _session_cookies(self, http_params: Any) -> str | None:
        """The jar's ``Cookie`` header value for ``http_params``' URL."""
        probe = httpx.Request(http_params.method.value, http_params.url)
        httpx.Cookies(self._cookie_jar).set_cookie_header(probe)
        return probe.headers.get("cookie")

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpxTransport used before open()")
        return self._client

    # --- Request execution --------------------------------------------------

    def _prepare(
        self, request: Request
    ) -> tuple[httpx.AsyncClient, dict[str, Any]]:
        """The client and the ``request``/``stream`` kwargs for a request.

        Shared by the resolve and stream paths so both put the same bytes on
        the wire.
        """
        http_params = request.request
        lane = getattr(request, "rate_limit", None) or DEFAULT_RATE_LIMIT
        client = self._client_for(http_params.verify, lane)
        content_param, data_param, content_type = _request_content_params(
            http_params.data
        )
        if http_params.method is HttpMethod.POST and isinstance(
            request.via, ViaFormSubmit
        ):
            # A browser labels a form POST urlencoded even with no fields to
            # send; httpx sets no Content-Type for an empty body. (The queue
            # stores an empty ``data`` as no body, so only the via tells.)
            content_type = _FORM_CONTENT_TYPE
        headers = merge_headers(self._default_headers, http_params.headers)
        if content_type is not None and not any(
            name.lower() == "content-type" for name in headers
        ):
            headers["Content-Type"] = content_type
        if http_params.cookies:
            _merge_cookies_into_headers(
                http_params.cookies,
                headers,
                session=self._session_cookies(http_params),
            )
        return client, {
            "method": http_params.method.value,
            "url": http_params.url,
            "headers": headers,
            "content": content_param,
            "data": data_param,
            "json": http_params.json,
            "follow_redirects": self._follow_redirects,
            "timeout": _httpx_timeout(http_params.timeout),
        }

    async def _resolve_request(self, request: Request) -> Response:
        """Fetch a Request and return the Response.

        Raises:
            HTTPResponseAssumptionException / PersistentHTTPResponseException /
                SpeculationHTTPFailure: per the scraper's status classifiers.
            RequestTimeoutException: if the request times out (retryable).
        """
        http_params = request.request
        client, send_kwargs = self._prepare(request)

        logger.info(
            "resolve_request: %s %s request_timeout=%r client_timeout=%r",
            http_params.method.value,
            http_params.url,
            http_params.timeout,
            client.timeout,
        )

        with _translate(http_params.url, http_params.timeout, self._timeout):
            http_response = await client.request(**send_kwargs)

        body = http_response.content
        hdrs = dict(http_response.headers)
        self.classify_and_raise(
            self._scraper,
            request,
            status_code=http_response.status_code,
            headers=hdrs,
            body=body,
            url=http_params.url,
        )

        return Response(
            status_code=http_response.status_code,
            headers=hdrs,
            content=body,
            url=str(http_response.url),
            request=request,
        )

    @contextlib.asynccontextmanager
    async def _stream_request(
        self, request: Request
    ) -> AsyncIterator[_AsyncStreamingResponse]:
        """Open a streaming HTTP request.

        Yields an :class:`_AsyncStreamingResponse` whose ``aiter_bytes`` can be
        consumed incrementally.  The underlying httpx connection is released
        when the context manager exits.
        """
        http_params = request.request
        client, send_kwargs = self._prepare(request)

        logger.info(
            "stream_request: opening stream %s %s "
            "request_timeout=%r client_timeout=%r",
            http_params.method.value,
            http_params.url,
            http_params.timeout,
            client.timeout,
        )

        with _translate(http_params.url, http_params.timeout, self._timeout):
            async with client.stream(**send_kwargs) as http_response:
                logger.info(
                    "stream_request: headers received url=%s status=%s",
                    http_params.url,
                    http_response.status_code,
                )
                hdrs = dict(http_response.headers)
                self.classify_and_raise(
                    self._scraper,
                    request,
                    status_code=http_response.status_code,
                    headers=hdrs,
                    body=None,
                    url=http_params.url,
                )
                yield _AsyncStreamingResponse(
                    http_response,
                    headers=hdrs,
                    timeout=http_params.timeout,
                    client_timeout=self._timeout,
                )
                logger.info(
                    "stream_request: stream closed url=%s", http_params.url
                )
