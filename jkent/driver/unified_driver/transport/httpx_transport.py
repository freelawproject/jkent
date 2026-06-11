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
:class:`~jkent.driver.unified_driver.rate_limiter.RateLimiter`. Requests flagged
``bypass_rate_limit`` still get a separate client pool.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import ssl
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, cast

import h11._events as _h11_events
import h11._headers as _h11_headers
import httpx

from jkent.common.exceptions import (
    RequestTimeoutException,
    TransientException,
)
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    Request,
    Response,
    TimeoutType,
)

if TYPE_CHECKING:
    from jkent.data_types import BaseScraper


from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    NoopHandle,
    Transport,
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
    seen = False
    out: list[tuple[Any, Any]] = []
    for name, value in headers:
        key = (
            name.lower()
            if isinstance(name, bytes)
            else name.lower().encode("ascii")
        )
        if key == b"transfer-encoding":
            if seen:
                continue
            seen = True
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
    _h11_headers.normalize_and_validate = _patched_normalize_and_validate  # type: ignore
    _h11_events.normalize_and_validate = _patched_normalize_and_validate  # type: ignore[attr-defined]


@contextlib.contextmanager
def lenient_te() -> Iterator[None]:
    token = _lenient_te.set(True)
    try:
        yield
    finally:
        _lenient_te.reset(token)


def lenient_te_for(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
) -> contextlib.AbstractContextManager[None]:  # type: ignore
    """Context manager that enables lenient TE iff the scraper opts in.

    Hoist this around the driver's ``run()`` body so child tasks (workers,
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


def _timeout_seconds_for_error(timeout: TimeoutType) -> float:
    """Best-effort numeric timeout for RequestTimeoutException reporting."""
    if isinstance(timeout, int | float):
        return float(timeout)
    if isinstance(timeout, tuple):
        return float(timeout[1])
    return 30.0


def _wants_follow_redirects(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
) -> bool:
    return DriverRequirement.FOLLOW_REDIRECTS in getattr(
        scraper, "driver_requirements", []
    )


def _merge_cookies_into_headers(
    cookies: dict[str, str] | Any | None,
    headers: dict[str, Any],
) -> None:
    """Merge per-request cookies into a Cookie header.

    httpx deprecated the per-request ``cookies`` kwarg.  This helper
    serialises cookies into the ``Cookie`` header instead.
    """
    if not cookies:
        return
    pairs = [f"{k}={v}" for k, v in cookies.items()]

    if not pairs:
        return

    cookie_str = "; ".join(pairs)
    for k in headers:
        if k.lower() == "cookie":
            headers[k] = f"{headers[k]}; {cookie_str}"
            return
    headers["Cookie"] = cookie_str


def _request_content_params(
    request_data: Any,
) -> tuple[bytes | None, dict[str, Any] | None]:
    """Split ``HTTPRequestParams.data`` into httpx content/data kwargs."""
    content_param: bytes | None = (
        request_data if isinstance(request_data, bytes) else None
    )
    data_param: dict[str, Any] | None = (
        cast(dict[str, Any], request_data)
        if isinstance(request_data, dict)
        else None
    )
    return content_param, data_param


class _AsyncStreamingResponse:
    """Async streaming wrapper around an open :class:`httpx.Response`."""

    def __init__(
        self,
        http_response: httpx.Response,
        url: str,
        *,
        headers: dict[str, Any],
        timeout: TimeoutType = None,
    ) -> None:
        self._response = http_response
        self.status_code = http_response.status_code
        self.headers = headers
        self.url = url
        self._timeout = timeout

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        # The body is consumed here, after the streaming context manager has
        # suspended at its ``yield`` — so a read timeout mid-download surfaces
        # in this loop, not in _stream_request's try/except. Translate it to
        # the retryable RequestTimeoutException to match the resolve() path.
        try:
            async for chunk in self._response.aiter_bytes(
                chunk_size=chunk_size
            ):
                yield chunk
        except httpx.TimeoutException as exc:
            raise RequestTimeoutException(
                url=self.url,
                timeout_seconds=_timeout_seconds_for_error(self._timeout),
            ) from exc
        except httpx.TransportError as exc:
            raise TransientException(
                f"{type(exc).__name__} from {self.url}: {exc}"
            ) from exc


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

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._streaming.aiter_bytes()

    async def aclose(self) -> None:
        await self._cm.__aexit__(None, None, None)


class HttpxTransport(Transport[NoopHandle]):
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over httpx."""

    def __init__(
        self,
        *,
        timeout: float | None = None,
        scraper: type[BaseScraper] | BaseScraper | None = None,
        ssl_context: ssl.SSLContext | None = None,
        proxy: str | None = None,
    ) -> None:
        self._timeout = timeout
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
        self._alt_clients: dict[str, httpx.AsyncClient] = {}
        self._bypass_client: httpx.AsyncClient | None = None
        self._handles: dict[int, NoopHandle] = {}

    async def open(self) -> None:
        self._client = self._new_client(True)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._bypass_client is not None:
            await self._bypass_client.aclose()
            self._bypass_client = None
        for client in self._alt_clients.values():
            await client.aclose()
        self._alt_clients.clear()

    async def acquire(self, worker_id: int) -> NoopHandle:
        """Get-or-create the worker's (stateless) handle, stable until release."""
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = NoopHandle()
            self._handles[worker_id] = handle
        return handle

    async def release(self, worker_id: int) -> None:
        """Drop the worker's handle; the next acquire makes a fresh one."""
        handle = self._handles.pop(worker_id, None)
        if handle is not None:
            await handle.close()

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
        decision: object | None = None,
    ) -> ArchiveStream:
        """Open a streaming download of ``queued.request`` and return its stream.

        The caller (worker) has already decided to download via the archive
        handler; this just opens the stream. ``decision`` is the worker's
        verdict, accepted for signature parity and not re-consulted here.
        """
        cm = self._stream_request(queued.request)
        streaming = await cm.__aenter__()
        return _HttpArchiveStream(cm, streaming)

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Close the streaming connection backing ``stream``."""
        if isinstance(stream, _HttpArchiveStream):
            await stream.aclose()

    # --- Client pool ------------------------------------------------------

    def _new_client(self, verify: bool | str) -> httpx.AsyncClient:
        """Create an httpx.AsyncClient with our timeout/proxy and the right verify.

        ``verify=True`` means "use the configured default": the supplied SSL
        context if any, otherwise httpx's own default verification. An explicit
        bool/path ``verify`` overrides the context. This is the single place any
        client is constructed, so connection options stay consistent across the
        default, alternate, and bypass pools.
        """
        verify_arg: bool | str | ssl.SSLContext = (
            self._ssl_context
            if (verify is True and self._ssl_context is not None)
            else verify
        )
        return httpx.AsyncClient(
            verify=verify_arg, timeout=self._timeout, proxy=self._proxy
        )

    def _client_for(self, verify: bool | str) -> httpx.AsyncClient:
        """Return the appropriate httpx.AsyncClient for the given verify value.

        Returns the default client when verify is True (the default).
        Otherwise returns a lazily-created cached alternate client.
        """
        if verify is True:
            return self._require_client()
        key = str(verify)
        if key not in self._alt_clients:
            self._alt_clients[key] = self._new_client(verify)
        return self._alt_clients[key]

    def _bypass_client_for(self, verify: bool | str) -> httpx.AsyncClient:
        """Return the client pool reserved for bypass_rate_limit requests."""
        if verify is not True:
            key = f"bypass_{verify}"
            if key not in self._alt_clients:
                self._alt_clients[key] = self._new_client(verify)
            return self._alt_clients[key]
        if self._bypass_client is None:
            self._bypass_client = self._new_client(True)
        return self._bypass_client

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HttpxTransport used before open()")
        return self._client

    # --- Request execution --------------------------------------------------

    def _prepare(
        self, request: Request
    ) -> tuple[
        httpx.AsyncClient, dict[str, Any], bytes | None, dict[str, Any] | None
    ]:
        """Build the (client, headers, content, data) inputs shared by the
        resolve and stream paths from a request's HTTP params.
        """
        http_params = request.request
        bypass = getattr(request, "bypass_rate_limit", False)
        client = (
            self._bypass_client_for(http_params.verify)
            if bypass
            else self._client_for(http_params.verify)
        )
        content_param, data_param = _request_content_params(http_params.data)
        explicit = dict(http_params.headers) if http_params.headers else {}
        explicit_names = {name.lower() for name in explicit}
        headers = {
            name: value
            for name, value in self._default_headers.items()
            if name.lower() not in explicit_names
        }
        headers.update(explicit)
        _merge_cookies_into_headers(http_params.cookies, headers)
        return client, headers, content_param, data_param

    async def _resolve_request(self, request: Request) -> Response:
        """Fetch a Request and return the Response.

        Raises:
            HTTPResponseAssumptionException / PersistentHTTPResponseException /
                SpeculationHTTPFailure: per the scraper's status classifiers.
            RequestTimeoutException: if the request times out (retryable).
        """
        http_params = request.request
        client, headers, content_param, data_param = self._prepare(request)

        logger.info(
            "resolve_request: %s %s request_timeout=%r client_timeout=%r",
            http_params.method.value,
            http_params.url,
            http_params.timeout,
            client.timeout,
        )

        try:
            http_response = await client.request(
                method=http_params.method.value,
                url=http_params.url,
                headers=headers,
                content=content_param,
                data=data_param,
                json=http_params.json,
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            )
        except httpx.TimeoutException as exc:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            ) from exc
        except httpx.TransportError as exc:
            # Connection resets, protocol errors, DNS failures — retryable
            # network-level failures, not scraper bugs.
            raise TransientException(
                f"{type(exc).__name__} from {http_params.url}: {exc}"
            ) from exc

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
            text=http_response.text,
            url=http_params.url,
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
        client, headers, content_param, data_param = self._prepare(request)

        logger.info(
            "stream_request: opening stream %s %s "
            "request_timeout=%r client_timeout=%r",
            http_params.method.value,
            http_params.url,
            http_params.timeout,
            client.timeout,
        )

        try:
            async with client.stream(
                method=http_params.method.value,
                url=http_params.url,
                headers=headers,
                content=content_param,
                data=data_param,
                json=http_params.json,
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            ) as http_response:
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
                    http_params.url,
                    headers=hdrs,
                    timeout=http_params.timeout,
                )
                logger.info(
                    "stream_request: stream closed url=%s", http_params.url
                )
        except httpx.TimeoutException as exc:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            ) from exc
        except httpx.TransportError as exc:
            raise TransientException(
                f"{type(exc).__name__} from {http_params.url}: {exc}"
            ) from exc
