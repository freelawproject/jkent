"""HTTP transport — a self-contained async httpx2 client.

The wire behavior is fixed and explicit: ``resolve`` sends ``method``,
``url`` (verbatim — query baked in), the scraper's ``default_headers``
overlaid with explicitly-set ``headers`` (per-request wins per name,
case-insensitively), the session's cookies with the request's own over
them in one ``Cookie`` header, the body
(``data`` bytes as content / dict or pair list as form), and a ``json``
payload (serialized by httpx2 as a JSON body). ``Response.url`` is the URL
the exchange landed on, after any followed redirects. ``params`` is not
re-sent — the queue folds it into the url upstream — but ``json`` is
carried through as its own column and re-sent here, so a request's JSON
body is preserved end-to-end.

Redirect-following is per-scraper: ``DriverRequirement.FOLLOW_REDIRECTS``
opts the whole transport into following redirects on every request
(resolve and stream paths alike). The transport follows them itself rather
than leaving it to httpx2, which rebuilds each hop's ``Cookie`` header from
the session jar alone and would drop the request's own cookies.
``DriverRequirement.H11_HEADER_FIXES`` likewise loosens h11's
response-header validation around this transport's own sends, and nowhere
else.

The proxy is the first choice made of the transport's ``proxy`` (the
run's config) and the scraper's ``proxy``, else the environment's
(:class:`~jkent.data_types.ProxySetting`). Certificates are verified
against, first to last: a request's ``verify``, the transport's
``ssl_context``, the scraper's ``get_ssl_context()``, ``SSL_CERT_FILE`` /
``SSL_CERT_DIR``, then the OS trust store.

There is deliberately no transport-level rate limiting: the unified driver
rate-limits in the worker via its own
:class:`~jkent.driver.unified_driver.rate_limiter.RateLimiter`. Each
rate-limit *lane* other than the default does get its own client pool, so a
lane's connections never queue behind another's.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import pathlib
import ssl
from collections.abc import Iterable, Iterator
from http.cookiejar import CookieJar
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import urlencode

import h11._events as _h11_events
import h11._headers as _h11_headers
import httpx2
from typing_extensions import override

from jkent.common.exceptions import (
    RequestTimeoutException,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.common.headers import merge_headers
from jkent.common.rate_limits import DEFAULT_RATE_LIMIT
from jkent.common.request import DEFAULT_TIMEOUT_S, encode_body
from jkent.data_types import (
    BaseScraper,
    CookiesType,
    DriverRequirement,
    HttpMethod,
    ProxySetting,
    Request,
    RequestData,
    Response,
    TimeoutType,
    ViaFormSubmit,
)
from jkent.driver.archive_handler import (
    content_encoded,
    plan_digests,
    verify_download,
)
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    NoopHandle,
    StatelessTransport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Sequence

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
    headers: Iterable[tuple[bytes | str, bytes | str]],
) -> list[tuple[bytes | str, bytes | str]]:
    """``headers`` with identical ``Transfer-Encoding`` repeats dropped.

    A repeat is dropped whether it came on its own line or comma-joined
    into one (``chunked, chunked``); a line left with no coding is dropped.
    Distinct values (``gzip`` then ``chunked``) all pass through, so h11
    reports the framing the server really sent.

    Args:
        headers: The parsed header pairs.

    Returns:
        The same pairs with identical ``Transfer-Encoding`` repeats dropped.
    """
    seen: set[bytes] = set()
    out: list[tuple[bytes | str, bytes | str]] = []
    for name, value in headers:
        key = (
            name.lower()
            if isinstance(name, bytes)
            else name.lower().encode("ascii")
        )
        if key == b"transfer-encoding":
            raw = (
                value if isinstance(value, bytes) else value.encode("latin-1")
            )
            fresh: list[bytes] = []
            for coding in (c.strip() for c in raw.split(b",")):
                if coding and coding.lower() not in seen:
                    seen.add(coding.lower())
                    fresh.append(coding)
            if not fresh:
                continue
            value = b", ".join(fresh)
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
    """Patch h11's header validation so lenient TE can be switched on."""
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
    """Accept duplicate ``Transfer-Encoding`` values for the block's duration.

    Yields:
        ``None``, for the body of the block.
    """
    token = _lenient_te.set(True)
    try:
        yield
    finally:
        _lenient_te.reset(token)


install()


def _httpx_timeout(timeout: TimeoutType) -> Any:
    """Translate jkent's TimeoutType to httpx2's per-request timeout.

    When ``timeout`` is ``None`` we return ``USE_CLIENT_DEFAULT`` so the
    client-level timeout is preserved; passing ``None`` directly would
    instead disable the timeout for this request.

    Args:
        timeout: jkent's timeout for this request.

    Returns:
        The value to pass httpx2 as ``timeout=``.
    """
    if timeout is None:
        return httpx2.USE_CLIENT_DEFAULT
    if isinstance(timeout, tuple):
        connect, read = timeout
        return httpx2.Timeout(read, connect=connect)
    return timeout


def _timeout_seconds_for_error(
    timeout: TimeoutType,
    client_timeout: float,
    exc: httpx2.TimeoutException,
) -> float:
    """The effective timeout, in seconds, for RequestTimeoutException.

    A request's own timeout, else ``client_timeout`` — the client-level
    timeout a request without one runs under. Of a ``(connect, read)``
    tuple, the element for the phase that timed out: connect for
    :class:`httpx2.ConnectTimeout`, read for the rest (httpx2 gives write and
    pool the read value).

    Args:
        timeout: The request's own timeout, if it has one.
        client_timeout: The client-level timeout, in seconds.
        exc: The timeout httpx2 raised.

    Returns:
        The effective timeout, in seconds.
    """
    if isinstance(timeout, int | float):
        return float(timeout)
    if isinstance(timeout, tuple):
        connect, read = timeout
        return float(
            connect if isinstance(exc, httpx2.ConnectTimeout) else read
        )
    return client_timeout


@contextlib.contextmanager
def _translate(
    url: str, timeout: TimeoutType, client_timeout: float
) -> Iterator[None]:
    """Re-raise httpx2 failures as the driver's exceptions.

    A timeout becomes :class:`RequestTimeoutException`, reporting the
    effective timeout (``timeout``, else ``client_timeout``). A request we
    built wrong — an unsupported scheme, an invalid URL, a header h11
    refuses to send — is the scraper's, not the network's, so it becomes
    :class:`ScraperConfigError` and is not retried. A response the site got
    wrong — a redirect loop, a body that will not decode — becomes a
    :class:`TransientException` of kind ``SITE_DEGRADED``. Any other
    ``httpx2.TransportError`` (connection reset, a server that hangs up,
    DNS failure, and a response h11 rejects as malformed) becomes a
    :class:`TransientException` of kind ``NETWORK``. A proxy that refuses
    the request is the run's configuration, a :class:`ScraperConfigError`;
    one that cannot reach the site is ``NETWORK``. (A redirect
    to a scheme we cannot fetch is the site's too; :meth:`HttpxTransport._send`
    raises that itself, before it can reach here as ``UnsupportedProtocol``.)

    Args:
        url: The request's URL, for the error messages.
        timeout: The request's own timeout, if it has one.
        client_timeout: The client-level timeout, in seconds.

    Yields:
        ``None``, for the body of the block.

    Raises:
        RequestTimeoutException: If the request timed out.
        ScraperConfigError: If the request could not be sent as built.
        TransientException: For a malformed response or any other httpx2
            transport failure.
    """
    try:
        yield
    except (
        httpx2.UnsupportedProtocol,
        httpx2.InvalidURL,
        httpx2.LocalProtocolError,
    ) as exc:
        raise ScraperConfigError(
            f"{type(exc).__name__} from {url}: {exc}"
        ) from exc
    except httpx2.TimeoutException as exc:
        raise RequestTimeoutException(
            url=url,
            timeout_seconds=_timeout_seconds_for_error(
                timeout, client_timeout, exc
            ),
        ) from exc
    except (httpx2.TooManyRedirects, httpx2.DecodingError) as exc:
        raise TransientException(
            f"{type(exc).__name__} from {url}: {exc}",
            url=url,
            kind=TransientKind.SITE_DEGRADED,
        ) from exc
    except httpx2.ProxyError as exc:
        if _proxy_refused(exc):
            raise ScraperConfigError(
                f"proxy refused the request to {url}: {exc}"
            ) from exc
        raise TransientException(
            f"{type(exc).__name__} from {url}: {exc}",
            url=url,
            kind=TransientKind.NETWORK,
        ) from exc
    # Includes httpx2.RemoteProtocolError. A simplification: a response h11
    # rejects as malformed is the site's fault, but h11 raises the same
    # type when the peer hangs up partway through one, and only the
    # message tells them apart — so both are NETWORK.
    except httpx2.TransportError as exc:
        raise TransientException(
            f"{type(exc).__name__} from {url}: {exc}",
            url=url,
            kind=TransientKind.NETWORK,
        ) from exc


# httpcore2's SOCKS5 reply phrases for a proxy that is reachable and
# working but forwards nothing for us: what it allows or speaks.
_SOCKS_REFUSALS: Final[tuple[str, ...]] = (
    "Connection not allowed by ruleset",
    "Command not supported",
    "Address type not supported",
)


def _proxy_refused(exc: httpx2.ProxyError) -> bool:
    """Whether the proxy refused us, rather than failed to reach the site.

    httpcore2 says which only in the message. An HTTP proxy answers a
    ``CONNECT`` with ``"<status> <reason>"``: a 4xx (407 wants credentials,
    403 denies the target) is its refusal, a 5xx (502, 503, 504) a failure
    upstream of it. A SOCKS5 proxy that rejects our credentials or auth
    method, or answers with a refusal in :data:`_SOCKS_REFUSALS`, refused
    us; its other replies (host unreachable, connection refused, ...) are
    the network's.

    Args:
        exc: The proxy failure.

    Returns:
        ``True`` if the proxy refused the request as configured.
    """
    message = str(exc)
    status = message.partition(" ")[0]
    if status.isdigit():
        return status.startswith("4")
    if message.startswith("Proxy Server could not connect: "):
        return message.endswith(tuple(f"{r}." for r in _SOCKS_REFUSALS))
    # Credentials rejected, or an auth method mismatch.
    return True


def _own_cookies(
    headers: dict[str, str], cookies: CookiesType
) -> dict[str, str]:
    """The request's own cookies — its per-request jar — keyed by name.

    Taken from any ``Cookie`` header (removed from ``headers``, so httpx2's
    session jar still applies) and then ``cookies``, which win by name.
    Each value is the ``name=value`` pair as it goes on the wire.

    Args:
        headers: The request's merged headers; ``Cookie`` is popped.
        cookies: The request's ``cookies``.

    Returns:
        The request's own cookie pairs, by name.
    """
    own: dict[str, str] = {}
    for name in [k for k in headers if k.lower() == "cookie"]:
        for pair in headers.pop(name).split(";"):
            pair = pair.strip()
            if pair:
                own[pair.partition("=")[0]] = pair
    for name, value in (cookies or {}).items():
        own[name] = f"{name}={value}"
    return own


def _lay_own_cookies(request: httpx2.Request, own: dict[str, str]) -> None:
    """Lay the request's own cookies over ``request``'s session cookies.

    httpx2 builds every hop's ``Cookie`` header from the session jar for
    that hop's URL; the request's own replace a session cookie of the same
    name and are added after the rest.

    Args:
        request: The hop about to be sent.
        own: The request's own cookie pairs, by name.
    """
    if not own:
        return
    session = request.headers.get("Cookie")
    kept = (
        [p for p in session.split("; ") if p.partition("=")[0] not in own]
        if session
        else []
    )
    request.headers["Cookie"] = "; ".join([*kept, *own.values()])


def _drop_reset_cookies(
    own: dict[str, str], response: httpx2.Response
) -> None:
    """Drop from ``own`` every cookie ``response`` sets or deletes.

    The request's own cookies are its per-request jar: host-only cookies
    for its host with path ``/``. A browser's jar keys a cookie by name,
    domain and path, so a ``Set-Cookie`` on a redirect from that host
    supersedes one of the same name only when it is host-only (no
    ``Domain``) with path ``/`` — given, or defaulted from the redirect's
    own path. The next hop then carries the session jar's value, or none
    if the response expired it. A ``Set-Cookie`` for another path or for
    the domain is a separate cookie, and the request's own one stays.

    Args:
        own: The request's own cookie pairs, by name; updated in place.
        response: A redirect response from the request's host.
    """
    path = response.url.path
    default_path = path[: path.rfind("/")] or "/"
    for header in response.headers.get_list("set-cookie"):
        # Split as RFC 6265 section 5.2 does: the pair, then attributes.
        pair, *attrs = header.split(";")
        attributes = {
            key.strip().lower(): value.strip()
            for key, _, value in (attr.partition("=") for attr in attrs)
        }
        cookie_path = attributes.get("path", "")
        if not cookie_path.startswith("/"):
            cookie_path = default_path
        if cookie_path == "/" and not attributes.get("domain"):
            own.pop(pair.partition("=")[0].strip(), None)


_FORM_CONTENT_TYPE: Final[str] = "application/x-www-form-urlencoded"

# The fetch spec's request-body-header names: a browser drops these when a
# redirect turns the request into a bodiless GET.
_BODY_HEADERS: Final[tuple[str, ...]] = (
    "Content-Encoding",
    "Content-Language",
    "Content-Location",
    "Content-Type",
)


async def _request_content(
    request_data: RequestData,
) -> tuple[bytes | None, str | None]:
    """``HTTPRequestParams.data`` as the bytes to send.

    The second element is a ``Content-Type`` to send when the request sets
    none. A mapping and a pair list are both form-encoded here by
    ``urlencode(..., doseq=True)``, so one form puts the same bytes on the
    wire either way, and as a GET would put them in its query — httpx2's
    ``data=`` takes only a mapping, would regroup ``a=1&b=2&a=3`` by name,
    and spells ``True`` and ``None`` its own way.
    A file-like body is read off the event loop and rewound, as
    :func:`~jkent.common.request.encode_body` reads it, so a retry of the
    same request sends it again. Anything else raises rather than going out
    as an empty body.

    Args:
        request_data: The request's ``data`` payload.

    Returns:
        The body bytes and a ``Content-Type`` to send when the request
        sets none.

    Raises:
        TypeError: If ``request_data`` is a type that cannot be sent.
    """
    if request_data is None:
        return None, None
    if isinstance(request_data, bytes):
        return request_data, None
    if isinstance(request_data, (dict, list, tuple)):
        # No fields is no body, as the queue stores it (``encode_body``).
        if not request_data:
            return None, None
        return (
            urlencode(request_data, doseq=True).encode(),
            _FORM_CONTENT_TYPE,
        )
    if hasattr(request_data, "read"):
        body, _ = await asyncio.to_thread(encode_body, request_data)
        return body, None
    raise TypeError(
        f"HTTPRequestParams.data of type {type(request_data).__name__} "
        "cannot be sent"
    )


async def _close_quietly(response: httpx2.Response) -> None:
    """Close ``response`` while another error is already propagating.

    A failure to close is logged and swallowed, so the caller sees the
    error that caused the close rather than the close's own.

    Args:
        response: The response to close.
    """
    try:
        await response.aclose()
    except Exception:
        logger.debug("closing %s failed", response.url, exc_info=True)


def _response_headers(headers: httpx2.Headers) -> dict[str, str]:
    """``headers`` as one value per (lowercased) name.

    Repeats are comma-joined, except ``Set-Cookie``, whose values are
    joined by newlines, as Playwright joins them: a cookie's ``Expires``
    date contains a comma, but no cookie contains a newline, so the
    stored value splits back into its cookies.

    Args:
        headers: The response's headers.

    Returns:
        The headers, by name.
    """
    out: dict[str, str] = {}
    for name, value in headers.multi_items():
        if name in out:
            sep = "\n" if name == "set-cookie" else ", "
            out[name] = f"{out[name]}{sep}{value}"
        else:
            out[name] = value
    return out


class _EncodedTally(httpx2.AsyncByteStream):
    """A response's encoded bytes, counted and hashed as httpx2 reads them.

    Sits under httpx2's decoder, so the archive still gets decoded bytes
    while the encoded ones its ``Content-Length`` and ``ETag`` describe are
    measured on the way past.
    """

    def __init__(
        self, stream: httpx2.AsyncByteStream, digests: dict[str, Any]
    ) -> None:
        self._stream = stream
        self.digests = digests
        self.bytes_total = 0

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            self.bytes_total += len(chunk)
            # Off the loop, as the archive handler hashes its own chunks.
            for digest in self.digests.values():
                await asyncio.to_thread(digest.update, chunk)
            yield chunk

    @override
    async def aclose(self) -> None:
        await self._stream.aclose()


class _HttpArchiveStream(ArchiveStream):
    """An ``ArchiveStream`` that owns an open httpx2 streaming response.

    ``url`` is the response's final URL, after any followed redirects. The
    response stays open until :meth:`HttpxTransport.finish_archiving`
    closes it, so the body must be consumed before then.

    An encoded body's ``Content-Length`` and ``ETag`` describe its encoded
    bytes, which only the transport sees: the archive handler skips them
    (:func:`~jkent.driver.archive_handler.content_encoded`) and this stream
    checks them against the encoded bytes once the body has been read, with
    the handler's own :func:`~jkent.driver.archive_handler.verify_download`.
    """

    def __init__(
        self,
        response: httpx2.Response,
        *,
        headers: dict[str, str],
        timeout: TimeoutType,
        client_timeout: float,
    ) -> None:
        super().__init__(
            status_code=response.status_code,
            headers=headers,
            url=str(response.url),
        )
        self._response = response
        self._timeout = timeout
        self._client_timeout = client_timeout
        self._tally: _EncodedTally | None = None
        # The validators, minus the coding: what the encoded bytes are.
        self._encoded_headers = {
            name: value
            for name, value in headers.items()
            if name.lower() != "content-encoding"
        }
        # A body httpx2 has already read is past checking: its encoded
        # bytes are gone.
        if content_encoded(headers) and not response.is_stream_consumed:
            # Only the ETag check needs hashing here; the handler's SHA-256
            # names the file it stores, which is the decoded bytes.
            self._tally = _EncodedTally(
                cast(httpx2.AsyncByteStream, response.stream),
                {
                    name: digest
                    for name, digest in plan_digests(
                        self._encoded_headers
                    ).items()
                    if name != "sha256"
                },
            )
            response.stream = self._tally
        self._iterator: AsyncGenerator[bytes, None] | None = None
        self._consumed = False
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Read the response body in chunks, once.

        Returns:
            An iterator over the body, closed by :meth:`aclose`.

        Raises:
            RuntimeError: If the body has already been iterated.
        """
        if self._consumed:
            raise RuntimeError(f"archive body already consumed: {self.url}")
        self._consumed = True
        self._iterator = self._aiter_bytes()
        return self._iterator

    async def _aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        # The body is read after resolve_archive has returned, so a read
        # timeout mid-download surfaces in this loop and is translated here.
        # aclosing: a read abandoned partway closes httpx2's generator here,
        # not later in asyncio's finalizer against a closed response.
        with _translate(self.url, self._timeout, self._client_timeout):
            async with contextlib.aclosing(
                self._response.aiter_bytes()
            ) as chunks:
                async for chunk in chunks:
                    yield chunk
        if self._tally is not None:
            verify_download(
                url=self.url,
                headers=self._encoded_headers,
                digests=self._tally.digests,
                bytes_total=self._tally.bytes_total,
            )

    @override
    async def aclose(self) -> None:
        """Close the iterator handed out, then the streaming response.

        The response is closed even if closing the iterator raises; that
        error, not one from closing the response, is the one raised.
        """
        if self._closed:
            return
        self._closed = True
        # Close the generator first, so a body read abandoned partway
        # unwinds here — inside its _translate — not later in asyncio's
        # async-generator finalizer, against a response already closed.
        iterator, self._iterator = self._iterator, None
        try:
            if iterator is not None:
                await iterator.aclose()
        except BaseException:
            await _close_quietly(self._response)
            raise
        with _translate(self.url, self._timeout, self._client_timeout):
            await self._response.aclose()


class HttpxTransport(StatelessTransport):
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over httpx2."""

    def __init__(
        self,
        *,
        timeout: float | None = None,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any] | None = None,
        ssl_context: ssl.SSLContext | None = None,
        proxy: str | ProxySetting | None = None,
    ) -> None:
        """Build the transport and its client pools.

        Args:
            timeout: The client-level timeout, in seconds; what a request
                without its own gets.
            scraper: The scraper whose classifier, default headers, and
                requirements this transport honours.
            ssl_context: SSL context for the default verification, over the
                scraper's ``get_ssl_context()``; ``None`` defers to it.
            proxy: Proxy URL or :class:`ProxySetting`, over the scraper's
                ``proxy``; ``None`` defers to it.
        """
        super().__init__()
        # The client-level timeout: what a request without its own gets.
        self._timeout = DEFAULT_TIMEOUT_S if timeout is None else timeout
        self._scraper: type[BaseScraper[Any]] | BaseScraper[Any] = (
            scraper if scraper is not None else BaseScraper
        )
        self._ssl_context = (
            ssl_context
            if ssl_context is not None
            else self._scraper.get_ssl_context()
        )
        self._proxy: str | ProxySetting = next(
            (p for p in (proxy, self._scraper.proxy) if p is not None),
            ProxySetting.ENVIRONMENT,
        )
        requirements = self._scraper.driver_requirements
        self._follow_redirects = (
            DriverRequirement.FOLLOW_REDIRECTS in requirements
        )
        self._lenient_te = DriverRequirement.H11_HEADER_FIXES in requirements
        self._default_headers = dict(self._scraper.default_headers)
        self._client: httpx2.AsyncClient | None = None
        # Set by aclose(), so a request after it is told apart from one
        # before open().
        self._closed = False
        # One pool per (lane, verify) pair other than the default lane with
        # default verification, which is _client. A tuple key: lane names
        # are scraper-chosen strings and verify may be a path, so a joined
        # string could alias two pairs onto one pool.
        self._alt_clients: dict[
            tuple[str, bool | str], httpx2.AsyncClient
        ] = {}
        # Every pool shares this jar (httpx2 adopts a CookieJar instance
        # rather than copying it): a session is the site's, not a lane's.
        self._cookie_jar = CookieJar()
        # One SSL context per verify value, shared by every pool using it:
        # loading CA material is slow and blocks the event loop.
        self._ssl_contexts: dict[bool | str, ssl.SSLContext] = {}

    @property
    @override
    def timeout(self) -> float:
        """Seconds a request without its own ``timeout`` waits."""
        return self._timeout

    async def open(self) -> None:
        """Build the default client pool.

        The default SSL context is loaded off the event loop here, so no
        pool that verifies by default loads it on the loop later.
        """
        await asyncio.to_thread(self._ssl_context_for, True)
        self._client = self._new_client(True)
        self._closed = False

    async def aclose(self) -> None:
        """Close every client pool this transport opened.

        Every pool is detached before the first await, so a request that
        arrives mid-close cannot create a lane pool this close misses, and
        every pool is closed even if closing another raises.
        """
        main, self._client = self._client, None
        self._closed = True
        alts = list(self._alt_clients.values())
        self._alt_clients.clear()
        async with contextlib.AsyncExitStack() as stack:
            for client in alts:
                stack.push_async_callback(client.aclose)
            if main is not None:
                stack.push_async_callback(main.aclose)

    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        """Fetch ``queued.request`` over HTTP (await conditions ignored).

        Args:
            handle: The worker's no-op handle.
            queued: The leased request.
            await_conditions: Ignored; there is no page to wait on.

        Returns:
            The fetched response.
        """
        return await self._resolve_request(queued.request)

    async def resolve_archive(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
    ) -> ArchiveStream:
        """Open a streaming download of ``queued.request`` and return its stream.

        The caller (worker) has already decided to download via the archive
        handler; this just opens the stream.

        Args:
            handle: The worker's no-op handle.
            queued: The leased archive request.

        Returns:
            The open archive stream.

        Raises:
            HTTPResponseAssumptionException: Per the scraper's classifier.
            PersistentHTTPResponseException: Per the scraper's classifier.
            SpeculationHTTPFailure: Per the scraper's classifier, for a
                speculative request.
            RequestTimeoutException: If the send times out (retryable).
        """  # noqa: DOC502 -- raised by _translate and classify_and_raise
        request = queued.request
        http_response = await self._exchange(request, stream=True)
        hdrs = _response_headers(http_response.headers)
        try:
            self.classify_and_raise(
                self._scraper,
                request,
                status_code=http_response.status_code,
                headers=hdrs,
                body=None,
                url=str(http_response.url),
            )
        except BaseException:
            await _close_quietly(http_response)
            raise
        return _HttpArchiveStream(
            http_response,
            headers=hdrs,
            timeout=request.request.timeout,
            client_timeout=self._timeout,
        )

    @override
    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Close ``stream``, and with it the streaming connection behind it."""
        await stream.aclose()

    # --- Client pool ------------------------------------------------------

    def _new_client(self, verify: bool | str) -> httpx2.AsyncClient:
        """Create an httpx2.AsyncClient with our timeout/proxy and the right verify.

        This is the single place any client is constructed, so connection
        options stay consistent across the default and every per-lane /
        per-verify pool.

        Args:
            verify: ``True`` for the configured default, else an explicit
                bool or a CA file/directory path.

        Returns:
            The new client.
        """
        return httpx2.AsyncClient(**self._client_kwargs(verify))

    def _client_kwargs(self, verify: bool | str) -> dict[str, Any]:
        """The ``httpx2.AsyncClient`` kwargs for a pool with this ``verify``.

        ``verify=True`` means "use the configured default": see
        :meth:`_ssl_context_for`. An explicit bool/path ``verify`` overrides
        it. ``trust_env`` reaches only proxies (the SSL context is built
        here), so it is on just for :attr:`ProxySetting.ENVIRONMENT`: a
        proxy URL or :attr:`ProxySetting.DIRECT` ignores the environment's.

        Args:
            verify: ``True`` for the configured default, else an explicit
                bool or a CA file/directory path.

        Returns:
            The kwargs to build the client with.
        """
        return {
            "verify": self._ssl_context_for(verify),
            "timeout": self._timeout,
            "proxy": self._proxy if isinstance(self._proxy, str) else None,
            "cookies": self._cookie_jar,
            "trust_env": self._proxy is ProxySetting.ENVIRONMENT,
        }

    def _ssl_context_for(self, verify: bool | str) -> ssl.SSLContext:
        """The SSL context for a ``verify`` value, built once and shared.

        ``True`` is the transport's or scraper's SSL context if any,
        otherwise httpx2's default verification: ``SSL_CERT_FILE`` /
        ``SSL_CERT_DIR`` when set, else the OS trust store.

        Args:
            verify: ``True`` for the configured default, else an explicit
                bool or a CA file/directory path.

        Returns:
            The SSL context every pool with this ``verify`` uses.

        Raises:
            ScraperConfigError: If ``verify`` is a CA path that cannot be
                loaded.
        """
        if verify is True and self._ssl_context is not None:
            return self._ssl_context
        if verify not in self._ssl_contexts:
            if isinstance(verify, str):
                # httpx2 deprecated str ``verify``; build the context it
                # would have: a directory is a ``capath``, else a ``cafile``.
                try:
                    context = (
                        ssl.create_default_context(capath=verify)
                        if pathlib.Path(verify).is_dir()
                        else ssl.create_default_context(cafile=verify)
                    )
                except OSError as exc:  # ssl.SSLError is an OSError
                    raise ScraperConfigError(
                        f"verify={verify!r} cannot be loaded: {exc}"
                    ) from exc
            else:
                context = httpx2.create_ssl_context(verify, trust_env=True)
            self._ssl_contexts[verify] = context
        return self._ssl_contexts[verify]

    def _client_for(
        self, verify: bool | str, lane: str = DEFAULT_RATE_LIMIT
    ) -> httpx2.AsyncClient:
        """The client pool for a rate-limit lane and ``verify`` value.

        The default lane with default verification is the main client;
        every other combination gets its own lazily-created pool — only
        while open, so a late request cannot create one ``aclose`` missed.

        Args:
            verify: The request's ``verify`` value.
            lane: The request's rate-limit lane.

        Returns:
            The client pool for that pair.
        """
        main = self._require_client()
        if lane == DEFAULT_RATE_LIMIT and verify is True:
            return main
        key = (lane, verify)
        if key not in self._alt_clients:
            self._alt_clients[key] = self._new_client(verify)
        return self._alt_clients[key]

    def _require_client(self) -> httpx2.AsyncClient:
        """The default client pool.

        Returns:
            The client built by :meth:`open`.

        Raises:
            TransientException: If the transport has been closed, so a
                request that raced shutdown is retried like one whose pool
                closed under it.
            RuntimeError: If the transport has not been opened.
        """
        if self._client is None:
            if self._closed:
                # aclose() ran while this request was on its way here.
                raise TransientException(
                    "transport closed before sending",
                    kind=TransientKind.NETWORK,
                )
            raise RuntimeError("HttpxTransport used before open()")
        return self._client

    # --- Request execution --------------------------------------------------

    def _h11_leniency(self) -> contextlib.AbstractContextManager[None]:
        """:func:`lenient_te` if the scraper declared ``H11_HEADER_FIXES``.

        Entered around this transport's own sends — the task awaiting a
        send is the one h11 parses the response headers in — so leniency
        reaches no other transport and needs no run-level wrapper.

        Returns:
            :func:`lenient_te`, or a null context.
        """
        return lenient_te() if self._lenient_te else contextlib.nullcontext()

    async def _prepare(
        self, request: Request
    ) -> tuple[httpx2.AsyncClient, httpx2.Request, dict[str, str]]:
        """The client, the built httpx2 request, and its own cookies.

        Shared by the resolve and stream paths so both put the same bytes on
        the wire.

        Args:
            request: The request to send.

        Returns:
            The client to send it on, the request to send, and the
            request's own cookies for :meth:`_send` to lay on every hop.

        Raises:
            ScraperConfigError: If the request promotes an incidental, which
                only a browser transport captures.
        """
        http_params = request.request
        if request.incidental is not None:
            raise ScraperConfigError(
                f"incidental= request to {http_params.url}: HttpxTransport "
                "captures no incidentals; use a browser transport"
            )
        client = self._client_for(
            http_params.verify, request.rate_limit or DEFAULT_RATE_LIMIT
        )
        content, content_type = await _request_content(http_params.data)
        if http_params.method is HttpMethod.POST and isinstance(
            request.via, ViaFormSubmit
        ):
            # A browser labels a form POST urlencoded even with no fields to
            # send; httpx2 sets no Content-Type for an empty body. (The queue
            # stores an empty ``data`` as no body, so only the via tells.)
            content_type = _FORM_CONTENT_TYPE
        headers = merge_headers(self._default_headers, http_params.headers)
        if content_type is not None and not any(
            name.lower() == "content-type" for name in headers
        ):
            headers["Content-Type"] = content_type
        own = _own_cookies(headers, http_params.cookies)
        http_request = client.build_request(
            method=http_params.method.value,
            url=http_params.url,
            headers=headers,
            content=content,
            json=http_params.json,
            timeout=_httpx_timeout(http_params.timeout),
        )
        return client, http_request, own

    async def _send(
        self,
        client: httpx2.AsyncClient,
        request: httpx2.Request,
        own: dict[str, str],
        *,
        stream: bool,
    ) -> httpx2.Response:
        """Send ``request``, following redirects when the scraper asked.

        Each hop carries the session's cookies for its URL; a hop to the
        first hop's host also carries the request's own laid over them, and
        a hop to any other host does not, as they were set for that host
        alone. After an https first hop, a plain-http hop does not carry
        them either, so a downgrade redirect cannot leak them unencrypted.
        A redirect from that host that sets or deletes one of them takes it
        out of the request's own, so later hops carry the jar's value. A
        hop that a redirect turned into a bodiless GET drops the body's
        headers, as a browser does. A redirect whose ``Location`` does not
        parse, or names a scheme other than http(s), is the site's fault: a
        ``SITE_DEGRADED`` transient, not a config error. A pool that
        :meth:`aclose` closed under the request is a ``NETWORK`` transient,
        so the request is retried rather than failed.

        Args:
            client: The pool to send on.
            request: The first hop.
            own: The request's own cookies, laid on hops to its host
                (over https only, if the first hop was).
            stream: Leave the final response's body unread.

        Returns:
            The final response.

        Raises:
            TransientException: For a redirect to a URL that cannot be
                fetched, or a pool closed before a hop was sent.
            httpx2.TooManyRedirects: Past the client's redirect limit.
        """
        hops = 0
        host = request.url.host
        secure = request.url.scheme == "https"
        while True:
            if request.url.host == host and (
                request.url.scheme == "https" or not secure
            ):
                _lay_own_cookies(request, own)
            if client.is_closed:
                # aclose() ran while this request was between awaits.
                raise TransientException(
                    f"transport closed before sending {request.url}",
                    url=str(request.url),
                    kind=TransientKind.NETWORK,
                )
            response = await client.send(
                request, stream=True, follow_redirects=False
            )
            if not (self._follow_redirects and response.has_redirect_location):
                break
            if response.url.host == host:
                _drop_reset_cookies(own, response)
            try:
                # Drained for keep-alive, never decoded: a hop's body is
                # discarded, so a mislabeled coding must not fail the chain.
                # A body already read (a mock's ``content=``) needs no drain.
                if not response.is_stream_consumed:
                    async for _ in response.aiter_raw():
                        pass
            finally:
                await response.aclose()
            next_request = response.next_request
            if next_request is None or next_request.url.scheme not in (
                "http",
                "https",
            ):
                raise TransientException(
                    f"redirect from {request.url} to unfetchable Location "
                    f"{response.headers['Location']!r}",
                    url=str(request.url),
                    kind=TransientKind.SITE_DEGRADED,
                )
            hops += 1
            if hops > client.max_redirects:
                raise httpx2.TooManyRedirects(
                    "Exceeded maximum allowed redirects.",
                    request=next_request,
                )
            # httpx2 drops only the framing headers when it switches to GET.
            if next_request.method != request.method:
                for name in _BODY_HEADERS:
                    next_request.headers.pop(name, None)
            request = next_request
        if not stream:
            try:
                await response.aread()
            except BaseException:
                await response.aclose()
                raise
        return response

    async def _exchange(
        self, request: Request, *, stream: bool
    ) -> httpx2.Response:
        """Build ``request``, send it, and return the final response.

        Leniency and translation cover the send only. A streamed body is
        read by the caller after this returns, possibly in another task, so
        a context variable set across that read would leak into the caller;
        the archive stream translates that read itself.

        Args:
            request: The request to send.
            stream: Leave the final response's body unread.

        Returns:
            The final response.
        """
        http_params = request.request
        logger.debug(
            "exchange: %s %s stream=%s request_timeout=%r",
            http_params.method.value,
            http_params.url,
            stream,
            http_params.timeout,
        )
        with (
            _translate(http_params.url, http_params.timeout, self._timeout),
            self._h11_leniency(),
        ):
            client, http_request, own = await self._prepare(request)
            return await self._send(client, http_request, own, stream=stream)

    async def _resolve_request(self, request: Request) -> Response:
        """Fetch a Request and return the Response.

        Args:
            request: The request to fetch.

        Returns:
            The fetched response.

        Raises:
            HTTPResponseAssumptionException: Per the scraper's classifier.
            PersistentHTTPResponseException: Per the scraper's classifier.
            SpeculationHTTPFailure: Per the scraper's classifier, for a
                speculative request.
            RequestTimeoutException: If the request times out (retryable).
        """  # noqa: DOC502 -- raised by _translate and classify_and_raise
        http_response = await self._exchange(request, stream=False)
        body = http_response.content
        hdrs = _response_headers(http_response.headers)
        url = str(http_response.url)
        self.classify_and_raise(
            self._scraper,
            request,
            status_code=http_response.status_code,
            headers=hdrs,
            body=body,
            url=url,
        )

        return Response(
            status_code=http_response.status_code,
            headers=hdrs,
            content=body,
            # UTF-8, as a stored row replays; ``@step(encoding=)`` covers
            # other charsets.
            text=body.decode("utf-8", errors="replace"),
            url=url,
            request=request,
        )
