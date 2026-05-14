"""Request managers for handling HTTP requests.

This module provides SyncRequestManager and AsyncRequestManager classes that
encapsulate the HTTP client, and request resolution logic.

The request manager is responsible for:
- Maintaining the HTTP client (httpx.Client or httpx.AsyncClient)
- Converting HTTP responses to Response objects

This separation allows drivers to focus on queue management and scraper
orchestration while delegating HTTP concerns to the request manager.
"""

from __future__ import annotations

import contextlib
import logging
import ssl
from typing import TYPE_CHECKING, Any, cast

import httpx

from kent.common.exceptions import (
    HTMLResponseAssumptionException,
    PersistentHTTPResponseException,
    RequestTimeoutException,
    SpeculationHTTPFailure,
)
from kent.data_types import (
    BaseRequest,
    BaseScraper,
    DriverRequirement,
    Response,
    TimeoutType,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from http.cookiejar import CookieJar

    from pyrate_limiter import Limiter, Rate


def _httpx_timeout(timeout: TimeoutType) -> Any:
    """Translate kent's TimeoutType to httpx's per-request timeout.

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


def _classify_and_raise(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
    http_response: httpx.Response,
    url: str,
    request: BaseRequest,
    *,
    body: bytes | None,
) -> None:
    """Consult the scraper's classifier and raise if the status is an error.

    ``body`` is ``None`` on streaming paths where the body hasn't been
    consumed yet. Successful / unclassified codes return silently; the
    caller then constructs and returns a :class:`Response`.

    For persistent-classified codes, speculative requests raise the
    narrower :class:`SpeculationHTTPFailure` so the worker can record the
    failure as a speculation outcome instead of an error row.
    """
    code = http_response.status_code
    hdrs = dict(http_response.headers)
    if scraper.is_transient_error(code, hdrs, body):
        raise HTMLResponseAssumptionException(
            status_code=code,
            expected_codes=[200],
            url=url,
        )
    if scraper.is_persistent_error(code, hdrs, body):
        if getattr(request, "is_speculative", False):
            raise SpeculationHTTPFailure(code, url)
        raise PersistentHTTPResponseException(code, url)


class SyncStreamingResponse:
    """Streaming wrapper around an open :class:`httpx.Response`.

    Exposes status/headers/url immediately and defers body consumption to the
    ``iter_bytes`` method. Only valid while the enclosing context manager
    returned by :meth:`SyncRequestManager.stream_request` is open.
    """

    def __init__(self, http_response: httpx.Response, url: str) -> None:
        self._response = http_response
        self.status_code = http_response.status_code
        self.headers = dict(http_response.headers)
        self.url = url

    def iter_bytes(self, chunk_size: int | None = None) -> Iterator[bytes]:
        return self._response.iter_bytes(chunk_size=chunk_size)


class AsyncStreamingResponse:
    """Async streaming wrapper around an open :class:`httpx.Response`."""

    def __init__(self, http_response: httpx.Response, url: str) -> None:
        self._response = http_response
        self.status_code = http_response.status_code
        self.headers = dict(http_response.headers)
        self.url = url

    def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        return self._response.aiter_bytes(chunk_size=chunk_size)


logger = logging.getLogger(__name__)


def _wants_follow_redirects(
    scraper: type[BaseScraper[Any]] | BaseScraper[Any],
) -> bool:
    return DriverRequirement.FOLLOW_REDIRECTS in getattr(
        scraper, "driver_requirements", []
    )


def _merge_cookies_into_headers(
    cookies: dict[str, str] | CookieJar | None,
    headers: dict[str, Any],
) -> None:
    """Merge per-request cookies into a Cookie header.

    httpx deprecated the per-request ``cookies`` kwarg.  This helper
    serialises cookies into the ``Cookie`` header instead.
    """
    if not cookies:
        return

    from http.cookiejar import CookieJar

    if isinstance(cookies, CookieJar):
        pairs = [f"{c.name}={c.value}" for c in cookies]
    else:
        pairs = [f"{k}={v}" for k, v in cookies.items()]

    if not pairs:
        return

    cookie_str = "; ".join(pairs)
    for k in headers:
        if k.lower() == "cookie":
            headers[k] = f"{headers[k]}; {cookie_str}"
            return
    headers["Cookie"] = cookie_str


class SyncRequestManager:
    """Manages HTTP requests for synchronous drivers.

    This class encapsulates:

    - httpx.Client lifecycle
    - Request resolution (URL fetching)
    - Response transformation

    Example::

        manager = SyncRequestManager(
            ssl_context=scraper.get_ssl_context(),
            timeout=30.0,
        )
        response = manager.resolve_request(request)
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext | None = None,
        timeout: float | None = None,
        rates: list[Rate] | None = None,
        proxy: str | None = None,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any] | None = None,
    ) -> None:
        """Initialize the request manager.

        Args:
            ssl_context: Optional SSL context for HTTPS connections. Use this
                for servers requiring specific cipher suites.
            timeout: Request timeout in seconds. None means no timeout (default).
            rates: Optional list of pyrate_limiter Rate objects. When provided,
                requests are throttled at the httpx transport layer.
            proxy: Optional proxy URL (e.g. ``"socks5://user:pass@host:1080"``
                or ``"http://host:3128"``). SOCKS schemes require the
                ``httpx[socks]`` extra (installed by default).
            scraper: Scraper (instance or class) whose
                ``is_transient_error`` / ``is_persistent_error`` classmethods
                decide how HTTP status codes are routed. When omitted, the
                ``BaseScraper`` defaults are used — equivalent to the old
                hard-coded "5xx raises, everything else returns" policy after
                the framework defaults were broadened.
        """
        self.timeout = timeout
        self._ssl_context = ssl_context
        self._rates = rates
        self._proxy = proxy
        self._scraper: type[BaseScraper[Any]] | BaseScraper[Any] = (
            scraper if scraper is not None else BaseScraper
        )
        self._follow_redirects = _wants_follow_redirects(self._scraper)
        self._limiter: Limiter | None = None
        self._alt_clients: dict[str, httpx.Client] = {}
        self._bypass_client: httpx.Client | None = None

        # Initialize httpx client, with rate-limited transport if rates given
        if rates:
            from pyrate_limiter import Limiter
            from pyrate_limiter.extras.httpx_limiter import (
                RateLimiterTransport,
            )

            self._limiter = Limiter(rates)
            transport_kwargs: dict[str, Any] = {}
            if ssl_context:
                transport_kwargs["verify"] = ssl_context
            if proxy:
                transport_kwargs["proxy"] = proxy
            transport = RateLimiterTransport(
                limiter=self._limiter, **transport_kwargs
            )
            self._client = httpx.Client(transport=transport, timeout=timeout)
        elif ssl_context:
            self._client = httpx.Client(
                verify=ssl_context, timeout=timeout, proxy=proxy
            )
        else:
            self._client = httpx.Client(timeout=timeout, proxy=proxy)

    def _make_client(self, verify: bool | str) -> httpx.Client:
        """Create a new httpx.Client with the given verify setting.

        Shares the same Limiter instance (if rate-limited) so that
        alternate-verify clients are still rate-limited together.
        """
        if self._limiter is not None:
            from pyrate_limiter.extras.httpx_limiter import (
                RateLimiterTransport,
            )

            transport_kwargs: dict[str, Any] = {"verify": verify}
            if self._proxy:
                transport_kwargs["proxy"] = self._proxy
            transport = RateLimiterTransport(
                limiter=self._limiter, **transport_kwargs
            )
            return httpx.Client(transport=transport, timeout=self.timeout)
        return httpx.Client(
            verify=verify, timeout=self.timeout, proxy=self._proxy
        )

    def _client_for(self, verify: bool | str) -> httpx.Client:
        """Return the appropriate httpx.Client for the given verify value.

        Returns the default client when verify is True (the default).
        Otherwise returns a lazily-created cached alternate client.
        """
        if verify is True:
            return self._client
        key = str(verify)
        if key not in self._alt_clients:
            self._alt_clients[key] = self._make_client(verify)
        return self._alt_clients[key]

    def _bypass_client_for(self, verify: bool | str) -> httpx.Client:
        """Return a non-rate-limited client for bypass requests."""
        if verify is not True:
            # For non-default verify, create a fresh non-rate-limited client
            key = f"bypass_{verify}"
            if key not in self._alt_clients:
                self._alt_clients[key] = httpx.Client(
                    verify=verify, timeout=self.timeout, proxy=self._proxy
                )
            return self._alt_clients[key]
        if self._bypass_client is None:
            if self._ssl_context:
                self._bypass_client = httpx.Client(
                    verify=self._ssl_context,
                    timeout=self.timeout,
                    proxy=self._proxy,
                )
            else:
                self._bypass_client = httpx.Client(
                    timeout=self.timeout, proxy=self._proxy
                )
        return self._bypass_client

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()
        if self._bypass_client is not None:
            self._bypass_client.close()
        for client in self._alt_clients.values():
            client.close()
        self._alt_clients.clear()

    def __enter__(self) -> SyncRequestManager:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - closes the client."""
        self.close()

    def resolve_request(self, request: BaseRequest) -> Response:
        """Fetch a BaseRequest and return the Response.

        Args:
            request: The BaseRequest to fetch. URL should be absolute.

        Returns:
            Response containing the HTTP response data.

        Raises:
            HTMLResponseAssumptionException: If server returns 5xx status code.
            httpx.TimeoutException: If request times out (for retry handling).
        """
        # Use the modified request for HTTP
        http_params = request.request
        bypass = getattr(request, "bypass_rate_limit", False)
        if bypass:
            client = self._bypass_client_for(http_params.verify)
        else:
            client = self._client_for(http_params.verify)

        headers = dict(http_params.headers) if http_params.headers else {}
        _merge_cookies_into_headers(http_params.cookies, headers)

        try:
            http_response = client.request(
                method=http_params.method.value,
                url=http_params.url,
                headers=headers,
                content=http_params.data
                if isinstance(http_params.data, bytes)
                else None,
                data=http_params.data  # type: ignore[arg-type]
                if isinstance(http_params.data, dict)
                else None,
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            )
        except httpx.TimeoutException:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            )

        _classify_and_raise(
            self._scraper,
            http_response,
            http_params.url,
            request,
            body=http_response.content,
        )

        response = Response(
            status_code=http_response.status_code,
            headers=dict(http_response.headers),
            content=http_response.content,
            text=http_response.text,
            url=http_params.url,
            request=request,
        )

        return response

    @contextlib.contextmanager
    def stream_request(
        self, request: BaseRequest
    ) -> Iterator[SyncStreamingResponse]:
        """Open a streaming HTTP request.

        Yields a :class:`SyncStreamingResponse` whose ``iter_bytes`` can be
        consumed incrementally.  The underlying httpx connection is released
        when the context manager exits.

        Raises:
            HTMLResponseAssumptionException: If server returns 5xx status code.
            RequestTimeoutException: If the request times out.
        """
        http_params = request.request
        bypass = getattr(request, "bypass_rate_limit", False)
        if bypass:
            client = self._bypass_client_for(http_params.verify)
        else:
            client = self._client_for(http_params.verify)

        headers = dict(http_params.headers) if http_params.headers else {}
        _merge_cookies_into_headers(http_params.cookies, headers)

        try:
            with client.stream(
                method=http_params.method.value,
                url=http_params.url,
                headers=headers,
                content=http_params.data
                if isinstance(http_params.data, bytes)
                else None,
                data=http_params.data  # type: ignore[arg-type]
                if isinstance(http_params.data, dict)
                else None,
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            ) as http_response:
                _classify_and_raise(
                    self._scraper,
                    http_response,
                    http_params.url,
                    request,
                    body=None,
                )
                yield SyncStreamingResponse(http_response, http_params.url)
        except httpx.TimeoutException:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            )


class AsyncRequestManager:
    """Manages HTTP requests for asynchronous drivers.

    This class encapsulates:

    - httpx.AsyncClient lifecycle
    - Request resolution (URL fetching)
    - Response transformation

    Example::

        manager = AsyncRequestManager(
            ssl_context=scraper.get_ssl_context(),
            timeout=30.0,
        )
        response = await manager.resolve_request(request)
    """

    def __init__(
        self,
        ssl_context: ssl.SSLContext | None = None,
        timeout: float | None = None,
        rates: list[Rate] | None = None,
        proxy: str | None = None,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any] | None = None,
    ) -> None:
        """Initialize the request manager.

        Args:
            ssl_context: Optional SSL context for HTTPS connections. Use this
                for servers requiring specific cipher suites.
            timeout: Request timeout in seconds. None means no timeout (default).
            rates: Optional list of pyrate_limiter Rate objects. When provided,
                requests are throttled at the httpx transport layer.
            proxy: Optional proxy URL (e.g. ``"socks5://user:pass@host:1080"``
                or ``"http://host:3128"``). SOCKS schemes require the
                ``httpx[socks]`` extra (installed by default).
            scraper: Scraper (instance or class) whose
                ``is_transient_error`` / ``is_persistent_error`` classmethods
                decide how HTTP status codes are routed.
        """
        self.timeout = timeout
        self._ssl_context = ssl_context
        self._rates = rates
        self._proxy = proxy
        self._scraper: type[BaseScraper[Any]] | BaseScraper[Any] = (
            scraper if scraper is not None else BaseScraper
        )
        self._follow_redirects = _wants_follow_redirects(self._scraper)
        self._limiter: Limiter | None = None
        self._alt_clients: dict[str, httpx.AsyncClient] = {}
        self._bypass_client: httpx.AsyncClient | None = None

        # Initialize httpx async client, with rate-limited transport if rates given
        if rates:
            from pyrate_limiter import Limiter
            from pyrate_limiter.extras.httpx_limiter import (
                AsyncRateLimiterTransport,
            )

            self._limiter = Limiter(rates)
            transport_kwargs: dict[str, Any] = {}
            if ssl_context:
                transport_kwargs["verify"] = ssl_context
            if proxy:
                transport_kwargs["proxy"] = proxy
            transport = AsyncRateLimiterTransport(
                limiter=self._limiter, **transport_kwargs
            )
            self._client = httpx.AsyncClient(
                transport=transport, timeout=timeout
            )
        elif ssl_context:
            self._client = httpx.AsyncClient(
                verify=ssl_context, timeout=timeout, proxy=proxy
            )
        else:
            self._client = httpx.AsyncClient(timeout=timeout, proxy=proxy)

    def _make_client(self, verify: bool | str) -> httpx.AsyncClient:
        """Create a new httpx.AsyncClient with the given verify setting.

        Shares the same Limiter instance (if rate-limited) so that
        alternate-verify clients are still rate-limited together.
        """
        if self._limiter is not None:
            from pyrate_limiter.extras.httpx_limiter import (
                AsyncRateLimiterTransport,
            )

            transport_kwargs: dict[str, Any] = {"verify": verify}
            if self._proxy:
                transport_kwargs["proxy"] = self._proxy
            transport = AsyncRateLimiterTransport(
                limiter=self._limiter, **transport_kwargs
            )
            return httpx.AsyncClient(transport=transport, timeout=self.timeout)
        return httpx.AsyncClient(
            verify=verify, timeout=self.timeout, proxy=self._proxy
        )

    def _client_for(self, verify: bool | str) -> httpx.AsyncClient:
        """Return the appropriate httpx.AsyncClient for the given verify value.

        Returns the default client when verify is True (the default).
        Otherwise returns a lazily-created cached alternate client.
        """
        if verify is True:
            return self._client
        key = str(verify)
        if key not in self._alt_clients:
            self._alt_clients[key] = self._make_client(verify)
        return self._alt_clients[key]

    def _bypass_client_for(self, verify: bool | str) -> httpx.AsyncClient:
        """Return a non-rate-limited client for bypass requests."""
        if verify is not True:
            key = f"bypass_{verify}"
            if key not in self._alt_clients:
                self._alt_clients[key] = httpx.AsyncClient(
                    verify=verify, timeout=self.timeout, proxy=self._proxy
                )
            return self._alt_clients[key]
        if self._bypass_client is None:
            if self._ssl_context:
                self._bypass_client = httpx.AsyncClient(
                    verify=self._ssl_context,
                    timeout=self.timeout,
                    proxy=self._proxy,
                )
            else:
                self._bypass_client = httpx.AsyncClient(
                    timeout=self.timeout, proxy=self._proxy
                )
        return self._bypass_client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.aclose()
        if self._bypass_client is not None:
            await self._bypass_client.aclose()
        for client in self._alt_clients.values():
            await client.aclose()
        self._alt_clients.clear()

    async def __aenter__(self) -> AsyncRequestManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit - closes the client."""
        await self.close()

    async def resolve_request(self, request: BaseRequest) -> Response:
        """Fetch a BaseRequest and return the Response.

        Args:
            request: The BaseRequest to fetch. URL should be absolute.

        Returns:
            Response containing the HTTP response data.

        Raises:
            HTMLResponseAssumptionException: If server returns 5xx status code.
            httpx.TimeoutException: If request times out (for retry handling).
        """

        # Use the modified request for HTTP
        http_params = request.request
        bypass = getattr(request, "bypass_rate_limit", False)
        if bypass:
            client = self._bypass_client_for(http_params.verify)
        else:
            client = self._client_for(http_params.verify)

        # Prepare content and data parameters for httpx
        request_data = http_params.data
        content_param: bytes | None = (
            request_data if isinstance(request_data, bytes) else None
        )
        data_param: dict[str, Any] | None = (
            cast(dict[str, Any], request_data)
            if isinstance(request_data, dict)
            else None
        )

        headers = dict(http_params.headers) if http_params.headers else {}
        _merge_cookies_into_headers(http_params.cookies, headers)

        logger.info(
            "resolve_request: %s %s request_timeout=%r client_timeout=%r",
            http_params.method.value,
            http_params.url,
            http_params.timeout,
            client.timeout,
        )

        # Make the HTTP request
        try:
            http_response = await client.request(
                method=http_params.method.value,
                url=http_params.url,
                headers=headers,
                content=content_param,
                data=data_param,
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            )
        except httpx.TimeoutException:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            )

        _classify_and_raise(
            self._scraper,
            http_response,
            http_params.url,
            request,
            body=http_response.content,
        )

        response = Response(
            status_code=http_response.status_code,
            headers=dict(http_response.headers),
            content=http_response.content,
            text=http_response.text,
            url=http_params.url,
            request=request,
        )

        return response

    @contextlib.asynccontextmanager
    async def stream_request(
        self, request: BaseRequest
    ) -> AsyncIterator[AsyncStreamingResponse]:
        """Open a streaming HTTP request.

        Yields an :class:`AsyncStreamingResponse` whose ``aiter_bytes`` can be
        consumed incrementally.  The underlying httpx connection is released
        when the context manager exits.
        """
        http_params = request.request
        bypass = getattr(request, "bypass_rate_limit", False)
        if bypass:
            client = self._bypass_client_for(http_params.verify)
        else:
            client = self._client_for(http_params.verify)

        request_data = http_params.data
        content_param: bytes | None = (
            request_data if isinstance(request_data, bytes) else None
        )
        data_param: dict[str, Any] | None = (
            cast(dict[str, Any], request_data)
            if isinstance(request_data, dict)
            else None
        )

        headers = dict(http_params.headers) if http_params.headers else {}
        _merge_cookies_into_headers(http_params.cookies, headers)

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
                follow_redirects=self._follow_redirects,
                timeout=_httpx_timeout(http_params.timeout),
            ) as http_response:
                logger.info(
                    "stream_request: headers received url=%s status=%s",
                    http_params.url,
                    http_response.status_code,
                )
                _classify_and_raise(
                    self._scraper,
                    http_response,
                    http_params.url,
                    request,
                    body=None,
                )
                yield AsyncStreamingResponse(http_response, http_params.url)
                logger.info(
                    "stream_request: stream closed url=%s", http_params.url
                )
        except httpx.TimeoutException:
            raise RequestTimeoutException(
                url=http_params.url,
                timeout_seconds=_timeout_seconds_for_error(
                    http_params.timeout
                ),
            )
