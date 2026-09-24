"""The request half of the scraper/driver contract.

:class:`HTTPRequestParams` is the wire-level description (method, URL,
params, body); :class:`Request` wraps it with the driver's bookkeeping —
step, location, ancestry, priority, dedup key, via, incidental match.
The dedup-key hash and the URL re-quoting used by :meth:`Request.resolve_url`
live here too, as they have no other consumer.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from typing import Any, BinaryIO, Final
from urllib.parse import quote, urljoin, urlparse

from jkent.common.coded_enum import CodedEnum
from jkent.common.decorator_metadata import DEFAULT_PRIORITY
from jkent.common.headers import merge_headers
from jkent.common.incidental import Multiple, Singular
from jkent.common.rate_limits import NO_RATE_LIMIT
from jkent.common.response import Response
from jkent.common.via import FieldValue, ViaFormSubmit, ViaLink
from jkent.contracts import ensure


class HttpMethod(CodedEnum):
    """HTTP methods supported by scrapers.

    A :class:`~jkent.common.coded_enum.CodedEnum`: handled in Python as the
    method name (so it still hashes, encodes, and compares against a literal
    the way the transports and cache-key hasher expect) and stored in the
    ``requests.method`` column as the integer in ``.code``.

    The codes are jkent's own, not anything the HTTP spec assigns — they are a
    storage detail and must not be renumbered. Nothing puts them on the wire.
    """

    GET = (1, "GET")
    OPTIONS = (2, "OPTIONS")
    POST = (3, "POST")
    PUT = (4, "PUT")
    DELETE = (5, "DELETE")
    PATCH = (6, "PATCH")
    HEAD = (7, "HEAD")


# Type aliases for complex parameter types
QueryParams = dict[str, Any] | list[tuple[str, Any]] | bytes | None
RequestData = dict[str, Any] | list[tuple[str, Any]] | bytes | BinaryIO | None
HeadersType = dict[str, str] | None
CookiesType = dict[str, str] | None
FileTuple = (
    tuple[str, BinaryIO]
    | tuple[str, BinaryIO, str]
    | tuple[str, BinaryIO, str, dict[str, str]]
)
# Values mirror requests' ``files=``: a file-like object, a file tuple,
# or raw str content. All forms round-trip through the queue: str content is
# stored as text, while bytes / file-like content is base64-encoded (see
# ``database_engine.queue._serialize_files``). Binary content reconstructs as
# ``bytes`` on deserialize.
FilesType = dict[str, BinaryIO | FileTuple | str] | None
AuthType = tuple[str, str] | None
TimeoutType = float | tuple[float, float] | None
ProxiesType = dict[str, str] | None
VerifyType = bool | str
CertType = str | tuple[str, str] | None


@dataclass(frozen=True)
class HTTPRequestParams:
    """Parameters for an HTTP request, mirroring the requests library interface.

    :param method: HTTP method for the request: ``GET``, ``OPTIONS``, ``HEAD``,
        ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
    :param url: URL for the request.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the request.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the request.
    :param json: (optional) A JSON serializable Python object to send in the
        body of the request.
    :param headers: (optional) Dictionary of HTTP Headers to send with the request.
    :param cookies: (optional) Dict of cookies to send with the request.
    :param files: (optional) Dictionary of ``'name': file-like-objects``
        (or ``{'name': file-tuple}``) for multipart encoding upload.
        ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``,
        3-tuple ``('filename', fileobj, 'content_type')``
        or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``,
        where ``'content_type'`` is a string defining the content type of the
        given file and ``custom_headers`` a dict-like object containing
        additional headers to add for the file.
    :param auth: (optional) Auth tuple to enable Basic/Digest/Custom HTTP Auth.
    :param timeout: (optional) How many seconds to wait for the server to send
        data before giving up, as a float, or a (connect timeout, read timeout) tuple.
    :param allow_redirects: (optional) Boolean. Enable/disable
        GET/OPTIONS/POST/PUT/PATCH/DELETE/HEAD redirection. Defaults to ``True``.
    :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
    :param verify: (optional) Either a boolean, in which case it controls whether
        we verify the server's TLS certificate, or a string, in which case it
        must be a path to a CA bundle to use. Defaults to ``True``.
    :param stream: (optional) if ``False``, the response content will be
        immediately downloaded.
    :param cert: (optional) if String, path to ssl client cert file (.pem).
        If Tuple, ('cert', 'key') pair.
    """

    method: HttpMethod
    url: str
    params: QueryParams = None
    data: RequestData = None
    json: Any = None
    headers: HeadersType = None
    cookies: CookiesType = None
    files: FilesType = None
    auth: AuthType = None
    timeout: TimeoutType = 60
    allow_redirects: bool = True
    proxies: ProxiesType = None
    verify: VerifyType = True
    stream: bool = False
    cert: CertType = None


@ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        len(result) == 64 and set(result) <= set("0123456789abcdef")
    ),
    "dedup key is a sha256 hex digest",
)
def _generate_deduplication_key(request_params: HTTPRequestParams) -> str:
    """Generate a deduplication key from HTTPRequestParams.

    Default deduplication key is a SHA256 hash of:
    - HTTP method
    - Full URL with parameters
    - Request data (sorted if dict/list of tuples)

    Args:
        request_params: The HTTP request parameters.

    Returns:
        A SHA256 hex digest string for deduplication.
    """
    # Start with the method and full URL. The method is part of a
    # request's identity: a GET search page and a bodyless POST search
    # submission to the same URL must not dedup each other away.
    url_str = f"{request_params.method.value} {request_params.url}"

    # Add query parameters if present
    if request_params.params:
        # Sort params for consistent hashing
        if isinstance(request_params.params, dict):
            sorted_params = sorted(request_params.params.items())
            params_str = str(sorted_params)
        elif isinstance(request_params.params, (list, tuple)):
            # Sort by repr: total over mixed value types (plain tuple
            # comparison raises TypeError when two entries share a name
            # and carry e.g. an int and a str).
            sorted_params = sorted(request_params.params, key=repr)
            params_str = str(sorted_params)
        else:
            # bytes or other type - use as-is
            params_str = str(request_params.params)
        url_str = f"{url_str}?{params_str}"

    # Add request data if present
    data_str = ""
    if request_params.data:
        if isinstance(request_params.data, dict):
            # Sort dict by key
            sorted_data = sorted(request_params.data.items())
            data_str = str(sorted_data)
        elif isinstance(request_params.data, list):
            # Sort full entries by repr so the key is invariant under
            # field order (sorting by name alone left duplicate names
            # in yield order) and total over mixed value types.
            sorted_data = sorted(request_params.data, key=repr)
            data_str = str(sorted_data)
        elif isinstance(request_params.data, bytes):
            data_str = str(request_params.data)
        elif hasattr(request_params.data, "read"):
            # File-like body: key on the content, not the object —
            # str(stream) renders a memory address, which would give
            # the same logical request a fresh key per construction.
            # Non-seekable streams can't be inspected without
            # consuming them, so they keep identity-based hashing.
            stream = request_params.data
            if stream.seekable():
                pos = stream.tell()
                data_str = str(stream.read())
                stream.seek(pos)
            else:
                data_str = str(stream)
        else:
            data_str = str(request_params.data)

    # Add JSON data if present
    if request_params.json is not None:
        if isinstance(request_params.json, dict):
            # Sort dict by key for consistent hashing
            json_str = json.dumps(request_params.json, sort_keys=True)
        else:
            json_str = json.dumps(request_params.json)
        data_str = f"{data_str}|{json_str}"

    # Combine URL and data, then hash
    combined = f"{url_str}|{data_str}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# Characters that never need percent-encoding (RFC 3986 unreserved set).
# Escapes of these are safe to decode during normalization; escapes of
# anything else (delimiters like %26 / %2F, non-ASCII bytes) must be
# preserved verbatim or the URL's meaning changes.
_UNRESERVED = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
# Reserved/sub-delim characters left raw when re-quoting a full URL,
# plus '%' so the escapes preserved above pass through untouched.
_REQUOTE_SAFE = "!#$%&'()*+,/:;=?@[]~"


def _requote_uri(uri: str) -> str:
    """Normalize a URL's percent-encoding without changing its meaning.

    Decodes escapes of unreserved characters (``%41`` → ``A``), keeps
    every other escape verbatim, percent-encodes stray ``%`` that don't
    start a valid escape, then quotes any remaining unsafe characters
    (spaces, non-ASCII). Idempotent, and — unlike a blanket
    unquote/quote round-trip — never turns an encoded delimiter such as
    ``%26`` into a live one.
    """
    out: list[str] = []
    i = 0
    while i < len(uri):
        char = uri[i]
        if char == "%":
            hex_pair = uri[i + 1 : i + 3]
            if len(hex_pair) == 2 and set(hex_pair) <= _HEX_DIGITS:
                decoded = chr(int(hex_pair, 16))
                if decoded in _UNRESERVED:
                    out.append(decoded)
                else:
                    out.append("%" + hex_pair)
                i += 3
                continue
            # Stray '%' that doesn't start an escape: encode it.
            out.append("%25")
            i += 1
            continue
        out.append(char)
        i += 1
    return quote("".join(out), safe=_REQUOTE_SAFE)


class SkipDeduplicationCheck:
    """Sentinel for ``deduplication_key`` that skips the dedup check.

    Pass an *instance* — ``deduplication_key=SkipDeduplicationCheck()`` — not
    the class itself, so the request opts out of deduplication entirely.
    """

    pass


# DEFAULT_PRIORITY (the priority for requests whose author didn't choose one)
# is defined in jkent.common.decorator_metadata and imported above so the
# decorators and data_types share one source of truth; re-exported here for
# the many callers that import it from data_types.
# Default priority for archive (file download) requests: downloads jump
# the queue because stale server-side state expires quickly.
ARCHIVE_DEFAULT_PRIORITY: Final = 1
# Soft-failure status the driver assigns to a speculative 2xx response that
# actually_successful() rejected, so the speculation callback sees a failure
# instead of a success.
SPECULATION_SOFT_FAILURE_STATUS: Final = 555


@dataclass(frozen=True)
class Request:
    """Unified request type for all scraper navigation patterns.

    Provides common functionality for URL resolution and HTTP parameters.
    Each request tracks its current_location and request ancestry.

    Controls behavior via boolean flags:
    - Default (nonnavigating=False, archive=False): Navigating request.
      Updates current_location when resolved. Supports speculation.
    - nonnavigating=True: Fetches data without changing current_location.
      Useful for API calls that provide supplementary data.
    - archive=True: Downloads and archives files. Preserves current_location.
      The driver returns an ArchiveResponse whose ``file_url`` holds the
      local filesystem path, injected into steps as ``local_filepath``.

    Attributes:
        request: HTTP request parameters (URL, method, headers, etc.).
        step: The method name to call with the Response, or a Callable.
                     When a Callable is provided, the @step decorator will automatically
                     resolve it to the function's name.
        current_location: The URL context for resolving relative URLs.
        parent_request: The immediate parent request that led to this one,
                        or None for entry requests. When we pop this off a request queue
                        we only take one parent, so we don't grow memory unbounded.
        accumulated_data: Data collected across the request chain.
        priority: Priority for request queue ordering (lower = higher
                  priority). None means "unset": the request inherits the
                  target step's priority when ``step`` is a
                  Callable, archive requests default to
                  ARCHIVE_DEFAULT_PRIORITY, and the queue falls back to
                  DEFAULT_PRIORITY (see effective_priority). An explicit
                  value — including an explicit 9 — is always kept.
        deduplication_key: Key for deduplication (defaults to hash of URL and
            data). Requests with the same deduplication_key will be resolved once,
            subsequent requests with the same key will be pruned. If you don't provide
            this key, it will default to a hash of the url and data. If you want "the same"
            request to be processed multiple times, you can pass different values here that
            embed some part of the context, or you can pass a ``SkipDeduplicationCheck()``
            instance to opt out.
        permanent: Persistent data (cookies, headers) that flows through the request chain.
        is_speculative: Whether this request is speculative (probing for content existence).
        speculation_tracking_id: Row id of the ``speculation_tracking`` entry for
                       the template that generated this request. The tracking row
                       is created before its probes are enqueued, so the id is
                       known by the time the request is built. None for
                       non-speculative requests.
        speculative_index: The integer passed to ``Speculative.from_int()`` to
                       build this probe — where it sits in the template's
                       sequence. None for non-speculative requests.
        via: Optional description of how the request was produced (ViaLink, ViaFormSubmit).
             Enables the Playwright driver to replay the browser action. HTTP driver ignores.
        incidental: Optional :class:`IncidentalMatch` (``Singular``/``Multiple``)
             selecting a sub-request captured by this request's *parent*
             navigation to promote into this request's Response. When set, the
             request does NOT navigate — the Playwright transport resolves it by
             matching against the parent's captured incidentals. Browser-only:
             the HTTP transport raises, since it captures no incidentals. Such a
             promoted response is not independently re-fetchable, so these
             requests are treated as non-reseedable.
        rate_limit: Which of the scraper's rate-limit lanes gates this
             request, by name (see :mod:`jkent.common.rate_limits`). None
             means the default lane (the scraper's ``rate_limits``), unless
             the target step declares one with ``@step(rate_limit=...)``,
             which an unset value inherits the way ``priority`` does.
             ``"none"`` (:data:`~jkent.common.rate_limits.NO_RATE_LIMIT`)
             is never throttled — for time-sensitive fetches off the scraped
             origin, such as presigned download links that expire. Any other
             name must appear in the scraper's ``named_rate_limits``; an
             unknown one fails at enqueue.
        reseedable: Tri-state marker for whether this request is safe to re-seed in isolation.
             True = stateless; can be re-fetched standalone. False = depends on server-mirrored
             client state (session, ViewState, CSRF token). None = unspecified.
             Consumed by replay tooling to choose how far up the parent chain
             to walk when re-seeding errored subtrees.
        nonnavigating: If True, does not update current_location.
        archive: If True, downloads and archives the file.
        expected_type: Optional file type hint for archive requests ("pdf", "audio", etc.).
        archive_hash_header: Reserved for future use, to contain ETag/SHA256 header ids.
    """

    request: HTTPRequestParams
    step: str | Callable[..., Any] = ""
    current_location: str = ""
    parent_request: Request | None = None
    accumulated_data: dict[str, Any] = field(default_factory=dict)
    priority: int | None = None
    deduplication_key: str | None | SkipDeduplicationCheck = None
    permanent: dict[str, Any] = field(default_factory=dict)
    is_speculative: bool = False
    speculation_tracking_id: int | None = None
    speculative_index: int | None = None
    via: ViaLink | ViaFormSubmit | None = None
    incidental: Singular | Multiple | None = None
    rate_limit: str | None = None
    reseedable: bool | None = None
    nonnavigating: bool = False
    archive: bool = False
    expected_type: str | None = None
    archive_hash_header: str | None = None
    #: Deprecated spelling of ``step``, accepted so scrapers written against
    #: the old keyword keep constructing; warns once per call site. Remove
    #: once juriscraper-prs and the hosts have been ported to ``step=``. Note
    #: the
    #: default lingers as a class attribute (``dataclasses.replace`` reads it
    #: back), so ``request.continuation`` is ``None``, never the step.
    # pyre-ignore[16]: pyre mishandles ``InitVar`` fields on dataclasses.
    continuation: InitVar[str | Callable[..., Any] | None] = None
    #: Deprecated spelling of ``rate_limit="none"``; same lifecycle as
    #: ``continuation``. ``True`` selects the unlimited lane, ``False`` is a
    #: no-op — both warn once per call site.
    # pyre-ignore[16]: pyre mishandles ``InitVar`` fields on dataclasses.
    bypass_rate_limit: InitVar[bool | None] = None

    def __post_init__(
        self,
        continuation: str | Callable[..., Any] | None,
        bypass_rate_limit: bool | None,
    ) -> None:
        """Deep copy accumulated_data and permanent to prevent unintended sharing.

        When a scraper yields multiple requests from the same method, they might
        share the same accumulated_data dict. Without deep copy, mutations in one
        branch would affect sibling branches. This is critical for correctness.

        Example problem without deep copy::

            shared_data = {"case_name": "Ant v. Bee"}
            yield Request(url="/detail/1", accumulated_data=shared_data)
            yield Request(url="/detail/2", accumulated_data=shared_data)
            # If detail/1 mutates the dict, detail/2 sees the mutation - BUG!

        The deep copy ensures each request gets its own independent copy of the data.
        """
        if continuation is not None:
            if self.step:
                raise TypeError(
                    "Request takes either step= or the deprecated "
                    "continuation=, not both"
                )
            warnings.warn(
                "Request(continuation=...) is deprecated; use step=",
                DeprecationWarning,
                stacklevel=3,
            )
            object.__setattr__(self, "step", continuation)
        if bypass_rate_limit is not None:
            warnings.warn(
                "Request(bypass_rate_limit=...) is deprecated; use "
                'rate_limit="none"',
                DeprecationWarning,
                stacklevel=3,
            )
            if bypass_rate_limit:
                if self.rate_limit not in (None, NO_RATE_LIMIT):
                    raise TypeError(
                        "Request takes either rate_limit= or the deprecated "
                        "bypass_rate_limit=True, not both"
                    )
                object.__setattr__(self, "rate_limit", NO_RATE_LIMIT)
        assert self.step and self.step != "", "Request made without step"
        # If archive=True and the author didn't choose a priority, default
        # to the higher archive priority for file downloads. An explicit
        # priority — even 9 — is kept.
        if self.archive and self.priority is None:
            object.__setattr__(self, "priority", ARCHIVE_DEFAULT_PRIORITY)

        # Since the dataclass is frozen, we need to use object.__setattr__
        object.__setattr__(
            self, "accumulated_data", deepcopy(self.accumulated_data)
        )
        object.__setattr__(self, "permanent", deepcopy(self.permanent))

        if self.permanent:
            new_request = self._merge_permanent_into_request()
            object.__setattr__(self, "request", new_request)

        if self.deduplication_key is None:
            object.__setattr__(
                self,
                "deduplication_key",
                _generate_deduplication_key(self.request),
            )

    @property
    def effective_priority(self) -> int:
        """The priority the queue should use.

        Resolves an unset (None) priority to DEFAULT_PRIORITY; explicit
        priorities are returned as-is.
        """
        if self.priority is None:
            return DEFAULT_PRIORITY
        return self.priority

    def _merge_permanent_into_request(self) -> HTTPRequestParams:
        """Merge permanent headers and cookies into the HTTPRequestParams.

        Returns:
            A new HTTPRequestParams with permanent data merged in.
        """
        req = self.request
        merged_headers: dict[str, str] | None = None
        merged_cookies: CookiesType = None
        # Merge headers. Permanent values are the base; an explicit
        # per-request header for the same name overrides the permanent one
        # by the transports' rule (case-insensitive, one value per name),
        # so the request carries the header set that reaches the wire.
        if "headers" in self.permanent:
            merged_headers = merge_headers(
                self.permanent["headers"], req.headers
            )
        else:
            merged_headers = req.headers

        # Merge cookies (only if both are dicts). Same precedence as
        # headers: permanent is the base, the per-request cookie wins.
        if "cookies" in self.permanent:
            if req.cookies is None:
                merged_cookies = dict(self.permanent["cookies"])
            elif isinstance(req.cookies, dict):
                merged_cookies = dict(self.permanent["cookies"])
                merged_cookies.update(req.cookies)
        else:
            merged_cookies = req.cookies

        return replace(req, headers=merged_headers, cookies=merged_cookies)

    @ensure(
        lambda result, current_location: (  # pyrefly: ignore[implicit-any-lambda]
            not urlparse(current_location).scheme
            or urlparse(result).scheme != ""
        ),
        "resolving against an absolute location yields an absolute URL",
    )
    def resolve_url(self, current_location: str) -> str:
        """Resolve the URL against the current location.

        Uses urllib.parse.urljoin to handle both relative and absolute URLs:
        - Absolute URLs (http://..., https://...) are returned unchanged
        - Relative URLs are resolved against current_location

        Args:
            current_location: The current page URL.

        Returns:
            The absolute URL.
        """
        # Normalize URL encoding. _requote_uri only decodes escapes of
        # unreserved characters and only encodes characters that are
        # invalid raw, so already-encoded URLs aren't double-encoded and
        # encoded delimiters (%26, %2F, %3D) keep their meaning.
        return urljoin(current_location, _requote_uri(self.request.url))

    def resolve_request_from(
        self, context: Response | Request
    ) -> tuple[HTTPRequestParams, str, Request]:
        if isinstance(context, Response):
            # Response from a Request - use its URL
            resolved_location = context.url
            parent_request = context.request
        else:
            # Request - use its current_location
            resolved_location = context.current_location
            parent_request = context
        return (
            replace(self.request, url=self.resolve_url(resolved_location)),
            resolved_location,
            parent_request,
        )

    def resolve_from(self, context: Response | Request) -> Request:
        """Create a new request with URL resolved from a Response or Request.

        - If context is a Response, use the response's URL as current_location
        - If context is a Request, use its current_location
        - accumulated_data is carried forward from the new request (self)

        Args:
            context: Response from a previous request or the originating Request.

        Returns:
            A new Request with resolved URL and updated context.
        """
        request, location, parent = self.resolve_request_from(context)
        # Merge permanent data - parent's permanent + this request's
        # permanent. "headers" and "cookies" merge by inner key (child wins
        # on conflicts): a child adding X-Requested-With must not silently
        # drop the chain's Authorization header. Header names are
        # case-insensitive; cookie names are not.
        merged_permanent = {**parent.permanent, **self.permanent}
        parent_headers = parent.permanent.get("headers")
        child_headers = self.permanent.get("headers")
        if isinstance(parent_headers, dict) and isinstance(
            child_headers, dict
        ):
            merged_permanent["headers"] = merge_headers(
                parent_headers, child_headers
            )
        parent_cookies = parent.permanent.get("cookies")
        child_cookies = self.permanent.get("cookies")
        if isinstance(parent_cookies, dict) and isinstance(
            child_cookies, dict
        ):
            merged_permanent["cookies"] = {**parent_cookies, **child_cookies}
        # An auto-generated key was hashed from the still-relative URL at
        # construction time; two "detail.aspx" yields from different pages
        # would collide. Detect auto keys by recomputing the hash for the
        # unresolved params and pass None so __post_init__ regenerates the
        # key from the resolved URL. Explicit keys (including a hand-built
        # hash, which behaves identically) and SkipDeduplicationCheck pass
        # through untouched.
        deduplication_key = self.deduplication_key
        if deduplication_key == _generate_deduplication_key(self.request):
            deduplication_key = None
        return replace(
            self,
            request=request,
            current_location=location,
            parent_request=parent,
            deduplication_key=deduplication_key,
            permanent=merged_permanent,
        )

    async def resolve_deferred_fields(self) -> Request:
        """Await any :data:`FieldResolver` values in this request's form data.

        A resolver (zero-arg async callable) can stand in for a field value
        in ``Form.submit`` data — and hence in the HTTP ``params``/``data``
        and ``ViaFormSubmit.field_data`` — so a scraper can fetch the value
        from an external service (e.g. an image-captcha solver). The driver
        awaits them here, when the yielded request is enqueued, so only
        concrete values reach serialization and the transports.

        ``Form.submit`` places the same value in both the HTTP payload and
        ``field_data``, so each distinct resolver is awaited once and its
        result substituted everywhere — the HTTP and browser transports must
        submit identical data.

        Returns self when there is nothing to resolve. Otherwise returns a
        new Request with resolved values; an auto-generated deduplication
        key (hashed from the resolver's repr at construction time) is
        regenerated from the resolved values, mirroring ``resolve_from``'s
        URL handling, while explicit keys pass through untouched.
        """
        field_dicts: list[dict[str, Any]] = [
            d
            for d in (self.request.params, self.request.data)
            if isinstance(d, dict)
        ]
        if isinstance(self.via, ViaFormSubmit):
            field_dicts.append(self.via.field_data)

        pending = {
            id(value): value
            for d in field_dicts
            for value in d.values()
            if callable(value)
        }
        if not pending:
            return self

        resolved: dict[int, FieldValue] = {}
        for key, fn in pending.items():
            out = fn()
            if not inspect.isawaitable(out):
                raise TypeError(
                    f"field resolver {fn!r} must be a zero-arg async "
                    f"callable; calling it returned "
                    f"{type(out).__name__} instead of an awaitable"
                )
            value = await out
            if not (
                isinstance(value, str)
                or (
                    isinstance(value, list)
                    and all(isinstance(item, str) for item in value)
                )
            ):
                raise TypeError(
                    f"field resolver {fn!r} must return str | list[str], "
                    f"got {type(value).__name__}"
                )
            resolved[key] = value

        def substitute(d: dict[str, Any]) -> dict[str, Any]:
            return {
                name: resolved[id(value)] if callable(value) else value
                for name, value in d.items()
            }

        new_params = self.request
        if isinstance(new_params.params, dict):
            new_params = replace(
                new_params, params=substitute(new_params.params)
            )
        if isinstance(new_params.data, dict):
            new_params = replace(new_params, data=substitute(new_params.data))
        new_via = self.via
        if isinstance(new_via, ViaFormSubmit):
            new_via = replace(
                new_via, field_data=substitute(new_via.field_data)
            )

        deduplication_key = self.deduplication_key
        if deduplication_key == _generate_deduplication_key(self.request):
            deduplication_key = None
        return replace(
            self,
            request=new_params,
            via=new_via,
            deduplication_key=deduplication_key,
        )

    def speculative(self, tracking_id: int, spec_id: int) -> Request:
        """Create a speculative copy of this request.

        Returns a new Request with is_speculative=True pointing at the
        ``speculation_tracking`` row of the template that produced it.

        Args:
            tracking_id: Row id of the template's ``speculation_tracking``
                entry, which the driver upserts before seeding its probes.
            spec_id: The integer ID from the Speculative.from_int() call.

        Returns:
            A new Request with speculation fields set.
        """
        return replace(
            self,
            is_speculative=True,
            speculation_tracking_id=tracking_id,
            speculative_index=spec_id,
        )
