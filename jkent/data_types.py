"""Data types for the scraper-driver architecture.

This module defines the core data types used for communication between
scrapers and drivers. These types are designed to be:

1. Exhaustive - Using Python 3.10's match statement to ensure all cases are handled
2. Serializable - Continuations are strings, not function references
3. Immutable - Dataclasses with frozen=True where appropriate
"""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import json
import ssl
from collections.abc import Awaitable, Callable, Generator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
    Final,
    Generic,
    TypeVar,
    cast,
    get_origin,
)
from urllib.parse import parse_qs, quote, urljoin, urlparse

from pydantic import BaseModel as PydanticBaseModel
from pydantic import TypeAdapter
from pyrate_limiter import Rate

from jkent.common.decorator_metadata import (
    DEFAULT_PRIORITY,
    EntryMetadata,
    get_entry_metadata,
    get_step_metadata,
)
from jkent.common.exceptions import ScraperConfigError
from jkent.common.speculative import Speculative

# Re-exported so scrapers can keep importing the wait conditions from
# jkent.data_types; not referenced in this module itself (hence the noqa).
from jkent.common.wait_conditions import (  # noqa: F401
    WaitCondition,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.contracts import ensure

if TYPE_CHECKING:
    from jkent.common.selector_observer import SelectorObserver

T = TypeVar("T")
ScraperReturnType = TypeVar("ScraperReturnType")


class ScraperStatus(Enum):
    """Status of a scraper's development lifecycle.

    Used for documentation and registry filtering.

    Values:
        IN_DEVELOPMENT: Scraper is being built, not ready for production.
        ACTIVE: Scraper is working and maintained.
        RETIRED: Scraper is no longer maintained (court changed, etc.).
    """

    IN_DEVELOPMENT = "in_development"
    ACTIVE = "active"
    RETIRED = "retired"


class DriverRequirement(Enum):
    """Capabilities a scraper requires from its driver.

    Scrapers declare these as a ClassVar list on the class body.
    ``jkent run`` reads them to auto-select the driver and browser profile.

    Values:
        JS_EVAL: Requires JavaScript evaluation (auto-selects Playwright).
        FF_ALIKE: Requires a Firefox-like browser profile.
        CHROME_ALIKE: Requires a Chrome-like browser profile.
        HCAP_HANDLER: Requires hCaptcha interstitial handling (auto-selects Camoufox).
        RCAP_HANDLER: Requires reCAPTCHA interstitial handling (auto-selects Camoufox).
        CFCAP_HANDLER: Requires Cloudflare interstitial handling (auto-selects Playwright).
        H11_HEADER_FIXES: Loosen h11 response-header validation.
        FOLLOW_REDIRECTS: Have httpx follow 3xx redirects automatically.
        STRICTLY_SERIAL: One worker; on transient retry, idle until the
            same request is ready instead of picking up other work
            (auto-selects Playwright).

    FF_ALIKE and CHROME_ALIKE are mutually exclusive: a requirement set
    should contain at most one. This is a convention the driver relies on,
    not a constraint enforced here — declaring both is unsupported and its
    behavior is undefined.
    """

    JS_EVAL = "js_eval"
    FF_ALIKE = "ff_alike"
    CHROME_ALIKE = "chrome_alike"
    HCAP_HANDLER = "hcap_handler"
    RCAP_HANDLER = "rcap_handler"
    CFCAP_HANDLER = "cfcap_handler"
    H11_HEADER_FIXES = "h11_header_fixes"
    FOLLOW_REDIRECTS = "follow_redirects"
    STRICTLY_SERIAL = "strictly_serial"


class HTTPCodeType(Enum):
    """How a scraper treats a given HTTP status code.

    A code maps to exactly one of these. Scrapers reclassify per-site codes
    by shadowing the ``HTTP_CODE_TYPES`` mapping on the class body; because a
    mapping holds one value per key, a code can never land in two buckets, so
    the framework needs no runtime overlap check.

    Values:
        SUCCESSFUL: Pass the response through to the scraper as a Response.
        TRANSIENT: Retryable error (the request manager may retry).
        PERSISTENT: Fail-fast error (no retry).
    """

    SUCCESSFUL = "successful"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"


@dataclass(frozen=True)
class StepInfo:
    """Metadata about a scraper step method.

    The introspection surface for hosts that enumerate a scraper's steps
    (via :meth:`BaseScraper.list_steps`) — e.g. jent's repo nodes.

    Attributes:
        name: The method name (continuation string).
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for text/HTML decoding.
    """

    name: str
    priority: int
    encoding: str


class BaseScraper(Generic[ScraperReturnType]):
    """Base class for all scrapers.

    Scrapers are generic over their return type, allowing drivers to
    be type-safe about what data they collect.

    Example:
        class MyScraper(BaseScraper[MyDataModel]):
            def parse_page(self, response: Response) -> Generator[ScraperYield, None, None]:
                yield ParsedData(MyDataModel(...))

    Class Attributes:
        court_ids: Set of court IDs this scraper covers (references courts.toml).
        court_url: The primary URL/origin for this scraper's court system.
        data_types: Set of data types this scraper produces (opinions, dockets, etc.).
        status: Development lifecycle status (IN_DEVELOPMENT, ACTIVE, RETIRED).
        version: Version string for this scraper (e.g., "2025-01-03").
        last_verified: Date when scraper was last verified working.
        oldest_record: Earliest date for which records are available.
        requires_auth: Whether authentication is required.
        rate_limits: pyrate_limiter Rate objects defining rate ceilings for this scraper.
        default_headers: Baseline HTTP headers for every httpx request.
    """

    # === METADATA FOR AUTODOC ===
    # These ClassVars are used by the registry builder to generate documentation.

    court_ids: ClassVar[set[str]] = set()

    # Primary URL/origin for this scraper
    court_url: ClassVar[str] = ""

    # Data types produced by this scraper (e.g., {"opinions", "dockets"})
    data_types: ClassVar[set[str]] = set()

    # Scraper lifecycle status
    status: ClassVar[ScraperStatus] = ScraperStatus.IN_DEVELOPMENT

    # Version tracking
    version: ClassVar[str] = ""
    last_verified: ClassVar[str] = ""

    # Data availability
    oldest_record: ClassVar[date | None] = None

    # Optional metadata
    requires_auth: ClassVar[bool] = False
    rate_limits: ClassVar[list[Rate] | None] = None

    # Baseline HTTP headers the httpx transport sends with every request.
    # A per-request header with the same name (matched case-insensitively,
    # including permanent headers merged into the request) overrides the
    # default. The Playwright/Camoufox transports ignore these — the
    # browser supplies its own headers.
    default_headers: ClassVar[Mapping[str, str]] = {}

    # Driver requirements — capabilities the scraper needs from its driver.
    # jkent run reads these to auto-select driver and browser profile.
    driver_requirements: ClassVar[list[DriverRequirement]] = []

    # SSL/TLS configuration for servers requiring specific ciphers or TLS versions.
    # If set, drivers will use this context for HTTPS connections.
    # Example usage for a scraper requiring specific ciphers:
    #     @classmethod
    #     def get_ssl_context(cls) -> ssl.SSLContext:
    #         ctx = ssl.create_default_context()
    #         ctx.set_ciphers("AES256-SHA256")
    #         return ctx
    ssl_context: ClassVar[ssl.SSLContext | None] = None

    # ------------------------------------------------------------------
    # HTTP status classification
    # ------------------------------------------------------------------
    # A single mapping from status code to HTTPCodeType is the source of
    # truth. DEFAULT_HTTP_CODE_TYPES is the framework baseline; scrapers
    # reclassify per-site codes by shadowing HTTP_CODE_TYPES on the class
    # body. The active map is ``{**defaults, **override}`` (override wins
    # per code), so a code lands in exactly one bucket by construction — no
    # overlap check needed. The ``is_transient_error`` / ``is_persistent_error``
    # classmethods (see further down) read this and expose the result to the
    # worker. A code in no bucket at all is persistent by default — an
    # unrecognized status fails fast instead of passing through as success.

    DEFAULT_HTTP_CODE_TYPES: Final[Mapping[int, HTTPCodeType]] = {
        **dict.fromkeys(
            {200, 201, 202, 203, 204, 205, 206, 207, 208, 226, 304},
            HTTPCodeType.SUCCESSFUL,
        ),
        **dict.fromkeys(
            {408, 425, 429, 502, 503, 504},
            HTTPCodeType.TRANSIENT,
        ),
        **dict.fromkeys(
            # All standard 4xx minus the transient 408/425/429, plus the
            # 5xx codes that aren't gateway-style.
            {
                400,
                401,
                402,
                403,
                404,
                405,
                406,
                407,
                409,
                410,
                411,
                412,
                413,
                414,
                415,
                416,
                417,
                418,
                421,
                422,
                423,
                424,
                426,
                428,
                431,
                451,
                500,
                501,
                505,
                506,
                507,
                508,
                510,
                511,
            },
            HTTPCodeType.PERSISTENT,
        ),
    }

    # Subclasses shadow this to reclassify specific codes; a code present
    # here wins over its DEFAULT_HTTP_CODE_TYPES classification.
    HTTP_CODE_TYPES: ClassVar[Mapping[int, HTTPCodeType]] = {}

    def get_entry(self) -> Generator[Request, None, None]:
        """Create the initial request(s) to start scraping.

        Subclasses should override this method (or use @entry decorators)
        to yield their entry point(s) and initial continuation method(s).

        Yields:
            Request for each entry point.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_entry() "
            f"or use @entry decorators"
        )

    @classmethod
    def get_ssl_context(cls) -> ssl.SSLContext | None:
        """Return an SSL context for HTTPS connections, if needed.

        Override this method in scrapers that require custom SSL configuration
        (e.g., specific ciphers or TLS versions for legacy servers).

        Returns:
            An ssl.SSLContext configured for this scraper, or None to use defaults.

        Example::

            @classmethod
            def get_ssl_context(cls) -> ssl.SSLContext:
                ctx = ssl.create_default_context()
                ctx.set_ciphers("AES256-SHA256")
                return ctx
        """
        return cls.ssl_context

    # ------------------------------------------------------------------
    # HTTP status classification helpers
    # ------------------------------------------------------------------

    @classmethod
    def active_http_code_types(cls) -> Mapping[int, HTTPCodeType]:
        """The effective code→type map: defaults with the override applied.

        A code in ``HTTP_CODE_TYPES`` wins over its
        ``DEFAULT_HTTP_CODE_TYPES`` classification.
        """
        return {**cls.DEFAULT_HTTP_CODE_TYPES, **cls.HTTP_CODE_TYPES}

    @classmethod
    def _codes_of_type(cls, type_: HTTPCodeType) -> frozenset[int]:
        return frozenset(
            code
            for code, code_type in cls.active_http_code_types().items()
            if code_type is type_
        )

    @classmethod
    def active_transient_http_error_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as transient (retryable)."""
        return cls._codes_of_type(HTTPCodeType.TRANSIENT)

    @classmethod
    def active_persistent_http_error_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as persistent (fail-fast, no retry)."""
        return cls._codes_of_type(HTTPCodeType.PERSISTENT)

    @classmethod
    def active_successful_http_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as successful (pass through as Response)."""
        return cls._codes_of_type(HTTPCodeType.SUCCESSFUL)

    @classmethod
    def is_transient_error(
        cls,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> bool:
        """Is ``status_code`` a transient (retryable) error for this scraper?

        The default implementation ignores ``headers`` and ``content`` and
        returns pure set membership. Override in scrapers with dynamic
        policy (e.g. "503 with body 'maintenance' is transient, anything
        else is persistent"). ``headers`` and ``content`` may be ``None``
        when the caller hasn't observed them (for example, on a streaming
        response whose body hasn't been consumed); dynamic overrides must
        tolerate that.
        """
        return status_code in cls.active_transient_http_error_codes()

    @classmethod
    def is_persistent_error(
        cls,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> bool:
        """Is ``status_code`` a persistent (no-retry) error for this scraper?

        Codes absent from the active map are persistent by default: an
        unrecognized status (a nonstandard 520, a redirect the scraper
        didn't opt into following) is a fail-fast, not a silent success.
        Reclassify per-site via ``HTTP_CODE_TYPES`` — any code placed in
        the map (as SUCCESSFUL or TRANSIENT) escapes this fallback.

        Same semantics for ``headers`` / ``content`` as
        :meth:`is_transient_error`.
        """
        code_type = cls.active_http_code_types().get(status_code)
        return code_type is None or code_type is HTTPCodeType.PERSISTENT

    def get_continuation(
        self, name: str
    ) -> Callable[
        [Response],
        Generator[ScraperYield[ScraperReturnType], bool | None, None],
    ]:
        """Resolve a continuation name to the actual method.

        This method looks up a continuation by name and returns the
        bound method. It provides a single point for continuation
        resolution, making it easy to add validation or caching later.

        Args:
            name: The name of the continuation method.

        Returns:
            The bound method that can be called with a Response.

        Raises:
            ScraperConfigError: If the continuation method doesn't exist.
        """
        try:
            method = getattr(self, name)
        except AttributeError:
            raise ScraperConfigError(
                "Nonexistent continuation referenced"
            ) from None
        return cast(
            Callable[
                [Response],
                Generator[ScraperYield[ScraperReturnType], bool | None, None],
            ],
            method,
        )

    @staticmethod
    def _iter_decorated(
        target: Any,
        get_metadata: Callable[[Any], Any],
    ) -> Generator[tuple[str, Any, Any], None, None]:
        """Yield (name, method, metadata) for each decorated attribute.

        Shared introspection loop for list_steps/list_entries/
        _list_entry_info: walks ``dir(target)`` (a class or an instance),
        skips private/dunder names, and turns any error raised while probing
        a candidate into a ScraperConfigError that names the attribute. Only
        attributes whose ``get_metadata`` returns non-None are yielded.
        """
        owner = target if isinstance(target, type) else type(target)
        for name in dir(target):
            if name.startswith("_"):
                continue
            try:
                method = getattr(target, name)
                metadata = get_metadata(method)
            except Exception as e:
                raise ScraperConfigError(
                    f"Introspecting candidate {owner.__name__}.{name} "
                    f"raised {type(e).__name__}: {e}"
                ) from e
            if metadata is not None:
                yield name, method, metadata

    @classmethod
    def list_steps(cls) -> list[StepInfo]:
        """List all step methods defined on this scraper.

        Introspects the class to find all methods decorated with @step
        and returns their metadata.

        This is useful for the web interface to display available steps,
        their priorities, and to populate dropdowns for pause_step/resume_step.

        Returns:
            List of StepInfo objects for each decorated step method.

        Example:
            >>> class MyScraper(BaseScraper[CaseData]):
            ...     @step
            ...     def parse_listing(self, lxml_tree): ...
            ...
            ...     @step(priority=5)
            ...     def parse_detail(self, lxml_tree): ...
            ...
            >>> MyScraper.list_steps()
            [StepInfo(name='parse_listing', priority=9, encoding='utf-8'),
             StepInfo(name='parse_detail', priority=5, encoding='utf-8')]
        """
        return [
            StepInfo(
                name=name,
                priority=metadata.priority,
                encoding=metadata.encoding,
            )
            for name, _method, metadata in cls._iter_decorated(
                cls, get_step_metadata
            )
        ]

    @classmethod
    def list_speculative_entries(cls) -> list[EntryMetadata]:
        """List all speculative entry point methods defined on this scraper.

        Returns:
            List of EntryMetadata objects for speculative entries only.
        """
        return [e for e in cls.list_entries() if e.speculative]

    @classmethod
    def list_entries(cls) -> list[EntryMetadata]:
        """List all entry point methods defined on this scraper.

        Introspects the class to find all methods decorated with @entry
        and returns their metadata.

        Returns:
            List of EntryMetadata objects for each decorated entry method.
        """
        return [
            metadata
            for _name, _method, metadata in cls._iter_decorated(
                cls, get_entry_metadata
            )
        ]

    def _list_entry_info(
        self,
    ) -> list[tuple[Any, Any]]:
        """List entry methods with their metadata for dispatch.

        Returns:
            List of (bound_method, EntryMetadata) tuples.
        """
        return [
            (method, metadata)
            for _name, method, metadata in self._iter_decorated(
                self, get_entry_metadata
            )
        ]

    def initial_seed(
        self, params: list[dict[str, dict[str, Any]]]
    ) -> Generator[Request, None, None]:
        """Dispatch parameter list to entry functions and yield combined requests.

        Takes a JSON-serializable list of parameter invocations and dispatches
        them to the appropriate @entry functions.

        For non-speculative entries, params are direct function arguments and
        the method yields Requests.

        For speculative entries (parameter subclassing the Speculative ABC),
        the validated model instance is stored in ``_speculation_templates``
        (paired with the raw seed value it was validated from) for the driver
        to consume during speculation seeding. No requests are yielded for
        speculative entries here.

        Args:
            params: List of single-key dicts mapping entry function name to kwargs.
                Example: [{"search_by_number": {"docket_number": "A10"}}]
                Speculative: [{"fetch_case": {"case_id": {"year": 2026, "number": 10}}}]

        Yields:
            Request instances from non-speculative entry functions.

        Raises:
            ValueError: If params is empty/None or references unknown entry names.
        """
        if not params:
            raise ValueError(
                "initial_seed() requires at least one parameter invocation"
            )

        entry_map = {
            info.func_name: (method, info)
            for method, info in self._list_entry_info()
        }

        for invocation in params:
            for func_name, kwargs_dict in invocation.items():
                if func_name not in entry_map:
                    available = list(entry_map.keys())
                    raise ValueError(
                        f"Unknown entry '{func_name}'. Available: {available}"
                    )
                method, meta = entry_map[func_name]  # type: ignore
                validated_kwargs = meta.validate_params(kwargs_dict)

                if meta.speculative:
                    # Store the validated Speculative model instance as a
                    # template for the driver, alongside the raw seed value
                    # it was validated from (persisted with the speculation
                    # state so hosts can map state rows back to their seed
                    if not hasattr(self, "_speculation_templates"):
                        self._speculation_templates: dict[  # type: ignore
                            str, list[tuple[Speculative, Any]]
                        ] = {}
                    if func_name not in self._speculation_templates:
                        self._speculation_templates[func_name] = []
                    assert meta.speculative_param is not None
                    template = validated_kwargs[meta.speculative_param]
                    raw_seed_value = kwargs_dict.get(meta.speculative_param)
                    self._speculation_templates[func_name].append(
                        (template, raw_seed_value)
                    )
                else:
                    yield from method(**validated_kwargs)

    @classmethod
    def schema(cls) -> dict[str, Any]:
        """Generate JSON Schema for all entry points.

        Returns a dict using Pydantic's model_json_schema() for BaseModel
        parameters and standard JSON Schema types for primitives.

        Returns:
            Dict with scraper name, entries, and $defs for referenced models.
        """
        entries: dict[str, Any] = {}
        all_defs: dict[str, Any] = {}

        for entry_info in cls.list_entries():
            # Build parameter schema
            properties: dict[str, Any] = {}
            required: list[str] = []

            for param_name, param_type in entry_info.param_types.items():
                required.append(param_name)

                # get_origin is a python 3.10 cludge
                if (
                    isinstance(param_type, type)
                    and get_origin(param_type) is None
                    and issubclass(param_type, PydanticBaseModel)
                ):
                    # Use Pydantic's schema generation
                    pydantic_type = cast(type[PydanticBaseModel], param_type)
                    model_schema = pydantic_type.model_json_schema()
                    # Extract $defs and add to top-level
                    if "$defs" in model_schema:
                        all_defs.update(model_schema["$defs"])
                        del model_schema["$defs"]
                    # Store the model definition
                    type_name = param_type.__name__
                    all_defs[type_name] = model_schema
                    properties[param_name] = {"$ref": f"#/$defs/{type_name}"}
                elif param_type is str:
                    properties[param_name] = {"type": "string"}
                elif param_type is int:
                    properties[param_name] = {"type": "integer"}
                elif param_type is date:
                    properties[param_name] = {
                        "type": "string",
                        "format": "date",
                    }
                else:
                    # Typed containers and other annotations (list[str],
                    # tuple[int, str], dict[...], etc.): let pydantic generate
                    # the field schema, hoisting any nested model definitions
                    # into the shared $defs.
                    field_schema = TypeAdapter(param_type).json_schema()
                    if "$defs" in field_schema:
                        all_defs.update(field_schema.pop("$defs"))
                    properties[param_name] = field_schema

            entry_schema: dict[str, Any] = {
                "returns": entry_info.return_type.__name__,
                "speculative": entry_info.speculative,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }

            entries[entry_info.func_name] = entry_schema

        result: dict[str, Any] = {
            "scraper": cls.__name__,
            "entries": entries,
        }
        if all_defs:
            result["$defs"] = all_defs

        return result

    def actually_successful(self, response: Response) -> bool:
        """Detect hidden error states in successful HTTP responses.

        Some websites return HTTP 200 status codes but embed error states
        in the page content or headers (e.g., "No results found" pages,
        session timeout pages, soft 404s). This method allows scrapers to
        detect these hidden failures.

        This is primarily used for speculation handling. When a
        speculative request gets a 2xx response, the driver calls this
        method to check if the response actually represents a failure.
        If this returns False, the driver sets the response status_code to
        SPECULATION_SOFT_FAILURE_STATUS (555) before calling the speculation
        callback.

        Args:
            response: The Response object to check for hidden errors.

        Returns:
            True if the response is genuinely successful (default behavior).
            False if the response contains a hidden error pattern.

        Example:
            Override this method to detect site-specific error patterns::

                def actually_successful(self, response: Response) -> bool:
                    # Detect "No results" page that returns 200
                    if "No results found" in response.text:
                        return False
                    # Detect session timeout
                    if response.url.endswith("/login"):
                        return False
                    return True
        """
        return True


@dataclass(frozen=True)
class ParsedData(Generic[T]):
    """Data yielded by a scraper after parsing a page.

    This is a simple wrapper around a bit of returned data to enable exhaustive pattern
    matching in the driver. When a scraper yields data, it should wrap
    it in ParsedData so the driver can distinguish it from other yield
    types (like Request).

    Example:
        yield ParsedData({"docket": "BCC-2024-001", "case_name": "..."})
    """

    data: T
    __match_args__ = ("data",)

    def unwrap(self) -> T:
        return self.data


class HttpMethod(Enum):
    """HTTP methods supported by scrapers."""

    GET = "GET"
    OPTIONS = "OPTIONS"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"


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
    timeout: TimeoutType = None
    allow_redirects: bool = True
    proxies: ProxiesType = None
    verify: VerifyType = True
    stream: bool = False
    cert: CertType = None


@ensure(
    lambda result: (
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
        elif isinstance(request_params.params, list | tuple):
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
class Selector:
    """A selector string together with the grammar it is written in.

    Attributes:
        value: The raw selector string.
        grammar: ``"css"`` or ``"xpath"`` — a ClassVar set by each subclass.
    """

    value: str
    grammar: ClassVar[str] = ""

    CSS: ClassVar[type[CSS]]  # type: ignore
    XPath: ClassVar[type[XPath]]  # type: ignore

    @classmethod
    def of(cls, value: str, grammar: str) -> Selector:
        """Rebuild a Selector from its serialized ``value``/``grammar`` parts."""
        if grammar == XPath.grammar:
            return XPath(value)
        if grammar == CSS.grammar:
            return CSS(value)
        raise ValueError(f"unknown selector grammar: {grammar!r}")

    def nth(self, position: int) -> Selector:
        """A selector for the 1-based ``position``-th match of this one.

        Each subclass encodes the positional wrapper in its own grammar so a
        single matched node can be replayed unambiguously.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CSS(Selector):
    """A CSS selector. Also reachable as :attr:`Selector.CSS`."""

    grammar: ClassVar[str] = "css"

    def nth(self, position: int) -> Selector:
        # Playwright's :nth-match() picks the position-th match document-wide,
        # mirroring how the parse enumerated the CSS matches.
        return CSS(f":nth-match({self.value}, {position})")


@dataclass(frozen=True)
class XPath(Selector):
    """An XPath selector. Also reachable as :attr:`Selector.XPath`."""

    grammar: ClassVar[str] = "xpath"

    def nth(self, position: int) -> Selector:
        # Parenthesize first so the positional predicate applies to the whole
        # node-set rather than only the last location step.
        return XPath(f"({self.value})[{position}]")


Selector.CSS = CSS  # type: ignore
Selector.XPath = XPath  # type: ignore


@dataclass(frozen=True)
class ViaLink:
    """Describes a request produced by following a link.

    Enables the Playwright driver to replay the browser action that corresponds
    to the request (clicking the link). The HTTP driver ignores this field.

    Attributes:
        selector: The :class:`Selector` that found the <a> element. Its grammar
            lets the driver route it to Playwright's engine without re-running
            the prefix heuristic.
        description: Human-readable description of the link.
    """

    selector: Selector
    description: str

    def to_json(self) -> str:
        """This via as the ``via_json`` wire format (see :func:`via_from_json`)."""
        return json.dumps(
            {
                "type": "link",
                "selector": self.selector.value,
                "selector_type": self.selector.grammar,
                "description": self.description,
            }
        )


# A form field's submitted value, or an async resolver for one. A resolver
# is a zero-arg async callable the driver awaits when the yielded request is
# enqueued — use it to fetch the value from an external service (e.g. an
# image-captcha solver), closing over whatever the call needs. Only concrete
# values ever reach the DB or the transports.
FieldValue = str | list[str]
FieldResolver = Callable[[], Awaitable[FieldValue]]


@dataclass(frozen=True)
class ViaFormSubmit:
    """Describes a request produced by submitting a form.

    Enables the Http and Browser based transports to have a unified interface for
    submitting a form.

    Attributes:
        form_selector: The :class:`Selector` that found the <form> element. Its
            grammar lets the driver route it to Playwright's engine without
            re-running the prefix heuristic.
        submit_selector: Selector relative to the form for the submit element.
        field_data: Merged field values (defaults + overrides). A list value
            means repeated keys (checkbox groups, multi-selects). A
            :data:`FieldResolver` value is awaited at enqueue time (see
            :meth:`Request.resolve_deferred_fields`).
        description: Human-readable description of the form.
    """

    form_selector: Selector
    submit_selector: str | None
    field_data: dict[str, FieldValue | FieldResolver]
    description: str

    def to_json(self) -> str:
        """This via as the ``via_json`` wire format (see :func:`via_from_json`).

        Requires concrete ``field_data`` values — a still-unresolved
        :data:`FieldResolver` is not JSON-serializable (the driver resolves
        them at enqueue time, before serialization).
        """
        return json.dumps(
            {
                "type": "form_submit",
                "form_selector": self.form_selector.value,
                "selector_type": self.form_selector.grammar,
                "submit_selector": self.submit_selector,
                "field_data": self.field_data,
                "description": self.description,
            }
        )


def _selector_grammar(selector: str) -> str:
    """Best-effort selector grammar for legacy via rows lacking selector_type.

    Mirrors ``find_form``/``find_links``: unambiguous XPath prefixes are
    "xpath", everything else "css". Only used as a fallback — rows written
    after selector_type was added carry the real value.
    """
    return "xpath" if selector.startswith(("//", "./", "(")) else "css"


def via_from_json(raw: str) -> ViaLink | ViaFormSubmit:
    """Rebuild a via from the ``via_json`` wire format.

    The inverse of :meth:`ViaLink.to_json` / :meth:`ViaFormSubmit.to_json` —
    the single reader for the ``via_json`` column consumers store and inspect
    (the driver's queue, jent's reconstruction). Shape:
    link → ``{type, selector, selector_type, description}``; form_submit →
    ``{type, form_selector, selector_type, submit_selector, field_data,
    description}``. Rows written before ``selector_type`` existed fall back
    to the prefix heuristic.

    Raises:
        ValueError: On an unknown ``type``.
    """
    data = json.loads(raw)
    kind = data.get("type")
    if kind == "form_submit":
        return ViaFormSubmit(
            form_selector=Selector.of(
                data["form_selector"],
                data.get(
                    "selector_type",
                    _selector_grammar(data["form_selector"]),
                ),
            ),
            submit_selector=data.get("submit_selector"),
            field_data=data["field_data"],
            description=data["description"],
        )
    if kind == "link":
        return ViaLink(
            selector=Selector.of(
                data["selector"],
                data.get("selector_type", _selector_grammar(data["selector"])),
            ),
            description=data["description"],
        )
    raise ValueError(f"unknown via type {kind!r} in via_json")


def _json_deep_contains(actual: Any, expected: Any) -> bool:
    """True if ``expected`` is a structural subset of ``actual``.

    Dicts match when every expected key is present and its value deep-contains;
    lists match when every expected item is deep-contained by some actual item;
    scalars match on equality. Extra keys/items in ``actual`` are ignored — this
    is an allowlist ("match what I name, ignore the rest"), so volatile fields
    the caller doesn't mention (tokens, nonces) never break the match.
    """
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(
            k in actual and _json_deep_contains(actual[k], v)
            for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_json_deep_contains(a, e) for a in actual) for e in expected
        )
    return actual == expected


@dataclass(frozen=True)
class IncidentalMatch:
    """Allowlist spec for selecting a captured incidental request to promote.

    Attached to a :class:`Request` via ``incidental=`` (as one of the
    :class:`Singular` / :class:`Multiple` subclasses, which fix the match
    cardinality). The request does not navigate: the Playwright transport
    matches this spec against the sub-requests its *parent* navigation captured
    and promotes the matching response into this request's :class:`Response`.

    Matching is an **allowlist** — only the fields you set are checked, and
    everything unspecified (the dozens of browser-added headers, cache-busting
    params, volatile JWTs) is ignored. That is what lets a spec survive tokens
    it can't predict: you don't redact them, you simply never mention them.

    Attributes:
        url: fnmatch-style glob (case-sensitive, ``*`` spans ``/``) tested
            against the full captured URL, e.g. ``*getcasedetaildata*``.
        method: HTTP method, matched case-insensitively.
        resource_type: the browser's request-initiation kind, matched
            case-insensitively (e.g. ``fetch``/``xhr``/``document``). Use it to
            pick the JSON ``fetch`` when a page navigation and a client-side
            fetch hit the same URL (the document response would otherwise
            collide with the fetch response).
        query_contains: subset of query params that must be present with the
            given value (parsed from the captured URL).
        header_contains: subset of request headers that must be present with
            the given value; header names are matched case-insensitively.
        body_contains: for a captured request body — a ``str`` is a substring
            test against the decoded body; a ``dict`` is a deep-contains against
            the body parsed as JSON, falling back to form-encoded params.
    """

    url: str | None = None
    method: str | None = None
    resource_type: str | None = None
    query_contains: Mapping[str, str] | None = None
    header_contains: Mapping[str, str] | None = None
    body_contains: Mapping[str, Any] | str | None = None

    def __post_init__(self) -> None:
        if not any(
            (
                self.url,
                self.method,
                self.resource_type,
                self.query_contains,
                self.header_contains,
                self.body_contains,
            )
        ):
            raise ValueError(
                "IncidentalMatch requires at least one match field (url/method/"
                "resource_type/query_contains/header_contains/body_contains)."
            )

    def matches(
        self,
        *,
        url: str,
        method: str | None,
        headers: Mapping[str, str],
        body: bytes | None,
        resource_type: str | None = None,
    ) -> bool:
        """True if a captured incidental request satisfies every set field."""
        if self.url is not None and not fnmatch.fnmatchcase(url, self.url):
            return False
        if (
            self.method is not None
            and (method or "").upper() != self.method.upper()
        ):
            return False
        if (
            self.resource_type is not None
            and (resource_type or "").lower() != self.resource_type.lower()
        ):
            return False
        if self.query_contains and not self._query_matches(url):
            return False
        if self.header_contains and not self._headers_match(headers):
            return False
        return self.body_contains is None or self._body_matches(body)

    def _query_matches(self, url: str) -> bool:
        assert self.query_contains is not None
        params = parse_qs(urlparse(url).query)
        for key, value in self.query_contains.items():
            if value not in params.get(key, []):
                return False
        return True

    def _headers_match(self, headers: Mapping[str, str]) -> bool:
        assert self.header_contains is not None
        lowered = {k.lower(): v for k, v in headers.items()}
        for key, value in self.header_contains.items():
            if lowered.get(key.lower()) != value:
                return False
        return True

    def _body_matches(self, body: bytes | None) -> bool:
        contains = self.body_contains
        assert contains is not None
        if body is None:
            return False
        text = body.decode("utf-8", errors="replace")
        if isinstance(contains, str):
            return contains in text
        # dict: JSON deep-contains, then form-encoded subset as a fallback.
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and _json_deep_contains(parsed, dict(contains)):
            return True
        form = parse_qs(text)
        return all(
            value in form.get(key, []) for key, value in contains.items()
        )


@dataclass(frozen=True)
class Singular(IncidentalMatch):
    """Promote exactly one captured incidental.

    Zero matches or more than one match raises
    :class:`~jkent.common.exceptions.IncidentalRequestAssumptionException` —
    the spec is meant to pin down a single sub-request, so ambiguity is a
    structural assumption violation to fix, not a silent pick.
    """


@dataclass(frozen=True)
class Multiple(IncidentalMatch):
    """Promote every captured incidental that matches (one child each).

    The transport enqueues one promoted request per match and lets the
    continuation disambiguate by inspecting each response. Zero matches raises
    :class:`~jkent.common.exceptions.IncidentalRequestAssumptionException`.
    """


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
        continuation: The method name to call with the Response, or a Callable.
                     When a Callable is provided, the @step decorator will automatically
                     resolve it to the function's name.
        current_location: The URL context for resolving relative URLs.
        parent_request: The immediate parent request that led to this one,
                        or None for entry requests. When we pop this off a request queue
                        we only take one parent, so we don't grow memory unbounded.
        accumulated_data: Data collected across the request chain.
        priority: Priority for request queue ordering (lower = higher
                  priority). None means "unset": the request inherits the
                  target step's priority when its continuation is a
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
        speculation_id: Tuple of (function_name, param_index, integer_id) identifying
                       which speculative template generated this request. None for
                       non-speculative requests.
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
        bypass_rate_limit: If True, skip the rate limiter for this request.
             Useful for time-sensitive requests (e.g., file downloads) where
             stale server-side state expires quickly and delays cause failures.
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
    continuation: str | Callable[..., Any]
    current_location: str = ""
    parent_request: Request | None = None
    accumulated_data: dict[str, Any] = field(default_factory=dict)
    priority: int | None = None
    deduplication_key: str | None | SkipDeduplicationCheck = None
    permanent: dict[str, Any] = field(default_factory=dict)
    is_speculative: bool = False
    speculation_id: tuple[str, int, int] | None = None
    via: ViaLink | ViaFormSubmit | None = None
    incidental: Singular | Multiple | None = None
    bypass_rate_limit: bool = False
    reseedable: bool | None = None
    nonnavigating: bool = False
    archive: bool = False
    expected_type: str | None = None
    archive_hash_header: str | None = None

    def __post_init__(self) -> None:
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
        assert self.continuation and self.continuation != "", (
            "Request made without continuation"
        )
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
        # per-request header for the same key overrides the permanent one.
        if "headers" in self.permanent:
            merged_headers = dict(self.permanent["headers"])
            if req.headers:
                merged_headers.update(req.headers)
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
        lambda result, current_location: (
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
        # drop the chain's Authorization header.
        merged_permanent = {**parent.permanent, **self.permanent}
        for key in ("headers", "cookies"):
            parent_value = parent.permanent.get(key)
            child_value = self.permanent.get(key)
            if isinstance(parent_value, dict) and isinstance(
                child_value, dict
            ):
                merged_permanent[key] = {**parent_value, **child_value}
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

    def speculative(
        self, func_name: str, param_index: int, spec_id: int
    ) -> Request:
        """Create a speculative copy of this request.

        Returns a new Request with is_speculative=True and
        speculation_id set to (func_name, param_index, spec_id).

        Args:
            func_name: Name of the entry function generating this request.
            param_index: Index of the template in the params list.
            spec_id: The integer ID from the Speculative.from_int() call.

        Returns:
            A new Request with speculation fields set.
        """
        return replace(
            self,
            is_speculative=True,
            speculation_id=(func_name, param_index, spec_id),
        )


@dataclass
class Response:
    """HTTP response from fetching a page.

    Modeled after httpx.Response to provide a familiar interface.
    The driver creates Response objects and passes them to scraper
    continuation methods.

    Attributes:
        status_code: HTTP status code (200, 404, etc.).
        headers: Response headers.
        content: Raw response bytes.
        text: Decoded response text.
        url: Final URL after any redirects.
        request: The Request that triggered this response.
        observer: SelectorObserver recorded while a @step with ``page``
            injection executed against this response. Set by the step
            wrapper; per-execution by construction (the driver owns one
            Response per execution), so drivers read autowait/debug
            telemetry here. None until a page-injecting step runs.
    """

    status_code: int
    headers: dict[str, str]
    content: bytes
    text: str
    url: str
    request: Request
    observer: SelectorObserver | None = None


@dataclass
class ArchiveResponse(Response):
    """HTTP response for an archived file.

    Extends Response with a ``file_url``  The @step machinery injects this
    value into steps as the ``local_filepath`` parameter;
    This lets scrapers include the file location in their ParsedData.

    Attributes:
        file_url: path where the downloaded file was stored.
            Injected into steps as ``local_filepath``.
    """

    file_url: str = ""


@dataclass
class ArchiveDecision:
    """Decision from an ArchiveHandler about whether to download a file.

    Attributes:
        download: If True, the driver should proceed with downloading.
        file_url: When download=False, the location of the existing file.
            When download=True, may be empty (save() determines final path).
    """

    download: bool
    file_url: str = ""


# =============================================================================
# Type Alias for Scraper Yields
# =============================================================================

# A scraper can yield ParsedData, Request, or None. This type alias enables
# exhaustive pattern matching in the driver.
ScraperYield = ParsedData[T] | Request | None

# =============================================================================
# Wait Conditions for Playwright Driver
# =============================================================================
# Defined in the leaf module jkent.common.wait_conditions (which imports
# nothing from jkent) so decorator_metadata can annotate ``await_list`` with
# WaitCondition without a data_types <-> decorator_metadata import cycle.
# Re-exported here because scrapers reach for them via jkent.data_types.
