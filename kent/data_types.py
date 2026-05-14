"""Data types for the scraper-driver architecture.

This module defines the core data types used for communication between
scrapers and drivers. These types are designed to be:

1. Exhaustive - Using Python 3.10's match statement to ensure all cases are handled
2. Serializable - Continuations are strings, not function references
3. Immutable - Dataclasses with frozen=True where appropriate

Step 1 introduces ParsedData.
Step 2 adds Request and Response.
Step 3 introduces BaseRequest and current_location tracking.
Step 4 adds archive support and ArchiveResponse for file downloads.
Step 5 adds accumulated_data to BaseRequest with deep copy semantics.
"""

from __future__ import annotations

import hashlib
import json
import ssl
from collections.abc import Callable, Generator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from http.cookiejar import CookieJar
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    ClassVar,
    Final,
    Generic,
    TypeVar,
    cast,
)
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse

from pyrate_limiter import Rate  # noqa: F401 (used in ClassVar annotation)

from kent.common.speculative import Speculative

if TYPE_CHECKING:
    pass

# =============================================================================
# Step 1: ParsedData
# =============================================================================

T = TypeVar("T")
ScraperReturnType = TypeVar("ScraperReturnType")
ScraperParamType = TypeVar("ScraperParamType")


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
    ``kent run`` reads them to auto-select the driver and browser profile.

    Values:
        JS_EVAL: Requires JavaScript evaluation (auto-selects Playwright).
        FF_ALIKE: Requires a Firefox-like browser profile.
        CHROME_ALIKE: Requires a Chrome-like browser profile.
        HCAP_HANDLER: Requires hCaptcha interstitial handling (auto-selects Playwright).
        RCAP_HANDLER: Requires reCAPTCHA interstitial handling (auto-selects Playwright).
        H11_HEADER_FIXES: Loosen h11 response-header validation.
        FOLLOW_REDIRECTS: Have httpx follow 3xx redirects automatically.

    FF_ALIKE and CHROME_ALIKE are mutually exclusive.
    """

    JS_EVAL = "js_eval"
    FF_ALIKE = "ff_alike"
    CHROME_ALIKE = "chrome_alike"
    HCAP_HANDLER = "hcap_handler"
    RCAP_HANDLER = "rcap_handler"
    H11_HEADER_FIXES = "h11_header_fixes"
    FOLLOW_REDIRECTS = "follow_redirects"
    HCAPTCHA_SOLVER = "hcaptcha_solver"
    IMAGE_CAPTCHA_SOLVER = "image_captcha_solver"


@dataclass(frozen=True)
class StepInfo:
    """Metadata about a scraper step method.

    Used by LocalDevDriver web interface to display available steps,
    their priorities, and to populate controls for pause_step/resume_step.

    Attributes:
        name: The method name (continuation string).
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for text/HTML decoding.
    """

    name: str
    priority: int
    encoding: str


@dataclass(frozen=True)
class EntryInfo:
    """Metadata about a scraper entry point method.

    Used by list_entries() to expose entry point metadata including
    return type, parameter types, and speculative param info.

    Attributes:
        name: The method name.
        return_type: The data type this entry produces.
        param_types: Mapping of parameter name to type.
        speculative_param: Name of the parameter implementing the Speculative
            protocol, or None if this entry is not speculative.
    """

    name: str
    return_type: type
    param_types: dict[str, type]
    speculative_param: str | None = None

    @property
    def speculative(self) -> bool:
        """Whether this is a speculative entry."""
        return self.speculative_param is not None


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

    # Driver requirements — capabilities the scraper needs from its driver.
    # kent run reads these to auto-select driver and browser profile.
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
    # DEFAULT_* sets are framework defaults; subclasses should leave them
    # alone. Scrapers reclassify per-site codes by shadowing any of the
    # three override sets below. The ``is_transient_error`` /
    # ``is_persistent_error`` classmethods (see further down) combine
    # defaults + overrides and expose the result to the request manager.

    DEFAULT_SUCCESSFUL_HTTP_CODES: Final[frozenset[int]] = frozenset(
        {200, 201, 202, 203, 204, 205, 206, 207, 208, 226, 304}
    )
    DEFAULT_TRANSIENT_HTTP_ERROR_CODES: Final[frozenset[int]] = frozenset(
        {408, 425, 429, 502, 503, 504}
    )
    DEFAULT_PERSISTENT_HTTP_ERROR_CODES: Final[frozenset[int]] = frozenset(
        # All standard 4xx minus the transient 408/425/429.
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
        }
        # 5xx codes that aren't gateway-style.
        | {500, 501, 505, 506, 507, 508, 510, 511}
    )

    # Subclasses shadow these to reclassify specific codes. A code appearing
    # in an override wins over a default in the opposite category (see the
    # active_*_http_* classmethods below).
    SUCCESSFUL_HTTP_CODES: ClassVar[frozenset[int]] = frozenset()
    TRANSIENT_HTTP_ERROR_CODES: ClassVar[frozenset[int]] = frozenset()
    PERSISTENT_HTTP_ERROR_CODES: ClassVar[frozenset[int]] = frozenset()

    def __init__(self, params: Any | None = None) -> None:
        """Initialize the scraper with optional search parameters.

        Args:
            params: Parameters for the scraper. During migration to @entry,
                this accepts the legacy ScraperParams or None.
        """
        self._params = params

    def get_params(self) -> Any | None:
        """Return the params instance for this scraper."""
        return self._params

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
    def params(cls) -> Any:
        """Build a params object for configuring scraper filters.

        Returns a stub params object for backward compat during
        migration to @entry decorators.
        """

        class _StubParams:
            def get_enabled_models(self) -> list[str]:
                return []

        return _StubParams()

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
    def active_transient_http_error_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as transient (retryable).

        Combines the framework default with the scraper-level override,
        and removes any code that the scraper has reassigned to
        successful or persistent.
        """
        return (
            (
                cls.DEFAULT_TRANSIENT_HTTP_ERROR_CODES
                | cls.TRANSIENT_HTTP_ERROR_CODES
            )
            - cls.SUCCESSFUL_HTTP_CODES
            - cls.PERSISTENT_HTTP_ERROR_CODES
        )

    @classmethod
    def active_persistent_http_error_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as persistent (fail-fast, no retry)."""
        return (
            (
                cls.DEFAULT_PERSISTENT_HTTP_ERROR_CODES
                | cls.PERSISTENT_HTTP_ERROR_CODES
            )
            - cls.SUCCESSFUL_HTTP_CODES
            - cls.TRANSIENT_HTTP_ERROR_CODES
        )

    @classmethod
    def active_successful_http_codes(cls) -> frozenset[int]:
        """Codes the scraper treats as successful (pass through as Response)."""
        return (
            (cls.DEFAULT_SUCCESSFUL_HTTP_CODES | cls.SUCCESSFUL_HTTP_CODES)
            - cls.TRANSIENT_HTTP_ERROR_CODES
            - cls.PERSISTENT_HTTP_ERROR_CODES
        )

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

        Same semantics for ``headers`` / ``content`` as
        :meth:`is_transient_error`.
        """
        return status_code in cls.active_persistent_http_error_codes()

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
            AttributeError: If the continuation method doesn't exist.
        """
        method = getattr(self, name)
        return cast(
            Callable[
                [Response],
                Generator[ScraperYield[ScraperReturnType], bool | None, None],
            ],
            method,
        )

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
        from kent.common.decorators import (
            get_step_metadata,
        )

        steps = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            try:
                method = getattr(cls, name)
                metadata = get_step_metadata(method)
                if metadata is not None:
                    steps.append(
                        StepInfo(
                            name=name,
                            priority=metadata.priority,
                            encoding=metadata.encoding,
                        )
                    )
            except Exception:
                continue
        return steps

    @classmethod
    def list_speculative_entries(cls) -> list[EntryInfo]:
        """List all speculative entry point methods defined on this scraper.

        Returns:
            List of EntryInfo objects for speculative entries only.
        """
        return [e for e in cls.list_entries() if e.speculative]

    @classmethod
    def list_entries(cls) -> list[EntryInfo]:
        """List all entry point methods defined on this scraper.

        Introspects the class to find all methods decorated with @entry
        and returns their metadata.

        Returns:
            List of EntryInfo objects for each decorated entry method.
        """
        from kent.common.decorators import (
            get_entry_metadata,
        )

        entries = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            try:
                method = getattr(cls, name)
                metadata = get_entry_metadata(method)
                if metadata is not None:
                    entries.append(
                        EntryInfo(
                            name=metadata.func_name,
                            return_type=metadata.return_type,
                            param_types=metadata.param_types,
                            speculative_param=metadata.speculative_param,
                        )
                    )
            except Exception:
                continue
        return entries

    def _list_entry_info(
        self,
    ) -> list[tuple[Any, Any]]:
        """List entry methods with their metadata for dispatch.

        Returns:
            List of (bound_method, EntryMetadata) tuples.
        """
        from kent.common.decorators import (
            get_entry_metadata,
        )

        entries = []
        for name in dir(self):
            if name.startswith("_"):
                continue
            try:
                method = getattr(self, name)
                metadata = get_entry_metadata(method)
                if metadata is not None:
                    entries.append((method, metadata))
            except Exception:
                continue
        return entries

    def initial_seed(
        self, params: list[dict[str, dict[str, Any]]]
    ) -> Generator[Request, None, None]:
        """Dispatch parameter list to entry functions and yield combined requests.

        Takes a JSON-serializable list of parameter invocations and dispatches
        them to the appropriate @entry functions.

        For non-speculative entries, params are direct function arguments and
        the method yields Requests.

        For speculative entries (parameter implementing Speculative protocol),
        the validated model instance is stored in ``_speculation_templates``
        for the driver to consume during speculation seeding. No requests are
        yielded for speculative entries here.

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
                method, meta = entry_map[func_name]
                validated_kwargs = meta.validate_params(kwargs_dict)

                if meta.speculative:
                    # Store the validated Speculative model instance
                    # as a template for the driver
                    if not hasattr(self, "_speculation_templates"):
                        self._speculation_templates: dict[
                            str, list[Speculative]  # type: ignore[type-arg]
                        ] = {}
                    if func_name not in self._speculation_templates:
                        self._speculation_templates[func_name] = []
                    assert meta.speculative_param is not None
                    template = validated_kwargs[meta.speculative_param]
                    self._speculation_templates[func_name].append(template)
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
        from pydantic import BaseModel as PydanticBaseModel

        entries: dict[str, Any] = {}
        all_defs: dict[str, Any] = {}

        for entry_info in cls.list_entries():
            # Build parameter schema
            properties: dict[str, Any] = {}
            required: list[str] = []

            for param_name, param_type in entry_info.param_types.items():
                required.append(param_name)

                if isinstance(param_type, type) and issubclass(
                    param_type, PydanticBaseModel
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

            entry_schema: dict[str, Any] = {
                "returns": entry_info.return_type.__name__,
                "speculative": entry_info.speculative,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }

            entries[entry_info.name] = entry_schema

        result: dict[str, Any] = {
            "scraper": cls.__name__,
            "entries": entries,
        }
        if all_defs:
            result["$defs"] = all_defs

        return result

    def fails_successfully(self, response: Response) -> bool:
        """Detect hidden error states in successful HTTP responses.

        Some websites return HTTP 200 status codes but embed error states
        in the page content or headers (e.g., "No results found" pages,
        session timeout pages, soft 404s). This method allows scrapers to
        detect these hidden failures.

        This is primarily used for speculation handling. When a
        speculative request gets a 2xx response, the driver calls this
        method to check if the response actually represents a failure.
        If this returns False, the driver sets status_code=555 before
        calling the speculation callback.

        Args:
            response: The Response object to check for hidden errors.

        Returns:
            True if the response is genuinely successful (default behavior).
            False if the response contains a hidden error pattern.

        Example:
            Override this method to detect site-specific error patterns::

                def fails_successfully(self, response: Response) -> bool:
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


@dataclass(frozen=True)
class EstimateData:
    """Estimate of downstream ParsedData items from a step.

    Yielded by steps that can predict how many items of certain types
    will be produced by follow-on requests (e.g., a search results page
    showing a total count). Used as a post-hoc integrity check in the
    LocalDevDriver debugger.

    Attributes:
        expected_types: Tuple of data model classes expected downstream.
        min_count: Minimum number of items expected.
        max_count: Maximum number of items expected, or None for "at least min_count".

    Example::

        @step
        def parse_search_results(self, response):
            total = int(tree.xpath("//span[@class='count']/text()")[0])
            yield EstimateData((CaseData,), min_count=total, max_count=total)

            for link in result_links:
                yield Request(...)
    """

    expected_types: tuple[type, ...]
    min_count: int
    max_count: int | None = None


# =============================================================================
# Step 2: Request and Response
# =============================================================================
# Step 3: BaseRequest, Request, and current_location tracking
# Step 4: ArchiveResponse for file downloads
# Step 5: accumulated_data with deep copy semantics


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
CookiesType = dict[str, str] | CookieJar | None
FileTuple = (
    tuple[str, BinaryIO]
    | tuple[str, BinaryIO, str]
    | tuple[str, BinaryIO, str, dict[str, str]]
)
FilesType = dict[str, BinaryIO | FileTuple] | None
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
    :param cookies: (optional) Dict or CookieJar object to send with the request.
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


def _generate_deduplication_key(request_params: HTTPRequestParams) -> str:
    """Generate a deduplication key from HTTPRequestParams.

    Step 16: Default deduplication key is a SHA256 hash of:
    - Full URL with parameters
    - Request data (sorted if dict/list of tuples)

    Args:
        request_params: The HTTP request parameters.

    Returns:
        A SHA256 hex digest string for deduplication.
    """
    # Start with the full URL
    url_str = request_params.url

    # Add query parameters if present
    if request_params.params:
        # Sort params for consistent hashing
        if isinstance(request_params.params, dict):
            sorted_params = sorted(request_params.params.items())
            params_str = str(sorted_params)
        elif isinstance(request_params.params, list | tuple):
            # List of tuples
            sorted_params = sorted(request_params.params)  # type: ignore
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
            # Assume list of tuples, sort by first element
            sorted_data = sorted(
                request_params.data,
                key=lambda x: x[0] if isinstance(x, tuple) else x,
            )
            data_str = str(sorted_data)
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


class SkipDeduplicationCheck:
    """Skip deduplication checks."""

    pass


@dataclass(frozen=True)
class BaseRequest:
    """Base class for all request types.

    Provides common functionality for URL resolution and HTTP parameters.
    Each request tracks its current_location and request ancestry.

    Attributes:
        request: HTTP request parameters (URL, method, headers, etc.).
        continuation: The method name to call with the Response, or a Callable.
                     When a Callable is provided, the @step decorator will automatically
                     resolve it to the function's name.
        current_location: The URL context for resolving relative URLs.
        previous_requests: Chain of requests that led to this one.
        accumulated_data: Data collected across the request chain.
        priority: Priority for request queue ordering (lower = higher priority).
        deduplication_key: Key for deduplication (defaults to hash of URL and data).
        permanent: Persistent data (cookies, headers) that flows through the request chain.
        is_speculative: Whether this request is speculative (probing for content existence).
        speculation_id: Tuple of (function_name, param_index, integer_id) identifying
                       which speculative template generated this request. None for
                       non-speculative requests.
        via: Optional description of how the request was produced (ViaLink, ViaFormSubmit).
             Enables the Playwright driver to replay the browser action. HTTP driver ignores.
        bypass_rate_limit: If True, skip the rate limiter for this request.
             Useful for time-sensitive requests (e.g., file downloads) where
             stale server-side state expires quickly and delays cause failures.
    """

    request: HTTPRequestParams
    continuation: str | Callable[..., Any]
    current_location: str = ""
    previous_requests: list[BaseRequest] = field(default_factory=list)
    accumulated_data: dict[str, Any] = field(default_factory=dict)
    priority: int = 9
    deduplication_key: str | None | SkipDeduplicationCheck = None
    permanent: dict[str, Any] = field(default_factory=dict)
    is_speculative: bool = False
    speculation_id: tuple[str, int, int] | None = None
    via: Any = None  # ViaLink | ViaFormSubmit | None - using Any to avoid circular import
    bypass_rate_limit: bool = False

    def __post_init__(self) -> None:
        """Deep copy accumulated_data and permanent to prevent unintended sharing.

        Step 16: Also generates default deduplication_key if not provided.
        Step 18: Also deep copies permanent dict and merges permanent headers/cookies
        into the HTTPRequestParams.

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
        # Since the dataclass is frozen, we need to use object.__setattr__
        object.__setattr__(
            self, "accumulated_data", deepcopy(self.accumulated_data)
        )
        object.__setattr__(self, "permanent", deepcopy(self.permanent))

        # Step 18: Merge permanent headers and cookies into HTTPRequestParams
        if self.permanent:
            new_request = self._merge_permanent_into_request()
            object.__setattr__(self, "request", new_request)

        # Step 16: Generate default deduplication key if not provided
        if self.deduplication_key is None:
            object.__setattr__(
                self,
                "deduplication_key",
                _generate_deduplication_key(self.request),
            )

    def _merge_permanent_into_request(self) -> HTTPRequestParams:
        """Merge permanent headers and cookies into the HTTPRequestParams.

        Returns:
            A new HTTPRequestParams with permanent data merged in.
        """
        req = self.request
        merged_headers: dict[str, str] | None = None
        # Merge headers
        if "headers" in self.permanent:
            merged_headers = dict(req.headers) if req.headers else {}
            merged_headers.update(self.permanent["headers"])
        else:
            merged_headers = req.headers

        # Merge cookies (only if both are dicts)
        if "cookies" in self.permanent:
            if req.cookies is None:
                merged_cookies: CookiesType = dict(self.permanent["cookies"])
            elif isinstance(req.cookies, dict):
                merged_cookies = dict(req.cookies)
                merged_cookies.update(self.permanent["cookies"])
            else:
                # CookieJar - can't merge, keep original
                merged_cookies = req.cookies
        else:
            merged_cookies = req.cookies

        return HTTPRequestParams(
            method=req.method,
            url=req.url,
            params=req.params,
            data=req.data,
            json=req.json,
            headers=merged_headers,
            cookies=merged_cookies,
            files=req.files,
            auth=req.auth,
            timeout=req.timeout,
            allow_redirects=req.allow_redirects,
            proxies=req.proxies,
            verify=req.verify,
            stream=req.stream,
            cert=req.cert,
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
        # Normalize URL encoding
        parsed = urlparse(self.request.url)

        # Decode then encode to normalize (prevents double-encoding)
        # safe='/:?&=' preserves URL structure while encoding special chars
        decoded_path = unquote(parsed.path)
        encoded_path = quote(decoded_path, safe="/")

        decoded_query = unquote(parsed.query)
        encoded_query = quote(decoded_query, safe="=&")

        reencoded_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                encoded_path,
                parsed.params,
                encoded_query,
                parsed.fragment,
            )
        )
        return urljoin(current_location, reencoded_url)

    def resolve_from(self, context: Response | Request) -> BaseRequest:
        """Create a new request with URL resolved from a Response or Request.

        Implemented by the Request subclass.

        Args:
            context: Response from a previous request or the originating Request.

        Returns:
            A new request with resolved URL and updated context.

        Raises:
            NotImplementedError: If called on BaseRequest directly.
        """
        raise NotImplementedError(
            "resolve_from must be implemented by subclasses"
        )

    def resolve_request_from(self, context: Response | BaseRequest):
        match context:
            case Response():
                # Response from a Request - use its URL
                resolved_location = context.url
                parent_request = context.request
            case BaseRequest():
                # Request - use its current_location
                resolved_location = context.current_location
                parent_request = context

        return [
            HTTPRequestParams(
                url=self.resolve_url(resolved_location),
                method=self.request.method,
                params=self.request.params,
                data=self.request.data,
                json=self.request.json,
                headers=self.request.headers,
                cookies=self.request.cookies,
                files=self.request.files,
                auth=self.request.auth,
                timeout=self.request.timeout,
                allow_redirects=self.request.allow_redirects,
                proxies=self.request.proxies,
                verify=self.request.verify,
                stream=self.request.stream,
                cert=self.request.cert,
            ),
            resolved_location,
            parent_request,
        ]


@dataclass(frozen=True)
class Request(BaseRequest):
    """Unified request type for all scraper navigation patterns.

    Controls behavior via boolean flags:
    - Default (nonnavigating=False, archive=False): Navigating request.
      Updates current_location when resolved. Supports speculation.
    - nonnavigating=True: Fetches data without changing current_location.
      Useful for API calls that provide supplementary data.
    - archive=True: Downloads and archives files. Preserves current_location.
      The driver returns an ArchiveResponse with a file_url field.

    Attributes:
        nonnavigating: If True, does not update current_location.
        archive: If True, downloads and archives the file.
        expected_type: Optional file type hint for archive requests ("pdf", "audio", etc.).
    """

    nonnavigating: bool = False
    archive: bool = False
    expected_type: str | None = None
    archive_hash_header: str | None = None

    def __post_init__(self) -> None:
        # If archive=True and priority was left at the BaseRequest default (9),
        # override to 1 to match the higher default priority for file downloads.
        if self.archive and self.priority == 9:
            object.__setattr__(self, "priority", 1)
        super().__post_init__()

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
        # Step 18: Merge permanent data - parent's permanent + this request's permanent
        merged_permanent = {**parent.permanent, **self.permanent}
        return Request(
            request=request,
            continuation=self.continuation,
            current_location=location,
            previous_requests=parent.previous_requests + [parent],
            accumulated_data=self.accumulated_data,
            priority=self.priority,
            deduplication_key=self.deduplication_key,
            permanent=merged_permanent,
            is_speculative=self.is_speculative,
            speculation_id=self.speculation_id,
            via=self.via,
            nonnavigating=self.nonnavigating,
            archive=self.archive,
            expected_type=self.expected_type,
            archive_hash_header=self.archive_hash_header,
            bypass_rate_limit=self.bypass_rate_limit,
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
        return Request(
            request=self.request,
            continuation=self.continuation,
            current_location=self.current_location,
            previous_requests=self.previous_requests,
            accumulated_data=self.accumulated_data,
            priority=self.priority,
            deduplication_key=self.deduplication_key,
            permanent=self.permanent,
            is_speculative=True,
            speculation_id=(func_name, param_index, spec_id),
            via=self.via,
            nonnavigating=self.nonnavigating,
            archive=self.archive,
            expected_type=self.expected_type,
            archive_hash_header=self.archive_hash_header,
            bypass_rate_limit=self.bypass_rate_limit,
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
        request: The BaseRequest that triggered this response.
    """

    status_code: int
    headers: dict[str, str]
    content: bytes
    text: str
    url: str
    request: BaseRequest


@dataclass
class ArchiveResponse(Response):
    """HTTP response for an archived file.

    Extends Response with a file_url field that contains the local storage
    path where the file was saved. This allows scrapers to include the
    file location in their ParsedData output.

    Attributes:
        file_url: Local file system path where the downloaded file was stored.
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
# Request preparation wrappers
# =============================================================================


class HTTPRequestPrep:
    """Wraps a Request with an httpx-driven preprocessor that runs at yield time.

    The prep callable receives ``(response, request, **kwargs)`` and returns
    the modified Request that actually enters the queue.
    """

    __slots__ = ("request", "prep_method", "kwargs")

    def __init__(
        self,
        request: BaseRequest,
        *,
        prep_method: str,
        **kwargs: Any,
    ) -> None:
        self.request = request
        self.prep_method = prep_method
        self.kwargs = kwargs


class JSRequestPrep:
    """Wraps a Request with a Playwright-driven preprocessor.

    The prep callable receives ``(response, request, page, **kwargs)`` —
    ``page`` is the live parent page so the prep can call ``page.evaluate``
    against an already-loaded DOM.
    """

    __slots__ = ("request", "prep_method", "kwargs")

    def __init__(
        self,
        request: BaseRequest,
        *,
        prep_method: str,
        **kwargs: Any,
    ) -> None:
        self.request = request
        self.prep_method = prep_method
        self.kwargs = kwargs


# =============================================================================
# Type Alias for Scraper Yields
# =============================================================================

# A scraper can yield ParsedData, EstimateData, Request, JSRequestPrep,
# HTTPRequestPrep, or None. This type alias enables exhaustive pattern
# matching in the driver.
ScraperYield = (
    ParsedData[T]
    | EstimateData
    | Request
    | JSRequestPrep
    | HTTPRequestPrep
    | None
)

# Type alias for scraper generator - what continuation methods return
# The second type parameter (bool | None) is the SendType - values sent back
# to the generator via .send(). Currently unused but kept for future compatibility.
ScraperGenerator = Generator[ScraperYield[T], bool | None, None]


# =============================================================================
# Wait Conditions for Playwright Driver
# =============================================================================


@dataclass(frozen=True)
class WaitForSelector:
    """Wait for a selector to appear in the DOM.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for an element before taking a DOM snapshot.

    Attributes:
        selector: CSS or XPath selector to wait for.
        state: Optional state to wait for ('attached', 'detached', 'visible', 'hidden').
               Defaults to 'visible'.
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    selector: str
    state: str = "visible"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForLoadState:
    """Wait for a specific load state.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for a load state before taking a DOM snapshot.

    Attributes:
        state: Load state to wait for ('load', 'domcontentloaded', 'networkidle').
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    state: str = "load"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForURL:
    """Wait for the URL to match a pattern.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for URL navigation before taking a DOM snapshot.

    Attributes:
        url: URL string or pattern to wait for. Can be a string, regex pattern, or callable.
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    url: str
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForTimeout:
    """Wait for a specific amount of time.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait before taking a DOM snapshot.

    Attributes:
        timeout: Time to wait in milliseconds.
    """

    timeout: int
