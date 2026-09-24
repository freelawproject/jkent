"""The scraper half of the contract: :class:`BaseScraper` and what it yields.

A scraper is parsing plus navigation intent: its steps take a
:class:`~jkent.common.response.Response` and yield :class:`ParsedData` or
further :class:`~jkent.common.request.Request` objects (the
:data:`ScraperYield` union). The driver-facing enums a scraper declares on
its class body — :class:`DriverRequirement`, :class:`ScraperStatus`,
:class:`HTTPCodeType` — live here with it.
"""

from __future__ import annotations

import ssl
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, ClassVar, Final, Generic, TypeVar, get_origin

from pydantic import BaseModel as PydanticBaseModel
from pydantic import TypeAdapter
from pyrate_limiter import Rate

from jkent.common.decorator_metadata import (
    EntryMetadata,
    StepMetadata,
    get_entry_metadata,
    get_step_metadata,
)
from jkent.common.exceptions import ScraperConfigError
from jkent.common.rate_limits import (
    RESERVED_RATE_LIMIT_NAMES,
    validate_named_rate_limits,
    validate_rate_limits,
)
from jkent.common.request import Request
from jkent.common.response import Response
from jkent.common.speculative import Speculative

T = TypeVar("T")
ScraperReturnType = TypeVar("ScraperReturnType")
M = TypeVar("M")


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
        name: The method name (step string).
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
        named_rate_limits: Extra rate-limit lanes by name, for requests that
            should be paced separately from the default (``Request(rate_limit=
            name)`` / ``@step(rate_limit=name)``). See
            :mod:`jkent.common.rate_limits`.
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

    # Additional rate-limit lanes, by name. The framework supplies "default"
    # (= rate_limits) and "none" (unlimited); anything here is stored in the
    # run database as 2 + its position, so APPEND lanes — never insert or
    # reorder — or a resumed run gates its pending rows at the wrong rate.
    # Validated in __init_subclass__.
    named_rate_limits: ClassVar[Mapping[str, list[Rate]]] = {}

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Only when the subclass declares (or re-declares) them: a scraper
        # that inherits its parent's rates was already checked with it.
        if "rate_limits" in cls.__dict__:
            validate_rate_limits(cls.rate_limits, owner=cls.__name__)
        if "named_rate_limits" in cls.__dict__:
            validate_named_rate_limits(
                cls.named_rate_limits, owner=cls.__name__
            )
        cls._check_step_lanes()

    @classmethod
    def _check_step_lanes(cls) -> None:
        """Reject a ``@step(rate_limit=...)`` naming a lane cls lacks.

        Otherwise the failure lands at the first enqueue of a request to
        that step, possibly hours into a run. Walks the MRO's ``__dict__``s
        rather than ``dir(cls)`` so no descriptor runs at class definition,
        and re-checks inherited steps: a subclass may redeclare
        ``named_rate_limits`` without a lane its parent's steps use.
        """
        lanes = set(RESERVED_RATE_LIMIT_NAMES) | set(cls.named_rate_limits)
        seen: set[str] = set()
        for klass in cls.__mro__:
            for name, attr in vars(klass).items():
                if name in seen:
                    continue
                seen.add(name)
                metadata = getattr(attr, "_step_metadata", None)
                if not isinstance(metadata, StepMetadata):
                    continue
                lane = metadata.rate_limit
                if lane is not None and lane not in lanes:
                    raise ScraperConfigError(
                        f"{cls.__name__}.{name}: @step(rate_limit={lane!r}) "
                        f"names a lane this scraper does not declare; its "
                        f"lanes are {sorted(lanes)}"
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

    def get_step(
        self, name: str
    ) -> Callable[
        [Response],
        Generator[ScraperYield[ScraperReturnType], bool | None, None],
    ]:
        """Resolve a step name to the actual method.

        This method looks up a step by name and returns the
        bound method. It provides a single point for step
        resolution, making it easy to add validation or caching later.

        Args:
            name: The name of the step method.

        Returns:
            The bound method that can be called with a Response.

        Raises:
            ScraperConfigError: If the step method doesn't exist.
        """
        try:
            method = getattr(self, name)
        except AttributeError:
            raise ScraperConfigError("Nonexistent step referenced") from None
        return method

    @staticmethod
    def _iter_decorated(
        target: object,
        get_metadata: Callable[[Callable[..., Any]], M | None],
    ) -> Generator[tuple[str, Callable[..., Any], M], None, None]:
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
    ) -> list[tuple[Callable[..., Any], EntryMetadata]]:
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
                method, meta = entry_map[func_name]
                validated_kwargs = meta.validate_params(kwargs_dict)

                if meta.speculative:
                    # Store the validated Speculative model instance as a
                    # template for the driver, alongside the raw seed value
                    # it was validated from (persisted with the speculation
                    # state so hosts can map state rows back to their seed
                    if not hasattr(self, "_speculation_templates"):
                        self._speculation_templates: dict[
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
                    model_schema = param_type.model_json_schema()
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


# A scraper can yield ParsedData, Request, or None. This type alias enables
# exhaustive pattern matching in the driver.
ScraperYield = ParsedData[T] | Request | None
