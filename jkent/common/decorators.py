"""Step and entry decorators for scraper methods.

Introduces a flexible @step decorator that uses argument inspection
to determine what to inject into scraper methods. Instead of having separate
decorators for each content type (lxml, json, text, etc.), a single decorator
inspects the function signature and injects values based on parameter names.

Supported parameter names:

- response: The Response object
- request: The current BaseRequest
- previous_request: The parent request from the chain
- accumulated_data: Data collected across the request chain (from request)
- json_content: Response content parsed as JSON
- lxml_tree: Response content parsed as CheckedHtmlElement
- page: Response content parsed as PageElement (wires up the observer)
- text: Response content as string
- local_filepath: Local file path from ArchiveResponse (None if not archive)

The decorator also handles:

- Attaching priority metadata to functions
- Attaching encoding, xsd, and json_model metadata for drivers to optionally use
- Auto-resolving Callable continuations to string names
- Automatic yielding from wrapped generators

The @entry decorator marks scraper methods as entry points with typed
parameters, replacing the old get_entry()/ScraperParams system.
"""

import inspect
import json
from collections.abc import Callable, Generator
from datetime import date
from functools import wraps
from typing import Any, TypeVar, get_type_hints

from lxml import html as lxml_html
from pydantic import BaseModel

from jkent.common.checked_html import CheckedHtmlElement
from jkent.common.decorator_metadata import (
    DEFAULT_PRIORITY,
    EntryMetadata,
    StepMetadata,
    get_entry_metadata,
    get_step_metadata,
)
from jkent.common.exceptions import (
    ScraperAssumptionException,
)
from jkent.common.lxml_page_element import (
    LxmlPageElement,
)
from jkent.common.selector_observer import (
    SelectorObserver,
)
from jkent.data_types import (
    ArchiveResponse,
    BaseRequest,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
    WaitCondition,
)

T = TypeVar("T")


def _parse_json(response: Response, encoding: str = "utf-8") -> Any:
    """Parse JSON from response content.

    Args:
        response: The HTTP response.
        encoding: Character encoding for decoding undecoded content.

    Returns:
        Parsed JSON data (dict, list, or other JSON types).

    Raises:
        ScraperAssumptionException: If JSON parsing fails.
    """
    try:
        text = response.text or response.content.decode(encoding)
        return json.loads(text)
    except Exception as e:
        raise ScraperAssumptionException(
            f"Failed to parse JSON: {e}",
            request_url=response.url,
            context={"error": str(e)},
        ) from e


def _parse_html(
    response: Response, encoding: str = "utf-8"
) -> CheckedHtmlElement:
    """Parse HTML from response content.

    Passes raw bytes to lxml so it can auto-detect encoding from the HTML
    meta charset tag (e.g., <meta charset="windows-1252">). This handles
    pages that declare non-UTF-8 encodings correctly.

    Args:
        response: The HTTP response.
        encoding: NOT used for parsing — lxml auto-detects from the raw
            bytes (BOM, XML declaration, meta charset). Only recorded in
            the exception context for debugging. The @step encoding
            governs ``text`` injection, not ``lxml_tree``/``page``.

    Returns:
        CheckedHtmlElement parsed from response content.

    Raises:
        ScraperAssumptionException: If HTML parsing fails.
    """
    try:
        # Pass raw bytes to lxml - it will detect encoding from:
        # 1. BOM
        # 2. XML declaration
        # 3. <meta charset="..."> or <meta http-equiv="Content-Type" content="...">
        # 4. Falls back to default if nothing found
        return CheckedHtmlElement(
            lxml_html.fromstring(response.content), response.url
        )
    except Exception as e:
        raise ScraperAssumptionException(
            f"Failed to parse HTML: {e}",
            request_url=response.url,
            context={"encoding": encoding, "error": str(e)},
        ) from e


def _get_text(response: Response, encoding: str = "utf-8") -> str:
    """Get text content from response.

    Args:
        response: The HTTP response.
        encoding: Character encoding for decoding.

    Returns:
        Response text as string.

    Raises:
        ScraperAssumptionException: If the content can't be decoded with the
            given encoding.
    """
    # Falsy check, matching _parse_json: an empty text with non-empty
    # content (e.g. the synthetic Response single_page() builds for
    # bytes input) means "not decoded yet", so decode with the step's
    # encoding rather than injecting "".
    if response.text:
        return response.text
    try:
        return response.content.decode(encoding)
    except UnicodeDecodeError as e:
        # Wrap in the assumption taxonomy like _parse_json/_parse_html, so an
        # unexpected encoding routes to the assumption-violation path instead
        # of the worker's unknown-exception branch.
        raise ScraperAssumptionException(
            f"Failed to decode response content with encoding "
            f"{encoding!r}: {e}",
            request_url=response.url,
            context={"encoding": encoding, "error": str(e)},
        ) from e


def _parse_page_element(
    response: Response, encoding: str = "utf-8"
) -> tuple[Any, Any]:
    """Parse HTML and create PageElement with SelectorObserver.

    Args:
        response: The HTTP response.
        encoding: Fallback encoding if lxml can't detect one.

    Returns:
        Tuple of (PageElement, SelectorObserver) for injection and debugging.

    Raises:
        ScraperAssumptionException: If HTML parsing fails.
    """
    try:
        # Parse HTML using lxml and wrap in CheckedHtmlElement
        checked_element = _parse_html(response, encoding)

        # Create observer to track selector queries. It records through the
        # get_active_observer() contextvar (activated per-resume in the step
        # wrapper), not through the PageElement — see SelectorObserver.
        observer = SelectorObserver()

        page_element = LxmlPageElement(
            element=checked_element, url=response.url
        )

        return page_element, observer
    except ScraperAssumptionException:
        # _parse_html already raised a well-formed assumption error; let it
        # propagate rather than double-wrapping ("Failed to parse HTML for
        # page element: Failed to parse HTML: ...").
        raise
    except Exception as e:
        raise ScraperAssumptionException(
            f"Failed to parse HTML for page element: {e}",
            request_url=response.url,
            context={"encoding": encoding, "error": str(e)},
        ) from e


def _process_yielded_request(yielded: Any) -> Any:
    """Process a yielded BaseRequest to resolve Callable continuations.

    When a decorated function yields a BaseRequest with a Callable continuation,
    this resolves it to the function name and attaches the target step's priority.

    Args:
        yielded: The value yielded by the step.

    Returns:
        The processed yield value.
    """
    if (
        isinstance(yielded, BaseRequest)
        and callable(yielded.continuation)
        and not isinstance(yielded.continuation, str)
    ):
        # Get the target function's step metadata (if decorated with @step)
        target_metadata = get_step_metadata(yielded.continuation)

        # Resolve Callable to function name
        func_name = yielded.continuation.__name__
        # Note: We use object.__setattr__ because dataclasses are frozen
        object.__setattr__(yielded, "continuation", func_name)

        # If the yielded request doesn't have a priority set,
        # inherit from the target step's metadata. Explicit priorities
        # (including an explicit 9) are kept.
        if yielded.priority is None and target_metadata is not None:
            object.__setattr__(yielded, "priority", target_metadata.priority)

    return yielded


def step(
    func: Callable[..., Generator[ScraperYield, Any, None]] | None = None,
    *,
    priority: int = DEFAULT_PRIORITY,
    encoding: str = "utf-8",
    xsd: str | None = None,
    json_model: str | None = None,
    await_list: list[WaitCondition] | None = None,
    auto_await_timeout: int | None = None,
) -> Any:
    """Decorator for scraper step methods with automatic argument injection.

    This decorator inspects the function signature and injects values based on
    parameter names:

    - response: The Response object
    - request: The current BaseRequest
    - previous_request: The parent request from the chain (if available)
    - accumulated_data: Data collected across the request chain (from request)
    - json_content: Response content parsed as JSON
    - lxml_tree: Response content parsed as CheckedHtmlElement
    - page: Response content parsed as PageElement (LxmlPageElement with observer)
    - text: Response content as string
    - local_filepath: Local file path from ArchiveResponse (None otherwise)

    Example::

        @step
        def parse_page(self, lxml_tree: CheckedHtmlElement, response: Response):
            # lxml_tree and response are automatically injected
            cases = lxml_tree.checked_xpath("//div[@class='case']", "cases")
            for case in cases:
                yield ParsedData(...)

        @step(priority=5)
        def parse_api(self, json_content: dict, response: Response):
            # json_content and response are automatically injected
            for item in json_content['items']:
                yield ParsedData(...)

        @step
        def parse_with_callable(self, text: str):
            # Can yield requests with Callable continuations
            yield Request(
                url="/next",
                continuation=self.parse_next_page  # Callable!
            )

        @step(xsd="schemas/court_page.xsd")
        def parse_court_page(self, lxml_tree: CheckedHtmlElement):
            # XSD reference available via get_step_metadata() for drivers
            # to optionally use when evaluating structural errors
            ...

        @step(json_model="api.publications.PublicationsResponse")
        def parse_api_response(self, json_content: dict):
            # JSON model reference available via get_step_metadata() for drivers
            # to optionally use for post-hoc validation
            ...

    Args:
        func: The scraper step method to decorate (when used without parens).
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for ``text`` injection (and JSON
            decoding fallback). HTML parsing (``lxml_tree``/``page``)
            auto-detects encoding from the raw bytes and ignores this.
        xsd: Optional path to XSD schema file. Drivers may use this hint
            when evaluating structural assumption errors.
        json_model: Optional dotted path to Pydantic model (e.g.,
            "api.publications.PublicationsResponse"). Resolved relative to
            scraper package. Drivers may use this for post-hoc validation.
        await_list: Optional list of wait conditions for Playwright driver
            (WaitForSelector, WaitForLoadState, WaitForURL, WaitForTimeout).
            HTTP driver ignores this parameter.
        auto_await_timeout: Optional timeout in milliseconds for autowait retry logic.
            When set, Playwright driver will retry the step if it raises
            HTMLStructuralAssumptionException. HTTP driver ignores this parameter.

    Returns:
        Decorated function with automatic argument injection.

    Raises:
        ScraperAssumptionException: If content parsing fails.
    """

    def decorator(
        fn: Callable[..., Generator[ScraperYield, Any, None]],
    ) -> Callable[..., Generator[ScraperYield, bool | None, None]]:
        # Inspect the function signature to determine what to inject
        sig = inspect.signature(fn)
        param_names = [p.name for p in sig.parameters.values()]

        # Create metadata
        metadata = StepMetadata(
            priority=priority,
            encoding=encoding,
            xsd=xsd,
            json_model=json_model,
            await_list=await_list,
            auto_await_timeout=auto_await_timeout,
        )

        @wraps(fn)
        def wrapper(
            scraper_self: Any,
            response: Response,
            *args: Any,
            **kwargs: Any,
        ) -> Generator[ScraperYield, bool | None, None]:
            # Build kwargs for injection based on parameter names
            injected_kwargs: dict[str, Any] = {}
            observer: SelectorObserver | None = None

            if "response" in param_names:
                injected_kwargs["response"] = response

            if "request" in param_names:
                injected_kwargs["request"] = response.request

            if "previous_request" in param_names:
                # The immediate parent request (None for entry requests)
                injected_kwargs["previous_request"] = (
                    response.request.parent_request
                )

            if "accumulated_data" in param_names:
                injected_kwargs["accumulated_data"] = (
                    response.request.accumulated_data
                )

            # Content transformations (lazy - only parse if requested)
            if "json_content" in param_names:
                injected_kwargs["json_content"] = _parse_json(
                    response, encoding
                )

            if "lxml_tree" in param_names:
                injected_kwargs["lxml_tree"] = _parse_html(response, encoding)

            if "page" in param_names:
                page_element, observer = _parse_page_element(
                    response, encoding
                )
                injected_kwargs["page"] = page_element
                # The observer is per-execution state and the Response is
                # the driver's per-execution handle, so it travels there —
                # never on the shared StepMetadata, where two in-flight
                # executions of the same step would clobber each other.
                response.observer = observer

            if "text" in param_names:
                injected_kwargs["text"] = _get_text(response, encoding)

            if "local_filepath" in param_names:
                if isinstance(response, ArchiveResponse):
                    injected_kwargs["local_filepath"] = response.file_url
                else:
                    injected_kwargs["local_filepath"] = None

            # Call the original function with injected kwargs
            gen = fn(scraper_self, *args, **injected_kwargs, **kwargs)

            # Yield from the generator, processing requests to resolve
            # Callables. When a page was injected, activate this
            # execution's observer around each resume of the scraper's
            # generator: queries record via the get_active_observer()
            # contextvar, and scoping activation per-resume keeps
            # interleaved executions of the same step from recording
            # into each other's observers.
            if observer is None:
                for yielded in gen:
                    yield _process_yielded_request(yielded)
            else:
                while True:
                    with observer:
                        try:
                            yielded = next(gen)
                        except StopIteration:
                            break
                    yield _process_yielded_request(yielded)

        # Attach metadata to the wrapper
        wrapper._step_metadata = metadata  # type: ignore[attr-defined]
        return wrapper

    # Support both @step and @step(priority=5) syntax
    if func is not None:
        return decorator(func)
    return decorator


def is_step(func: Callable[..., Any]) -> bool:
    """Check if a method is a decorated step.

    Args:
        func: A method to check.

    Returns:
        True if the method has step decorator metadata.
    """
    return get_step_metadata(func) is not None


# =============================================================================
# @entry decorator for scraper entry points
# =============================================================================

# Allowed primitive types for @entry parameters
_ENTRY_PRIMITIVE_TYPES = (str, int, date)


def _implements_speculative(cls: type) -> bool:
    """Return True if ``cls`` structurally satisfies the Speculative protocol.

    Used instead of ``issubclass(cls, Speculative)`` because the Protocol
    has a non-method member (``should_advance``), which makes the runtime
    ``issubclass`` check raise at import time. We check attribute presence
    on the class and its annotations — Pydantic fields show up in
    ``model_fields``, methods show up on the class itself.
    """
    for method_name in ("seed_range", "from_int", "max_gap"):
        if not callable(getattr(cls, method_name, None)):
            return False
    # ``should_advance`` may be a Pydantic field (visible on the class via
    # model_fields for BaseModel subclasses), a class attribute, or a
    # property. Any of the three satisfies the Protocol attribute.
    if hasattr(cls, "should_advance"):
        return True
    model_fields = getattr(cls, "model_fields", None)
    return isinstance(model_fields, dict) and "should_advance" in model_fields


def entry(
    return_type: type | Any,
) -> Callable[..., Any]:
    """Decorator for scraper entry point methods with typed parameters.

    Marks a method as an entry point and attaches EntryMetadata describing
    the return type and parameter schema. Does NOT modify the function's
    runtime behavior.

    If a parameter's type implements the ``Speculative`` protocol, the
    entry is automatically detected as speculative. The driver will use
    the protocol methods to seed, track, and extend speculation.

    Parameters can be Pydantic BaseModel subclasses or primitives
    (str, int, date). Tuples are not supported.

    Example::

        @entry(Docket)
        def search_by_number(self, docket_number: str) -> Generator[Request, None, None]:
            ...

        @entry(CaseData)
        def fetch_case(self, case_id: DocketId) -> Request:
            # DocketId implements Speculative — auto-detected
            ...

    Args:
        return_type: The data type this entry produces.

    Returns:
        Decorator that attaches EntryMetadata to the function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Inspect function signature to extract parameter types
        # Skip 'self' for instance methods
        # Use get_type_hints with the function's module globals for proper
        # resolution when `from __future__ import annotations` is used
        hints: dict[str, Any] = {}
        # Preserve the first get_type_hints failure so the per-parameter
        # "unresolvable type annotation" error below can chain from the real
        # import/forward-ref cause instead of swallowing it.
        hint_resolution_error: Exception | None = None
        try:
            module = inspect.getmodule(fn)
            globalns = getattr(module, "__dict__", None) if module else None
            hints = get_type_hints(fn, globalns=globalns)
        except Exception as e:
            hint_resolution_error = e
            # Fallback: try raw annotations (may be strings with PEP 563)
            try:
                hints = get_type_hints(fn)
            except Exception as e2:
                hint_resolution_error = e2
                hints = {}

        sig = inspect.signature(fn)
        param_types: dict[str, type] = {}
        speculative_param: str | None = None

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            # Get the type from hints, fallback to annotation
            param_type = hints.get(param_name)
            if param_type is None:
                # Try raw annotation (might be a string)
                ann = param.annotation
                if ann is inspect.Parameter.empty:
                    raise TypeError(
                        f"Entry function '{fn.__name__}' parameter "
                        f"'{param_name}' must have a type annotation"
                    )
                # If annotation is a string, try to resolve it
                if isinstance(ann, str):
                    module = inspect.getmodule(fn)
                    globalns = (
                        getattr(module, "__dict__", {}) if module else {}
                    )
                    try:
                        param_type = eval(ann, globalns)
                    except Exception:
                        raise TypeError(
                            f"Entry function '{fn.__name__}' parameter "
                            f"'{param_name}' has unresolvable type "
                            f"annotation '{ann}'"
                        ) from hint_resolution_error
                else:
                    param_type = ann

            # Validate the parameter type
            if isinstance(param_type, type) and issubclass(
                param_type, BaseModel
            ):
                # Check if this BaseModel implements Speculative. Structural
                # check (not ``issubclass(param_type, Speculative)``) because
                # the Protocol now has a non-method member (``should_advance``),
                # which ``issubclass`` refuses for @runtime_checkable Protocols.
                if _implements_speculative(param_type):
                    if speculative_param is not None:
                        raise TypeError(
                            f"Entry function '{fn.__name__}' has multiple "
                            f"Speculative parameters: '{speculative_param}' "
                            f"and '{param_name}'. Only one is allowed."
                        )
                    speculative_param = param_name
            elif param_type in _ENTRY_PRIMITIVE_TYPES:
                pass  # Primitive is fine
            elif param_type is tuple or (
                hasattr(param_type, "__origin__")
                and getattr(param_type, "__origin__", None) is tuple
            ):
                raise TypeError(
                    f"Entry function '{fn.__name__}' parameter "
                    f"'{param_name}' uses tuple type, which is not supported. "
                    f"Use a Pydantic BaseModel instead."
                )
            else:
                raise TypeError(
                    f"Entry function '{fn.__name__}' parameter "
                    f"'{param_name}' has unsupported type {param_type}. "
                    f"Use a Pydantic BaseModel subclass or one of: "
                    f"str, int, date"
                )

            param_types[param_name] = param_type

        metadata = EntryMetadata(
            return_type=return_type,
            param_types=param_types,
            func_name=fn.__name__,
            speculative_param=speculative_param,
        )

        fn._entry_metadata = metadata  # type: ignore[attr-defined]
        return fn

    return decorator


def is_entry(func: Callable[..., Any]) -> bool:
    """Check if a method is a decorated entry point.

    Args:
        func: A method to check.

    Returns:
        True if the method has entry decorator metadata.
    """
    return get_entry_metadata(func) is not None


def single_page(
    scraper_cls: type,
    step_name: str,
    *,
    params: Any | None = None,
) -> Callable[..., list[Any]]:
    """Create a function that runs a single @step method on provided content.

    Useful for unit-testing scraper parsing logic without a driver or HTTP
    server.  The returned callable constructs a synthetic Response, feeds
    it through the @step wrapper (so all argument injection works normally),
    and returns the unwrapped ParsedData items.

    Only ``ParsedData`` yields are returned: any ``Request`` the step yields
    (e.g. on a list page that follows links) is dropped, so pointing this at
    a navigation step returns ``[]``. It is built for leaf parse steps that
    yield data.

    One scraper instance is constructed when ``single_page()`` is called and
    reused across every ``run()`` invocation, so the step must be stateless
    (carry per-page state in ``accumulated_data``, not on ``self``).

    Args:
        scraper_cls: A BaseScraper subclass.
        step_name: Name of a @step-decorated method on the scraper.
        params: Optional params passed to the scraper constructor.

    Returns:
        A callable ``run(content, *, url, accumulated_data, status_code,
        headers)`` that returns ``list[T]`` of unwrapped ParsedData items
        (non-ParsedData yields are dropped).

    Example::

        from my_scraper import MyScraper

        run = single_page(MyScraper, "parse_results")
        results = run("<html>...</html>", accumulated_data={"page": 1})
        assert len(results) == 5
    """
    scraper = scraper_cls(params=params)
    method = scraper.get_continuation(step_name)

    def run(
        content: str | bytes,
        *,
        url: str = "https://test.example.com",
        accumulated_data: dict[str, Any] | None = None,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
            text = content
        else:
            content_bytes = content
            # Empty text means "not decoded yet": _get_text falls back
            # to decoding content with the step's declared encoding.
            text = ""

        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=url,
            ),
            continuation=step_name,
            accumulated_data=accumulated_data or {},
        )

        response = Response(
            status_code=status_code,
            headers=headers or {},
            content=content_bytes,
            text=text,
            url=url,
            request=request,
        )

        results: list[Any] = []
        for item in method(response):
            if isinstance(item, ParsedData):
                results.append(item.unwrap())
        return results

    return run
