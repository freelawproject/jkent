"""Step and entry decorators for scraper methods.

Introduces a flexible @step decorator that uses argument inspection
to determine what to inject into scraper methods. Instead of having separate
decorators for each content type (lxml, json, text, etc.), a single decorator
inspects the function signature and injects values based on parameter names.

Supported parameter names live in one registry (``INJECTORS``), which also
generates the documented list on ``@step`` — see :func:`register_injector`
for adding one.

The decorator also handles:

- Attaching priority metadata to functions
- Attaching encoding metadata for drivers to optionally use
- Auto-resolving Callable steps to string names
- Automatic yielding from wrapped generators

The @entry decorator marks scraper methods as entry points with typed
parameters; it is the only way a scraper declares where a run starts.
"""

import inspect
from collections.abc import Callable, Generator
from functools import wraps
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from lxml import html as lxml_html
from pydantic_core import from_json

# typing.Concatenate rejects a trailing ``...`` before Python 3.11.
from typing_extensions import Concatenate  # noqa: UP035

from jkent.common.decorator_metadata import (
    DEFAULT_PRIORITY,
    EntryMetadata,
    StepMetadata,
    attach_entry_metadata,
    attach_step_metadata,
    get_step_metadata,
)
from jkent.common.exceptions import (
    ScraperAssumptionException,
)
from jkent.common.lxml_page_element import (
    LxmlPageElement,
)
from jkent.common.request import Request
from jkent.common.response import ArchiveResponse, Response
from jkent.common.scraper import BaseScraper, ScraperYield
from jkent.common.selector_observer import (
    SelectorObserver,
    get_active_observer,
)
from jkent.common.speculative import Speculative
from jkent.common.wait_conditions import WaitCondition

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
        return from_json(text)
    except Exception as e:
        raise ScraperAssumptionException(
            f"Failed to parse JSON: {e}",
            request_url=response.url,
            context={"error": str(e)},
        ) from e


def _parse_html(
    response: Response, encoding: str = "utf-8", *, text: str | None = None
) -> LxmlPageElement:
    """Parse HTML from response content (or preprocessed text).

    Passes raw bytes to lxml so it can auto-detect encoding from the HTML
    meta charset tag (e.g., <meta charset="windows-1252">). This handles
    pages that declare non-UTF-8 encodings correctly.

    Args:
        response: The HTTP response.
        encoding: NOT used for parsing — lxml auto-detects from the raw
            bytes (BOM, XML declaration, meta charset). Only recorded in
            the exception context for debugging. The @step encoding
            governs ``text`` injection, not ``lxml_tree``/``page``.
        text: Already-decoded (typically ``preprocess``-repaired) document
            text to parse instead of the response bytes. When set, lxml's
            byte-level encoding auto-detection is moot — the text was
            decoded with the @step encoding before repair.

    Returns:
        LxmlPageElement parsed from response content.

    Raises:
        ScraperAssumptionException: If HTML parsing fails.
    """
    try:
        # Pass raw bytes to lxml - it will detect encoding from:
        # 1. BOM
        # 2. XML declaration
        # 3. <meta charset="..."> or <meta http-equiv="Content-Type" content="...">
        # 4. Falls back to default if nothing found
        source = text if text is not None else response.content
        return LxmlPageElement(lxml_html.fromstring(source), response.url)
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
    # content (a synthetic Response built from raw bytes) means "not
    # decoded yet", so decode with the step's encoding rather than
    # injecting "".
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
    response: Response, encoding: str = "utf-8", *, text: str | None = None
) -> tuple[Any, Any]:
    """Parse HTML and create PageElement with SelectorObserver.

    Args:
        response: The HTTP response.
        encoding: Fallback encoding if lxml can't detect one.
        text: Already-decoded (typically ``preprocess``-repaired) document
            text to parse instead of the response bytes.

    Returns:
        Tuple of (PageElement, SelectorObserver) for injection and debugging.

    Raises:
        ScraperAssumptionException: If HTML parsing fails.
    """
    try:
        # Parse HTML straight into a LxmlPageElement (the count-validated
        # PageElement — no separate wrapper object).
        page_element = _parse_html(response, encoding, text=text)

        # Observer to track selector queries. It records through the
        # get_active_observer() contextvar (activated per-resume in the step
        # wrapper), not through the PageElement — see SelectorObserver.
        # An observer already active here was injected by a caller running
        # the step under its own (possibly subclassed) observer —
        # e.g. a host's selector-coverage analysis — so reuse it instead of
        # shadowing it with a fresh one. Under the driver no observer is
        # active at parse time, so each execution gets its own as before.
        observer = get_active_observer() or SelectorObserver()

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


class InjectionContext:
    """Everything an injector may read, for one execution of one step.

    Holds the per-execution state the injectors share: the response, the
    step's ``encoding``, and the ``preprocess`` repair hook. :attr:`document`
    memoises the repaired text so ``text``, ``lxml_tree``, and ``page`` in the
    same signature all see one repair rather than three.

    :attr:`observer` is written by the ``page`` injector and read back by the
    wrapper — per-execution state, so it lives here and on the ``Response``,
    never on the shared :class:`StepMetadata` where two in-flight executions
    of one step would clobber each other.
    """

    __slots__ = ("_document", "encoding", "observer", "preprocess", "response")

    def __init__(
        self,
        response: Response,
        encoding: str,
        preprocess: Callable[[str], str] | None,
    ) -> None:
        self.response = response
        self.encoding = encoding
        self.preprocess = preprocess
        self.observer: SelectorObserver | None = None
        self._document: str | None = None

    @property
    def document(self) -> str | None:
        """The repaired document text, or ``None`` when no hook is set.

        Computed at most once per execution, and only if an injector that
        feeds on the document is actually in the signature — asking for
        ``json_content`` alone never runs the repair.
        """
        if self.preprocess is None:
            return None
        if self._document is None:
            try:
                self._document = self.preprocess(
                    _get_text(self.response, self.encoding)
                )
            except ScraperAssumptionException:
                raise
            except Exception as e:
                raise ScraperAssumptionException(
                    f"Step preprocess hook failed: {e}",
                    request_url=self.response.url,
                    context={"error": str(e)},
                ) from e
        return self._document


#: One injector per supported ``@step`` parameter name: ``name -> (builder,
#: docstring line)``. The registry is the single source for both the
#: injection behaviour and the documented list in :func:`step` and this
#: module's docstring, so a new injection cannot be added without being
#: documented. Extend it with :func:`register_injector`.
INJECTORS: dict[str, tuple[Callable[[InjectionContext], Any], str]] = {}


def register_injector(
    name: str,
    builder: Callable[[InjectionContext], Any],
    *,
    doc: str,
) -> None:
    """Register a ``@step`` parameter name and what to inject for it.

    Hosts embedding jkent (a replay worker, for one) register their own
    injections here rather than patching the decorator.

    Args:
        name: The parameter name a step declares to receive this value.
        builder: Called with the :class:`InjectionContext` once per execution
            of a step whose signature names ``name``.
        doc: One-line description for the generated documentation.

    Raises:
        ValueError: ``name`` is already registered.
    """
    if name in INJECTORS:
        raise ValueError(f"@step injector {name!r} is already registered")
    INJECTORS[name] = (builder, doc)


def _injector_doc(indent: str = "    ") -> str:
    """The registry rendered as the documented parameter list."""
    return "\n".join(
        f"{indent}- {name}: {doc}" for name, (_, doc) in INJECTORS.items()
    )


def _inject_page(ctx: InjectionContext) -> Any:
    page_element, observer = _parse_page_element(
        ctx.response, ctx.encoding, text=ctx.document
    )
    ctx.observer = observer
    # The Response is the driver's per-execution handle, so the observer
    # travels there for the transports to read back.
    ctx.response.observer = observer
    return page_element


def _inject_local_filepath(ctx: InjectionContext) -> Any:
    if isinstance(ctx.response, ArchiveResponse):
        return ctx.response.file_url
    return None


register_injector(
    "response",
    lambda ctx: ctx.response,
    doc="The Response object",
)
register_injector(
    "request",
    lambda ctx: ctx.response.request,
    doc="The current Request",
)
register_injector(
    "previous_request",
    lambda ctx: ctx.response.request.parent_request,
    doc="The parent request from the chain (None for entry requests)",
)
register_injector(
    "accumulated_data",
    lambda ctx: ctx.response.request.accumulated_data,
    doc="Data collected across the request chain (from request)",
)
register_injector(
    "json_content",
    lambda ctx: _parse_json(ctx.response, ctx.encoding),
    doc="Response content parsed as JSON",
)
register_injector(
    "lxml_tree",
    lambda ctx: _parse_html(ctx.response, ctx.encoding, text=ctx.document),
    doc="Response content parsed as LxmlPageElement",
)
register_injector(
    "page",
    _inject_page,
    doc=(
        "Response content parsed as PageElement (LxmlPageElement with "
        "observer)"
    ),
)
register_injector(
    "text",
    lambda ctx: (
        ctx.document
        if ctx.document is not None
        else _get_text(ctx.response, ctx.encoding)
    ),
    doc="Response content as string",
)
register_injector(
    "local_filepath",
    _inject_local_filepath,
    doc="Local file path from ArchiveResponse (None otherwise)",
)


def _process_yielded_request(yielded: Any) -> Any:
    """Process a yielded Request to resolve Callable steps.

    When a decorated function yields a Request with a Callable step, this
    resolves it to the function name and fills in what the request left
    unset from the target step's metadata: its priority and its rate-limit
    lane.

    Args:
        yielded: The value yielded by the step.

    Returns:
        The processed yield value.
    """
    if isinstance(yielded, Request) and callable(yielded.step):
        # Get the target function's step metadata (if decorated with @step)
        target_metadata = get_step_metadata(yielded.step)

        # Resolve Callable to function name
        func_name = yielded.step.__name__
        # Note: We use object.__setattr__ because dataclasses are frozen
        object.__setattr__(yielded, "step", func_name)

        # If the yielded request doesn't have a priority set,
        # inherit from the target step's metadata. Explicit priorities
        # (including an explicit 9) are kept.
        if yielded.priority is None and target_metadata is not None:
            object.__setattr__(yielded, "priority", target_metadata.priority)
        # Same rule for the rate-limit lane: unset inherits, explicit wins.
        if (
            yielded.rate_limit is None
            and target_metadata is not None
            and target_metadata.rate_limit is not None
        ):
            object.__setattr__(
                yielded, "rate_limit", target_metadata.rate_limit
            )

    return yielded


StepYield = TypeVar("StepYield", bound=ScraperYield[Any])
StepScraper = TypeVar("StepScraper", bound=BaseScraper[Any])

# The method as the scraper author writes it: ``self`` followed by whatever
# injectable names it asks for (see :func:`step`).
StepFunction = Callable[
    Concatenate[StepScraper, ...], Generator[StepYield, Any, None]
]
# The method as the driver calls it: ``self`` and the Response, with the
# injected arguments supplied by the wrapper.
StepMethod = Callable[
    Concatenate[StepScraper, Response, ...],
    Generator[StepYield, bool | None, None],
]


@overload
def step(
    func: StepFunction[StepScraper, StepYield],
    *,
    priority: int = ...,
    encoding: str = ...,
    await_list: list[WaitCondition] | None = ...,
    auto_await_timeout: int | None = ...,
    preprocess: Callable[[str], str] | None = ...,
    rate_limit: str | None = ...,
) -> StepMethod[StepScraper, StepYield]: ...
@overload
def step(
    func: None = None,
    *,
    priority: int = ...,
    encoding: str = ...,
    await_list: list[WaitCondition] | None = ...,
    auto_await_timeout: int | None = ...,
    preprocess: Callable[[str], str] | None = ...,
    rate_limit: str | None = ...,
) -> Callable[
    [StepFunction[StepScraper, StepYield]], StepMethod[StepScraper, StepYield]
]: ...
def step(
    func: StepFunction[StepScraper, StepYield] | None = None,
    *,
    priority: int = DEFAULT_PRIORITY,
    encoding: str = "utf-8",
    await_list: list[WaitCondition] | None = None,
    auto_await_timeout: int | None = None,
    preprocess: Callable[[str], str] | None = None,
    rate_limit: str | None = None,
) -> (
    StepMethod[StepScraper, StepYield]
    | Callable[
        [StepFunction[StepScraper, StepYield]],
        StepMethod[StepScraper, StepYield],
    ]
):
    """Decorator for scraper step methods with automatic argument injection.

    This decorator inspects the function signature and injects values based on
    parameter names. The list below is generated from :data:`INJECTORS`, so a
    host that calls :func:`register_injector` sees its own name here too:

    {injections}

    Example::

        @step
        def parse_page(self, lxml_tree: LxmlPageElement, response: Response):
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
            # Can yield requests with Callable steps
            yield Request(
                url="/next",
                step=self.parse_next_page  # Callable!
            )

    Args:
        func: The scraper step method to decorate (when used without parens).
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for ``text`` injection (and JSON
            decoding fallback). HTML parsing (``lxml_tree``/``page``)
            auto-detects encoding from the raw bytes and ignores this.
        await_list: Optional list of wait conditions for Playwright driver
            (WaitForSelector, WaitForLoadState, WaitForURL, WaitForTimeout).
            HTTP driver ignores this parameter.
        auto_await_timeout: Optional timeout in milliseconds for autowait retry logic.
            When set, Playwright driver will retry the step if it raises
            HTMLStructuralAssumptionException. HTTP driver ignores this parameter.
        preprocess: Optional document repair hook, ``str -> str``. When set,
            the response text (decoded with ``encoding``) is run through it
            once, and the repaired text feeds the ``text``, ``lxml_tree``,
            and ``page`` injections — so a step can fix malformed HTML (e.g.
            unclosed ``<style>`` tags swallowing the document) *before* lxml
            parses it, while still receiving a normal ``page`` with its
            selector observer wired up. ``json_content`` is unaffected.
        rate_limit: Rate-limit lane for requests routed to this step — a
            name from the scraper's ``named_rate_limits``, or ``"none"`` for
            no limit. A yielded Request whose own ``rate_limit`` is unset
            inherits it (an explicit value on the request wins), the way
            ``priority`` is inherited. None leaves requests in the default
            lane.

    Returns:
        Decorated function with automatic argument injection.

    Raises:
        ScraperAssumptionException: If content parsing or ``preprocess``
            fails.
    """

    def decorator(
        fn: StepFunction[StepScraper, StepYield],
    ) -> StepMethod[StepScraper, StepYield]:
        # Resolve, once at decoration time, which registered injections
        # this signature asks for. Registry order, not signature order, so
        # the shared document repair is driven by the injectors themselves.
        sig = inspect.signature(fn)
        param_names = {p.name for p in sig.parameters.values()}
        wanted = [name for name in INJECTORS if name in param_names]

        # Create metadata
        metadata = StepMetadata(
            priority=priority,
            encoding=encoding,
            await_list=await_list,
            auto_await_timeout=auto_await_timeout,
            rate_limit=rate_limit,
        )

        @wraps(fn)
        def wrapper(
            scraper_self: StepScraper,
            response: Response,
            *args: Any,
            **kwargs: Any,
        ) -> Generator[StepYield, bool | None, None]:
            # Build kwargs from the injector registry (see INJECTORS).
            ctx = InjectionContext(response, encoding, preprocess)
            injected_kwargs: dict[str, Any] = {
                name: INJECTORS[name][0](ctx) for name in wanted
            }
            observer = ctx.observer

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
        attach_step_metadata(wrapper, metadata)
        return wrapper

    # Support both @step and @step(priority=5) syntax
    if func is not None:
        return decorator(func)
    return decorator


#: ``step.__doc__`` with the ``{injections}`` placeholder still in it, kept so
#: every refresh renders from the template rather than from the last render.
#: ``None`` under ``python -OO``, which strips docstrings.
_STEP_DOC_TEMPLATE = step.__doc__


def refresh_step_doc() -> None:
    """Render :data:`INJECTORS` into ``step``'s docstring.

    Call after :func:`register_injector` to have a host's own injection show
    up in ``help(step)``. No-op under ``python -OO``.
    """
    if _STEP_DOC_TEMPLATE is not None:
        step.__doc__ = _STEP_DOC_TEMPLATE.replace(
            "{injections}", _injector_doc()
        )


refresh_step_doc()


# =============================================================================
# @entry decorator for scraper entry points
# =============================================================================


def _is_bare_tuple(param_type: Any) -> bool:
    """Return True for an unparameterized ``tuple`` annotation.

    A bare ``tuple`` (or ``typing.Tuple`` with no arguments) is positional
    and untyped — could be a list, we'll never know!
    """
    if param_type is tuple:
        return True
    return get_origin(param_type) is tuple and not get_args(param_type)


EntryParams = ParamSpec("EntryParams")
EntryReturn = TypeVar("EntryReturn")


def entry(
    return_type: type,
) -> Callable[
    [Callable[EntryParams, EntryReturn]], Callable[EntryParams, EntryReturn]
]:
    """Decorator for scraper entry point methods with typed parameters.

    Marks a method as an entry point and attaches EntryMetadata describing
    the return type and parameter schema. Does NOT modify the function's
    runtime behavior.

    If a parameter's type subclasses the ``Speculative`` ABC, the
    entry is automatically detected as speculative. The driver will use
    the abstract methods to seed, track, and extend speculation.

    Parameter types may be anything pydantic can validate: Pydantic models
    (including ``RootModel`` for single-value wrappers), primitives (``str``,
    ``int``, ``date``), and typed containers (``list[str]``, ``tuple[int,
    str]``, ``dict[...]``). A bare, unparameterized ``tuple`` is rejected;
    values must be JSON-serializable for run replay.

    Example::

        @entry(Docket)
        def search_by_number(self, docket_number: str) -> Generator[Request, None, None]:
            ...

        @entry(CaseData)
        def fetch_case(self, case_id: DocketId) -> Request:
            # DocketId subclasses Speculative — auto-detected
            ...

    Args:
        return_type: The data type this entry produces.

    Returns:
        Decorator that attaches EntryMetadata to the function.
    """

    def decorator(
        fn: Callable[EntryParams, EntryReturn],
    ) -> Callable[EntryParams, EntryReturn]:
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
        param_types: dict[str, Any] = {}
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

            # Param values are validated and coerced by a per-entry pydantic
            # model (see EntryMetadata.validate_params), so any annotation
            # pydantic can build a schema for is accepted. The lone exception
            # is a bare, unparameterized ``tuple``: positional and untyped, it
            # is a poor fit for the name-keyed seed format. Use a typed
            # annotation (``tuple[int, str]``) or a Pydantic BaseModel instead.
            if _is_bare_tuple(param_type):
                raise TypeError(
                    f"Entry function '{fn.__name__}' parameter "
                    f"'{param_name}' uses a bare, untyped `tuple`. Use a typed "
                    f"annotation (e.g. tuple[int, str]) or a Pydantic BaseModel."
                )

            # A parameter may subclass the Speculative ABC; detect the
            # (at most one) speculative param. The ``get_origin`` check
            # is a cludge for python 3.10.
            if (
                isinstance(param_type, type)
                and get_origin(param_type) is None
                and issubclass(param_type, Speculative)
            ):
                if speculative_param is not None:
                    raise TypeError(
                        f"Entry function '{fn.__name__}' has multiple "
                        f"Speculative parameters: '{speculative_param}' "
                        f"and '{param_name}'. Only one is allowed."
                    )
                speculative_param = param_name

            param_types[param_name] = param_type

        metadata = EntryMetadata(
            return_type=return_type,
            param_types=param_types,
            func_name=fn.__name__,
            speculative_param=speculative_param,
        )

        attach_entry_metadata(fn, metadata)
        return fn

    return decorator
