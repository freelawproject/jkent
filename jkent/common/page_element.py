"""The page element (:class:`PageElement`) and its value objects.

A page element is always backed by static parsed HTML: it wraps a raw lxml
``HtmlElement`` (``self._element``) and the base URL the HTML came from. The
driver is responsible for obtaining that HTML, whether over HTTP or by
serializing a rendered Playwright DOM — there is no browser-backed variant,
so there is one implementation and one name.

:class:`PageElement` provides:

- the high-level API scrapers use — ``query(XPath(...))`` /
  ``query(CSS(...))``, ``query_strings``, ``find_form``, ``find_links`` —
  which takes explicit :class:`~jkent.common.selectors.Selector` values, and
- the low-level ``checked_xpath``/``checked_css`` string-selector forms,
  whose grammar is fixed by the method name (scrapers rarely call them
  directly).

Every one of them routes through ``_checked``, the single count-validated
query engine. Element results are re-wrapped as :class:`PageElement`, so a
query on a page element yields page elements — there is no separate wrapper
object and no re-wrapping pass.

Query recording is driven entirely by the active ``SelectorObserver`` (see
:mod:`jkent.common.selector_observer`); the checked queries report to it, so
this class holds no observer state.

The value objects a query produces — :class:`Form`, :class:`FormField` and
:class:`Link` — live here too, since the page-element API is where scrapers
obtain them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeGuard, cast, overload
from urllib.parse import urljoin

from lxml import html
from lxml.html import HtmlElement

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperConfigError,
)
from jkent.common.request import HttpMethod, HTTPRequestParams, Request
from jkent.common.selector_observer import get_active_observer
from jkent.common.selectors import (
    CSS,
    ElementQuery,
    QueryResult,
    Selector,
    XPath,
)

# ViaLink and ViaFormSubmit are defined in via so that Request.via can be
# typed directly. They are imported here because the page-element API is
# where scrapers produce them.
from jkent.common.via import FieldResolver, FieldValue, ViaFormSubmit, ViaLink
from jkent.contracts import ensure, require

# A ``submit_selector`` picks the activated submit control. We resolve it
# against the parsed FormFields, which retain only the id, name and value
# attributes, so we extract attribute-equality predicates on those three:
# CSS ``[attr=val]``/``#id`` and XPath ``[@attr=val]``. The activated control
# is the first submit satisfying every extracted predicate, mirroring how the
# Playwright transport's ``form.query_selector`` picks the first DOM-order
# match. Selectors we can't express this way (positional, class- or
# structure-based) yield no predicates and fall back to the first submit.
_SUBMIT_ATTR_PREDICATE = re.compile(
    r"""\[\s*@?(?P<attr>id|name|value)\s*=\s*
        (?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^\]\s'"]+))\s*\]""",
    re.VERBOSE,
)
_CSS_ID_SELECTOR = re.compile(r"#(?P<id>[-\w]+)")


def _normalize_selector_key(key: str | Selector) -> str:
    """Unwrap a ``CSS``/``XPath`` ``data`` key to the plain field name.

    Scrapers may wrap a field name in a :class:`Selector` for symmetry with the
    rest of a form spec (``data={CSS("ctl00$name"): ...}``). The override map is
    keyed by raw ``FormField.name``, so we use the selector's ``value`` — which
    for these name-as-selector keys is the field name itself.
    """
    return key.value if isinstance(key, Selector) else key


def _normalize_submit_selector(
    submit_selector: Selector | None,
) -> str | None:
    """Render a ``submit_selector`` as a string the transport can resolve.

    The selector is tagged with its grammar engine (``css=``/``xpath=``) so
    the Playwright transport's ``query_selector`` resolves it unambiguously —
    important for XPath, whose ``nth`` form (``(//…)[1]``) Playwright would
    not auto-detect as XPath. A bare string is not accepted: leaving the
    grammar to Playwright's auto-detection is the guess :class:`Selector`
    exists to remove. The ``id``/``name``/``value`` predicate extraction in
    :func:`_submit_selector_predicates` is unaffected by the engine prefix.
    """
    if submit_selector is None:
        return None
    return submit_selector.for_playwright()


def _submit_selector_predicates(selector: str) -> list[tuple[str, str]]:
    """Extract the ``(attr, value)`` equality predicates a selector implies.

    Only ``id``/``name``/``value`` are recognized — the attributes a
    ``FormField`` retains. Returns an empty list for selectors with no such
    predicate, which then fall back to the first submit control.
    """
    predicates: list[tuple[str, str]] = []
    for m in _SUBMIT_ATTR_PREDICATE.finditer(selector):
        value = next(
            g
            for g in (m.group("dq"), m.group("sq"), m.group("bare"))
            if g is not None
        )
        predicates.append((m.group("attr"), value))
    # Match a CSS ``#id`` only outside any ``[...]`` predicate, so a '#' inside
    # e.g. ``[value="#fff"]`` is not mistaken for an id selector.
    outside_predicates = re.sub(r"\[[^\]]*\]", "", selector)
    predicates += [
        ("id", m.group("id"))
        for m in _CSS_ID_SELECTOR.finditer(outside_predicates)
    ]
    return predicates


@dataclass(frozen=True)
class FormField:
    """Represents a single form field.

    Attributes:
        name: The field's name attribute.
        field_type: Type of field (input, select, textarea, etc).
        value: Current/default value.
        options: For select elements, list of option values.
        element_id: The control's ``id`` attribute, when present. Lets
            ``Form.submit`` match a ``#id`` ``submit_selector`` to the activated
            submit button so only that button's name/value is sent.
    """

    name: str
    field_type: str
    value: str | None
    options: list[str] | None = None
    element_id: str | None = None


def _has_no_none(items: list[str | None]) -> TypeGuard[list[str]]:
    return None not in items


def _merge_override(
    default: str | list[str] | None,
    override: str | list[str | None] | FieldResolver,
) -> FieldValue | FieldResolver:
    """Resolve one ``Form.submit(data=)`` override against the rendered default.

    A scalar, a :data:`FieldResolver`, or a list with no ``None`` replaces the
    default wholesale (the historical behaviour). A list containing ``None``
    fills repeated same-named controls positionally, with each ``None`` keeping
    the parsed default at that position — so the result is concrete (no
    ``None`` reaches the wire) and the HTTP and Playwright transports submit
    identical data. Positions past the rendered defaults take the override
    verbatim; a trailing ``None`` with no default to fall back to is dropped.
    """
    if not isinstance(override, list) or _has_no_none(override):
        return override
    defaults = (
        default
        if isinstance(default, list)
        else ([] if default is None else [default])
    )
    resolved: list[str] = []
    for i, item in enumerate(override):
        if item is None:
            if i < len(defaults):
                resolved.append(defaults[i])
        else:
            resolved.append(item)
    return resolved


@dataclass(frozen=True)
class Form:
    """Represents an HTML <form> element with its fields and submission details.

    Form is a pure value object constructed from parsed HTML — it performs no I/O.

    Attributes:
        action: Resolved absolute URL for form submission.
        method: HTTP method (GET or POST).
        fields: List of form fields.
        selector: The :class:`Selector` that found this form (for replay by
            Playwright). Its grammar travels with it, so submit() need not
            re-derive it.
    """

    action: str
    method: str
    fields: list[FormField]
    selector: Selector

    def get_field(self, name: str) -> FormField | None:
        """Get a specific field by name.

        Args:
            name: The field name to find.

        Returns:
            The FormField with the matching name, or None if not found.
        """
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def _activated_submit(
        self, submit_selector: str | None
    ) -> FormField | None:
        """The one submit/image control whose name/value a browser would send.

        A browser includes only the activated submit control. When
        ``submit_selector`` carries an id/name/value predicate (``#id``,
        ``[name=…]``, ``[value=…]`` in CSS, or the ``[@attr=…]`` XPath form) we
        return the first submit matching every such predicate — the same
        control the Playwright transport's ``form.query_selector`` would click,
        so a non-first button sends *that* button's name/value. Otherwise — no
        selector, or one we can't resolve against the parsed fields (positional,
        class- or structure-based) — we fall back to the first submit control,
        matching a browser's implicit-submission default and the prior behavior.

        Returns:
            The activated ``FormField``, or None when the form has no submit
            control (e.g. a JS/``__EVENTTARGET`` submission).
        """
        submits = [
            f for f in self.fields if f.field_type in ("submit", "image")
        ]
        if not submits:
            return None
        if submit_selector:
            predicates = _submit_selector_predicates(submit_selector.strip())
            if predicates:
                for field in submits:
                    field_attrs = {
                        "id": field.element_id,
                        "name": field.name,
                        "value": field.value,
                    }
                    if all(
                        field_attrs.get(attr) == value
                        for attr, value in predicates
                    ):
                        return field
        return submits[0]

    def submit(
        self,
        data: Mapping[str | Selector, str | list[str | None] | FieldResolver]
        | None = None,
        submit_selector: Selector | None = None,
        request_params: dict[str, Any] | None = None,
        **request_kwargs: Any,
    ) -> Request:
        """Submit the form as a request.

        Only one submit control's name/value is included, matching what a
        browser sends. ``submit_selector`` chooses which (by id/name/value —
        see :meth:`_activated_submit`); without it, the first submit control is
        used, as on implicit submission. A different button's value can also be
        forced via ``data``.

        Args:
            data: Optional field overrides (merged with defaults). A list
                value submits repeated keys (checkbox groups, multi-selects)
                and fills repeated same-named controls positionally; a ``None``
                entry keeps that position's rendered default (it is still
                submitted, just unchanged), so a caller can override some of a
                group's controls without restating the rest. A zero-arg async
                callable defers the value: the driver awaits it when the
                yielded request is enqueued (see
                :meth:`Request.resolve_deferred_fields`), so a scraper can
                fetch e.g. a captcha solution from an external service.
            submit_selector: Optional ``CSS``/``XPath`` selector for the
                submit element (relative to the form).
            request_params: Optional HTTPRequestParams field overrides (e.g.
                {"timeout": 30}). Wins over the form-derived values, except for
                url/method/params/data which are always set by the form.
            **request_kwargs: Additional kwargs passed to Request constructor.
                Common ones: step, accumulated_data, archive, expected_type,
                priority, deduplication_key, permanent.

        Returns:
            Request with the form's action as URL, method as HTTP method,
            and via set to ViaFormSubmit for Playwright replay.
        """

        # Merge field defaults with overrides. Repeated names (checkbox
        # groups, multi-selects) accumulate into a list — httpx encodes a
        # list value as repeated keys, like a browser does.
        # The engine-prefixed string form: what the transport resolves and
        # what the via stores.
        submit_string = _normalize_submit_selector(submit_selector)
        activated_submit = self._activated_submit(submit_string)
        defaults: dict[str, str | list[str]] = {}
        for field in self.fields:
            if (
                field.field_type in ("submit", "image")
                and field is not activated_submit
            ):
                # A browser submits only the activated submit control's
                # name/value, not every submit button in the form.
                continue
            value = field.value or ""
            existing = defaults.get(field.name)
            if existing is None:
                defaults[field.name] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                defaults[field.name] = [existing, value]
        field_data: dict[str, FieldValue | FieldResolver] = {}
        field_data.update(defaults)
        if data:
            for key, override in data.items():
                name = _normalize_selector_key(key)
                field_data[name] = _merge_override(
                    defaults.get(name), override
                )

        # Create request based on method
        method_enum = (
            HttpMethod.POST
            if self.method.upper() == "POST"
            else HttpMethod.GET
        )

        # For GET forms, field data becomes query parameters
        # For POST forms, field data becomes form-encoded body
        if method_enum == HttpMethod.GET:
            http_params = HTTPRequestParams(
                url=self.action, method=method_enum, params=field_data
            )
        else:
            http_params = HTTPRequestParams(
                url=self.action,
                method=method_enum,
                data=field_data,
            )

        # Set defaults for step if not provided
        request_kwargs.setdefault("step", "")
        if request_params:
            overrides = {
                k: v
                for k, v in request_params.items()
                if k not in {"url", "method", "params", "data"}
            }
            http_params = replace(http_params, **overrides)
        return Request(
            request=http_params,
            via=ViaFormSubmit(
                form_selector=self.selector,
                submit_selector=submit_string,
                field_data=field_data,
                description=f"form at {self.selector}",
            ),
            **request_kwargs,
        )


@dataclass(frozen=True)
class Link:
    """Represents an HTML <a> element with its resolved URL and text.

    Link is a pure value object — it performs no I/O.

    Attributes:
        url: Resolved absolute URL from the href attribute.
        text: Visible text content of the link.
        selector: The :class:`Selector` that found this link (for replay by
            Playwright). Its grammar travels with it, so follow() need not
            re-derive it.
    """

    url: str
    text: str
    selector: Selector

    def follow(self, **request_kwargs: Any) -> Request:
        """Follow the link as a request.

        Args:
            **request_kwargs: Additional kwargs passed to the Request
                constructor. Common ones: step, accumulated_data,
                archive, expected_type, priority, deduplication_key,
                permanent.

        Returns:
            Request with the link's URL and via set to ViaLink
            for Playwright replay.
        """
        return Request(
            request=HTTPRequestParams(url=self.url, method=HttpMethod.GET),
            via=ViaLink(
                selector=self.selector,
                description=f"link: {self.text}",
            ),
            **request_kwargs,
        )


class PageElement:
    """Page element backed by a raw lxml ``HtmlElement``.

    Holds the wrapped element as ``self._element`` and the base URL as
    ``self._request_url`` (used both for error context and as the base for
    resolving relative URLs). The checked queries wrap their element results
    in ``PageElement``, so nested queries return page elements directly.
    """

    def __init__(self, element: HtmlElement, request_url: str = "") -> None:
        """Initialize the page element.

        Args:
            element: The lxml HtmlElement to wrap.
            request_url: Base URL for resolving relative URLs and for error
                context.
        """
        self._element = element
        self._request_url = request_url

    @overload
    def checked_xpath(
        self,
        xpath: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
        *,
        type: type[str],
    ) -> list[str]: ...

    @overload
    def checked_xpath(
        self,
        xpath: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[PageElement]: ...

    def checked_xpath(
        self,
        xpath: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
        *,
        type: type[str] | None = None,
    ) -> list[PageElement] | list[str]:
        """Execute XPath query with count validation.

        Args:
            xpath: XPath expression to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).
            type: Pass `str` to return only string results (text/attributes).
                If omitted, returns only PageElements (filtering out
                any string results).

        Returns:
            List of matching results filtered by type. By default returns
            PageElements; pass type=str for string results.
            Filtering happens before count validation: min/max bounds
            apply to results of the requested type only, so a
            string-returning XPath without type=str counts as 0 elements.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = PageElement(lxml.html.fromstring(html))
            # Get elements (default)
            cases = tree.checked_xpath("//tr[@class='case']", "cases")
            # Get text/attributes
            hrefs = tree.checked_xpath("//a/@href", "links", type=str)
        """
        if type is str:
            return self._checked(
                XPath(xpath), description, min_count, max_count, strings=True
            )
        return self._checked(XPath(xpath), description, min_count, max_count)

    def checked_css(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[PageElement]:
        """Execute CSS selector query with count validation.

        Args:
            selector: CSS selector expression.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching PageElements. Each element is wrapped to
            support nested checked queries.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = PageElement(lxml.html.fromstring(html))
            # Expect exactly 1 case name
            case_name = tree.checked_css("h1.case-name", "case name")
            # Expect at least 5 case divs
            cases = tree.checked_css("div.case", "case divs", min_count=5)
            # Nested queries work
            for case in cases:
                title = case.checked_css("h2.title", "title", min_count=1)
        """
        return self._checked(CSS(selector), description, min_count, max_count)

    @overload
    def _checked(
        self,
        selector: Selector,
        description: str,
        min_count: int,
        max_count: int | None,
        *,
        strings: Literal[True],
    ) -> list[str]: ...

    @overload
    def _checked(
        self,
        selector: Selector,
        description: str,
        min_count: int,
        max_count: int | None,
        *,
        strings: Literal[False] = ...,
    ) -> list[PageElement]: ...

    @require(
        lambda min_count, max_count: (  # pyrefly: ignore[implicit-any-lambda]
            min_count >= 0 and (max_count is None or max_count >= min_count)
        ),
        "expected-count bounds form a valid (possibly open) interval",
    )
    @ensure(
        lambda result, min_count, max_count: (  # pyrefly: ignore[implicit-any-lambda]
            min_count <= len(result)
            and (max_count is None or len(result) <= max_count)
        ),
        "a returned result list always satisfies the caller's bounds — "
        "out-of-bounds counts raise instead",
    )
    def _checked(
        self,
        selector: Selector,
        description: str,
        min_count: int,
        max_count: int | None,
        *,
        strings: bool = False,
    ) -> list[str] | list[PageElement]:
        """The count-validated query engine behind every selector method.

        Runs ``selector`` in its own grammar, keeps only results of the
        requested kind (strings, or elements wrapped as page elements),
        reports them to the active observer, and enforces the bounds.
        Filtering happens before the count, so the observer's recorded
        match count is the count the bounds were checked against.
        """
        try:
            raw = selector.query(self)
        except Exception as e:
            # A selector that doesn't parse is a bug in the scraper, not
            # a change in the website — never report it as structural.
            raise ScraperConfigError(
                f"Invalid {selector.label} selector {selector.value!r} for "
                f"'{description}' (url: {self._request_url}): {e}"
            ) from e

        # A nodeset XPath (//a/@href) returns a list; a scalar XPath
        # (string()/count()/concat()/normalize-space()/boolean()/…) returns a
        # bare value — a str subclass, float, or bool — not a list. Wrap the
        # scalar so it counts as one result; iterating a bare str would count
        # its characters (a 2+ digit count() would report len(str) results and
        # spuriously trip the count check).
        results = raw if isinstance(raw, list) else [raw]
        typed_results: list[Any] = (
            [r for r in results if isinstance(r, str)]
            if strings
            else [
                PageElement(r, self._request_url)
                for r in results
                if isinstance(r, HtmlElement)
            ]
        )

        observer = get_active_observer()
        if observer is not None:
            observer.record_query(
                selector=selector.value,
                selector_type=selector.grammar,
                description=description,
                results=typed_results,
                expected_min=min_count,
                expected_max=max_count,
                parent_element=self._element,
            )

        self._enforce_count(
            selector,
            description,
            len(typed_results),
            min_count,
            max_count,
            is_element_query=not strings,
        )
        return typed_results

    def _enforce_count(
        self,
        selector: Selector,
        description: str,
        actual_count: int,
        min_count: int,
        max_count: int | None,
        *,
        is_element_query: bool = True,
    ) -> None:
        """Raise :class:`HTMLStructuralAssumptionException` if out of bounds."""
        if actual_count < min_count or (
            max_count is not None and actual_count > max_count
        ):
            raise HTMLStructuralAssumptionException(
                selector=selector.value,
                selector_type=selector.grammar,
                description=description,
                expected_min=min_count,
                expected_max=max_count,
                actual_count=actual_count,
                request_url=self._request_url,
                is_element_query=is_element_query,
            )

    def query(
        self,
        selector: Selector,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[PageElement]:
        """Query elements by selector, dispatching on its grammar.

        Runs the count-validated engine in ``selector.grammar``. The caller
        wraps the selector in
        ``Selector.XPath``/``Selector.CSS``, so the grammar is explicit — no
        prefix heuristic to guess it back, and no bare string can slip in.

        Args:
            selector: ``Selector.XPath``/``Selector.CSS`` to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching PageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        return self._checked(selector, description, min_count, max_count)

    def query_strings(
        self,
        selector: XPath,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[str]:
        """Query string values by XPath selector.

        Only XPath can yield strings, so this takes a ``Selector.XPath`` —
        a CSS selector is a type error rather than a query that matches nothing.

        Args:
            selector: ``Selector.XPath`` returning strings (text nodes, attributes).
            description: Human-readable description of what's being selected.
            min_count: Minimum number of strings expected (default: 1).
            max_count: Maximum number of strings expected (None = unlimited).

        Returns:
            List of matching string values.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        return self._checked(
            selector, description, min_count, max_count, strings=True
        )

    def text_content(self) -> str:
        """Extract the visible text content.

        Returns:
            Visible text content of the element and its descendants.
        """
        return self._element.text_content()

    def get_attribute(self, name: str) -> str | None:
        """Extract an attribute value.

        Args:
            name: Name of the attribute.

        Returns:
            Value of the attribute, or None if it doesn't exist.
        """
        return self._element.get(name)

    def inner_html(self) -> str:
        """Get the inner HTML content.

        Returns:
            Inner HTML content of the element as a string.
        """
        elem = self._element

        # elem.text is the leading text node before the first child; lxml
        # keeps it off the children list, so serialize it separately or it
        # vanishes ("<td>Case No. <a>123</a></td>" would lose "Case No. ").
        leading = elem.text or ""
        inner = leading + "".join(
            html.tostring(child, encoding="unicode") for child in elem
        )
        return inner

    def tag_name(self) -> str:
        """Get the element's tag name.

        Returns:
            Tag name as a lowercase string (e.g., "div", "a", "form").
        """
        # An HtmlElement's tag is always a str (lxml types it as the wider
        # str | bytes | QName union shared with raw XML nodes).
        return cast("str", self._element.tag).lower()

    @staticmethod
    def _option_value(option: PageElement) -> str:
        """The value an <option> submits: value attribute, else label text.

        The attribute wins even when empty — value="" is how placeholder
        options ("All case types") request an empty filter.
        """
        value = option.get_attribute("value")
        if value is not None:
            return value
        return option.text_content()

    def find_form(
        self,
        selector: Selector,
        description: str,
    ) -> Form:
        """Find a form by selector.

        Args:
            selector: ``Selector.XPath``/``Selector.CSS`` locating the form.
            description: Human-readable description of the form.

        Returns:
            Form value object with action, method, and fields.

        Raises:
            HTMLStructuralAssumptionException: If no form matches the selector.
        """
        # A form selector must match exactly one element.
        form_elements = self.query(
            selector, description, min_count=1, max_count=1
        )

        form_elem = form_elements[0]

        # Extract form action and method
        action = form_elem.get_attribute("action") or ""
        method = (form_elem.get_attribute("method") or "GET").upper()

        # Resolve action URL against base URL
        if action:
            action = urljoin(self._request_url, action)
        else:
            action = self._request_url

        # Extract form fields
        fields: list[FormField] = []

        # Collect every submittable control in ONE document-order pass. A
        # browser submits fields in document order regardless of tag, so the
        # union XPath (which preserves document order across tags) keeps the
        # reconstructed request's field order matching the browser — querying
        # inputs/buttons, then selects, then textareas separately would group
        # by tag and reorder a <textarea>/<select> that sits among inputs.
        #
        # The submittability filters live in the XPath, not a Python loop: a
        # control submits only when it carries a non-empty name (`@name != ''`)
        # and is not disabled (`not(@disabled)` — disabled controls are barred
        # from submission and must not be filled on the Playwright replay path).
        # Encoding both predicates per tag means the union yields only the
        # controls a browser would send, so the loop never re-tests them.
        control_elements = form_elem.checked_xpath(
            ".//input[@name != ''][not(@disabled)] | "
            ".//button[@name != ''][not(@disabled)] | "
            ".//select[@name != ''][not(@disabled)] | "
            ".//textarea[@name != ''][not(@disabled)]",
            "form controls",
            min_count=0,
        )

        for elem in control_elements:
            tag = elem.tag_name()
            if tag in ("input", "button"):
                self._collect_input_or_button(elem, fields)
            elif tag == "select":
                self._collect_select(elem, fields)
            else:  # textarea
                self._collect_textarea(elem, fields)

        return Form(
            action=action,
            method=method,
            fields=fields,
            selector=selector,
        )

    def _collect_input_or_button(
        self, elem: PageElement, fields: list[FormField]
    ) -> None:
        """Append the field for one ``<input>``/``<button>`` if it submits."""
        name = elem.get_attribute("name")
        assert name, (
            "find_form's union XPath restricts controls to [@name != '']"
        )

        value = elem.get_attribute("value")
        element_id = elem.get_attribute("id")

        if elem.tag_name() == "button":
            # A <button> without a type attribute is a submit button.
            # type=button/reset never contribute to form submission.
            button_type = (elem.get_attribute("type") or "submit").lower()
            if button_type != "submit":
                return
            fields.append(
                FormField(
                    name=name,
                    field_type="submit",
                    value=value,
                    element_id=element_id,
                )
            )
            return

        field_type = (elem.get_attribute("type") or "text").lower()

        # Per HTML spec, reset and push buttons never submit.
        if field_type in ("reset", "button"):
            return

        # Per HTML spec, unchecked radios/checkboxes contribute nothing to
        # form submission; omit them so request bodies match real browsers.
        if (
            field_type in ("radio", "checkbox")
            and elem.get_attribute("checked") is None
        ):
            return

        # A checked checkbox/radio without an explicit value submits as "on".
        if field_type in ("checkbox", "radio") and value is None:
            value = "on"

        fields.append(
            FormField(
                name=name,
                field_type=field_type,
                value=value,
                element_id=element_id,
            )
        )

    def _collect_select(
        self, elem: PageElement, fields: list[FormField]
    ) -> None:
        """Append the field(s) for one ``<select>`` if it submits."""
        name = elem.get_attribute("name")
        assert name, (
            "find_form's union XPath restricts controls to [@name != '']"
        )

        options = elem.checked_xpath(
            ".//option", "select options", min_count=0
        )
        # Per HTML spec an option's value is its value attribute when present —
        # including value="" (placeholder "All" options) — and its label text
        # only when the attribute is absent.
        option_values = [self._option_value(opt) for opt in options]

        selected_options = elem.checked_xpath(
            ".//option[@selected]", "selected option", min_count=0
        )

        if elem.get_attribute("multiple") is not None:
            # A multi-select submits one pair per selected option and nothing
            # when none are selected (no first-option default).
            for selected in selected_options:
                fields.append(
                    FormField(
                        name=name,
                        field_type="select",
                        value=self._option_value(selected),
                        options=option_values,
                    )
                )
            return

        if selected_options:
            value = self._option_value(selected_options[0])
        elif options:
            value = option_values[0]
        else:
            value = None

        fields.append(
            FormField(
                name=name,
                field_type="select",
                value=value,
                options=option_values,
            )
        )

    def _collect_textarea(
        self, elem: PageElement, fields: list[FormField]
    ) -> None:
        """Append the field for one ``<textarea>`` if it submits."""
        name = elem.get_attribute("name")
        assert name, (
            "find_form's union XPath restricts controls to [@name != '']"
        )
        fields.append(
            FormField(
                name=name, field_type="textarea", value=elem.text_content()
            )
        )

    def find_links(
        self,
        selector: Selector,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[Link]:
        """Find links matching a selector.

        Args:
            selector: ``Selector.XPath``/``Selector.CSS`` locating <a> elements.
            description: Human-readable description of the links.
            min_count: Minimum number of links expected (default: 1).
            max_count: Maximum number of links expected (None = unlimited).

        Returns:
            List of Link value objects with resolved URLs and text.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        # Check min_count against raw matches (fewer matches than min is
        # already a failure); max_count waits until href-less anchors are
        # filtered below, since only returned links count.
        link_elements = self.query(
            selector, description, min_count, max_count=None
        )

        links: list[Link] = []
        for i, elem in enumerate(link_elements):
            href = elem.get_attribute("href")
            if not href:
                raise HTMLStructuralAssumptionException(
                    selector=selector.value,
                    selector_type=selector.grammar,
                    description=f"{description} missing href",
                    expected_min=min_count,
                    expected_max=max_count,
                    actual_count=len(links),
                    request_url=self._request_url,
                )

            # Resolve URL against base URL
            url = urljoin(self._request_url, href)
            text = elem.text_content().strip()

            # A unique selector for this specific link, positional in the
            # matched element's own grammar (Selector.nth wraps it). The index
            # counts all matched elements (pre-href-filter) so replay selects
            # the same node the parse saw.
            links.append(
                Link(
                    url=url,
                    text=text,
                    selector=selector.nth(i + 1),
                )
            )

        # Validate the bounds against the links actually returned: a page
        # that swaps real anchors for href-less JS handlers must fail the
        # structural contract, not silently return fewer links.
        self._enforce_count(
            selector, description, len(links), min_count, max_count
        )

        return links

    def cssselect(self, expr: str, /) -> Sequence[ElementQuery]:
        """Every element matching the CSS selector ``expr``, unchecked.

        Half of the :class:`~jkent.common.selectors.ElementQuery` surface a
        :class:`~jkent.common.selectors.CSS` selector dispatches into. Raw:
        no count validation and no observer record, so scrapers want
        ``query``/``checked_css`` instead. Explicit rather than left to
        ``__getattr__`` below, which would type the result as ``Any``.
        """
        return self._element.cssselect(expr)

    def xpath(self, expr: str, /) -> QueryResult:
        """``expr`` evaluated against this element, unchecked.

        The other half of :class:`~jkent.common.selectors.ElementQuery`. A
        node-set comes back as a list; a scalar XPath (``count()``,
        ``string()``, …) as a bare value. Same caveat as :meth:`cssselect`:
        raw, so scrapers want ``query_strings``/``checked_xpath``.
        """
        return self._element.xpath(expr)

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped element.

        This lets PageElement stand in for the raw HtmlElement for any
        attribute the explicit methods above don't cover.
        """
        return getattr(self._element, name)
