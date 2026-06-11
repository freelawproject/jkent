"""PageElement ABC for unified data extraction across drivers.

This module provides a driver-agnostic interface for querying HTML elements,
extracting text and attributes, and navigating the DOM. PageElement is always
backed by static parsed HTML (LXML). The driver is responsible for obtaining
the HTML, whether via HTTP or by serializing a rendered Playwright DOM.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from typing_extensions import Self

# ViaLink and ViaFormSubmit are defined in data_types so that Request.via can be
# typed directly (data_types cannot import from this module). They are imported
# here because the page-element API is where scrapers produce them.
from jkent.data_types import (
    FieldResolver,
    FieldValue,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
    ViaFormSubmit,
    ViaLink,
    XPath,
)

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
    submit_selector: str | Selector | None,
) -> str | None:
    """Render a ``submit_selector`` as a string the transport can resolve.

    Accepts a plain string (passed through unchanged, the historical form) or a
    :class:`Selector`. A wrapped selector is tagged with its grammar engine
    (``css=``/``xpath=``) so the Playwright transport's ``query_selector``
    resolves it unambiguously — important for XPath, whose ``nth`` form
    (``(//…)[1]``) Playwright would not auto-detect as XPath. The
    ``id``/``name``/``value`` predicate extraction in
    :func:`_submit_selector_predicates` is unaffected by the engine prefix.
    """
    if isinstance(submit_selector, Selector):
        engine = "xpath" if submit_selector.grammar == "xpath" else "css"
        return f"{engine}={submit_selector.value}"
    return submit_selector


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
    if not isinstance(override, list) or None not in override:
        return override  # type: ignore[return-value]
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
        submit_selector: str | Selector | None = None,
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
            submit_selector: Optional selector for submit element (relative to form).
            request_params: Optional HTTPRequestParams field overrides (e.g.
                {"timeout": 30}). Wins over the form-derived values, except for
                url/method/params/data which are always set by the form.
            **request_kwargs: Additional kwargs passed to Request constructor.
                Common ones: continuation, accumulated_data, archive, expected_type,
                priority, deduplication_key, permanent.

        Returns:
            Request with the form's action as URL, method as HTTP method,
            and via set to ViaFormSubmit for Playwright replay.
        """

        # Merge field defaults with overrides. Repeated names (checkbox
        # groups, multi-selects) accumulate into a list — httpx encodes a
        # list value as repeated keys, like a browser does.
        submit_selector = _normalize_submit_selector(submit_selector)
        activated_submit = self._activated_submit(submit_selector)
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
                data=field_data,  # type: ignore[arg-type]
            )

        # Set defaults for continuation if not provided
        request_kwargs.setdefault("continuation", "")
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
                submit_selector=submit_selector,
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
                constructor. Common ones: continuation, accumulated_data,
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


class PageElement(ABC):
    """Abstract base for driver-agnostic data extraction from HTML elements.

    PageElement is always backed by static parsed HTML (LXML). The driver is
    responsible for obtaining the HTML, whether via HTTP or by serializing a
    rendered Playwright DOM.

    All query methods support count validation and raise
    HTMLStructuralAssumptionException if the actual count doesn't match
    expectations.
    """

    @abstractmethod
    def query(
        self,
        selector: Selector,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[Self]:
        """Query elements by selector, dispatching on its grammar.

        Accepts ``Selector.XPath``/``Selector.CSS`` and routes to the matching
        engine off ``selector.grammar`` — callers no longer pick the method
        (``query_xpath``/``query_css``) by hand, and a bare string can't reach
        this entry point.

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
        ...

    @abstractmethod
    def query_strings(
        self,
        selector: XPath,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[str]:
        """Query string values by XPath selector.

        Useful for extracting text nodes or attribute values directly. Only
        XPath can yield strings (CSS selectors always match elements), so this
        requires a :class:`~jkent.data_types.XPath` selector — a CSS selector is
        rejected by the type checker rather than silently matching nothing.

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
        ...

    @abstractmethod
    def text_content(self) -> str:
        """Extract the visible text content.

        Returns:
            Visible text content of the element and its descendants.
        """
        ...

    @abstractmethod
    def get_attribute(self, name: str) -> str | None:
        """Extract an attribute value.

        Args:
            name: Name of the attribute.

        Returns:
            Value of the attribute, or None if it doesn't exist.
        """
        ...

    @abstractmethod
    def inner_html(self) -> str:
        """Get the inner HTML content.

        Returns:
            Inner HTML content of the element as a string.
        """
        ...

    @abstractmethod
    def tag_name(self) -> str:
        """Get the element's tag name.

        Returns:
            Tag name as a lowercase string (e.g., "div", "a", "form").
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...
