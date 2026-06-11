"""PageElement protocol for unified data extraction across drivers.

This module provides a driver-agnostic interface for querying HTML elements,
extracting text and attributes, and navigating the DOM. PageElement is always
backed by static parsed HTML (LXML). The driver is responsible for obtaining
the HTML, whether via HTTP or by serializing a rendered Playwright DOM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Protocol

# ViaLink and ViaFormSubmit are defined in data_types so that Request.via can be
# typed directly (data_types cannot import from this module). They are imported
# here because the page-element API is where scrapers produce them.
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    ViaFormSubmit,
    ViaLink,
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


@dataclass(frozen=True)
class Form:
    """Represents an HTML <form> element with its fields and submission details.

    Form is a pure value object constructed from parsed HTML — it performs no I/O.

    Attributes:
        action: Resolved absolute URL for form submission.
        method: HTTP method (GET or POST).
        fields: List of form fields.
        selector: The selector that found this form (for replay by Playwright).
        selector_type: Which grammar ``selector`` uses ("xpath" or "css"),
            carried from find_form so submit() need not re-derive it.
    """

    action: str
    method: str
    fields: list[FormField]
    selector: str
    selector_type: str

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
        data: dict[str, str | list[str]] | None = None,
        submit_selector: str | None = None,
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
                value submits repeated keys (checkbox groups, multi-selects).
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
        activated_submit = self._activated_submit(submit_selector)
        field_data: dict[str, str | list[str]] = {}
        for field in self.fields:
            if (
                field.field_type in ("submit", "image")
                and field is not activated_submit
            ):
                # A browser submits only the activated submit control's
                # name/value, not every submit button in the form.
                continue
            value = field.value or ""
            existing = field_data.get(field.name)
            if existing is None:
                field_data[field.name] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                field_data[field.name] = [existing, value]
        if data:
            field_data.update(data)

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
                selector_type=self.selector_type,
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
        selector: The selector that found this link (for replay by Playwright).
        selector_type: Which grammar ``selector`` uses ("xpath" or "css"),
            carried from find_links so follow() need not re-derive it.
    """

    url: str
    text: str
    selector: str
    selector_type: str

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
        assert "continuation" in request_kwargs
        assert request_kwargs["continuation"] != ""
        return Request(
            request=HTTPRequestParams(url=self.url, method=HttpMethod.GET),
            via=ViaLink(
                selector=self.selector,
                selector_type=self.selector_type,
                description=f"link: {self.text}",
            ),
            **request_kwargs,
        )


class PageElement(Protocol):
    """Protocol for driver-agnostic data extraction from HTML elements.

    PageElement is always backed by static parsed HTML (LXML). The driver is
    responsible for obtaining the HTML, whether via HTTP or by serializing a
    rendered Playwright DOM.

    All query methods support count validation and raise
    HTMLStructuralAssumptionException if the actual count doesn't match
    expectations.
    """

    def query_xpath(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[PageElement]:
        """Query elements by XPath selector.

        Args:
            selector: XPath expression to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching PageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        ...

    def query_xpath_strings(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[str]:
        """Query string values by XPath selector.

        Useful for extracting text nodes or attribute values directly.

        Args:
            selector: XPath expression returning strings (text nodes, attributes).
            description: Human-readable description of what's being selected.
            min_count: Minimum number of strings expected (default: 1).
            max_count: Maximum number of strings expected (None = unlimited).

        Returns:
            List of matching string values.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        ...

    def query_css(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[PageElement]:
        """Query elements by CSS selector.

        Args:
            selector: CSS selector expression.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching PageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        ...

    def text_content(self) -> str:
        """Extract the visible text content.

        Returns:
            Visible text content of the element and its descendants.
        """
        ...

    def get_attribute(self, name: str) -> str | None:
        """Extract an attribute value.

        Args:
            name: Name of the attribute.

        Returns:
            Value of the attribute, or None if it doesn't exist.
        """
        ...

    def inner_html(self) -> str:
        """Get the inner HTML content.

        Returns:
            Inner HTML content of the element as a string.
        """
        ...

    def tag_name(self) -> str:
        """Get the element's tag name.

        Returns:
            Tag name as a lowercase string (e.g., "div", "a", "form").
        """
        ...

    def find_form(
        self,
        selector: str,
        description: str,
    ) -> Form:
        """Find a form by selector.

        Args:
            selector: XPath or CSS selector to find the form.
            description: Human-readable description of the form.

        Returns:
            Form value object with action, method, and fields.

        Raises:
            HTMLStructuralAssumptionException: If no form matches the selector.
        """
        ...

    def find_links(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[Link]:
        """Find links matching a selector.

        Args:
            selector: XPath or CSS selector to find <a> elements.
            description: Human-readable description of the links.
            min_count: Minimum number of links expected (default: 1).
            max_count: Maximum number of links expected (None = unlimited).

        Returns:
            List of Link value objects with resolved URLs and text.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        ...

    def links(self) -> list[Link]:
        """Discover all links in the element.

        Returns:
            List of all <a> elements with href attributes as Link objects.
        """
        ...
