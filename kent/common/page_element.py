"""PageElement protocol for unified data extraction across drivers.

This module provides a driver-agnostic interface for querying HTML elements,
extracting text and attributes, and navigating the DOM. PageElement is always
backed by static parsed HTML (LXML). The driver is responsible for obtaining
the HTML, whether via HTTP or by serializing a rendered Playwright DOM.

Step: unified-page-interface
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from kent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
)


@dataclass(frozen=True)
class ViaLink:
    """Describes a request produced by following a link.

    Enables the Playwright driver to replay the browser action that corresponds
    to the request (clicking the link). The HTTP driver ignores this field.

    Attributes:
        selector: The XPath or CSS selector that found the <a> element.
        description: Human-readable description of the link.
    """

    selector: str
    description: str


@dataclass(frozen=True)
class ViaFormSubmit:
    """Describes a request produced by submitting a form.

    Enables the Playwright driver to replay the browser action that corresponds
    to the request (filling and submitting the form). The HTTP driver ignores
    this field.

    Attributes:
        form_selector: The selector that found the <form> element.
        submit_selector: Selector relative to the form for the submit element.
        field_data: Merged field values (defaults + overrides).
        description: Human-readable description of the form.
    """

    form_selector: str
    submit_selector: str | None
    field_data: dict[str, str]
    description: str


@dataclass(frozen=True)
class FormField:
    """Represents a single form field.

    Attributes:
        name: The field's name attribute.
        field_type: Type of field (input, select, textarea, etc).
        value: Current/default value.
        options: For select elements, list of option values.
    """

    name: str
    field_type: str
    value: str | None
    options: list[str] | None = None


@dataclass(frozen=True)
class Form:
    """Represents an HTML <form> element with its fields and submission details.

    Form is a pure value object constructed from parsed HTML — it performs no I/O.

    Attributes:
        action: Resolved absolute URL for form submission.
        method: HTTP method (GET or POST).
        fields: List of form fields.
        selector: The selector that found this form (for replay by Playwright).
    """

    action: str
    method: str
    fields: list[FormField]
    selector: str

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

    def submit(
        self,
        data: dict[str, str] | None = None,
        submit_selector: str | None = None,
        request_params: dict[str, Any] | None = None,
        **request_kwargs: Any,
    ) -> Request:
        """Submit the form as a request.

        Args:
            data: Optional field overrides (merged with defaults).
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

        # Merge field defaults with overrides
        field_data = {field.name: field.value or "" for field in self.fields}
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
            http_params = HTTPRequestParams(
                **(asdict(http_params) | overrides)
            )
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
        selector: The selector that found this link (for replay by Playwright).
    """

    url: str
    text: str
    selector: str

    def follow(self) -> Request:
        """Follow the link as a request.

        Returns:
            Request with the link's URL and via set to ViaLink
            for Playwright replay.
        """

        return Request(
            request=HTTPRequestParams(url=self.url, method=HttpMethod.GET),
            continuation="",  # Will be set by caller
            via=ViaLink(
                selector=self.selector, description=f"link: {self.text}"
            ),
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
