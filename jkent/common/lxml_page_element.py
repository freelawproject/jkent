"""LxmlPageElement implementation wrapping CheckedHtmlElement.

This module provides an implementation of the PageElement protocol that wraps
CheckedHtmlElement and delegates to it. This is the standard implementation
used by all drivers.
"""

from __future__ import annotations

from urllib.parse import urljoin

from lxml import html

from jkent.common.checked_html import (
    CheckedHtmlElement,
    raise_if_count_out_of_bounds,
)
from jkent.common.page_element import (
    Form,
    FormField,
    Link,
)


class LxmlPageElement:
    """Implementation of PageElement protocol wrapping CheckedHtmlElement.

    This is the standard implementation used by all drivers. It wraps
    CheckedHtmlElement to provide the PageElement interface.

    Query recording is driven entirely by the active SelectorObserver
    contextvar (see :mod:`jkent.common.selector_observer`); the wrapped
    CheckedHtmlElement reports to it, so this class holds no observer state.

    Attributes:
        _element: The underlying CheckedHtmlElement.
        _url: The base URL for resolving relative URLs.
    """

    def __init__(
        self,
        element: CheckedHtmlElement,
        url: str = "",
    ):
        """Initialize LxmlPageElement.

        Args:
            element: The CheckedHtmlElement to wrap.
            url: Base URL for resolving relative URLs.
        """
        self._element = element
        self._url = url

    def query_xpath(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[LxmlPageElement]:
        """Query elements by XPath selector.

        Args:
            selector: XPath expression to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching LxmlPageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        # Delegate to CheckedHtmlElement
        checked_elements = self._element.checked_xpath(
            selector, description, min_count, max_count
        )

        # Wrap each result in LxmlPageElement
        return [LxmlPageElement(elem, self._url) for elem in checked_elements]

    def query_xpath_strings(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[str]:
        """Query string values by XPath selector.

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
        # Delegate to CheckedHtmlElement with type=str
        return self._element.checked_xpath(
            selector, description, min_count, max_count, type=str
        )

    def query_css(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[LxmlPageElement]:
        """Query elements by CSS selector.

        Args:
            selector: CSS selector expression.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching LxmlPageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        # Delegate to CheckedHtmlElement
        checked_elements = self._element.checked_css(
            selector, description, min_count, max_count
        )

        # Wrap each result in LxmlPageElement
        return [LxmlPageElement(elem, self._url) for elem in checked_elements]

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
        # Use lxml's tostring to get inner HTML
        # Access the wrapped lxml element
        elem = self._element._element

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
        return self._element.tag.lower()

    @staticmethod
    def _option_value(option: LxmlPageElement) -> str:
        """The value an <option> submits: value attribute, else label text.

        The attribute wins even when empty — value="" is how placeholder
        options ("All case types") request an empty filter.
        """
        value = option.get_attribute("value")
        if value is not None:
            return value
        return option.text_content()

    def _query_selector(
        self,
        selector: str,
        description: str,
        min_count: int,
        max_count: int | None,
    ) -> tuple[list[LxmlPageElement], bool]:
        """Route ``selector`` to query_xpath or query_css.

        Returns the matched elements and whether the selector was treated as
        XPath (callers need that to record ``selector_type`` and to build
        positional replay selectors). Routes on unambiguous XPath prefixes
        only: a bare "." prefix is far more likely a CSS class selector
        (".search-form") than an abbreviated relative XPath, which always
        continues with "/".
        """
        is_xpath = selector.startswith(("//", "./", "("))
        if is_xpath:
            elements = self.query_xpath(
                selector, description, min_count, max_count
            )
        else:
            elements = self.query_css(
                selector, description, min_count, max_count
            )
        return elements, is_xpath

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
        # A form selector must match exactly one element.
        form_elements, is_xpath = self._query_selector(
            selector, description, min_count=1, max_count=1
        )

        form_elem = form_elements[0]

        # Extract form action and method
        action = form_elem.get_attribute("action") or ""
        method = (form_elem.get_attribute("method") or "GET").upper()

        # Resolve action URL against base URL
        if action:
            action = urljoin(self._url, action)
        else:
            action = self._url

        # Extract form fields
        fields: list[FormField] = []

        # Collect every submittable control in ONE document-order pass. A
        # browser submits fields in document order regardless of tag, so the
        # union XPath (which preserves document order across tags) keeps the
        # reconstructed request's field order matching the browser — querying
        # inputs/buttons, then selects, then textareas separately would group
        # by tag and reorder a <textarea>/<select> that sits among inputs.
        control_elements = form_elem.query_xpath(
            ".//input | .//button | .//select | .//textarea",
            "form controls",
            min_count=0,
        )

        for elem in control_elements:
            # Per HTML spec a disabled control is barred from submission;
            # omit it so the reconstructed request matches a real browser
            # (and the Playwright fill path doesn't try to type into it).
            if elem.get_attribute("disabled") is not None:
                continue
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
            selector_type="xpath" if is_xpath else "css",
        )

    def _collect_input_or_button(
        self, elem: LxmlPageElement, fields: list[FormField]
    ) -> None:
        """Append the field for one ``<input>``/``<button>`` if it submits."""
        name = elem.get_attribute("name")
        if not name:
            return  # controls without a name don't submit

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
        self, elem: LxmlPageElement, fields: list[FormField]
    ) -> None:
        """Append the field(s) for one ``<select>`` if it submits."""
        name = elem.get_attribute("name")
        if not name:
            return

        options = elem.query_xpath(".//option", "select options", min_count=0)
        # Per HTML spec an option's value is its value attribute when present —
        # including value="" (placeholder "All" options) — and its label text
        # only when the attribute is absent.
        option_values = [self._option_value(opt) for opt in options]

        selected_options = elem.query_xpath(
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
        self, elem: LxmlPageElement, fields: list[FormField]
    ) -> None:
        """Append the field for one ``<textarea>`` if it submits."""
        name = elem.get_attribute("name")
        if not name:
            return
        fields.append(
            FormField(
                name=name, field_type="textarea", value=elem.text_content()
            )
        )

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
        # Check min_count against raw matches (fewer matches than min is
        # already a failure); max_count waits until href-less anchors are
        # filtered below, since only returned links count.
        link_elements, is_xpath = self._query_selector(
            selector, description, min_count, max_count=None
        )

        links: list[Link] = []
        for i, elem in enumerate(link_elements):
            href = elem.get_attribute("href")
            if not href:
                continue  # Skip links without href

            # Resolve URL against base URL
            url = urljoin(self._url, href)
            text = elem.text_content().strip()

            # Create a unique selector for this specific link, positional
            # in the matched element's grammar: an XPath positional
            # predicate for XPath, Playwright's :nth-match() for CSS.
            # The index counts all matched elements (pre-href-filter) so
            # replay selects the same node the parse saw.
            if is_xpath:
                link_selector = f"({selector})[{i + 1}]"
            else:
                link_selector = f":nth-match({selector}, {i + 1})"

            links.append(
                Link(
                    url=url,
                    text=text,
                    selector=link_selector,
                    selector_type="xpath" if is_xpath else "css",
                )
            )

        # Validate the bounds against the links actually returned: a page
        # that swaps real anchors for href-less JS handlers must fail the
        # structural contract, not silently return fewer links.
        raise_if_count_out_of_bounds(
            selector=selector,
            selector_type="xpath" if is_xpath else "css",
            description=description,
            actual_count=len(links),
            min_count=min_count,
            max_count=max_count,
            request_url=self._url,
        )

        return links

    def links(self) -> list[Link]:
        """Discover all links in the element.

        Returns:
            List of all <a> elements with href attributes as Link objects.
        """
        return self.find_links(".//a[@href]", "all links", min_count=0)
