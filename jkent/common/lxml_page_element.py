"""LxmlPageElement: the count-validated PageElement implementation backed by lxml.

This is the single, standard ``PageElement`` implementation used by all
drivers. It wraps a raw lxml ``HtmlElement`` directly (``self._element``) and
provides:

- the high-level ``PageElement`` API scrapers use — ``query(XPath(...))`` /
  ``query(CSS(...))``, ``query_strings``, ``find_form``, ``find_links`` —
  which takes explicit :class:`~jkent.data_types.Selector` values, and
- the low-level, count-validated ``checked_xpath``/``checked_css`` engine
  those methods route through (string selector + expected-count validation;
  scrapers rarely call it directly).

Element results are re-wrapped as ``LxmlPageElement`` so a query on a page
element yields page elements — there is no separate wrapper object and no
re-wrapping pass.

Query recording is driven entirely by the active ``SelectorObserver``
contextvar (see :mod:`jkent.common.selector_observer`); the checked queries
report to it, so this class holds no observer state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, overload
from urllib.parse import urljoin

from lxml import html
from lxml.html import HtmlElement

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperConfigError,
)
from jkent.common.page_element import (
    Form,
    FormField,
    Link,
    PageElement,
)
from jkent.common.selector_observer import get_active_observer
from jkent.contracts import ensure, require
from jkent.data_types import Selector, XPath


class LxmlPageElement(PageElement):
    """PageElement implementation backed by a raw lxml ``HtmlElement``.

    Holds the wrapped element as ``self._element`` and the base URL as
    ``self._request_url`` (used both for error context and as the base for
    resolving relative URLs). The checked queries wrap their element results
    in ``LxmlPageElement``, so nested queries return page elements directly.
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
    ) -> list[LxmlPageElement]: ...

    @require(
        lambda min_count, max_count: (
            min_count >= 0 and (max_count is None or max_count >= min_count)
        ),
        "expected-count bounds form a valid (possibly open) interval",
    )
    @ensure(
        lambda result, min_count, max_count: (
            min_count <= len(result)
            and (max_count is None or len(result) <= max_count)
        ),
        "a returned result list always satisfies the caller's bounds — "
        "out-of-bounds counts raise instead",
    )
    # pyre-ignore[43]: contracts decorate only the implementation, not
    # the @overload stubs — they're identity functions to type checkers.
    def checked_xpath(
        self,
        xpath: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
        *,
        type: type[str] | None = None,
    ) -> list[LxmlPageElement] | list[str]:
        """Execute XPath query with count validation.

        Args:
            xpath: XPath expression to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).
            type: Pass `str` to return only string results (text/attributes).
                If omitted, returns only LxmlPageElements (filtering out
                any string results).

        Returns:
            List of matching results filtered by type. By default returns
            LxmlPageElements; pass type=str for string results.
            Filtering happens before count validation: min/max bounds
            apply to results of the requested type only, so a
            string-returning XPath without type=str counts as 0 elements.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = LxmlPageElement(lxml.html.fromstring(html))
            # Get elements (default)
            cases = tree.checked_xpath("//tr[@class='case']", "cases")
            # Get text/attributes
            hrefs = tree.checked_xpath("//a/@href", "links", type=str)
        """
        try:
            raw = self._element.xpath(xpath)
        except Exception as e:
            # A selector that doesn't parse is a bug in the scraper, not
            # a change in the website — never report it as structural.
            raise ScraperConfigError(
                f"Invalid XPath selector {xpath!r} for "
                f"'{description}' (url: {self._request_url}): {e}"
            ) from e

        # A nodeset XPath (//a/@href) returns a list; a scalar XPath
        # (string()/count()/concat()/normalize-space()/boolean()/…) returns a
        # bare value — a str subclass, float, or bool — not a list. Wrap the
        # scalar so it counts as one result; iterating a bare str would count
        # its characters (a 2+ digit count() would report len(str) results and
        # spuriously trip the count check).
        results = raw if isinstance(raw, list) else [raw]

        if type is str:
            # Return only string results
            typed_results: list[Any] = [
                r for r in results if isinstance(r, str)
            ]
            is_element_query = False
        else:
            # Wrap element results so nested queries return page elements.
            typed_results = [
                LxmlPageElement(r, self._request_url)
                for r in results
                if isinstance(r, HtmlElement)
            ]
            is_element_query = True

        actual_count = len(typed_results)

        # Report to the active observer using the post-filter results, so the
        # recorded match_count matches the count the structural check below
        # enforces. Recording the raw results would make simple_tree() show
        # ✓ for a query that just raised "found 0" — e.g. a string-returning
        # XPath called without type=str, whose string results are filtered out
        # here.
        observer = get_active_observer()
        if observer is not None:
            observer.record_query(
                selector=xpath,
                selector_type="xpath",
                description=description,
                results=typed_results,
                expected_min=min_count,
                expected_max=max_count,
                parent_element=self._element,
            )

        if actual_count < min_count or (
            max_count is not None and actual_count > max_count
        ):
            raise HTMLStructuralAssumptionException(
                selector=xpath,
                selector_type="xpath",
                description=description,
                expected_min=min_count,
                expected_max=max_count,
                actual_count=actual_count,
                request_url=self._request_url,
                is_element_query=is_element_query,
            )
        return typed_results

    @require(
        lambda min_count, max_count: (
            min_count >= 0 and (max_count is None or max_count >= min_count)
        ),
        "expected-count bounds form a valid (possibly open) interval",
    )
    @ensure(
        lambda result, min_count, max_count: (
            min_count <= len(result)
            and (max_count is None or len(result) <= max_count)
        ),
        "a returned result list always satisfies the caller's bounds — "
        "out-of-bounds counts raise instead",
    )
    def checked_css(
        self,
        selector: str,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[LxmlPageElement]:
        """Execute CSS selector query with count validation.

        Args:
            selector: CSS selector expression.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching LxmlPageElements. Each element is wrapped to
            support nested checked queries.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = LxmlPageElement(lxml.html.fromstring(html))
            # Expect exactly 1 case name
            case_name = tree.checked_css("h1.case-name", "case name")
            # Expect at least 5 case divs
            cases = tree.checked_css("div.case", "case divs", min_count=5)
            # Nested queries work
            for case in cases:
                title = case.checked_css("h2.title", "title", min_count=1)
        """
        # Use lxml's built-in cssselect() method
        try:
            results = self._element.cssselect(selector)
        except Exception as e:
            # A selector that doesn't parse is a bug in the scraper, not
            # a change in the website — never report it as structural.
            raise ScraperConfigError(
                f"Invalid CSS selector {selector!r} for "
                f"'{description}' (url: {self._request_url}): {e}"
            ) from e

        # Report to active observer if present
        observer = get_active_observer()
        if observer is not None:
            # Pin the element arm of QueryResults: cssselect yields raw
            # HtmlElements, so without the annotation the checker tries the
            # Sequence[str] arm first and rejects the list.
            css_results: Sequence[HtmlElement] = list(results)
            observer.record_query(
                selector=selector,
                selector_type="css",
                description=description,
                results=css_results,
                expected_min=min_count,
                expected_max=max_count,
                parent_element=self._element,
            )

        actual_count = len(results)
        if actual_count < min_count or (
            max_count is not None and actual_count > max_count
        ):
            raise HTMLStructuralAssumptionException(
                selector=selector,
                selector_type="css",
                description=description,
                expected_min=min_count,
                expected_max=max_count,
                actual_count=actual_count,
                request_url=self._request_url,
            )

        # CSS selectors always return elements (never text/attributes); wrap
        # each so nested queries return page elements.
        return [
            LxmlPageElement(result, self._request_url) for result in results
        ]

    def query(
        self,
        selector: Selector,
        description: str,
        min_count: int = 1,
        max_count: int | None = None,
    ) -> list[LxmlPageElement]:
        """Query elements by selector, dispatching on its grammar.

        Routes to the count-validated ``checked_xpath``/``checked_css`` engine
        off ``selector.grammar``. The caller wraps the selector in
        ``Selector.XPath``/``Selector.CSS``, so the grammar is explicit — no
        prefix heuristic to guess it back, and no bare string can slip in.

        Args:
            selector: ``Selector.XPath``/``Selector.CSS`` to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching LxmlPageElement instances.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.
        """
        if selector.grammar == "xpath":
            return self.checked_xpath(
                selector.value, description, min_count, max_count
            )
        return self.checked_css(
            selector.value, description, min_count, max_count
        )

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
        return self.checked_xpath(
            selector.value, description, min_count, max_count, type=str
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
    def _option_value(option: LxmlPageElement) -> str:
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
        self, elem: LxmlPageElement, fields: list[FormField]
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
        self, elem: LxmlPageElement, fields: list[FormField]
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
        self, elem: LxmlPageElement, fields: list[FormField]
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
        if len(links) < min_count or (
            max_count is not None and len(links) > max_count
        ):
            raise HTMLStructuralAssumptionException(
                selector=selector.value,
                selector_type=selector.grammar,
                description=description,
                expected_min=min_count,
                expected_max=max_count,
                actual_count=len(links),
                request_url=self._request_url,
            )

        return links

    def __getattr__(self, name: str) -> Any:
        """Delegate all other attributes to the wrapped element.

        This lets LxmlPageElement stand in for the raw HtmlElement for any
        attribute the explicit methods above don't cover.
        """
        return getattr(self._element, name)
