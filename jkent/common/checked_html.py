"""Checked HTML element wrapper for safe XPath/CSS querying.

This module provides CheckedHtmlElement, a wrapper around lxml.html.HtmlElement
that validates selector results against expected counts. This helps catch
structural assumption violations early.
"""

from __future__ import annotations

from typing import Any, overload

from lxml.html import HtmlElement

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperConfigError,
)
from jkent.common.selector_observer import (
    get_active_observer,
)
from jkent.contracts import ensure, require


def raise_if_count_out_of_bounds(
    *,
    selector: str,
    selector_type: str,
    description: str,
    actual_count: int,
    min_count: int,
    max_count: int | None,
    request_url: str,
    is_element_query: bool = True,
) -> None:
    """Raise HTMLStructuralAssumptionException if ``actual_count`` is outside
    the ``[min_count, max_count]`` interval (``max_count=None`` is unbounded).

    Shared by :meth:`CheckedHtmlElement.checked_xpath`/``checked_css`` and
    :meth:`LxmlPageElement.find_links` so the bounds rule and the exception's
    field set live in one place.
    """
    if actual_count < min_count or (
        max_count is not None and actual_count > max_count
    ):
        raise HTMLStructuralAssumptionException(
            selector=selector,
            selector_type=selector_type,
            description=description,
            expected_min=min_count,
            expected_max=max_count,
            actual_count=actual_count,
            request_url=request_url,
            is_element_query=is_element_query,
        )


class CheckedHtmlElement:
    """Wrapper around HtmlElement with validated selectors.

    This class wraps an lxml HtmlElement and provides checked_xpath() and
    checked_css() methods that validate the number of results against expected
    min/max counts. If the actual count doesn't match expectations, it raises
    HTMLStructuralAssumptionException with clear error context.

    This helps catch website structure changes early and provides clear error
    messages for debugging.
    """

    def __init__(self, element: HtmlElement, request_url: str = "") -> None:
        """Initialize the checked element wrapper.

        Args:
            element: The lxml HtmlElement to wrap.
            request_url: Optional URL for error context.
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
    ) -> list[CheckedHtmlElement]: ...

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
    ) -> list[CheckedHtmlElement] | list[str]:
        """Execute XPath query with count validation.

        Args:
            xpath: XPath expression to execute.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).
            type: Pass `str` to return only string results (text/attributes).
                If omitted, returns only CheckedHtmlElements (filtering out
                any string results).

        Returns:
            List of matching results filtered by type. By default returns
            CheckedHtmlElements; pass type=str for string results.
            Filtering happens before count validation: min/max bounds
            apply to results of the requested type only, so a
            string-returning XPath without type=str counts as 0 elements.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = CheckedHtmlElement(lxml.html.fromstring(html))
            # Get elements (default)
            cases = tree.checked_xpath("//tr[@class='case']", "cases")
            # Get text/attributes
            hrefs = tree.checked_xpath("//a/@href", "links", type=str)
        """
        try:
            results = self._element.xpath(xpath)
        except Exception as e:
            # A selector that doesn't parse is a bug in the scraper, not
            # a change in the website — never report it as structural.
            raise ScraperConfigError(
                f"Invalid XPath selector {xpath!r} for "
                f"'{description}' (url: {self._request_url}): {e}"
            ) from e

        if type is str:
            # Return only string results
            typed_results: list[Any] = [
                r for r in results if isinstance(r, str)
            ]
            is_element_query = False
        else:
            # Return only element results, wrapped in CheckedHtmlElement
            typed_results = [
                CheckedHtmlElement(r, self._request_url)
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

        raise_if_count_out_of_bounds(
            selector=xpath,
            selector_type="xpath",
            description=description,
            actual_count=actual_count,
            min_count=min_count,
            max_count=max_count,
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
    ) -> list[CheckedHtmlElement]:
        """Execute CSS selector query with count validation.

        Args:
            selector: CSS selector expression.
            description: Human-readable description of what's being selected.
            min_count: Minimum number of elements expected (default: 1).
            max_count: Maximum number of elements expected (None = unlimited).

        Returns:
            List of matching CheckedHtmlElements. Each element is wrapped to support
            nested checked queries.

        Raises:
            HTMLStructuralAssumptionException: If count doesn't match expectations.

        Example::

            tree = CheckedHtmlElement(lxml.html.fromstring(html))
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
            observer.record_query(
                selector=selector,
                selector_type="css",
                description=description,
                results=list(results),  # Convert to list for consistency
                expected_min=min_count,
                expected_max=max_count,
                parent_element=self._element,
            )

        # Validate count
        actual_count = len(results)
        raise_if_count_out_of_bounds(
            selector=selector,
            selector_type="css",
            description=description,
            actual_count=actual_count,
            min_count=min_count,
            max_count=max_count,
            request_url=self._request_url,
        )

        # Wrap results in CheckedHtmlElement for nested queries
        # CSS selectors always return elements (never text/attributes)
        wrapped_results = [
            CheckedHtmlElement(result, self._request_url) for result in results
        ]

        return wrapped_results  # type: ignore[return-value]

    def __getattr__(self, name: str):
        """Delegate all other attributes to the wrapped element.

        This allows CheckedHtmlElement to be used as a drop-in replacement for
        HtmlElement, while adding the checked methods.
        """
        return getattr(self._element, name)
