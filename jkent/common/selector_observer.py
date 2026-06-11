"""SelectorObserver for debugging selector queries.

Records XPath/CSS queries for debugging. Can be used either by direct
injection into a PageElement, or as a context manager whose active instance
LxmlPageElement picks up via ``get_active_observer()``.

The observer records query trees, deduplicates repeated selectors, captures
sample content, and provides human-readable and JSON output formats.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pydantic_core

if TYPE_CHECKING:
    from lxml.html import HtmlElement

    from jkent.common.lxml_page_element import LxmlPageElement

    # A query yields either text/attribute strings or elements — never a
    # mix within one query. Element queries arrive either wrapped
    # (LxmlPageElement, from checked_xpath) or raw (HtmlElement, from
    # checked_css), so the element side is the union of both. Sequence (not
    # list) keeps the element side covariant, so a list[HtmlElement] from
    # cssselect is accepted.
    Element: TypeAlias = "LxmlPageElement | HtmlElement"
    QueryResults: TypeAlias = "Sequence[str] | Sequence[Element]"
    # The per-item view of QueryResults: a single sequence whose elements
    # may each be a str or an element. Slicing a QueryResults union yields
    # this joined form, and the per-item consumers (_extract_samples) don't
    # rely on the homogeneity invariant, so they take this wider type.
    MixedResults: TypeAlias = "Sequence[str | Element]"

_active_observer: contextvars.ContextVar[SelectorObserver | None] = (  # type: ignore
    contextvars.ContextVar["SelectorObserver | None"](  # type: ignore
        "selector_observer", default=None
    )
)


@dataclass
class SelectorQuery:
    """A single XPath or CSS selector query.

    Attributes:
        selector: The XPath or CSS selector string.
        selector_type: "xpath" or "css".
        description: Human-readable description.
        match_count: Number of elements matched.
        expected_min: Minimum expected count.
        expected_max: Maximum expected count (None = unlimited).
        sample_elements: Sample content from matched elements.
        children: Nested child queries.
        element_id: Unique ID for highlighting in UI.
        parent_element_id: ID of parent query (for scoped highlights).
        parent: Reference to parent SelectorQuery for tree navigation.
    """

    selector: str
    selector_type: str  # "xpath" or "css"
    description: str
    match_count: int
    expected_min: int
    expected_max: int | None
    sample_elements: list[str] = field(default_factory=list)
    children: list[SelectorQuery] = field(default_factory=list)
    element_id: str | None = None
    parent_element_id: str | None = None
    parent: SelectorQuery | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of the query.
        """
        return pydantic_core.to_jsonable_python(self, exclude={"parent"})


class SelectorObserver:
    """Observer that collects selector query information.

    Used as a context manager; any LxmlPageElement queried inside the
    ``with`` block picks the observer up via ``get_active_observer()`` and
    reports its queries to it::

        with SelectorObserver() as observer:
            tree = LxmlPageElement(lxml_html.fromstring(content), url)
            rows = tree.checked_xpath("//tr", "table rows", min_count=1)

        print(observer.simple_tree())  # Human-readable tree
        print(observer.json())  # JSON for UI highlighting

    Deduplication:

    When the same selector is used multiple times with the same parent query
    (e.g., iterating over rows and selecting the same column from each),
    the observer deduplicates these into a single query entry. Match counts
    and sample elements are aggregated.
    """

    def __init__(self, max_sample_length: int = 100, max_samples: int = 3):
        """Initialize the observer.

        Args:
            max_sample_length: Maximum characters per sample element.
            max_samples: Maximum number of sample elements to capture.
        """
        self.max_sample_length = max_sample_length
        self.max_samples = max_samples
        self.queries: list[SelectorQuery] = []
        self._element_counter: int = 0
        # Maps element id() to the query that produced it
        self._element_to_query: dict[int, SelectorQuery] = {}
        # Maps (parent_element_id, selector) to existing SelectorQuery for deduplication
        self._dedup_index: dict[tuple[str | None, str], SelectorQuery] = {}
        self._token: contextvars.Token[SelectorObserver | None] | None = None

    def __enter__(self) -> SelectorObserver:
        """Activate this observer for any LxmlPageElement in this context."""
        self._token = _active_observer.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Restore the previously-active observer, if any."""
        if self._token is not None:
            _active_observer.reset(self._token)
            self._token = None

    def record_query(
        self,
        selector: str,
        selector_type: str,
        description: str,
        results: QueryResults,
        expected_min: int,
        expected_max: int | None,
        parent_element: HtmlElement | None = None,
    ) -> None:
        """Record a selector query and its results.

        Args:
            selector: The XPath or CSS selector string.
            selector_type: "xpath" or "css".
            description: Human-readable description from query method.
            results: The elements/values returned by the query.
            expected_min: Minimum expected count.
            expected_max: Maximum expected count (None = unlimited).
            parent_element: The element the query was executed on.

        Note:
            Queries with the same (parent_element_id, selector) are deduplicated.
            Match counts and samples are aggregated into the existing query.
        """
        # Find parent query if this query was executed on a child element
        parent_query_id: str | None = None
        parent_query: SelectorQuery | None = None
        if parent_element is not None:
            parent_elem_id = id(parent_element)
            parent_query = self._element_to_query.get(parent_elem_id)
            if parent_query is not None:
                parent_query_id = parent_query.element_id

        # Check for existing query with same parent + selector (deduplication)
        dedup_key = (parent_query_id, selector)
        existing_query = self._dedup_index.get(dedup_key)

        if existing_query is not None:
            # Aggregate into existing query
            existing_query.match_count += len(results)

            # Add more samples if we haven't hit the limit
            samples_needed = self.max_samples - len(
                existing_query.sample_elements
            )
            if samples_needed > 0:
                new_samples = self._extract_samples(results[:samples_needed])
                existing_query.sample_elements.extend(new_samples)

            # Track which elements came from this query (use existing query)
            for result in results:
                elem = self._unwrap_element(result)
                if elem is not None:
                    self._element_to_query[id(elem)] = existing_query

            return

        # Generate sample content from results
        samples = self._extract_samples(results[: self.max_samples])

        # Generate unique element ID for highlighting
        self._element_counter += 1
        element_id = f"selector_match_{self._element_counter}"

        query = SelectorQuery(
            selector=selector,
            selector_type=selector_type,
            description=description,
            match_count=len(results),
            expected_min=expected_min,
            expected_max=expected_max,
            sample_elements=samples,
            element_id=element_id,
            parent_element_id=parent_query_id,
            parent=parent_query,
        )

        # Register in dedup index
        self._dedup_index[dedup_key] = query

        # Track which elements came from this query (for future child queries)
        for result in results:
            elem = self._unwrap_element(result)
            if elem is not None:
                self._element_to_query[id(elem)] = query

        # Add to current context (nested or top-level)
        if parent_query is not None:
            parent_query.children.append(query)
        else:
            self.queries.append(query)

    def _unwrap_element(self, result: str | Element) -> HtmlElement | None:
        """Unwrap a result to the underlying HtmlElement.

        Args:
            result: A query result: a string, a wrapped LxmlPageElement,
                or a raw HtmlElement.

        Returns:
            The underlying HtmlElement, or None for a string result.
        """
        match result:
            case str():
                return None
            case _:
                # LxmlPageElement exposes the raw element via _element;
                # a raw HtmlElement is already unwrapped.
                element = getattr(result, "_element", result)
                return cast("HtmlElement", element)

    def _extract_samples(self, results: MixedResults) -> list[str]:
        """Extract sample text content from results.

        Args:
            results: List of query results.

        Returns:
            List of sample text strings.
        """
        samples = []
        for result in results:
            match result:
                case str():
                    text = result
                case _:
                    # LxmlPageElement defines text_content() directly; a raw
                    # HtmlElement also has it.
                    text = result.text_content()

            # Normalize whitespace and truncate
            text = " ".join(text.split())
            if len(text) > self.max_sample_length:
                text = text[: self.max_sample_length] + "..."
            samples.append(text)
        return samples

    def simple_tree(self, indent: int = 0) -> str:
        """Generate a human-readable tree representation.

        Args:
            indent: Initial indentation level (internal use).

        Returns:
            Formatted string showing query hierarchy with match counts.

        Example output::

            - //div[@id='mainContent']/table "Main Table" ✓ (1 match)
              - //tr "Main Table Rows" ✓ (5 matches)
                - (//td)[2] "Important Column" ✗ (0 matches, expected 1+)
        """
        lines = []
        for query in self.queries:
            lines.extend(self._format_query(query, indent))
        return "\n".join(lines)

    def _format_query(self, query: SelectorQuery, indent: int) -> list[str]:
        """Format a single query and its children.

        Args:
            query: The query to format.
            indent: Indentation level.

        Returns:
            List of formatted lines.
        """
        prefix = "  " * indent + "- "

        # Status indicator
        if query.match_count >= query.expected_min:
            if (
                query.expected_max is None
                or query.match_count <= query.expected_max
            ):
                status = "✓"
            else:
                status = "✗"
        else:
            status = "✗"

        # Match count display
        match_text = f"{query.match_count} match" + (
            "es" if query.match_count != 1 else ""
        )
        if status == "✗":
            if query.match_count < query.expected_min:
                match_text += f", expected {query.expected_min}+"
            elif (
                query.expected_max is not None
                and query.match_count > query.expected_max
            ):
                # `is not None`, not truthiness: expected_max=0 ("must not be
                # present") is a legitimate bound that would be skipped by a
                # falsy check, matching the status logic above.
                match_text += f", expected max {query.expected_max}"

        line = f'{prefix}{query.selector} "{query.description}" {status} ({match_text})'
        lines = [line]

        # Add sample content preview if available
        if query.sample_elements and query.match_count > 0:
            sample_preview = query.sample_elements[0]
            if sample_preview:
                sample_line = "  " * (indent + 1) + f'→ "{sample_preview}"'
                lines.append(sample_line)

        # Recurse for children
        for child in query.children:
            lines.extend(self._format_query(child, indent + 1))

        return lines

    def json(self) -> list[dict[str, Any]]:
        """Generate JSON representation for UI highlighting.

        Returns:
            List of query dictionaries suitable for JavaScript processing.
        """
        return [q.to_dict() for q in self.queries]

    def compose_absolute_selector(self, query: SelectorQuery) -> str | None:
        """Compose an absolute selector from a query's parent chain.

        Used by the Playwright driver's autowait mechanism to construct a
        complete selector from relative queries.

        Args:
            query: The query to compose an absolute selector for.

        Returns:
            An absolute selector string, or None if the chain contains
            mixed selector types (XPath + CSS).

        Example::

            # Root query: //table
            # Child query: .//tr
            # Result: //table//tr
        """
        if query.parent is None:
            # Already absolute (no parent)
            return query.selector

        # Walk up the parent chain to collect selectors
        selectors: list[tuple[str, str]] = []  # (selector_type, selector)
        current: SelectorQuery | None = query

        while current is not None:
            selectors.append((current.selector_type, current.selector))
            current = current.parent

        # Reverse to get root-to-leaf order
        selectors.reverse()

        # Check for mixed selector types
        selector_types = {sel_type for sel_type, _ in selectors}
        if len(selector_types) > 1:
            # Mixed types - can't compose
            return None

        selector_type = selectors[0][0]

        if selector_type == "xpath":
            # Compose XPath selectors
            # The first selector is the root (absolute), subsequent ones are relative
            result = selectors[0][1]  # Root selector

            for i in range(1, len(selectors)):
                _, sel = selectors[i]
                # Strip leading "./" or "." from relative selectors
                if sel.startswith(".//"):
                    # ".//tr" becomes "//tr" - descendant
                    result += sel[1:]  # Keep the "//" part
                elif sel.startswith("./"):
                    # "./tr" becomes "/tr" - child
                    result += sel[1:]
                elif sel.startswith("."):
                    # Rare case, just strip the dot
                    result += sel[1:]
                else:
                    # Not relative - join with //
                    result += "//" + sel

            return result

        elif selector_type == "css":
            # Compose CSS selectors: join with a space (descendant combinator).
            return " ".join(sel for _, sel in selectors)

        return None


def get_active_observer() -> SelectorObserver | None:
    """Return the SelectorObserver active in the current context, if any."""
    return _active_observer.get()
