"""Tests for SelectorObserver.

Tests recording logic, deduplication, output formats, and absolute selector composition.
"""

import pytest
from lxml import html

from jkent.common.decorators import step
from jkent.common.exceptions import HTMLStructuralAssumptionException
from jkent.common.lxml_page_element import LxmlPageElement
from jkent.common.selector_observer import (
    SelectorObserver,
    SelectorQuery,
    get_active_observer,
)
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    Selector,
)


@pytest.fixture
def simple_html():
    """Simple HTML document for testing."""
    return """
    <html>
    <body>
        <div id="main">
            <table>
                <tr class="row"><td>Cell 1</td><td>Cell 2</td></tr>
                <tr class="row"><td>Cell 3</td><td>Cell 4</td></tr>
                <tr class="row"><td>Cell 5</td><td>Cell 6</td></tr>
            </table>
        </div>
    </body>
    </html>
    """


def test_context_manager_sets_active_observer():
    """Entering the context makes the observer the active one."""
    assert get_active_observer() is None

    with SelectorObserver() as observer:
        assert get_active_observer() is observer

    assert get_active_observer() is None


def test_nested_context_managers_restore_previous():
    """Nested observers take over, and the outer is restored on exit."""
    with SelectorObserver() as outer:
        assert get_active_observer() is outer

        with SelectorObserver() as inner:
            assert get_active_observer() is inner

        assert get_active_observer() is outer

    assert get_active_observer() is None


def test_page_element_reports_to_active_observer(simple_html):
    """LxmlPageElement records queries to the active observer."""
    tree = LxmlPageElement(html.fromstring(simple_html), "http://example.com")

    with SelectorObserver() as observer:
        rows = tree.checked_xpath("//tr[@class='row']", "rows")
        assert len(rows) == 3

    assert len(observer.queries) == 1
    assert observer.queries[0].selector == "//tr[@class='row']"
    assert observer.queries[0].match_count == 3


def test_no_active_observer_outside_context(simple_html):
    """LxmlPageElement still works when no observer is active."""
    assert get_active_observer() is None

    tree = LxmlPageElement(html.fromstring(simple_html), "http://example.com")
    rows = tree.checked_xpath("//tr[@class='row']", "rows")
    assert len(rows) == 3

    assert get_active_observer() is None


def test_record_simple_query(simple_html):
    """Observer should record a simple query."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    results = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=results,
        expected_min=1,
        expected_max=None,
    )

    assert len(observer.queries) == 1
    query = observer.queries[0]
    assert query.selector == "//tr[@class='row']"
    assert query.selector_type == "xpath"
    assert query.description == "table rows"
    assert query.match_count == 3
    assert query.expected_min == 1
    assert query.expected_max is None


def test_record_nested_queries(simple_html):
    """Observer should record nested queries with parent-child relationships."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    # First query: find rows
    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )

    # Second query: find cells within first row
    cells = rows[0].xpath(".//td")
    observer.record_query(
        selector=".//td",
        selector_type="xpath",
        description="cells",
        results=cells,
        expected_min=1,
        expected_max=None,
        parent_element=rows[0],
    )

    # Should have one top-level query
    assert len(observer.queries) == 1

    # Top-level query should have one child
    parent_query = observer.queries[0]
    assert len(parent_query.children) == 1

    # Child query should reference parent
    child_query = parent_query.children[0]
    assert child_query.selector == ".//td"
    assert child_query.parent == parent_query
    assert child_query.parent_element_id == parent_query.element_id


def test_deduplication_same_selector(simple_html):
    """Observer should deduplicate repeated queries with the same parent."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    # Find rows
    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )

    # Query cells for each row (same selector, same parent type)
    for row in rows:
        cells = row.xpath(".//td")
        observer.record_query(
            selector=".//td",
            selector_type="xpath",
            description="cells",
            results=cells,
            expected_min=1,
            expected_max=None,
            parent_element=row,
        )

    # Should have one top-level query
    assert len(observer.queries) == 1

    # Should have ONE child query (deduplicated)
    parent_query = observer.queries[0]
    assert len(parent_query.children) == 1

    # Child query should have aggregated match count (2 cells * 3 rows = 6)
    child_query = parent_query.children[0]
    assert child_query.match_count == 6


def test_sample_extraction(simple_html):
    """Observer should extract sample content from results."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver(max_samples=2)

    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )

    query = observer.queries[0]
    assert len(query.sample_elements) == 2  # max_samples
    # First row contains "Cell 1Cell 2" (normalized whitespace)
    assert "Cell 1" in query.sample_elements[0]
    assert "Cell 2" in query.sample_elements[0]


def test_simple_tree_output(simple_html):
    """Observer should generate human-readable tree output."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )

    cells = rows[0].xpath(".//td")
    observer.record_query(
        selector=".//td",
        selector_type="xpath",
        description="cells",
        results=cells,
        expected_min=1,
        expected_max=None,
        parent_element=rows[0],
    )

    tree = observer.simple_tree()

    # Should contain parent query
    assert "//tr[@class='row']" in tree
    assert "table rows" in tree
    assert "✓" in tree  # Success indicator

    # Should contain child query (indented)
    assert ".//td" in tree
    assert "cells" in tree


def test_simple_tree_failure_indicator(simple_html):
    """Observer should show ✗ for failed queries."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    # Query that finds nothing
    results = doc.xpath("//nonexistent")
    observer.record_query(
        selector="//nonexistent",
        selector_type="xpath",
        description="missing elements",
        results=results,
        expected_min=1,
        expected_max=None,
    )

    tree = observer.simple_tree()

    assert "✗" in tree
    assert "0 matches" in tree
    assert "expected 1+" in tree


def test_json_output(simple_html):
    """Observer should generate JSON output."""
    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )

    json_output = observer.json()

    assert isinstance(json_output, list)
    assert len(json_output) == 1

    query_dict = json_output[0]
    assert query_dict["selector"] == "//tr[@class='row']"
    assert query_dict["selector_type"] == "xpath"
    assert query_dict["description"] == "table rows"
    assert query_dict["match_count"] == 3
    assert query_dict["element_id"] is not None


def test_json_output_nested_has_no_parent_cycle(simple_html):
    """json() must serialize a scoped query tree without hitting the parent
    backref cycle (regression: children carry a ``parent`` ref pointing back
    up, so serialization has to drop ``parent`` at every depth, not just the
    top)."""
    import json as _json

    doc = html.fromstring(simple_html)
    observer = SelectorObserver()

    rows = doc.xpath("//tr[@class='row']")
    observer.record_query(
        selector="//tr[@class='row']",
        selector_type="xpath",
        description="table rows",
        results=rows,
        expected_min=1,
        expected_max=None,
    )
    # A scoped query on a matched row gives its query a non-None parent and
    # appends it to the parent's children -- the shape that used to cycle.
    cells = rows[0].xpath(".//td")
    observer.record_query(
        selector=".//td",
        selector_type="xpath",
        description="cells",
        results=cells,
        expected_min=1,
        expected_max=None,
        parent_element=rows[0],
    )

    json_output = observer.json()

    # Serializable end to end, and no "parent" key leaks at any level.
    assert _json.dumps(json_output)
    top = json_output[0]
    assert "parent" not in top
    child = top["children"][0]
    assert "parent" not in child
    assert child["selector"] == ".//td"


def test_compose_absolute_selector_simple():
    """compose_absolute_selector should return selector for root query."""
    query = SelectorQuery(
        selector="//table",
        selector_type="xpath",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=None,
    )

    observer = SelectorObserver()
    absolute = observer.compose_absolute_selector(query)

    assert absolute == "//table"


def test_compose_absolute_selector_nested_xpath():
    """compose_absolute_selector should compose nested XPath selectors."""
    parent = SelectorQuery(
        selector="//table",
        selector_type="xpath",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=None,
    )

    child = SelectorQuery(
        selector=".//tr",
        selector_type="xpath",
        description="rows",
        match_count=3,
        expected_min=1,
        expected_max=None,
        parent=parent,
    )

    observer = SelectorObserver()
    absolute = observer.compose_absolute_selector(child)

    # Should strip "./" and compose
    assert absolute == "//table//tr"


def test_compose_absolute_selector_mixed_types():
    """compose_absolute_selector should return None for mixed selector types."""
    parent = SelectorQuery(
        selector="table",
        selector_type="css",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=None,
    )

    child = SelectorQuery(
        selector=".//tr",
        selector_type="xpath",
        description="rows",
        match_count=3,
        expected_min=1,
        expected_max=None,
        parent=parent,
    )

    observer = SelectorObserver()
    absolute = observer.compose_absolute_selector(child)

    # Mixed types - cannot compose
    assert absolute is None


def test_compose_absolute_selector_css():
    """compose_absolute_selector should compose CSS selectors with space."""
    parent = SelectorQuery(
        selector="div.container",
        selector_type="css",
        description="container",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=None,
    )

    child = SelectorQuery(
        selector="table",
        selector_type="css",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=parent,
    )

    observer = SelectorObserver()
    absolute = observer.compose_absolute_selector(child)

    assert absolute == "div.container table"


def test_compose_absolute_selector_three_levels():
    """compose_absolute_selector should handle deep nesting."""
    level1 = SelectorQuery(
        selector="//div",
        selector_type="xpath",
        description="div",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=None,
    )

    level2 = SelectorQuery(
        selector=".//table",
        selector_type="xpath",
        description="table",
        match_count=1,
        expected_min=1,
        expected_max=None,
        parent=level1,
    )

    level3 = SelectorQuery(
        selector=".//tr",
        selector_type="xpath",
        description="rows",
        match_count=3,
        expected_min=1,
        expected_max=None,
        parent=level2,
    )

    observer = SelectorObserver()
    absolute = observer.compose_absolute_selector(level3)

    assert absolute == "//div//table//tr"


def test_max_samples_limit():
    """Observer should respect max_samples limit."""
    doc = html.fromstring(
        "<div><p>1</p><p>2</p><p>3</p><p>4</p><p>5</p></div>"
    )
    observer = SelectorObserver(max_samples=2)

    results = doc.xpath("//p")
    observer.record_query(
        selector="//p",
        selector_type="xpath",
        description="paragraphs",
        results=results,
        expected_min=1,
        expected_max=None,
    )

    query = observer.queries[0]
    assert len(query.sample_elements) == 2


def test_max_sample_length():
    """Observer should truncate long sample text."""
    long_text = "A" * 200
    doc = html.fromstring(f"<div><p>{long_text}</p></div>")
    observer = SelectorObserver(max_sample_length=50)

    results = doc.xpath("//p")
    observer.record_query(
        selector="//p",
        selector_type="xpath",
        description="paragraph",
        results=results,
        expected_min=1,
        expected_max=None,
    )

    query = observer.queries[0]
    assert len(query.sample_elements[0]) <= 53  # 50 + "..."
    assert query.sample_elements[0].endswith("...")


# ── per-execution observer through @step ───────────────────────────


def _step_response(url: str, html_content: str):
    request = Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse",
    )
    return Response(
        status_code=200,
        headers={},
        content=html_content.encode("utf-8"),
        text=html_content,
        url=url,
        request=request,
    )


def test_interleaved_step_executions_keep_their_own_observers():
    """Each step execution gets its own observer, on its own Response.

    The observer used to be stored on the function's shared StepMetadata,
    so two in-flight executions of the same step clobbered each other and
    the driver could read autowait/debug telemetry from the wrong request.
    """

    class TwoPageScraper(BaseScraper[dict]):
        @step
        def parse(self, page, response):
            page.query(Selector.XPath("//h1"), "title", min_count=0)
            yield ParsedData({"url": response.url})

    scraper = TwoPageScraper()
    response_a = _step_response(
        "https://example.com/a", "<html><h1>A</h1></html>"
    )
    response_b = _step_response(
        "https://example.com/b", "<html><h1>B</h1></html>"
    )

    # Interleave two executions of the same step: pump A, pump B, then
    # finish both. With shared metadata, B's observer overwrites A's.
    generator_a = scraper.parse(response_a)
    generator_b = scraper.parse(response_b)
    next(generator_a)
    next(generator_b)
    list(generator_a)
    list(generator_b)

    assert response_a.observer is not None
    assert response_b.observer is not None
    assert response_a.observer is not response_b.observer
    # Each observer saw exactly its own execution's queries.
    assert len(response_a.observer.queries) == 1
    assert len(response_b.observer.queries) == 1


def test_step_without_page_injection_leaves_observer_unset():
    """Steps that never parse a page attach no observer."""

    class TextScraper(BaseScraper[dict]):
        @step
        def parse(self, text):
            yield ParsedData({"text": text})

    scraper = TextScraper()
    response = _step_response("https://example.com/a", "plain")
    list(scraper.parse(response))

    assert response.observer is None


def test_simple_tree_matches_raised_count_for_missing_type_str():
    """simple_tree() reflects the post-filter count the structural check uses.

    A string-returning XPath called without ``type=str`` has its string
    results filtered out, so the structural check raises "found 0". The
    observer must record that same 0 — not the raw pre-filter string count —
    or simple_tree() would show ✓ for the very query that just failed.
    """
    tree = LxmlPageElement(
        html.fromstring('<div><a href="/a">A</a><a href="/b">B</a></div>')
    )

    with (
        SelectorObserver() as obs,
        pytest.raises(HTMLStructuralAssumptionException),
    ):
        tree.checked_xpath("//a/@href", "case links")

    rendered = obs.simple_tree()
    assert "✗" in rendered
    assert "✓" not in rendered
    # Recorded match is the post-filter 0, agreeing with the raised error —
    # not the 2 raw @href strings the XPath returned.
    assert "0 match" in rendered
