"""Tests for PageElement implementation.

Tests checked-query behavior, observer integration, form and link handling.
"""

import pytest
from lxml import html

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
)
from jkent.common.page_element import PageElement, ViaLink
from jkent.common.selector_observer import (
    SelectorObserver,
)
from jkent.data_types import CSS, Selector, XPath


@pytest.fixture
def simple_page() -> PageElement:
    """Simple HTML page for testing."""
    html_content = """
    <html>
    <body>
        <div id="main">
            <h1>Test Page</h1>
            <table>
                <tr class="row"><td>Cell 1</td><td>Cell 2</td></tr>
                <tr class="row"><td>Cell 3</td><td>Cell 4</td></tr>
            </table>
        </div>
    </body>
    </html>
    """
    doc = html.fromstring(html_content)
    return PageElement(doc, "https://example.com/page")


@pytest.fixture
def form_page() -> PageElement:
    """HTML page with a form for testing."""
    html_content = """
    <html>
    <body>
        <form id="search" action="/search" method="POST">
            <input type="text" name="query" value="" />
            <input type="hidden" name="token" value="abc123" />
            <select name="category">
                <option value="all">All</option>
                <option value="news" selected>News</option>
                <option value="blog">Blog</option>
            </select>
            <textarea name="description">Default text</textarea>
            <button type="submit">Search</button>
        </form>
    </body>
    </html>
    """
    doc = html.fromstring(html_content)
    return PageElement(doc, "https://example.com/")


@pytest.fixture
def links_page() -> PageElement:
    """HTML page with links for testing."""
    html_content = """
    <html>
    <body>
        <nav>
            <a href="/page1" class="nav-link">Page 1</a>
            <a href="/page2" class="nav-link">Page 2</a>
            <a href="https://external.com/page3">External</a>
        </nav>
        <div>
            <a>No href</a>
        </div>
    </body>
    </html>
    """
    doc = html.fromstring(html_content)
    return PageElement(doc, "https://example.com/")


def test_query_xpath_returns_page_elements(simple_page: PageElement):
    """query should return PageElement instances."""
    rows = simple_page.query(Selector.XPath("//tr"), "rows", min_count=2)

    # Each result should be PageElement
    for row in rows:
        assert isinstance(row, PageElement)
        # Should be able to query nested elements
        cells = row.query(Selector.XPath(".//td"), "cells")
        assert len(cells) == 2


def test_query_xpath_strings(simple_page: PageElement):
    """query_strings should return string values."""
    cell_texts = simple_page.query_strings(
        Selector.XPath("//td/text()"), "cell texts", min_count=4, max_count=4
    )

    assert len(cell_texts) == 4
    assert all(isinstance(text, str) for text in cell_texts)
    assert "Cell 1" in cell_texts


def test_query_strings_scalar_counts_as_one_result():
    """A scalar string() XPath is one result, not len(str) characters.

    lxml returns a bare _ElementUnicodeResult (a str subclass) for a scalar
    XPath, not a list. Iterating it would count characters, so a 2+ digit
    count() (10 preceding tables here) would report actual_count=2 and trip
    the max_count check. It must count as a single result of value "10".
    """
    markup = "<table class='t'></table>" * 10 + "<td class='c'>x</td>"
    cell = PageElement(html.fromstring(markup)).query(
        Selector.XPath("//td[@class='c']"), "cell", min_count=1
    )[0]

    result = cell.query_strings(
        Selector.XPath("string(count(preceding::table[@class='t']))"),
        "preceding count",
        min_count=0,
        max_count=1,
    )

    assert result == ["10"]


def test_query_strings_scalar_number_and_bool_count_as_one():
    """count()/boolean() (float/bool, not str) each count as one result.

    Wrapping the bare scalar before filtering means a non-str scalar is a
    single result the isinstance(str) filter drops — actual_count=0 — rather
    than iterating a non-iterable or miscounting.
    """
    page = PageElement(html.fromstring("<div><p>a</p><p>b</p></div>"))

    # count(...) returns a float; boolean(...) returns a bool. Neither is a
    # str, so with type=str both filter down to zero string results.
    for xpath in ("count(//p)", "boolean(//p)"):
        assert (
            page.query_strings(
                Selector.XPath(xpath), "scalar", min_count=0, max_count=0
            )
            == []
        )


def test_inner_html(simple_page: PageElement):
    """inner_html should return inner HTML content."""
    div = simple_page.query(Selector.XPath("//div[@id='main']"), "main div")[0]

    inner = div.inner_html()

    assert "<h1>Test Page</h1>" in inner
    assert "<table>" in inner


def test_inner_html_preserves_leading_text():
    """inner_html keeps text that precedes the first child element."""
    doc = html.fromstring("<td>Case No. <a href='/c/1'>123</a></td>")
    cell = PageElement(doc, "https://example.com/")

    inner = cell.inner_html()

    assert inner.startswith("Case No. ")
    assert '<a href="/c/1">123</a>' in inner


def test_child_queries_record_to_active_observer():
    """Queries on child elements record to the active observer.

    Recording flows through the get_active_observer() contextvar, not
    through any observer held on the PageElement, so a query on a child
    element returned by query_xpath must still be captured.
    """
    doc = html.fromstring("<div><section><p>Test</p></section></div>")
    page = PageElement(doc, "https://example.com/")

    with SelectorObserver() as observer:
        sections = page.query(Selector.XPath("//section"), "sections")
        sections[0].query(Selector.XPath(".//p"), "paragraphs")

    # The parent query records at the top level, and the query made on the
    # child element nests beneath it — proving the child still reports
    # through the active observer with no observer held on the element.
    assert [q.selector for q in observer.queries] == ["//section"]
    assert [c.selector for c in observer.queries[0].children] == [".//p"]


def test_find_form_by_xpath(form_page: PageElement):
    """find_form should find form by XPath selector."""
    form = form_page.find_form(
        Selector.XPath("//form[@id='search']"), "search form"
    )

    assert form.action == "https://example.com/search"
    assert form.method == "POST"
    assert form.selector.value == "//form[@id='search']"
    assert len(form.fields) == 4  # query, token, category, description


def test_find_form_by_css(form_page: PageElement):
    """find_form should find form by CSS selector."""
    form = form_page.find_form(Selector.CSS("form#search"), "search form")

    assert form.action == "https://example.com/search"
    assert form.method == "POST"


def test_form_fields_extraction(form_page: PageElement):
    """find_form should extract all form fields correctly."""
    form = form_page.find_form(Selector.XPath("//form"), "form")

    # Check text input
    query_field = form.get_field("query")
    assert query_field is not None
    assert query_field.field_type == "text"
    assert query_field.value == ""

    # Check hidden input
    token_field = form.get_field("token")
    assert token_field is not None
    assert token_field.field_type == "hidden"
    assert token_field.value == "abc123"

    # Check select
    category_field = form.get_field("category")
    assert category_field is not None
    assert category_field.field_type == "select"
    assert category_field.value == "news"  # Selected option
    assert category_field.options == ["all", "news", "blog"]

    # Check textarea
    desc_field = form.get_field("description")
    assert desc_field is not None
    assert desc_field.field_type == "textarea"
    assert desc_field.value == "Default text"


def test_form_action_resolution(form_page: PageElement):
    """find_form should resolve relative action URLs."""
    # Absolute URL in action
    html_content = (
        '<form action="https://other.com/submit"><input name="test"/></form>'
    )
    doc = html.fromstring(html_content)
    page = PageElement(doc, "https://example.com/")

    form = page.find_form(Selector.XPath("//form"), "form")
    assert form.action == "https://other.com/submit"


def test_form_no_action_uses_base_url(form_page: PageElement):
    """find_form should use base URL when form has no action."""
    html_content = '<form><input name="test"/></form>'
    doc = html.fromstring(html_content)
    page = PageElement(doc, "https://example.com/page")

    form = page.find_form(Selector.XPath("//form"), "form")
    assert form.action == "https://example.com/page"


def test_find_links_by_xpath(links_page: PageElement):
    """find_links should find links by XPath selector."""
    links = links_page.find_links(
        Selector.XPath("//a[@class='nav-link']"), "nav links"
    )

    assert len(links) == 2
    assert links[0].url == "https://example.com/page1"
    assert links[0].text == "Page 1"
    assert links[1].url == "https://example.com/page2"
    assert links[1].text == "Page 2"


def test_find_links_by_css(links_page: PageElement):
    """find_links should find links by CSS selector."""
    links = links_page.find_links(Selector.CSS("a.nav-link"), "nav links")

    assert len(links) == 2
    assert links[0].url == "https://example.com/page1"


def test_find_links_resolves_urls(links_page: PageElement):
    """find_links should resolve relative URLs."""
    links = links_page.find_links(
        Selector.XPath("//a[@class='nav-link']"), "nav links"
    )

    # Relative URLs should be resolved
    assert links[0].url == "https://example.com/page1"
    assert links[1].url == "https://example.com/page2"


def test_find_links_skips_links_without_href(links_page: PageElement):
    """find_links should skip <a> elements without href."""
    all_links = links_page.find_links(
        Selector.XPath(".//a[@href]"), "all links", min_count=0
    )

    # Should find 3 links (not the one without href)
    assert len(all_links) == 3


def test_find_links_min_count_validates_links_with_href():
    """min_count applies to returned links, not raw matched <a> elements.

    A site swapping real anchors for <a onclick=...> JS handlers is
    exactly the structural change the count contract must catch loudly —
    matching five anchors and returning zero links must not pass.
    """
    page = _page_from_html(
        '<div><a class="case" href="/c1">One</a>'
        '<a class="case">Two</a>'
        '<a class="case">Three</a></div>'
    )

    with pytest.raises(HTMLStructuralAssumptionException) as exc_info:
        page.find_links(Selector.CSS("a.case"), "case links", min_count=3)

    assert exc_info.value.actual_count == 1


def test_find_links_throws_error_for_missing_hrefs():
    """max_count is satisfied by the href-bearing link count."""
    page = _page_from_html(
        '<div><a class="case" href="/c1">One</a>'
        '<a class="case" href="/c2">Two</a>'
        '<a class="case">Three</a></div>'
    )
    with pytest.raises(HTMLStructuralAssumptionException):
        _links = page.find_links(
            Selector.CSS("a.case"), "case links", min_count=1, max_count=2
        )


def test_find_links_returns_all_links(links_page: PageElement):
    """find_links with .//a[@href] should return all linked <a> elements."""
    all_links = links_page.find_links(
        Selector.XPath(".//a[@href]"), "all links", min_count=0
    )

    assert len(all_links) == 3
    # Should include both relative and absolute URLs
    urls = [link.url for link in all_links]
    assert "https://example.com/page1" in urls
    assert "https://external.com/page3" in urls


def test_link_follow_creates_navigating_request(links_page: PageElement):
    """Link.follow() should create a Request with ViaLink."""
    links = links_page.find_links(
        Selector.XPath("//a[@class='nav-link']"), "nav links"
    )
    link = links[0]

    request = link.follow(step="testing")

    assert request.request.url == "https://example.com/page1"
    assert request.via is not None
    assert isinstance(request.via, ViaLink)
    assert "nav-link" in request.via.selector.value
    assert request.step == "testing"


def test_find_form_raises_on_no_match(simple_page: PageElement):
    """find_form should raise if no form matches."""
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.find_form(
            Selector.XPath("//form[@id='nonexistent']"), "missing form"
        )


def _page_from_html(html_content: str, base_url: str = "https://example.com/"):
    doc = html.fromstring(html_content)
    return PageElement(doc, base_url)


# Shared markup fragments for the table below.
_Q = '<input name="q" value="bees"/>'
_TWO_SUBMITS = (
    '<input type="submit" name="action" value="search"/>'
    '<input type="submit" name="action" value="clear"/>'
)
_TWO_ID_SUBMITS = (
    '<input type="submit" name="action" id="btn_search" value="search"/>'
    '<input type="submit" name="action" id="btn_clear" value="clear"/>'
)


# --- find_form submission semantics ----------------------------------------
#
# What a browser would submit for a given form, as a table: each row is
# (name, controls, submit_selector, submitted data). The expectation is the
# *whole* dict, so a row pins both what is sent and what is left out.
#
# The cross-transport version of this contract — the same markup shapes driven
# through every transport against a vanilla-browser oracle — lives in
# tests/form_conformance/.

_FORM_CASES: list[tuple[str, str, Selector | None, dict[str, object]]] = [
    # --- checkboxes: only checked ones submit, defaulting to "on" ---------
    (
        "unchecked checkbox is omitted",
        '<input name="cb" type="checkbox" value="true"/>',
        None,
        {},
    ),
    (
        "checked checkbox submits its value",
        '<input name="cb" type="checkbox" value="yes" checked/>',
        None,
        {"cb": "yes"},
    ),
    (
        "checked checkbox without a value submits 'on'",
        '<input name="cb" type="checkbox" checked/>',
        None,
        {"cb": "on"},
    ),
    (
        "unchecked checkbox without a value is omitted",
        '<input name="cb" type="checkbox"/>',
        None,
        {},
    ),
    (
        "same-named checkboxes submit only the checked one",
        '<input name="cb" type="checkbox" value="a"/>'
        '<input name="cb" type="checkbox" value="b" checked/>',
        None,
        {"cb": "b"},
    ),
    (
        "checkbox group submits every checked box as a repeated key",
        '<input name="court" type="checkbox" value="ca1" checked/>'
        '<input name="court" type="checkbox" value="ca2"/>'
        '<input name="court" type="checkbox" value="ca3" checked/>',
        None,
        {"court": ["ca1", "ca3"]},
    ),
    # --- radios: same rules, but at most one is checked -------------------
    (
        "checked radio without a value submits 'on'",
        '<input name="choice" type="radio" checked/>',
        None,
        {"choice": "on"},
    ),
    (
        "radio group submits only the checked radio",
        '<input name="choice" type="radio" value="a"/>'
        '<input name="choice" type="radio" value="b" checked/>'
        '<input name="choice" type="radio" value="c"/>',
        None,
        {"choice": "b"},
    ),
    # --- <select> ---------------------------------------------------------
    (
        # Falling through to the label would turn an "All case types"
        # placeholder into a bogus filter that silently returns no results.
        "option value='' submits empty, not its label",
        '<select name="case_type">'
        '<option value="" selected>All</option>'
        '<option value="civil">Civil</option>'
        "</select>",
        None,
        {"case_type": ""},
    ),
    (
        "option with no value attribute submits its label, per spec",
        '<select name="category">'
        "<option selected>News</option><option>Blog</option>"
        "</select>",
        None,
        {"category": "News"},
    ),
    (
        "select multiple submits every selected option",
        '<select name="court" multiple>'
        '<option value="ca1" selected>First</option>'
        '<option value="ca2">Second</option>'
        '<option value="ca3" selected>Third</option>'
        "</select>",
        None,
        {"court": ["ca1", "ca3"]},
    ),
    (
        "select multiple with nothing selected contributes nothing",
        _Q + '<select name="court" multiple>'
        '<option value="ca1">First</option><option value="ca2">Second</option>'
        "</select>",
        None,
        {"q": "bees"},
    ),
    # --- submit controls --------------------------------------------------
    (
        # Sending every button at once is a request no browser produces, and
        # changes server behaviour on e.g. ASP.NET sites.
        "only the first submit travels when none is selected",
        _Q + '<input type="submit" name="action" value="search"/>'
        '<input type="submit" name="action_clear" value="clear"/>',
        None,
        {"q": "bees", "action": "search"},
    ),
    (
        # An input submit's id has to be captured (not just a <button>'s) or
        # _activated_submit never matches and falls back to the first button.
        "submit_selector '#id' activates an input submit",
        _Q + _TWO_ID_SUBMITS,
        CSS("#btn_clear"),
        {"q": "bees", "action": "clear"},
    ),
    (
        # The Playwright transport runs the raw selector against the DOM, so
        # the HTTP path must resolve value/name selectors too, not just "#id",
        # or the two transports submit different buttons.
        "submit_selector resolves a css value predicate",
        _Q + _TWO_SUBMITS,
        CSS('input[value="clear"]'),
        {"q": "bees", "action": "clear"},
    ),
    (
        "submit_selector resolves an xpath value predicate",
        _Q + _TWO_SUBMITS,
        XPath('.//input[@value="clear"]'),
        {"q": "bees", "action": "clear"},
    ),
    (
        # data= can add a key but cannot remove a wrongly-included button, so
        # resolution has to be right here.
        "a differently-named submit sends only its own name",
        _Q + '<input type="submit" name="search" value="Search"/>'
        '<input type="submit" name="clear" value="Clear"/>',
        CSS('[name="clear"]'),
        {"q": "bees", "clear": "Clear"},
    ),
    (
        "an unresolvable submit_selector falls back to the first submit",
        _Q + _TWO_SUBMITS,
        XPath("(.//input[@type='submit'])[2]"),
        {"q": "bees", "action": "search"},
    ),
    (
        "a named <button> is an implicit type=submit",
        _Q + '<button name="action" value="search">Search</button>',
        None,
        {"q": "bees", "action": "search"},
    ),
    (
        "type=button and type=reset never contribute, either spelling",
        _Q + '<button type="button" name="toggle" value="1">Toggle</button>'
        '<button type="reset" name="reset" value="1">Reset</button>'
        '<input type="reset" name="reset2" value="1"/>'
        '<input type="button" name="btn" value="1"/>',
        None,
        {"q": "bees"},
    ),
]


@pytest.mark.parametrize(
    ("controls", "submit_selector", "expected"),
    [row[1:] for row in _FORM_CASES],
    ids=[row[0] for row in _FORM_CASES],
)
def test_form_submits_what_a_browser_would(
    controls: str,
    submit_selector: Selector | None,
    expected: dict[str, object],
):
    page = _page_from_html(
        f'<form id="f" action="/x" method="post">{controls}</form>'
    )
    form = page.find_form(Selector.XPath("//form[@id='f']"), "f")

    request = form.submit(submit_selector=submit_selector, step="test")

    assert request.request.data == expected


# The parsed fields behind three of the rows above: a group collapses to its
# checked members, a <select> keeps every option, and an input submit's id is
# recorded so a "#id" submit_selector has something to match.
def test_find_form_collapses_a_checkbox_group_to_its_checked_boxes():
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="court" type="checkbox" value="ca1" checked/>'
        '<input name="court" type="checkbox" value="ca2"/>'
        '<input name="court" type="checkbox" value="ca3" checked/>'
        "</form>"
    )
    form = page.find_form(Selector.XPath("//form[@id='f']"), "f")

    assert [f.value for f in form.fields if f.name == "court"] == [
        "ca1",
        "ca3",
    ]


def test_find_form_keeps_every_option_of_a_select():
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<select name="case_type">'
        '<option value="" selected>All</option>'
        '<option value="civil">Civil</option>'
        "</select></form>"
    )
    form = page.find_form(Selector.XPath("//form[@id='f']"), "f")

    field = form.get_field("case_type")
    assert field is not None
    assert field.value == ""
    assert field.options == ["", "civil"]


def test_find_form_records_the_id_of_an_input_submit():
    page = _page_from_html(
        f'<form id="f" action="/x" method="post">{_TWO_ID_SUBMITS}</form>'
    )
    form = page.find_form(Selector.XPath("//form[@id='f']"), "f")

    assert {f.element_id for f in form.fields if f.field_type == "submit"} == {
        "btn_search",
        "btn_clear",
    }


def test_query_count_validation(simple_page: PageElement):
    """Query methods should validate count constraints."""
    # Too few
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.query(Selector.XPath("//tr"), "rows", min_count=10)

    # Too many
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.query(
            Selector.XPath("//tr"), "rows", min_count=1, max_count=1
        )


def test_link_selector_includes_position(links_page: PageElement):
    """find_links should create positional selectors for each link."""
    links = links_page.find_links(
        Selector.XPath("//a[@class='nav-link']"), "nav links"
    )

    # Each link should have a unique positional selector
    assert "[1]" in links[0].selector.value
    assert "[2]" in links[1].selector.value


def test_link_selector_xpath_uses_positional_predicate(
    links_page: PageElement,
):
    """XPath-found links get XPath positional predicate syntax."""
    links = links_page.find_links(
        Selector.XPath("//a[@class='nav-link']"), "nav links"
    )

    assert links[0].selector.value == "(//a[@class='nav-link'])[1]"
    assert links[1].selector.value == "(//a[@class='nav-link'])[2]"


def test_link_selector_css_uses_nth_match(links_page: PageElement):
    """CSS-found links get Playwright :nth-match syntax.

    Wrapping a CSS selector in an XPath positional predicate produces a
    string that is valid in neither grammar, which would break ViaLink
    replay in the Playwright driver.
    """
    links = links_page.find_links(Selector.CSS("a.nav-link"), "nav links")

    assert links[0].selector.value == ":nth-match(a.nav-link, 1)"
    assert links[1].selector.value == ":nth-match(a.nav-link, 2)"


# ── XPath/CSS routing ──────────────────────────────────────────────


def test_find_form_by_css_class_selector():
    """A bare .class selector is CSS, not relative XPath.

    Class selectors are the most common CSS selector scraper authors
    reach for; routing them to XPath raises a confusing XPathEvalError.
    """
    page = _page_from_html(
        '<form class="search-form" action="/search" method="post">'
        '<input name="q" value="bees" />'
        "</form>"
    )
    form = page.find_form(Selector.CSS(".search-form"), "search form")

    assert form.action == "https://example.com/search"
    field = form.get_field("q")
    assert field is not None
    assert field.value == "bees"


def test_find_links_by_css_class_selector():
    page = _page_from_html(
        '<div><a class="pdf-link" href="/doc1.pdf">Doc 1</a>'
        '<a class="pdf-link" href="/doc2.pdf">Doc 2</a></div>'
    )
    links = page.find_links(Selector.CSS(".pdf-link"), "pdf links")

    assert len(links) == 2
    assert links[0].url == "https://example.com/doc1.pdf"


def test_find_form_by_relative_xpath():
    """Relative XPath (.//) still routes to the XPath engine."""
    page = _page_from_html(
        '<div><form id="f" action="/x" method="post">'
        '<input name="q" value="" />'
        "</form></div>"
    )
    form = page.find_form(Selector.XPath(".//form[@id='f']"), "form")
    assert form.action == "https://example.com/x"


def test_find_links_by_relative_xpath():
    page = _page_from_html('<div><a href="/page1">One</a></div>')
    links = page.find_links(Selector.XPath(".//a[@href]"), "links")
    assert len(links) == 1
    assert links[0].url == "https://example.com/page1"
