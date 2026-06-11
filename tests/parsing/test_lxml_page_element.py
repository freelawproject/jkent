"""Tests for LxmlPageElement implementation.

Tests delegation to CheckedHtmlElement, observer integration, form and link handling.
"""

import pytest
from lxml import html

from jkent.common.checked_html import CheckedHtmlElement
from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
)
from jkent.common.lxml_page_element import (
    LxmlPageElement,
)
from jkent.common.page_element import ViaLink
from jkent.common.selector_observer import (
    SelectorObserver,
)


@pytest.fixture
def simple_page():
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
    checked = CheckedHtmlElement(doc, "https://example.com/page")
    return LxmlPageElement(checked, "https://example.com/page")


@pytest.fixture
def form_page():
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
    checked = CheckedHtmlElement(doc, "https://example.com/")
    return LxmlPageElement(checked, "https://example.com/")


@pytest.fixture
def links_page():
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
    checked = CheckedHtmlElement(doc, "https://example.com/")
    return LxmlPageElement(checked, "https://example.com/")


def test_query_xpath_delegation(simple_page):
    """query_xpath should delegate to CheckedHtmlElement."""
    rows = simple_page.query_xpath("//tr[@class='row']", "rows")

    assert len(rows) == 2
    # Results should be wrapped in LxmlPageElement
    assert all(isinstance(row, LxmlPageElement) for row in rows)


def test_query_xpath_returns_lxml_page_elements(simple_page):
    """query_xpath should return LxmlPageElement instances."""
    rows = simple_page.query_xpath("//tr", "rows", min_count=2)

    # Each result should be LxmlPageElement
    for row in rows:
        assert isinstance(row, LxmlPageElement)
        # Should be able to query nested elements
        cells = row.query_xpath(".//td", "cells")
        assert len(cells) == 2


def test_query_xpath_strings(simple_page):
    """query_xpath_strings should return string values."""
    cell_texts = simple_page.query_xpath_strings(
        "//td/text()", "cell texts", min_count=4, max_count=4
    )

    assert len(cell_texts) == 4
    assert all(isinstance(text, str) for text in cell_texts)
    assert "Cell 1" in cell_texts


def test_query_css_delegation(simple_page):
    """query_css should delegate to CheckedHtmlElement."""
    rows = simple_page.query_css("tr.row", "rows")

    assert len(rows) == 2
    assert all(isinstance(row, LxmlPageElement) for row in rows)


def test_text_content(simple_page):
    """text_content should return element text."""
    h1 = simple_page.query_xpath("//h1", "heading", min_count=1, max_count=1)[
        0
    ]

    assert h1.text_content() == "Test Page"


def test_get_attribute(links_page):
    """get_attribute should return attribute value."""
    link = links_page.query_xpath("//a[@class='nav-link']", "nav links")[0]

    assert link.get_attribute("href") == "/page1"
    assert link.get_attribute("class") == "nav-link"
    assert link.get_attribute("nonexistent") is None


def test_inner_html(simple_page):
    """inner_html should return inner HTML content."""
    div = simple_page.query_xpath("//div[@id='main']", "main div")[0]

    inner = div.inner_html()

    assert "<h1>Test Page</h1>" in inner
    assert "<table>" in inner


def test_inner_html_preserves_leading_text():
    """inner_html keeps text that precedes the first child element."""
    doc = html.fromstring("<td>Case No. <a href='/c/1'>123</a></td>")
    checked = CheckedHtmlElement(doc, "https://example.com/")
    cell = LxmlPageElement(checked, "https://example.com/")

    inner = cell.inner_html()

    assert inner.startswith("Case No. ")
    assert '<a href="/c/1">123</a>' in inner


def test_tag_name(simple_page):
    """tag_name should return lowercase tag name."""
    h1 = simple_page.query_xpath("//h1", "heading")[0]
    table = simple_page.query_xpath("//table", "table")[0]

    assert h1.tag_name() == "h1"
    assert table.tag_name() == "table"


def test_child_queries_record_to_active_observer():
    """Queries on child elements record to the active observer.

    Recording flows through the get_active_observer() contextvar, not
    through any observer held on the PageElement, so a query on a child
    element returned by query_xpath must still be captured.
    """
    doc = html.fromstring("<div><section><p>Test</p></section></div>")
    checked = CheckedHtmlElement(doc, "https://example.com/")
    page = LxmlPageElement(checked, "https://example.com/")

    with SelectorObserver() as observer:
        sections = page.query_xpath("//section", "sections")
        sections[0].query_xpath(".//p", "paragraphs")

    # The parent query records at the top level, and the query made on the
    # child element nests beneath it — proving the child still reports
    # through the active observer with no observer held on the element.
    assert [q.selector for q in observer.queries] == ["//section"]
    assert [c.selector for c in observer.queries[0].children] == [".//p"]


def test_find_form_by_xpath(form_page):
    """find_form should find form by XPath selector."""
    form = form_page.find_form("//form[@id='search']", "search form")

    assert form.action == "https://example.com/search"
    assert form.method == "POST"
    assert form.selector == "//form[@id='search']"
    assert len(form.fields) == 4  # query, token, category, description


def test_find_form_by_css(form_page):
    """find_form should find form by CSS selector."""
    form = form_page.find_form("form#search", "search form")

    assert form.action == "https://example.com/search"
    assert form.method == "POST"


def test_form_fields_extraction(form_page):
    """find_form should extract all form fields correctly."""
    form = form_page.find_form("//form", "form")

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


def test_form_action_resolution(form_page):
    """find_form should resolve relative action URLs."""
    # Absolute URL in action
    html_content = (
        '<form action="https://other.com/submit"><input name="test"/></form>'
    )
    doc = html.fromstring(html_content)
    checked = CheckedHtmlElement(doc, "https://example.com/")
    page = LxmlPageElement(checked, "https://example.com/")

    form = page.find_form("//form", "form")
    assert form.action == "https://other.com/submit"


def test_form_no_action_uses_base_url(form_page):
    """find_form should use base URL when form has no action."""
    html_content = '<form><input name="test"/></form>'
    doc = html.fromstring(html_content)
    checked = CheckedHtmlElement(doc, "https://example.com/page")
    page = LxmlPageElement(checked, "https://example.com/page")

    form = page.find_form("//form", "form")
    assert form.action == "https://example.com/page"


def test_find_links_by_xpath(links_page):
    """find_links should find links by XPath selector."""
    links = links_page.find_links("//a[@class='nav-link']", "nav links")

    assert len(links) == 2
    assert links[0].url == "https://example.com/page1"
    assert links[0].text == "Page 1"
    assert links[1].url == "https://example.com/page2"
    assert links[1].text == "Page 2"


def test_find_links_by_css(links_page):
    """find_links should find links by CSS selector."""
    links = links_page.find_links("a.nav-link", "nav links")

    assert len(links) == 2
    assert links[0].url == "https://example.com/page1"


def test_find_links_resolves_urls(links_page):
    """find_links should resolve relative URLs."""
    links = links_page.find_links("//a[@class='nav-link']", "nav links")

    # Relative URLs should be resolved
    assert links[0].url == "https://example.com/page1"
    assert links[1].url == "https://example.com/page2"


def test_find_links_skips_links_without_href(links_page):
    """find_links should skip <a> elements without href."""
    all_links = links_page.links()

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
        page.find_links("a.case", "case links", min_count=3)

    assert exc_info.value.actual_count == 1


def test_find_links_max_count_applies_to_links_with_href():
    """max_count is satisfied by the href-bearing link count."""
    page = _page_from_html(
        '<div><a class="case" href="/c1">One</a>'
        '<a class="case" href="/c2">Two</a>'
        '<a class="case">Three</a></div>'
    )

    links = page.find_links("a.case", "case links", min_count=1, max_count=2)

    assert len(links) == 2


def test_links_returns_all_links(links_page):
    """links() should return all <a> elements with href."""
    all_links = links_page.links()

    assert len(all_links) == 3
    # Should include both relative and absolute URLs
    urls = [link.url for link in all_links]
    assert "https://example.com/page1" in urls
    assert "https://external.com/page3" in urls


def test_link_follow_creates_navigating_request(links_page):
    """Link.follow() should create a Request with ViaLink."""
    links = links_page.find_links("//a[@class='nav-link']", "nav links")
    link = links[0]

    request = link.follow(continuation="testing")

    assert request.request.url == "https://example.com/page1"
    assert request.via is not None
    assert isinstance(request.via, ViaLink)
    assert "nav-link" in request.via.selector
    assert request.continuation == "testing"


def test_find_form_raises_on_no_match(simple_page):
    """find_form should raise if no form matches."""
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.find_form("//form[@id='nonexistent']", "missing form")


def _page_from_html(html_content: str, base_url: str = "https://example.com/"):
    doc = html.fromstring(html_content)
    checked = CheckedHtmlElement(doc, base_url)
    return LxmlPageElement(checked, base_url)


def test_unchecked_checkbox_with_value_is_omitted():
    """Unchecked checkboxes must not appear in the submitted form data."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="cb" type="checkbox" value="true" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    assert form.get_field("cb") is None
    request = form.submit()
    assert "cb" not in request.request.data


def test_checked_checkbox_with_value_is_submitted():
    """Checked checkboxes submit their explicit value attribute."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="cb" type="checkbox" value="yes" checked />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    field = form.get_field("cb")
    assert field is not None
    assert field.value == "yes"
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["cb"] == "yes"


def test_checked_checkbox_without_value_defaults_to_on():
    """Checked checkboxes without a value attribute submit as 'on'."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="cb" type="checkbox" checked />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    field = form.get_field("cb")
    assert field is not None
    assert field.value == "on"
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["cb"] == "on"


def test_unchecked_checkbox_without_value_is_omitted():
    """Unchecked checkboxes without a value attribute are still omitted."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="cb" type="checkbox" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    assert form.get_field("cb") is None
    request = form.submit()
    assert "cb" not in request.request.data


def test_mixed_checkboxes_only_checked_submitted():
    """When two checkboxes share a name, only the checked one is submitted."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="cb" type="checkbox" value="a" />'
        '<input name="cb" type="checkbox" value="b" checked />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    cb_fields = [f for f in form.fields if f.name == "cb"]
    assert len(cb_fields) == 1
    assert cb_fields[0].value == "b"
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["cb"] == "b"


def test_checked_radio_without_value_defaults_to_on():
    """Checked radios without a value attribute submit as 'on', per spec."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="choice" type="radio" checked />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    field = form.get_field("choice")
    assert field is not None
    assert field.value == "on"
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["choice"] == "on"


def test_radio_group_only_checked_submitted():
    """When radios share a name, only the checked one is submitted."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="choice" type="radio" value="a" />'
        '<input name="choice" type="radio" value="b" checked />'
        '<input name="choice" type="radio" value="c" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["choice"] == "b"


def test_option_with_empty_value_submits_empty_string():
    """<option value=""> submits "", not its label text.

    Falling through to the label turns "All case types" placeholder
    options into bogus filter values, silently returning zero results.
    """
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<select name="case_type">'
        '<option value="" selected>All</option>'
        '<option value="civil">Civil</option>'
        "</select>"
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    field = form.get_field("case_type")
    assert field is not None
    assert field.value == ""
    assert field.options == ["", "civil"]
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["case_type"] == ""


def test_option_without_value_attribute_uses_label():
    """An option with no value attribute submits its label text, per spec."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<select name="category">'
        "<option selected>News</option>"
        "<option>Blog</option>"
        "</select>"
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["category"] == "News"


def test_only_first_submit_button_is_submitted():
    """Only the activated (default: first) submit control's value submits.

    Sending every submit button's name/value at once is a request no
    browser produces, and changes server behavior on e.g. ASP.NET sites.
    """
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="action" value="search" />'
        '<input type="submit" name="action_clear" value="clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["q"] == "bees"
    assert data["action"] == "search"
    assert "action_clear" not in data


def test_submit_selector_activates_input_submit_by_id():
    """submit_selector='#id' must activate the matching <input type=submit>.

    The control's ``id`` has to be captured for input submits (not just
    <button>s) or ``_activated_submit`` can never match and silently falls
    back to the first submit button, so the request carries the wrong
    button's name/value.
    """
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="action" id="btn_search" value="search" />'
        '<input type="submit" name="action" id="btn_clear" value="clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    # The id attribute is captured for input submits, so the selector can
    # resolve to a specific control.
    submit_ids = {
        f.element_id for f in form.fields if f.field_type == "submit"
    }
    assert submit_ids == {"btn_search", "btn_clear"}

    request = form.submit(submit_selector="#btn_clear")
    data = request.request.data
    assert isinstance(data, dict)
    assert data["q"] == "bees"
    # The activated (clear) button's value submits, not the first button's.
    assert data["action"] == "clear"


def test_submit_selector_resolves_input_submit_by_value_css():
    """A CSS attribute submit_selector resolves against the parsed fields.

    The Playwright transport runs the raw selector against the DOM, so the
    HTTP path must resolve the same value/name/attribute selectors (not just
    ``#id``) or the two transports submit different buttons.
    """
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="action" value="search" />'
        '<input type="submit" name="action" value="clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit(submit_selector='input[value="clear"]')
    data = request.request.data
    assert isinstance(data, dict)
    assert data["action"] == "clear"


def test_submit_selector_resolves_input_submit_by_value_xpath():
    """An XPath attribute submit_selector resolves the same way."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="action" value="search" />'
        '<input type="submit" name="action" value="clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit(submit_selector='.//input[@value="clear"]')
    data = request.request.data
    assert isinstance(data, dict)
    assert data["action"] == "clear"


def test_submit_selector_by_name_excludes_other_button():
    """Selecting a differently-named submit sends only that button's name.

    A browser POSTs only the activated submit, so the non-activated submit's
    name must be absent — the data= override can add a key but can't remove
    the wrongly-included first button, so resolution has to be correct here.
    """
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="search" value="Search" />'
        '<input type="submit" name="clear" value="Clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit(submit_selector='[name="clear"]')
    data = request.request.data
    assert isinstance(data, dict)
    assert data["clear"] == "Clear"
    assert "search" not in data


def test_submit_selector_unresolvable_falls_back_to_first():
    """A selector we can't resolve against parsed fields (no id/name/value
    predicate, e.g. positional) falls back to the first submit, unchanged."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<input type="submit" name="action" value="search" />'
        '<input type="submit" name="action" value="clear" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit(submit_selector="(.//input[@type='submit'])[2]")
    data = request.request.data
    assert isinstance(data, dict)
    assert data["action"] == "search"


def test_button_element_participates_in_submission():
    """A named <button> (implicit type=submit) submits its value."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<button name="action" value="search">Search</button>'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["action"] == "search"


def test_button_type_button_and_reset_not_submitted():
    """type=button/reset controls never contribute to form data."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<button type="button" name="toggle" value="1">Toggle</button>'
        '<button type="reset" name="reset" value="1">Reset</button>'
        '<input type="reset" name="reset2" value="1" />'
        '<input type="button" name="btn" value="1" />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data == {"q": "bees"}


def test_checkbox_group_submits_all_checked_values():
    """All checked same-named checkboxes submit, as repeated keys."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="court" type="checkbox" value="ca1" checked />'
        '<input name="court" type="checkbox" value="ca2" />'
        '<input name="court" type="checkbox" value="ca3" checked />'
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    court_fields = [f for f in form.fields if f.name == "court"]
    assert [f.value for f in court_fields] == ["ca1", "ca3"]
    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["court"] == ["ca1", "ca3"]


def test_select_multiple_submits_all_selected_options():
    """<select multiple> submits every selected option."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<select name="court" multiple>'
        '<option value="ca1" selected>First</option>'
        '<option value="ca2">Second</option>'
        '<option value="ca3" selected>Third</option>'
        "</select>"
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert data["court"] == ["ca1", "ca3"]


def test_select_multiple_nothing_selected_submits_nothing():
    """<select multiple> with no selected options contributes no value."""
    page = _page_from_html(
        '<form id="f" action="/x" method="post">'
        '<input name="q" value="bees" />'
        '<select name="court" multiple>'
        '<option value="ca1">First</option>'
        '<option value="ca2">Second</option>'
        "</select>"
        "</form>"
    )
    form = page.find_form("//form[@id='f']", "f")

    request = form.submit()
    data = request.request.data
    assert isinstance(data, dict)
    assert "court" not in data


def test_query_count_validation(simple_page):
    """Query methods should validate count constraints."""
    # Too few
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.query_xpath("//tr", "rows", min_count=10)

    # Too many
    with pytest.raises(HTMLStructuralAssumptionException):
        simple_page.query_xpath("//tr", "rows", min_count=1, max_count=1)


def test_link_selector_includes_position(links_page):
    """find_links should create positional selectors for each link."""
    links = links_page.find_links("//a[@class='nav-link']", "nav links")

    # Each link should have a unique positional selector
    assert "[1]" in links[0].selector
    assert "[2]" in links[1].selector


def test_link_selector_xpath_uses_positional_predicate(links_page):
    """XPath-found links get XPath positional predicate syntax."""
    links = links_page.find_links("//a[@class='nav-link']", "nav links")

    assert links[0].selector == "(//a[@class='nav-link'])[1]"
    assert links[1].selector == "(//a[@class='nav-link'])[2]"


def test_link_selector_css_uses_nth_match(links_page):
    """CSS-found links get Playwright :nth-match syntax.

    Wrapping a CSS selector in an XPath positional predicate produces a
    string that is valid in neither grammar, which would break ViaLink
    replay in the Playwright driver.
    """
    links = links_page.find_links("a.nav-link", "nav links")

    assert links[0].selector == ":nth-match(a.nav-link, 1)"
    assert links[1].selector == ":nth-match(a.nav-link, 2)"


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
    form = page.find_form(".search-form", "search form")

    assert form.action == "https://example.com/search"
    field = form.get_field("q")
    assert field is not None
    assert field.value == "bees"


def test_find_links_by_css_class_selector():
    page = _page_from_html(
        '<div><a class="pdf-link" href="/doc1.pdf">Doc 1</a>'
        '<a class="pdf-link" href="/doc2.pdf">Doc 2</a></div>'
    )
    links = page.find_links(".pdf-link", "pdf links")

    assert len(links) == 2
    assert links[0].url == "https://example.com/doc1.pdf"


def test_find_form_by_relative_xpath():
    """Relative XPath (.//) still routes to the XPath engine."""
    page = _page_from_html(
        '<div><form id="f" action="/x" method="post">'
        '<input name="q" value="" />'
        "</form></div>"
    )
    form = page.find_form(".//form[@id='f']", "form")
    assert form.action == "https://example.com/x"


def test_find_links_by_relative_xpath():
    page = _page_from_html('<div><a href="/page1">One</a></div>')
    links = page.find_links(".//a[@href]", "links")
    assert len(links) == 1
    assert links[0].url == "https://example.com/page1"
