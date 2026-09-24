"""Tests for ``can_playwright_wait`` (steps.py).

The autowait loop's Playwright-compatibility gate: which selectors can
be handed to ``page.wait_for_selector``. A thin reader over
``Selector.can_playwright_wait`` in ``jkent/common/selectors.py``.
"""

from collections.abc import Callable

import pytest

from jkent.common.selectors import _EXSLT_PREFIXES
from jkent.driver.unified_driver.steps import can_playwright_wait


class TestCanPlaywrightWait:
    """Tests for can_playwright_wait function."""

    def test_css_selectors_always_compatible(self):
        """CSS selectors should always be compatible with Playwright."""
        assert can_playwright_wait("div.content", "css") is True
        assert can_playwright_wait("#main-content", "css") is True
        assert can_playwright_wait("table > tr:first-child", "css") is True
        assert can_playwright_wait("a[href]", "css") is True

    def test_element_targeting_xpath_compatible(self):
        """XPath selectors targeting elements should be compatible."""
        assert can_playwright_wait("//div", "xpath") is True
        assert can_playwright_wait("//div[@class='content']", "xpath") is True
        assert can_playwright_wait("//table//tr", "xpath") is True
        assert can_playwright_wait("//a[@href]", "xpath") is True
        assert can_playwright_wait("(//div)[1]", "xpath") is True

    def test_text_node_xpath_incompatible(self):
        """XPath selectors ending with /text() should be incompatible."""
        assert can_playwright_wait("//div/text()", "xpath") is False
        assert (
            can_playwright_wait("//p[@class='title']/text()", "xpath") is False
        )
        assert can_playwright_wait("//span//text()", "xpath") is False

    def test_attribute_xpath_incompatible(self):
        """XPath selectors targeting attributes should be incompatible."""
        assert can_playwright_wait("//a/@href", "xpath") is False
        assert can_playwright_wait("//div/@class", "xpath") is False
        assert can_playwright_wait("//input/@value", "xpath") is False
        assert can_playwright_wait("//table//td/@data-id", "xpath") is False

    def test_xpath_variable_references_incompatible(self):
        """Playwright can't bind XPath variable references ($var)."""
        assert can_playwright_wait("//div[@id=$section]", "xpath") is False
        assert can_playwright_wait("//a[position()=$n]", "xpath") is False

    def test_dollar_inside_string_literal_compatible(self):
        """A literal dollar sign in quoted text is not a variable."""
        assert can_playwright_wait("//a[text()='Price: $5']", "xpath") is True
        assert can_playwright_wait('//td[contains(., "US$")]', "xpath") is True

    def test_css_attribute_suffix_selector_compatible(self):
        """CSS [attr$=value] uses $ legitimately; css is always waitable."""
        assert can_playwright_wait("a[href$='.pdf']", "css") is True

    def test_css_contains_incompatible(self):
        """cssselect's ``:contains()`` is not a pseudo-class Playwright has."""
        assert can_playwright_wait("td:contains('Docket')", "css") is False
        assert can_playwright_wait('a:CONTAINS("x"), b', "css") is False

    def test_css_contains_inside_string_literal_compatible(self):
        assert can_playwright_wait("a[title=':contains(x)']", "css") is True
        assert can_playwright_wait("a:has-text('x')", "css") is True

    def test_following_sibling_compatible(self):
        assert (
            can_playwright_wait(
                "//div[@id='first']/following-sibling::div", "xpath"
            )
            is True
        )

    def test_attribute_in_middle_of_path(self):
        """Attribute in middle of path should still be considered compatible."""
        # This is unusual but the selector targets an element (the div)
        # Even though it has @id in the middle, it's not selecting @id at the end
        assert can_playwright_wait("//div[@id]/span", "xpath") is True


@pytest.mark.parametrize("case", [str.lower, str.upper, str.title])
@pytest.mark.parametrize("prefix", _EXSLT_PREFIXES)
def test_an_exslt_prefix_in_a_predicate_is_incompatible(
    prefix: str, case: Callable[[str], str]
) -> None:
    """Caught in any case: Playwright binds none of them."""
    selector = f"//div[{case(prefix)}f(., 'x')]"
    assert can_playwright_wait(selector, "xpath") is False


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        # Functions returning numbers or strings select no nodes.
        ("count(//a)", False),
        ("normalize-space(//h1)", False),
        # A comparison is a boolean, not a node-set.
        ("//a = 'x'", False),
        # Non-element last steps, spelled with an explicit axis or node test.
        ("//a/attribute::href", False),
        ("//td/following-sibling::text()", False),
        ("//div/comment()", False),
        ("//div/processing-instruction('x')", False),
        ("//div/node()", False),
        ("//div/namespace::*", False),
        # One non-element branch of a union is enough to rule it out.
        ("//div | //a/@href", False),
        ("//div | //span", True),
        # A union bar inside a string literal is not a union.
        ("//a[text()='a | @b']", True),
        # Trailing and wrapped positional predicates keep the node kind.
        ("//div/text()[1]", False),
        ("(//div/text())[1]", False),
        ("(//a/@href)[1]", False),
        ("(//div)[1]", True),
        # id() selects elements.
        ("id('x')", True),
        # Does not compile.
        ("//div[", False),
        # Not XML-compatible text at all.
        ("//div[@x='\x08']", False),
        # Only XPath whitespace is insignificant; U+0085 is not.
        ("//div\x85", False),
    ],
)
def test_xpath_waitability(selector: str, expected: bool):
    assert can_playwright_wait(selector, "xpath") is expected
