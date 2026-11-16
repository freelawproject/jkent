"""The ``Selector`` grammar hierarchy (``jkent.common.selectors``).

Construction and pydantic round-tripping, then each grammar's Playwright
waitability rule.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from jkent.common.selectors import (
    _EXSLT_PREFIXES,
    CSS,
    Selector,
    XPath,
)


class _Holder(BaseModel):
    selector: Selector


def test_base_selector_cannot_be_constructed():
    # Abstract: a selector without a grammar has no meaning downstream.
    with pytest.raises(TypeError):
        Selector("x")  # pyrefly: ignore[bad-instantiation]


@pytest.mark.parametrize(
    ("grammar", "expected_class"), [("css", CSS), ("xpath", XPath)]
)
def test_of_builds_the_named_grammar(
    grammar: str, expected_class: type[Selector]
):
    selector = Selector.of("x", grammar)

    assert type(selector) is expected_class
    assert selector.grammar == grammar


def test_of_rejects_an_unknown_grammar():
    with pytest.raises(ValueError, match="unknown selector grammar"):
        Selector.of("x", "jsonpath")


@pytest.mark.parametrize("selector", [CSS("div.x"), XPath("//div")])
def test_pydantic_round_trips_the_grammar(selector: Selector):
    dumped = _Holder(selector=selector).model_dump_json()

    assert _Holder.model_validate_json(dumped).selector == selector


# --- Playwright waitability -------------------------------------------------
#
# Which selectors ``page.wait_for_selector()`` can be handed: the rule is per
# grammar, so it is tested on the grammar. CSS is always waitable except for
# cssselect's ``:contains()``; XPath is waitable only when its last step
# selects elements and Playwright can bind everything in it.
# ``jkent.driver.unified_driver.steps.can_playwright_wait`` is a thin reader
# over this for callers holding a ``(value, grammar)`` pair (see
# test_steps.py).


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        # Plain element-targeting selectors, both grammars.
        ("div.content", True),
        ("#main-content", True),
        ("table > tr:first-child", True),
        ("a[href]", True),
        # [attr$=value] uses $ legitimately; css has no variable references.
        ("a[href$='.pdf']", True),
        # ``:contains()`` is cssselect's, not a pseudo-class Playwright has.
        ("td:contains('Docket')", False),
        ('a:CONTAINS("x"), b', False),  # caught in any case, in any branch
        # ...but not when it is only the text of an attribute value.
        ("a[title=':contains(x)']", True),
        ("a:has-text('x')", True),
    ],
)
def test_css_waitability(selector: str, expected: bool):
    assert CSS(selector).can_playwright_wait() is expected


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        # Element-targeting paths, including axes and positional wrappers.
        ("//div", True),
        ("//div[@class='content']", True),
        ("//table//tr", True),
        ("//a[@href]", True),
        ("(//div)[1]", True),
        ("//div[@id='first']/following-sibling::div", True),
        ("//div[@id]/span", True),  # @id mid-path still targets the span
        ("id('x')", True),  # id() selects elements
        # Text nodes and attributes are not elements.
        ("//div/text()", False),
        ("//p[@class='title']/text()", False),
        ("//span//text()", False),
        ("//a/@href", False),
        ("//div/@class", False),
        ("//table//td/@data-id", False),
        # Playwright cannot bind XPath variable references...
        ("//div[@id=$section]", False),
        ("//a[position()=$n]", False),
        # ...but a literal dollar sign in quoted text is not one.
        ("//a[text()='Price: $5']", True),
        ('//td[contains(., "US$")]', True),
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
        # Does not compile.
        ("//div[", False),
        # Not XML-compatible text at all.
        ("//div[@x='\x08']", False),
        # Only XPath whitespace is insignificant; U+0085 is not.
        ("//div\x85", False),
    ],
)
def test_xpath_waitability(selector: str, expected: bool):
    assert XPath(selector).can_playwright_wait() is expected


@pytest.mark.parametrize("case", [str.lower, str.upper, str.title])
@pytest.mark.parametrize("prefix", _EXSLT_PREFIXES)
def test_an_exslt_prefix_in_a_predicate_is_incompatible(
    prefix: str, case: Callable[[str], str]
) -> None:
    """Caught in any case: Playwright binds none of them."""
    assert XPath(f"//div[{case(prefix)}f(., 'x')]").can_playwright_wait() is (
        False
    )
