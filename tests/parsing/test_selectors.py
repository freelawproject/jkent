"""Tests for the ``Selector`` grammar hierarchy (``jkent.common.selectors``)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from jkent.common.selectors import CSS, Selector, XPath


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


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        pytest.param(["//table"], "//table", id="absolute-root"),
        pytest.param([".//td"], "/*//td", id="relative-root-descendant"),
        pytest.param(["body/div"], "/*/body/div", id="relative-root-child"),
        pytest.param(["./body"], "/*/body", id="dot-relative-root"),
        pytest.param(
            ["//table", ".//td"], "//table//td", id="descendant-child"
        ),
        pytest.param(["//table", "./tr"], "//table/tr", id="dot-child-step"),
        pytest.param(
            ["//table", "td[3]"], "//table/td[3]", id="bare-child-step"
        ),
        pytest.param(["//table", "../div"], "//table/../div", id="parent"),
        pytest.param(
            ["//table", "//span"], "//span", id="absolute-child-ignores-parent"
        ),
        pytest.param(
            ["//table", "td[@title='a | b']"],
            "//table/td[@title='a | b']",
            id="pipe-inside-literal",
        ),
    ],
)
def test_xpath_compose_selects_what_lxml_ran(
    parts: list[str], expected: str
) -> None:
    """Each part continues from what the part before it matched.

    The first ran on the parsed root, the document element (``/*``); the
    browser evaluates from the document node above it.
    """
    assert XPath.compose(parts) == expected


@pytest.mark.parametrize(
    "parts",
    [
        pytest.param(["//table", ".//td | .//th"], id="union"),
        pytest.param(["//table", "(.//td)[1]"], id="parenthesized"),
        pytest.param(["id('main')", ".//td"], id="function-call"),
    ],
)
def test_xpath_compose_refuses_a_part_with_no_single_path(
    parts: list[str],
) -> None:
    assert XPath.compose(parts) is None
