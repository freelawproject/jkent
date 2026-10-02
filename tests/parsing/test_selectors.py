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
        Selector("x")  # type: ignore[abstract]  # pyrefly: ignore[bad-instantiation]


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
