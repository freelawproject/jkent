"""Tests for the ``Selector`` grammar hierarchy (``jkent.common.selectors``)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from jkent.common.selectors import CSS, Selector, XPath


class _Holder(BaseModel):
    selector: Selector


def test_base_selector_cannot_be_constructed():
    with pytest.raises(TypeError, match="grammar"):
        Selector("x")


def test_grammarless_subclass_cannot_be_constructed():
    @dataclass(frozen=True)
    class Intermediate(Selector):
        pass

    with pytest.raises(TypeError, match="grammar"):
        Intermediate("x")


@pytest.mark.parametrize(
    ("grammar", "expected_class"), [("css", CSS), ("xpath", XPath)]
)
def test_of_builds_the_registered_grammar(
    grammar: str, expected_class: type[Selector]
):
    selector = Selector.of("x", grammar)

    assert type(selector) is expected_class
    assert selector.grammar == grammar


@pytest.mark.parametrize("selector", [CSS("div.x"), XPath("//div")])
def test_pydantic_round_trips_the_grammar(selector: Selector):
    dumped = _Holder(selector=selector).model_dump_json()

    assert _Holder.model_validate_json(dumped).selector == selector
