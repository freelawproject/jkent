"""The public ``via_json`` wire format (ViaLink/ViaFormSubmit ↔ JSON).

This is the format the driver's queue persists and consumers (a host's
request reconstruction) read back, so the shape itself is pinned — not just
the round-trip.

The selector is nested (``{value, grammar}``) rather than flattened into a
``selector``/``selector_type`` pair, and the reader is strict about that:
the grammar is read, never guessed from the selector's prefix.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jkent.data_types import (
    CSS,
    ViaFormSubmit,
    ViaLink,
    XPath,
    via_from_json,
)


def test_via_link_round_trip_pins_wire_shape() -> None:
    via = ViaLink(selector=XPath("//a[@id='next']"), description="next page")
    raw = via.to_json()
    assert json.loads(raw) == {
        "type": "link",
        "selector": {"value": "//a[@id='next']", "grammar": "xpath"},
        "description": "next page",
    }
    assert via_from_json(raw) == via


def test_via_form_submit_round_trip_pins_wire_shape() -> None:
    via = ViaFormSubmit(
        form_selector=CSS("#search"),
        submit_selector='button[type="submit"]',
        field_data={"q": "smith", "court": ["a", "b"]},
        description="search form",
    )
    raw = via.to_json()
    assert json.loads(raw) == {
        "type": "form_submit",
        "form_selector": {"value": "#search", "grammar": "css"},
        "submit_selector": 'button[type="submit"]',
        "field_data": {"q": "smith", "court": ["a", "b"]},
        "description": "search form",
    }
    assert via_from_json(raw) == via


def test_round_trip_restores_the_selector_subclass() -> None:
    """The grammar travels as data, so the right subclass comes back.

    Not incidental: the subclass is what carries the grammar-specific
    behaviour (``for_playwright``, ``nth``, ``query``), so a via that
    round-trips to a bare ``Selector`` would replay wrong.
    """
    link = via_from_json(
        ViaLink(selector=CSS("a.next"), description="d").to_json()
    )
    assert isinstance(link, ViaLink)
    assert type(link.selector) is CSS
    assert link.selector.for_playwright() == "css=a.next"


def test_from_json_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        via_from_json(json.dumps({"type": "teleport"}))


def test_from_json_rejects_an_unknown_grammar() -> None:
    raw = json.dumps(
        {
            "type": "link",
            "selector": {"value": "//a", "grammar": "jsonpath"},
            "description": "d",
        }
    )
    with pytest.raises(ValidationError):
        via_from_json(raw)
