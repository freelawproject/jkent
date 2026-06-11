"""The public ``via_json`` wire format (ViaLink/ViaFormSubmit ↔ JSON).

This is the format the driver's queue persists and consumers (jent's
request reconstruction) read back, so the shape itself is pinned — not just
the round-trip.
"""

from __future__ import annotations

import json

import pytest

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
        "selector": "//a[@id='next']",
        "selector_type": "xpath",
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
        "form_selector": "#search",
        "selector_type": "css",
        "submit_selector": 'button[type="submit"]',
        "field_data": {"q": "smith", "court": ["a", "b"]},
        "description": "search form",
    }
    assert via_from_json(raw) == via


def test_from_json_defaults_grammar_for_legacy_rows() -> None:
    # Rows written before selector_type existed fall back to the prefix
    # heuristic: unambiguous XPath prefixes are xpath, everything else css.
    legacy = json.dumps(
        {"type": "link", "selector": "//a", "description": "d"}
    )
    legacy_via = via_from_json(legacy)
    assert isinstance(legacy_via, ViaLink)
    assert legacy_via.selector.grammar == "xpath"
    legacy_css = json.dumps(
        {"type": "link", "selector": "a.next", "description": "d"}
    )
    legacy_css_via = via_from_json(legacy_css)
    assert isinstance(legacy_css_via, ViaLink)
    assert legacy_css_via.selector.grammar == "css"


def test_from_json_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown via type"):
        via_from_json(json.dumps({"type": "teleport"}))
