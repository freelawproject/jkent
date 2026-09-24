"""Tests for the run database's JSON column codec (``jkent.common.serialization``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest
from pydantic import BaseModel

from jkent.common.serialization import dump_json, dump_json_or_none


@dataclass
class _Doc:
    body: Any


class _Model(BaseModel):
    body: Any


class _Opaque:
    def __str__(self) -> str:
        return "opaque"


@pytest.mark.parametrize(
    ("value", "path"),
    [
        (b"raw", "value"),
        (bytearray(b"raw"), "value"),
        (memoryview(b"raw"), "value"),
        ({"doc": [1, 2, b"raw"]}, "value['doc'][2]"),
        ((1, (b"raw",)), "value[1][0]"),
        ({"s": {b"raw"}}, "value['s']{...}"),
        ({b"key": 1}, "value (key b'key')"),
        (_Doc(body={"x": b"raw"}), "value.body['x']"),
        (_Model(body=[b"raw"]), "value.body[0]"),
    ],
)
def test_bytes_anywhere_are_refused_with_their_path(value: Any, path: str):
    with pytest.raises(TypeError) as excinfo:
        dump_json(value)

    assert f"{path}:" in str(excinfo.value)


def test_refusal_names_the_caller_supplied_root():
    with pytest.raises(TypeError, match=r"accumulated_data\['doc'\]\[2\]:"):
        dump_json_or_none({"doc": [0, 1, b"raw"]}, name="accumulated_data")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"a": [1, "b", None]}, {"a": [1, "b", None]}),
        (date(2026, 1, 2), "2026-01-02"),
        (_Doc(body=[1]), {"body": [1]}),
        (_Model(body={"k": "v"}), {"body": {"k": "v"}}),
        ({"o": _Opaque()}, {"o": "opaque"}),
    ],
)
def test_non_bytes_values_encode(value: Any, expected: Any):
    assert json.loads(dump_json(value)) == expected


def test_none_stays_none():
    assert dump_json_or_none(None) is None
