"""Tests for the run database's JSON column codec (``jkent.common.serialization``)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field
from pydantic_core import PydanticSerializationError
from typing_extensions import override

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


@pytest.mark.parametrize("value", [0, False, "", [], {}])
def test_a_falsy_but_real_value_is_still_encoded(value: Any):
    """Only ``None`` means a NULL column — the overloads cannot say this."""
    assert dump_json_or_none(value) == dump_json(value)


def _cycle(build: Any, *extra: Any) -> Any:
    """A *build*-shaped value that refers back to itself, holding *extra*."""
    value: Any
    if build is list:
        value = list(extra)
        value.append(value)
    elif build is dict:
        value = {f"item{i}": item for i, item in enumerate(extra)}
        value["self"] = value
    else:
        value = build(body=None)
        value.body = [value, *extra]
    return value


class _CountingList(list[Any]):
    """A list that records how many times it was walked."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.walks = 0

    @override
    def __iter__(self) -> Iterator[Any]:
        self.walks += 1
        return super().__iter__()


def test_a_shared_branch_is_walked_once():
    shared = _CountingList([1, 2, 3])

    dump_json({"a": {"deep": shared}, "b": shared, "c": [shared]})

    assert shared.walks == 1


def test_bytes_behind_a_shared_branch_are_refused_at_first_sight():
    shared = {"x": [b"raw"]}

    with pytest.raises(TypeError, match=r"value\['a'\]\['x'\]\[0\]:"):
        dump_json({"a": shared, "b": shared})


def test_two_equal_but_distinct_branches_are_both_walked():
    with pytest.raises(TypeError, match=r"value\[1\]\['x'\]:"):
        dump_json([{"x": "ok"}, {"x": b"raw"}])


@pytest.mark.parametrize("build", [list, dict, _Doc, _Model])
def test_a_cycle_is_left_for_to_json_to_report(build: Any):
    """The walk terminates on a cycle; ``to_json`` is what refuses it."""
    value = _cycle(build)

    with pytest.raises(PydanticSerializationError, match="Circular reference"):
        dump_json(value)


def test_a_cycle_holding_bytes_is_still_refused():
    inner: dict[str, Any] = {"raw": b"raw"}
    outer = {"inner": inner}
    inner["back"] = outer

    with pytest.raises(TypeError, match=r"value\['inner'\]\['raw'\]:"):
        dump_json(outer)


def test_a_cycle_through_a_shared_branch_terminates():
    """A node reached twice, once inside its own cycle, is walked once."""
    leaf = _CountingList(["ok"])
    node: dict[str, Any] = {"leaf": leaf}
    node["cycle"] = [node, leaf]

    with pytest.raises(PydanticSerializationError):
        dump_json(node)

    assert leaf.walks == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (b"raw", "b'raw'"),
        ({"doc": [b"raw"]}, {"doc": ["b'raw'"]}),
        ({b"key": b"raw"}, {"b'key'": "b'raw'"}),
        (_Doc(body={"x": b"raw"}), {"body": {"x": "b'raw'"}}),
        (_Model(body=[b"raw"]), {"body": ["b'raw'"]}),
        ({"s": {b"raw"}}, {"s": ["b'raw'"]}),
    ],
)
def test_bytes_as_repr_writes_them_instead_of_refusing(
    value: Any, expected: Any
):
    assert json.loads(dump_json(value, bytes_as_repr=True)) == expected


@pytest.mark.parametrize("build", [list, dict, _Doc, _Model])
def test_bytes_as_repr_survives_a_cycle(build: Any):
    """A diagnostic column must not fail on what it is recording."""
    value = _cycle(build, b"raw")

    written = json.loads(dump_json(value, bytes_as_repr=True))

    assert "<circular reference>" in json.dumps(written)


def test_bytes_as_repr_keeps_a_shared_branch_on_both_paths():
    """Only a genuine cycle is elided, not a branch referenced twice."""
    shared = {"raw": b"raw"}

    written = json.loads(
        dump_json({"a": shared, "b": shared}, bytes_as_repr=True)
    )

    assert written == {"a": {"raw": "b'raw'"}, "b": {"raw": "b'raw'"}}


class _WithExcluded(BaseModel):
    body: Any
    secret: Annotated[Any, Field(exclude=True)] = None


@pytest.mark.parametrize("bytes_as_repr", [False, True])
def test_an_excluded_model_field_is_not_walked(bytes_as_repr: bool):
    """``to_json`` never writes it, so neither path should look at it."""
    value = _WithExcluded(body="ok", secret=b"raw")

    written = json.loads(dump_json(value, bytes_as_repr=bytes_as_repr))

    assert written == {"body": "ok"}
