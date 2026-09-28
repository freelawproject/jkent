"""The one JSON codec for the run database's ``*_json`` columns.

Every JSON column is written through :func:`dump_json` (or
:func:`dump_json_or_none` — NULL means ``None``, nothing else), which is
``pydantic_core.to_json``: it serializes pydantic models, dataclasses, dates,
``Decimal``, sets and nested containers natively, and falls back to ``str()``
for any other type it does not know — so a scraped value of an unexpected
type lands as text instead of raising from inside the write path. Output is
compact (no spaces after separators).

Raw bytes (``bytes``, ``bytearray``, ``memoryview``) are the exception: JSON
has no bytes type, and nothing on the read side would decode an encoding of
them, so they would come back as a different value. Anywhere in the value —
nested containers, dict keys, dataclass and pydantic-model fields — they
raise :class:`TypeError` naming where they were found. Decode them to text
(or keep them out of JSON columns) before writing. Diagnostic columns — an
error's context and failed document, an invalid result — pass
``bytes_as_repr=True`` instead, which writes each as its ``repr`` (and
elides a circular reference), so recording a failure never fails on what
it is recording.

Reads decode on the row model: a field annotated with :data:`JsonColumn`
(``Annotated[T, JsonColumn]``) parses the column text as the model is
validated from the row.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, TypeVar, overload

from pydantic import BaseModel, BeforeValidator
from pydantic_core import to_json

__all__ = ["JsonColumn", "dump_json", "dump_json_or_none", "parse_json_column"]

_BYTES_TYPES = (bytes, bytearray, memoryview)

T = TypeVar("T")


def _refuse_bytes(value: Any, path: str, seen: set[int]) -> None:
    """Raise :class:`TypeError` for the first raw-bytes value in *value*.

    Walks what :func:`~pydantic_core.to_json` would: mappings (keys and
    values), lists, tuples, sets, dataclass fields and pydantic-model
    fields — skipping a model field ``to_json`` would not serialize
    (``Field(exclude=True)``). *seen* holds every container already walked, so each is walked
    once however many times it is referenced — a container that is reached
    a second time was walked in full the first time and raised then if it
    held bytes — and a cycle stops the walk, left for ``to_json`` to
    report. Nothing is removed from *seen*: everything reached is held
    alive by the value being walked, so an id cannot be reused mid-walk.
    """
    if isinstance(value, _BYTES_TYPES):
        raise TypeError(
            f"{path}: {type(value).__name__} cannot be stored in a JSON "
            "column; decode it to str first"
        )
    if isinstance(value, str | int | float) or value is None:
        return
    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, _BYTES_TYPES):
                _refuse_bytes(key, f"{path} (key {key!r})", seen)
            _refuse_bytes(item, f"{path}[{key!r}]", seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _refuse_bytes(item, f"{path}[{index}]", seen)
    elif isinstance(value, (set, frozenset)):
        for item in value:
            _refuse_bytes(item, f"{path}{{...}}", seen)
    elif isinstance(value, BaseModel):
        for field, info in type(value).model_fields.items():
            if info.exclude:
                continue
            _refuse_bytes(getattr(value, field), f"{path}.{field}", seen)
        for field, item in (value.__pydantic_extra__ or {}).items():
            _refuse_bytes(item, f"{path}.{field}", seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for dc_field in dataclasses.fields(value):
            _refuse_bytes(
                getattr(value, dc_field.name),
                f"{path}.{dc_field.name}",
                seen,
            )


def _bytes_to_repr(value: Any, seen: set[int]) -> Any:
    """*value* with every raw-bytes value (and dict key) replaced by its repr.

    Models and dataclasses become plain dicts of their fields (minus a
    model field ``Field(exclude=True)`` keeps out of JSON) so those are
    reached; everything else is returned as is. *seen* holds the containers
    on the current path: a value that refers back into itself is replaced
    by ``"<circular reference>"`` rather than recursed into, since this is
    the path a diagnostic column takes and it must not fail on what it is
    recording.
    """
    if isinstance(value, _BYTES_TYPES):
        return repr(value)
    if isinstance(value, str | int | float) or value is None:
        return value
    if id(value) in seen:
        return "<circular reference>"
    seen.add(id(value))
    try:
        if isinstance(value, BaseModel):
            value = {
                field: getattr(value, field)
                for field, info in type(value).model_fields.items()
                if not info.exclude
            } | (value.__pydantic_extra__ or {})
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            value = {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        if isinstance(value, dict):
            return {
                _bytes_to_repr(key, seen): _bytes_to_repr(item, seen)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_bytes_to_repr(item, seen) for item in value]
        return value
    finally:
        seen.discard(id(value))


def dump_json(
    value: Any, *, name: str = "value", bytes_as_repr: bool = False
) -> str:
    """*value* as compact JSON text.

    Args:
        value: What to encode.
        name: What *value* is, as the root of the path a refusal names
            (``accumulated_data`` → ``accumulated_data['doc'][2]``).
        bytes_as_repr: Write raw bytes as their ``repr`` instead of
            refusing them — for diagnostic columns only.

    Raises:
        TypeError: *value* holds raw bytes somewhere and ``bytes_as_repr``
            is false.
    """
    if bytes_as_repr:
        value = _bytes_to_repr(value, set())
    else:
        _refuse_bytes(value, name, set())
    return to_json(value, fallback=str).decode()


@overload
def dump_json_or_none(
    value: None, *, name: str = ..., bytes_as_repr: bool = ...
) -> None: ...
@overload
def dump_json_or_none(
    value: Any, *, name: str = ..., bytes_as_repr: bool = ...
) -> str: ...
def dump_json_or_none(
    value: Any, *, name: str = "value", bytes_as_repr: bool = False
) -> str | None:
    """:func:`dump_json`, except ``None`` stays ``None`` (a NULL column).

    Only ``None`` is treated as absent: a falsy-but-real value (``0``,
    ``[]``, ``False``) is still encoded.
    """
    if value is None:
        return None
    return dump_json(value, name=name, bytes_as_repr=bytes_as_repr)


@overload
def parse_json_column(value: str) -> Any: ...
@overload
def parse_json_column(value: T) -> T: ...
def parse_json_column(value: Any) -> Any:
    """``BeforeValidator`` for a model field fed from a ``*_json`` column.

    Text is parsed; anything else passes through untouched, so the same
    field validates both from a row (JSON text) and from a re-validated
    ``model_dump`` (already decoded).
    """
    return json.loads(value) if isinstance(value, str) else value


#: ``Annotated`` metadata for a row-model field fed from a ``*_json`` column.
JsonColumn = BeforeValidator(parse_json_column)
