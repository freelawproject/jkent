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
``bytes_as_repr=True`` instead, which writes each as its ``repr``, so
recording a failure never fails on what it is recording.

Reads decode on the row model: a field annotated with :data:`JsonColumn`
(``Annotated[T, JsonColumn]``) parses the column text as the model is
validated from the row.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from pydantic import BaseModel, BeforeValidator
from pydantic_core import to_json

__all__ = ["JsonColumn", "dump_json", "dump_json_or_none", "parse_json_column"]

_BYTES_TYPES = (bytes, bytearray, memoryview)


def _refuse_bytes(value: Any, path: str, seen: set[int]) -> None:
    """Raise :class:`TypeError` for the first raw-bytes value in *value*.

    Walks what :func:`~pydantic_core.to_json` would: mappings (keys and
    values), lists, tuples, sets, dataclass fields and pydantic-model
    fields. *seen* holds the containers on the current path, so a cycle
    stops the walk and is left for ``to_json`` to report.
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
    try:
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
            for field in type(value).model_fields:
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
    finally:
        seen.discard(id(value))


def _bytes_to_repr(value: Any) -> Any:
    """*value* with every raw-bytes value (and dict key) replaced by its repr.

    Models and dataclasses are dumped to plain containers first so their
    fields are reached; everything else is returned as is.
    """
    if isinstance(value, _BYTES_TYPES):
        return repr(value)
    if isinstance(value, BaseModel):
        value = value.model_dump()
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, dict):
        return {
            _bytes_to_repr(key): _bytes_to_repr(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_bytes_to_repr(item) for item in value]
    return value


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
        value = _bytes_to_repr(value)
    else:
        _refuse_bytes(value, name, set())
    return to_json(value, fallback=str).decode()


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


def parse_json_column(value: Any) -> Any:
    """``BeforeValidator`` for a model field fed from a ``*_json`` column.

    Text is parsed; anything else passes through untouched, so the same
    field validates both from a row (JSON text) and from a re-validated
    ``model_dump`` (already decoded).
    """
    return json.loads(value) if isinstance(value, str) else value


#: ``Annotated`` metadata for a row-model field fed from a ``*_json`` column.
JsonColumn = BeforeValidator(parse_json_column)
