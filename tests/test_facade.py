"""The import direction that keeps :mod:`jkent.data_types` cycle-free.

Leaves never import the facade; the facade imports leaves; ``jkent``
re-exports from the facade only, so every name ``jkent`` offers is also
reachable through ``jkent.data_types``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import jkent
import jkent.data_types

_ROOT = Path(jkent.__file__).parent


def _jkent_imports(path: Path) -> set[str]:
    """Every ``jkent…`` module *path* imports, at any nesting level."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return {m for m in found if m == "jkent" or m.startswith("jkent.")}


@pytest.mark.parametrize(
    "path",
    sorted((_ROOT / "common").rglob("*.py")),
    ids=lambda p: str(p.relative_to(_ROOT)),
)
def test_leaves_never_import_the_facade(path: Path) -> None:
    assert "jkent.data_types" not in _jkent_imports(path)


def test_package_reexports_from_the_facade_only() -> None:
    assert _jkent_imports(_ROOT / "__init__.py") == {"jkent.data_types"}


def test_every_package_name_is_on_the_facade() -> None:
    assert set(jkent.__all__) <= set(jkent.data_types.__all__)


@pytest.mark.parametrize("module", [jkent, jkent.data_types])
def test_all_names_resolve(module: object) -> None:
    for name in module.__all__:  # type: ignore[attr-defined]
        assert hasattr(module, name), name


def test_transient_kind_is_public() -> None:
    assert "TransientKind" in jkent.__all__
