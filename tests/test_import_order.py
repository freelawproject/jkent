"""Every jkent module imports first, in a cold interpreter.

An in-process test cannot see an import cycle: by the time it runs,
``sys.modules`` is already populated in whatever order hid it. So each
module is imported in a fresh subprocess, as the first jkent import.
Packages alone are not enough: a cycle entered through a leaf module (one
that imports a sibling package whose ``__init__`` imports the leaf back)
is invisible when the package is imported first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import jkent

_ROOT = Path(jkent.__file__).parent


def _modules() -> list[str]:
    names = []
    for path in sorted(_ROOT.rglob("*.py")):
        parts = path.relative_to(_ROOT.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


MODULES = _modules()


def test_module_list_is_populated() -> None:
    assert "jkent.driver.database_engine.errors" in MODULES


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_cold(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
