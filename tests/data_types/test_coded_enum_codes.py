"""``CodedEnum`` subclasses keep their codes: a storage format.

A member's ``.code`` is written into run databases that outlive the
process, so renumbering one silently reinterprets every stored row (see
``jkent.common.coded_enum``). The generic laws — codes unique, at least 1,
``from_code`` inverting ``.code``, the label round-tripping through the
enum's by-value lookup — hold for every ``class X(CodedEnum):`` found in the
source tree;
the golden snapshot at the bottom pins the actual numbers, so a renumber,
a recycled code, or a deleted member fails here before it corrupts a
database. Appending a member means appending to the snapshot.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

import jkent
from jkent.common.coded_enum import CodedEnum

_SUBCLASS_DEF = re.compile(r"^class (\w+)\(CodedEnum\):", re.MULTILINE)
_PACKAGE_ROOT = Path(jkent.__file__).parent


def _coded_enums() -> dict[str, type[CodedEnum]]:
    """Every ``class X(CodedEnum)`` defined under the ``jkent`` package.

    Found by scanning source rather than ``__subclasses__`` so a subclass in
    a module nothing else imports still counts. Docstring examples (the
    ``Color`` in ``coded_enum.py``'s module docstring) are indented, and
    the anchored pattern skips them.
    """
    found: dict[str, type[CodedEnum]] = {}
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        names = _SUBCLASS_DEF.findall(path.read_text())
        if not names:
            continue
        parts = list(path.relative_to(_PACKAGE_ROOT).with_suffix("").parts)
        dotted = ".".join(["jkent", *parts]).removesuffix(".__init__")
        module = importlib.import_module(dotted)
        for name in names:
            found[name] = getattr(module, name)
    return found


CODED_ENUMS = _coded_enums()

#: Golden snapshot of every code in the package. Generated from the enums
#: as they stand; a diff here is a storage-format change and needs a
#: migration story, not a snapshot update.
CODE_SNAPSHOT: dict[str, dict[str, int]] = {
    "ErrorType": {
        "STRUCTURAL": 1,
        "VALIDATION": 2,
        "TRANSIENT": 3,
        "PERSISTENT": 4,
        "UNKNOWN": 5,
    },
    "HttpMethod": {
        "GET": 1,
        "OPTIONS": 2,
        "POST": 3,
        "PUT": 4,
        "DELETE": 5,
        "PATCH": 6,
        "HEAD": 7,
    },
    "RequestStatus": {
        "PENDING": 1,
        "IN_PROGRESS": 2,
        "COMPLETED": 3,
        "FAILED": 4,
        "STUBBED": 6,
    },
    "RequestType": {
        "NAVIGATING": 1,
        "NON_NAVIGATING": 2,
        "ARCHIVE": 3,
    },
    "RunStatus": {
        "CREATED": 1,
        "RUNNING": 2,
        "COMPLETED": 3,
        "ERROR": 4,
        "INTERRUPTED": 5,
    },
    "SelectorType": {
        "CSS": 1,
        "XPATH": 2,
    },
    "SpeculationOutcome": {
        "HIT": 1,
        "STOPPED": 2,
        "TERMINATED_EARLY": 3,
        "MISS": 4,
    },
    "TransientKind": {
        "NETWORK": 1,
        "BROWSER_CRASH": 2,
        "NAVIGATION": 3,
        "INTERACTION": 4,
        "ARCHIVE": 5,
        "SNAPSHOT": 6,
        "INTERSTITIAL": 7,
        "SITE_DEGRADED": 8,
    },
}


def test_every_coded_enum_is_in_the_snapshot():
    assert set(CODED_ENUMS) == set(CODE_SNAPSHOT)


def test_coded_enum_laws():
    """``CodedEnum``'s own accessors, against a throwaway vocabulary.

    Uniqueness and ``code >= 1`` are enforced in ``CodedEnum.__new__`` at
    class-creation time, so what is left to assert is the accessors:
    ``codes()`` ascending, ``from_code`` inverting ``.code``, the by-label
    lookup, and ``str``. They live on the base class, so one vocabulary
    exercises them for every subclass.
    """

    class Color(CodedEnum):
        BLUE = (2, "blue")
        RED = (1, "red")

    assert Color.codes() == [1, 2]
    for member in Color:
        assert Color.from_code(member.code) is member
        assert Color(member.value) is member
        assert str(member) == member.value


@pytest.mark.parametrize(
    "enum_class", CODED_ENUMS.values(), ids=list(CODED_ENUMS)
)
def test_coded_enum_codes_match_snapshot(enum_class: type[CodedEnum]):
    actual = {member.name: member.code for member in enum_class}
    assert actual == CODE_SNAPSHOT[enum_class.__name__]
