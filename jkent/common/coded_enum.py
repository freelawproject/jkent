"""An enum whose members carry both a readable label and a stored integer code.

Some enums in jkent are persisted to the run database as small integers, but
are read, compared, and reported as their names. :class:`CodedEnum` declares
both in one place: a member's value is its label, and ``.code`` is the integer
the database holds.

Declare members as ``NAME = (code, "label")``::

    class Colour(CodedEnum):
        RED = (1, "red")
        BLUE = (2, "blue")

    Colour.RED.code       # 1  -- what the database stores
    Colour.RED.value      # "red"
    Colour.RED == "red"   # True
    f"{Colour.RED}"       # "red"
    Colour.from_code(1)   # <Colour.RED>

Members are ``str`` subclasses so they drop into string contexts unchanged —
comparisons against literals, dict keys, JSON payloads, log lines, and the
progress/callback surfaces the driver hands to hosts. ``__str__`` is pinned to
the value because plain ``str, Enum`` renders ``Colour.RED`` on 3.10
(:class:`enum.StrEnum` does this for us from 3.11).

**Codes are a storage format.** A member's code is written into databases that
outlive the process, so changing one silently reinterprets every stored row,
and reusing a retired member's code silently resurrects it as something else.
Codes may be appended, never renumbered or recycled. Nothing here can enforce
that — it is why each code is spelled out at its member rather than derived
from declaration order, where inserting a member would shift everything after
it.

Codes start at 1 rather than 0 so no member is falsy when a raw integer is read
straight out of the database, outside the mapping this class provides.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["CodedEnum"]


class CodedEnum(str, enum.Enum):
    """Base for enums stored as an integer but handled as their label.

    Has no members of its own, so it can be subclassed (an ``Enum`` that
    already has members cannot be).
    """

    # Both are set after the fact — ``code`` per member in ``__new__``, the
    # cache on first ``_code_index`` call — which pyre reports as an
    # uninitialized attribute.
    code: int  # pyre-ignore[13]
    _code_index_cache: ClassVar[dict[int, Any]]  # pyre-ignore[13]

    if TYPE_CHECKING:
        # Type checkers resolve a *functional* call — ``RequestStatus("pending")``,
        # the by-label lookup that ``EnumMeta.__call__`` actually serves — against
        # ``__new__``. Declaring the real ``(code, label)`` signature here would
        # make every such lookup an error, so checkers see the lookup form and
        # the member-construction signature stays runtime-only.
        def __new__(cls, value: object) -> Self: ...

    else:

        def __new__(cls, code, label):
            """Build a member whose ``str`` payload is *label*, coded *code*."""
            if not isinstance(code, int) or isinstance(code, bool):
                raise TypeError(
                    f"{cls.__name__}: code must be an int, got {code!r}"
                )
            if code < 1:
                raise ValueError(
                    f"{cls.__name__}: codes start at 1, got {code!r} "
                    "(0 would make the member falsy as a raw database value)"
                )
            obj = str.__new__(cls, label)
            obj._value_ = label
            obj.code = code
            return obj

    __str__ = str.__str__

    @classmethod
    def from_code(cls, code: int) -> Self:
        """Return the member stored as *code*.

        The decode path for anything reading these columns outside the ORM —
        raw ``SELECT``s, ``sqlite3`` sessions, tooling in sibling repos.

        Raises:
            LookupError: If no member carries that code.
        """
        try:
            return _code_index(cls)[code]
        except KeyError:
            raise LookupError(
                f"{cls.__name__} has no member with code {code!r}; "
                f"known codes: {sorted(_code_index(cls))}"
            ) from None

    @classmethod
    def codes(cls) -> list[int]:
        """Every code this enum defines, ascending."""
        return sorted(_code_index(cls))


def _code_index(cls: type[CodedEnum]) -> dict[int, Any]:
    """Cached ``{code: member}`` for *cls*, verifying codes are unique.

    Built on first use rather than at class creation: ``__new__`` runs per
    member, so there is no point during class body execution at which the
    whole set is known.
    """
    # Read off ``__dict__`` rather than with ``getattr`` so a subclass builds
    # its own index instead of inheriting the base's.
    cached = cls.__dict__.get("_code_index_cache")
    if cached is not None:
        return cached
    index: dict[int, Any] = {}
    for member in cls:
        if member.code in index:
            raise ValueError(
                f"{cls.__name__}: code {member.code} is used by both "
                f"{index[member.code].name} and {member.name}; codes "
                "identify stored rows and must be unique"
            )
        index[member.code] = member
    cls._code_index_cache = index
    return index
