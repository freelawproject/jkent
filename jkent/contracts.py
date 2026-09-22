"""Dev-time-only contract decorators.

A thin gate over :mod:`icontract`. When ``JKENT_ENFORCE_CONTRACTS`` is
unset or ``0`` (any production install), :func:`require` and
:func:`ensure` return the decorated function untouched — no wrapper, no
condition evaluation, and ``icontract`` itself is never imported, so it
can live in the dev dependency group rather than the SDK's runtime
dependencies.

When the variable is set to anything else, the real icontract
decorators are applied and violations raise. The test suite flips it on
for every run (top of ``tests/conftest.py``); CrossHair runs against
production modules need it too::

    JKENT_ENFORCE_CONTRACTS=1 uv run crosshair check \
        --analysis_kind=icontract <module>

The gate is evaluated at decoration (i.e. import) time, so the variable
must be set before any ``jkent`` module is imported.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])

ENFORCE_CONTRACTS: bool = os.environ.get(
    "JKENT_ENFORCE_CONTRACTS", "0"
) not in ("", "0")


class ContractDecorator(Protocol):
    """A decorator that hands back the decorated callable's own type.

    The return type of :func:`require` / :func:`ensure`. (A plain
    ``Callable[[F], F]`` return annotation leaves the type variable
    free in the signature, which pyre rejects; a callback protocol
    scopes it to ``__call__``.)
    """

    def __call__(self, fn: F) -> F: ...


# Class-statement keywords for Protocols whose method stubs carry
# contracts::
#
#     class RateLimiter(Protocol, **DBC_PROTOCOL_KW): ...
#
# With contracts on, this injects a metaclass composing icontract's
# DBCMeta (which propagates contracts to overriding methods) with
# Protocol's own metaclass, so the Protocol's contracts are inherited
# by every implementation that *explicitly subclasses* it. Purely
# structural conformers get no enforcement — contract attachment
# happens at class creation. ``runtime_checkable`` isinstance checks
# keep working: the MRO finds ``_ProtocolMeta.__instancecheck__``
# before ``ABCMeta``'s.
#
# With contracts off the dict is empty and the class statement is
# exactly a plain Protocol. The keywords go through ``**`` because
# type checkers (mypy and pyre both) reject a dynamic ``metaclass=``
# expression outright — passed this way they see a plain Protocol,
# which is also precisely what production gets.
DBC_PROTOCOL_KW: dict[str, type] = {}
if not TYPE_CHECKING and ENFORCE_CONTRACTS:
    import icontract  # noqa: PLC0415 — dev-only dep, contracts are on

    # pyre-ignore[31]: pyre can't model type(Protocol) as a base class;
    # this branch is runtime-only (dev, contracts on) anyway.
    class _DBCProtocolMeta(icontract.DBCMeta, type(Protocol)):
        """DBCMeta composed with Protocol's metaclass."""

    DBC_PROTOCOL_KW = {"metaclass": _DBCProtocolMeta}


def require(
    condition: Callable[..., object], description: str | None = None
) -> ContractDecorator:
    """``icontract.require`` when contracts are on; identity otherwise."""
    if not ENFORCE_CONTRACTS:
        return lambda fn: fn
    # Dev-only dependency, imported only when contracts are enabled.
    import icontract  # noqa: PLC0415

    # icontract's decorators are classes with a generic __call__; type
    # checkers won't structurally match them against the protocol.
    return cast(
        "ContractDecorator",
        icontract.require(condition, description=description),
    )


def ensure(
    condition: Callable[..., object], description: str | None = None
) -> ContractDecorator:
    """``icontract.ensure`` when contracts are on; identity otherwise."""
    if not ENFORCE_CONTRACTS:
        return lambda fn: fn
    # Dev-only dependency, imported only when contracts are enabled.
    import icontract  # noqa: PLC0415

    # icontract's decorators are classes with a generic __call__; type
    # checkers won't structurally match them against the protocol.
    return cast(
        "ContractDecorator",
        icontract.ensure(condition, description=description),
    )
