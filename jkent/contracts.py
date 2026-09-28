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
from typing import Any, Literal, Protocol, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

ENFORCE_CONTRACTS: bool = os.environ.get(
    "JKENT_ENFORCE_CONTRACTS", "0"
) not in ("", "0")


class ContractDecorator(Protocol):
    """A decorator that hands back the decorated callable's own type.

    The return type of :func:`require` / :func:`ensure`. (A callback
    protocol scopes the type variable to ``__call__`` instead of leaving
    it free in the factory's own signature, as a plain ``Callable[[F], F]``
    return annotation would.) The parameter is positional-only so that
    icontract's decorator classes, whose ``__call__`` names it ``func``,
    satisfy the protocol structurally.
    """

    def __call__(self, fn: F, /) -> F: ...


def require(
    condition: Callable[..., object], description: str | None = None
) -> ContractDecorator:
    """``icontract.require`` when contracts are on; identity otherwise."""
    return _gate("require", condition, description)


def ensure(
    condition: Callable[..., object], description: str | None = None
) -> ContractDecorator:
    """``icontract.ensure`` when contracts are on; identity otherwise."""
    return _gate("ensure", condition, description)


def _gate(
    kind: Literal["require", "ensure"],
    condition: Callable[..., object],
    description: str | None,
) -> ContractDecorator:
    """The named icontract decorator, or the identity when contracts are off."""
    if not ENFORCE_CONTRACTS:
        return lambda fn: fn
    # Dev-only dependency, imported only when contracts are enabled.
    import icontract  # noqa: PLC0415

    factory = icontract.require if kind == "require" else icontract.ensure
    return factory(condition, description=description)
