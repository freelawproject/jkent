"""Selectors and their grammars.

A :class:`Selector` carries its value *and* the grammar it is written in, so
nothing downstream has to re-run a prefix heuristic to find out.

A leaf: it imports nothing from jkent. The via models, the page-element
layer, the drivers, and the authoring facade (``jkent.data_types``) all sit
above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from typing_extensions import override


@dataclass(frozen=True)
class Selector:
    """A selector string together with the grammar it is written in.

    Attributes:
        value: The raw selector string.
        grammar: ``"css"`` or ``"xpath"`` — a ClassVar set by each subclass.
    """

    value: str
    grammar: ClassVar[str] = ""

    CSS: ClassVar[type[CSS]]
    XPath: ClassVar[type[XPath]]

    @classmethod
    def of(cls, value: str, grammar: str) -> Selector:
        """Rebuild a Selector from its serialized ``value``/``grammar`` parts."""
        if grammar == XPath.grammar:
            return XPath(value)
        if grammar == CSS.grammar:
            return CSS(value)
        raise ValueError(f"unknown selector grammar: {grammar!r}")

    def nth(self, position: int) -> Selector:
        """A selector for the 1-based ``position``-th match of this one.

        Each subclass encodes the positional wrapper in its own grammar so a
        single matched node can be replayed unambiguously.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class CSS(Selector):
    """A CSS selector. Also reachable as :attr:`Selector.CSS`."""

    grammar: ClassVar[str] = "css"

    @override
    def nth(self, position: int) -> Selector:
        # Playwright's :nth-match() picks the position-th match document-wide,
        # mirroring how the parse enumerated the CSS matches.
        return CSS(f":nth-match({self.value}, {position})")


@dataclass(frozen=True)
class XPath(Selector):
    """An XPath selector. Also reachable as :attr:`Selector.XPath`."""

    grammar: ClassVar[str] = "xpath"

    @override
    def nth(self, position: int) -> Selector:
        # Parenthesize first so the positional predicate applies to the whole
        # node-set rather than only the last location step.
        return XPath(f"({self.value})[{position}]")


Selector.CSS = CSS
Selector.XPath = XPath
