"""Selectors and their grammars — the one place a grammar is dispatched on.

A :class:`Selector` carries its value *and* the grammar it is written in, so
nothing downstream has to re-run a prefix heuristic to find out. Each
grammar-specific behaviour is a method here, overridden per subclass: the
lxml query (:meth:`Selector.query`), the Playwright-engine prefix
(:meth:`Selector.for_playwright`), relative-chain composition
(:meth:`Selector.compose`), and the Playwright-waitability predicate
(:meth:`Selector.can_playwright_wait`). Adding a grammar is one subclass — it
registers itself, and :meth:`Selector.of` finds it.

A leaf: it imports nothing from jkent. The via models, the page-element
layer, the drivers, and the authoring facade (``jkent.data_types``) all sit
above it.

:class:`Selector` also carries its own pydantic schema, so a model that holds
one (a via) serializes it as ``{"value": …, "grammar": …}`` and validates it
straight back through :meth:`Selector.of` — the grammar round-trips as data,
never as a guess.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from lxml import etree
from pydantic_core import core_schema
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import GetCoreSchemaHandler

__all__ = ["CSS", "Selector", "XPath"]


@dataclass(frozen=True)
class Selector:
    """A selector string together with the grammar it is written in.

    Attributes:
        value: The raw selector string.
        grammar: ``"css"`` or ``"xpath"`` — a ClassVar set by each subclass,
            and the key it registers itself under.
        label: Human-readable grammar name, for error messages.
        playwright_engine: Playwright's name for this grammar, used as the
            explicit ``engine=`` prefix.
    """

    value: str
    grammar: ClassVar[str] = ""
    label: ClassVar[str] = ""
    playwright_engine: ClassVar[str] = ""

    CSS: ClassVar[type[CSS]]
    XPath: ClassVar[type[XPath]]

    #: ``grammar`` → subclass, populated by :meth:`__init_subclass__`. The
    #: registry :meth:`of` reads, so deserialization needs no if-chain.
    _BY_GRAMMAR: ClassVar[dict[str, type[Selector]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        grammar = cls.__dict__.get("grammar")
        if not grammar:
            # An intermediate subclass that adds behaviour but not a grammar
            # (nothing does today) stays out of the registry, and
            # ``__post_init__`` refuses to construct it.
            return
        existing = Selector._BY_GRAMMAR.get(grammar)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"selector grammar {grammar!r} is already registered to "
                f"{existing.__name__}"
            )
        Selector._BY_GRAMMAR[grammar] = cls

    def __post_init__(self) -> None:
        """Refuse a selector with no grammar.

        The base class and any intermediate subclass that sets no
        ``grammar`` are not constructible: a selector's grammar is what every
        downstream dispatch reads, so one without it has no meaning. Build a
        concrete grammar (:class:`CSS`, :class:`XPath`) or go through
        :meth:`of`.

        Raises:
            TypeError: ``type(self)`` has no ``grammar``.
        """
        if not type(self).grammar:
            raise TypeError(
                f"{type(self).__name__} has no grammar; construct a concrete "
                "selector (CSS, XPath) or use Selector.of(value, grammar)"
            )

    @classmethod
    def grammar_class(cls, grammar: str) -> type[Selector] | None:
        """The registered subclass for ``grammar``, or ``None``.

        For callers holding a grammar name but no value — they want the
        grammar's classmethods (:meth:`compose`), not an instance.
        """
        return Selector._BY_GRAMMAR.get(grammar)

    @classmethod
    def of(cls, value: str, grammar: str) -> Selector:
        """Rebuild a Selector from its serialized ``value``/``grammar`` parts.

        Raises:
            ValueError: ``grammar`` names no registered subclass.
        """
        subclass = Selector._BY_GRAMMAR.get(grammar)
        if subclass is None:
            raise ValueError(f"unknown selector grammar: {grammar!r}")
        return subclass(value)

    @classmethod
    def infer(cls, value: str) -> Selector:
        """Best-effort Selector for a bare string with no recorded grammar.

        Mirrors ``find_form``/``find_links``: unambiguous XPath prefixes are
        XPath, everything else CSS. A fallback only: guessing is exactly what
        :class:`Selector` exists to stop.
        """
        return (
            XPath(value) if value.startswith(("//", "./", "(")) else CSS(value)
        )

    def nth(self, position: int) -> Selector:
        """A selector for the 1-based ``position``-th match of this one.

        Each subclass encodes the positional wrapper in its own grammar so a
        single matched node can be replayed unambiguously.
        """
        raise NotImplementedError

    def query(self, element: Any) -> Any:
        """Run this selector against an lxml element, in its own grammar."""
        raise NotImplementedError

    def for_playwright(self) -> str:
        """This selector as a Playwright selector string.

        Always engine-prefixed rather than left to Playwright's
        auto-detection: :meth:`nth` wraps XPath as ``(...)[n]``, a form
        Playwright does *not* recognize as XPath.
        """
        return f"{self.playwright_engine}={self.value}"

    def can_playwright_wait(self) -> bool:
        """Whether ``page.wait_for_selector()`` can wait on this selector.

        Playwright can only wait on something that targets *elements*.
        """
        return True

    @classmethod
    def compose(cls, parts: Sequence[str]) -> str | None:
        """Compose a root-to-leaf chain of same-grammar selectors into one.

        ``parts[0]`` is the absolute root; the rest are relative to their
        parent. ``None`` when this grammar cannot express the composition.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.value

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        """Serialize as ``{"value", "grammar"}``; validate via :meth:`of`.

        Declared on the base so a field annotated ``Selector`` accepts any
        registered grammar and rebuilds the right subclass — the discriminator
        is the stored ``grammar``, not the shape of the value.
        """

        def from_parts(data: dict[str, Any]) -> Selector:
            try:
                return Selector.of(data["value"], data["grammar"])
            except KeyError as e:
                raise ValueError(f"selector is missing {e.args[0]!r}") from e

        parts_schema = core_schema.chain_schema(
            [
                core_schema.typed_dict_schema(
                    {
                        "value": core_schema.typed_dict_field(
                            core_schema.str_schema()
                        ),
                        "grammar": core_schema.typed_dict_field(
                            core_schema.str_schema()
                        ),
                    }
                ),
                core_schema.no_info_plain_validator_function(from_parts),
            ]
        )
        return core_schema.json_or_python_schema(
            json_schema=parts_schema,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(Selector), parts_schema]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda sel: {"value": sel.value, "grammar": sel.grammar},
                return_schema=core_schema.typed_dict_schema(
                    {
                        "value": core_schema.typed_dict_field(
                            core_schema.str_schema()
                        ),
                        "grammar": core_schema.typed_dict_field(
                            core_schema.str_schema()
                        ),
                    }
                ),
            ),
        )


# CSS string literals, backslash escapes included; checked out like XPath's
# so an attribute value spelling ":contains(" is not a pseudo-class.
_CSS_STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
_CSS_CONTAINS = re.compile(r":contains\s*\(", re.IGNORECASE)


@dataclass(frozen=True)
class CSS(Selector):
    """A CSS selector. Also reachable as :attr:`Selector.CSS`."""

    grammar: ClassVar[str] = "css"
    label: ClassVar[str] = "CSS"
    playwright_engine: ClassVar[str] = "css"

    @override
    def nth(self, position: int) -> Selector:
        # Playwright's :nth-match() picks the position-th match document-wide,
        # mirroring how the parse enumerated the CSS matches.
        return CSS(f":nth-match({self.value}, {position})")

    @override
    def query(self, element: Any) -> Any:
        return element.cssselect(self.value)

    @override
    def can_playwright_wait(self) -> bool:
        """False for cssselect's ``:contains()``, which Playwright lacks.

        lxml's cssselect evaluates it, so a parse can use it; Playwright
        hands a pseudo-class it does not define to the browser, which
        rejects the selector. Its equivalent there is ``:has-text()``.

        Examples:
            >>> CSS("td.docket").can_playwright_wait()
            True
            >>> CSS("td:contains('Docket')").can_playwright_wait()
            False
        """
        structural = _CSS_STRING_LITERAL.sub("''", self.value)
        return _CSS_CONTAINS.search(structural) is None

    @classmethod
    @override
    def compose(cls, parts: Sequence[str]) -> str | None:
        # The descendant combinator is a space.
        return " ".join(parts)


# XPath string literals have no escaping, so a regex can excise them exactly.
# The structural checks below run on the literal-free form: content inside
# quotes ("score: 5", "a | b") must not trip them.
_XPATH_STRING_LITERAL = re.compile(r"'[^']*'|\"[^\"]*\"")
# Common EXSLT namespace prefixes, lowercase; matched case-insensitively.
# lxml evaluates these; Playwright binds none of them.
_EXSLT_PREFIXES = (
    "re:",
    "str:",
    "math:",
    "set:",
    "dyn:",
    "exsl:",
    "func:",
    "date:",
)
# Axes whose nodes are never elements.
_NON_ELEMENT_AXES = frozenset({"attribute", "namespace"})
# Node tests that match (or may match) non-element nodes.
_NON_ELEMENT_NODE_TEST = re.compile(
    r"(?:text|comment|node)\s*\(\s*\)|processing-instruction\s*\(.*\)"
)
_OPENERS = {"[": "]", "(": ")"}


def _at_depth_zero(expression: str, char: str) -> list[int]:
    """Indices of *char* in *expression* outside all brackets and parens.

    *expression* must already have its string literals excised.
    """
    positions: list[int] = []
    depth = 0
    for index, current in enumerate(expression):
        if current == char and depth == 0:
            positions.append(index)
        if current in _OPENERS:
            depth += 1
        elif current in _OPENERS.values():
            depth -= 1
    return positions


def _split_union(expression: str) -> list[str]:
    """*expression*'s top-level ``|`` branches (literals already excised)."""
    bounds = [-1, *_at_depth_zero(expression, "|"), len(expression)]
    return [
        expression[start + 1 : stop] for start, stop in zip(bounds, bounds[1:])
    ]


def _selects_elements(branch: str) -> bool:
    """Whether a union branch's last location step can select elements.

    The last step is what follows the branch's last top-level ``/``, with
    its predicates dropped (``//div/text()[1]`` still selects text). A
    parenthesized step (``(//a/@href)[1]``) is judged by what it wraps.
    """
    branch = branch.strip()
    slashes = _at_depth_zero(branch, "/")
    step = branch[slashes[-1] + 1 :] if slashes else branch
    predicates = _at_depth_zero(step, "[")
    if predicates:
        step = step[: predicates[0]]
    step = step.strip()
    if step.startswith("(") and step.endswith(")"):
        return all(
            _selects_elements(inner) for inner in _split_union(step[1:-1])
        )
    if step.startswith("@"):
        return False
    axis, separator, node_test = step.partition("::")
    if separator and axis.strip() in _NON_ELEMENT_AXES:
        return False
    return not _NON_ELEMENT_NODE_TEST.fullmatch(
        (node_test if separator else step).strip()
    )


@functools.lru_cache(maxsize=1024)
def _evaluates_to_node_set(expression: str) -> bool:
    """Whether *expression* compiles and yields a node-set.

    Evaluated against an empty probe document: a node-set comes back as a
    list (empty here), while a number, string or boolean result means the
    expression selects no nodes at all. EXSLT regexp support is off, so
    only XPath 1.0 — what the browser evaluates — compiles. Predicates never
    run against the empty probe, so a namespace prefix inside one is not
    detected (see :meth:`XPath.can_playwright_wait`).
    """
    try:
        result = etree.XPath(expression, regexp=False)(
            etree.fromstring("<_/>")
        )
    except (etree.XPathError, ValueError):
        # ValueError: text lxml refuses outright (NUL, control characters).
        return False
    return isinstance(result, list)


@dataclass(frozen=True)
class XPath(Selector):
    """An XPath selector. Also reachable as :attr:`Selector.XPath`."""

    grammar: ClassVar[str] = "xpath"
    label: ClassVar[str] = "XPath"
    playwright_engine: ClassVar[str] = "xpath"

    @override
    def nth(self, position: int) -> Selector:
        # Parenthesize first so the positional predicate applies to the whole
        # node-set rather than only the last location step.
        return XPath(f"({self.value})[{position}]")

    @override
    def query(self, element: Any) -> Any:
        return element.xpath(self.value)

    @override
    def can_playwright_wait(self) -> bool:
        """False for anything that doesn't resolve to elements.

        Playwright's ``wait_for_selector()`` waits on elements only. This is
        False for XPath variable references (``$var``, which it cannot bind),
        EXSLT functions, anything that does not compile, an expression whose
        value is not a node-set (``count(//a)``, ``//a = 'x'``), and a union
        with any branch whose last step selects attributes, namespaces, text,
        comments, processing instructions or ``node()``.

        Namespace prefixes are unsupported in a wait. Playwright binds none,
        so a prefixed name or function makes the browser's evaluation fail.
        A prefix on a step (``//foo:div``) or a common EXSLT prefix
        (``re:``, ``str:``, …) is detected and returns False; one inside a
        predicate (``//div[foo:bar(.)]``) is not, and returns True — a wait
        on it errors or burns its timeout.

        Examples:
            >>> XPath("//div[@class='content']").can_playwright_wait()
            True
            >>> XPath("//div/@href").can_playwright_wait()
            False
            >>> XPath("//div/text()").can_playwright_wait()
            False
            >>> XPath("//div[@id=$section]").can_playwright_wait()
            False
            >>> XPath("//a[text()='Price: $5']").can_playwright_wait()
            True
            >>> XPath("count(//a)").can_playwright_wait()
            False
        """
        # XPath's own whitespace only: str.strip() would also drop characters
        # (U+0085, U+00A0) the browser's XPath parser rejects.
        expression = self.value.strip(" \t\r\n")
        # Checked on the literal-free form so text like 'Price: $5' is fine.
        structural = _XPATH_STRING_LITERAL.sub("''", expression)
        if "$" in structural:
            return False
        folded = structural.casefold()
        if any(prefix in folded for prefix in _EXSLT_PREFIXES):
            return False
        if not _evaluates_to_node_set(expression):
            return False
        return all(_selects_elements(b) for b in _split_union(structural))

    @classmethod
    @override
    def compose(cls, parts: Sequence[str]) -> str | None:
        # parts[0] is the absolute root; the rest are relative, so strip the
        # leading "." and keep whatever combinator followed it.
        result = parts[0]
        for part in parts[1:]:
            if part.startswith("."):
                # ".//tr" -> "//tr" (descendant); "./tr" -> "/tr" (child);
                # a bare "." is the rare case and just loses the dot.
                result += part[1:]
            else:
                result += "//" + part
        return result


Selector.CSS = CSS
Selector.XPath = XPath
