"""Selectors and their grammars — the one place a grammar is dispatched on.

A :class:`Selector` carries its value *and* the grammar it is written in, so
nothing downstream has to re-run a prefix heuristic to find out. There are
exactly two grammars, :class:`CSS` and :class:`XPath`, and each
grammar-specific behaviour is an abstract method here implemented by both:
the lxml query (:meth:`Selector.query`), the positional wrapper
(:meth:`Selector.nth`), relative-chain composition (:meth:`Selector.compose`),
and the Playwright-waitability predicate (:meth:`Selector.can_playwright_wait`).

A selector always states its grammar: there is no bare-string form and no
inference. :meth:`Selector.of` exists only to rebuild one from a stored
``(value, grammar)`` pair.

A leaf: it imports nothing from jkent. It never names the parse-tree
implementation either — :meth:`Selector.query` takes an
:class:`ElementQuery`, the structural protocol a page element satisfies, so
the lxml backing stays behind :class:`~jkent.common.page_element.PageElement`
rather than surfacing in a selector's signature. The via models, the
page-element layer, the drivers, and the authoring facade
(``jkent.data_types``) all sit above it.

:class:`Selector` also carries its own pydantic schema, so a model that holds
one (a via) serializes it as ``{"value": …, "grammar": …}`` and validates it
straight back through :meth:`Selector.of` — the grammar round-trips as data,
never as a guess.
"""

from __future__ import annotations

import functools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeAlias

from lxml import etree
from pydantic_core import core_schema
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic import GetCoreSchemaHandler

__all__ = ["CSS", "ElementQuery", "Grammar", "Selector", "XPath"]

#: The selector grammars jkent supports. Stored verbatim in a serialized
#: selector and in ``HTMLStructuralAssumptionException.selector_type``.
Grammar = Literal["css", "xpath"]

#: What running a selector against an element yields: a node-set of elements
#: and/or strings (attributes, text nodes), or a bare scalar for an XPath
#: like ``count()``/``string()``. ``Sequence`` rather than ``list`` so an
#: implementer's ``list[ConcreteElement]`` satisfies it — ``list`` is
#: invariant, ``Sequence`` is covariant.
QueryResult: TypeAlias = "Sequence[ElementQuery | str] | str | float | bool"


class ElementQuery(Protocol):
    """The raw query surface :meth:`Selector.query` dispatches into.

    One method per grammar, named for it. Structural rather than nominal so
    this module stays a leaf: a
    :class:`~jkent.common.page_element.PageElement` satisfies it, and so does
    the lxml element it wraps, without either being named here.

    Self-referential on purpose: a query yields things you can query again,
    stated without naming a concrete element type. That is as far as this
    protocol goes, though — it is a *query* surface, not an element one, so
    it carries no ``text_content``/``get``/``tag``. Reading a result means
    narrowing it to a concrete type, which is what
    ``PageElement._checked`` does before wrapping each node.

    Both methods take their expression positionally: lxml names the
    parameter ``_path`` and accepts it positional-only, so a keyword-capable
    protocol parameter would exclude ``HtmlElement`` from satisfying this.

    These are the *unchecked* queries — no count validation, no observer
    record, results as the parser hands them back. Scrapers go through
    ``PageElement.query``/``checked_xpath``; only a Selector calls these.
    """

    def cssselect(self, expr: str, /) -> Sequence[ElementQuery]:
        """Every element matching the CSS selector ``expr``.

        CSS can only select elements, so this never yields strings.
        """
        ...

    def xpath(self, expr: str, /) -> QueryResult:
        """``expr`` evaluated: a node-set, or a scalar for ``count()`` etc.

        The node-set is mixed: ``//a`` yields elements, ``//a/@href`` and
        ``//a/text()`` yield strings.
        """
        ...


@dataclass(frozen=True)
class Selector(ABC):
    """A selector string together with the grammar it is written in.

    Abstract: construct a :class:`CSS` or an :class:`XPath`, or rebuild one
    from stored parts with :meth:`of`. A selector without a grammar has no
    meaning, since the grammar is what every downstream dispatch reads.

    Attributes:
        value: The raw selector string.
        grammar: ``"css"`` or ``"xpath"`` — a ClassVar set by each subclass.
        label: Human-readable grammar name, for error messages.
        playwright_engine: Playwright's name for this grammar, used as the
            explicit ``engine=`` prefix.
    """

    value: str
    grammar: ClassVar[Grammar]
    label: ClassVar[str]
    playwright_engine: ClassVar[Grammar]

    CSS: ClassVar[type[CSS]]
    XPath: ClassVar[type[XPath]]

    @classmethod
    def of(cls, value: str, grammar: str) -> Selector:
        """Rebuild a Selector from its serialized ``value``/``grammar`` parts.

        The only way to reach a Selector from a grammar *name* — for the
        deserializers and the driver call sites that hold a stored
        ``(value, grammar)`` pair rather than a Selector.

        Raises:
            ValueError: ``grammar`` is neither ``"css"`` nor ``"xpath"``.
        """
        if grammar == CSS.grammar:
            return CSS(value)
        if grammar == XPath.grammar:
            return XPath(value)
        raise ValueError(f"unknown selector grammar: {grammar!r}")

    @abstractmethod
    def nth(self, position: int) -> Selector:
        """A selector for the 1-based ``position``-th match of this one.

        Each subclass encodes the positional wrapper in its own grammar so a
        single matched node can be replayed unambiguously.
        """

    @abstractmethod
    def query(self, element: ElementQuery) -> QueryResult:
        """Run this selector against an lxml element, in its own grammar.

        A node-set comes back as a list; a scalar XPath
        (``count()``/``string()``/…) comes back as a bare value.
        """

    def for_playwright(self) -> str:
        """This selector as a Playwright selector string.

        Always engine-prefixed rather than left to Playwright's
        auto-detection: :meth:`nth` wraps XPath as ``(...)[n]``, a form
        Playwright does *not* recognize as XPath.
        """
        return f"{self.playwright_engine}={self.value}"

    def can_playwright_wait(self) -> bool:
        """Whether ``page.wait_for_selector()`` can wait on this selector.

        Playwright can only wait on something that targets *elements*. This
        is a property of the *wait* context only: a selector this rejects is
        still perfectly valid for parsing, which is why the constructors do
        not reject it. The driver reads this to decide whether a failed
        structural assumption can be retried behind an autowait, and falls
        back to re-raising when it cannot.
        """
        return True

    @classmethod
    @abstractmethod
    def compose(cls, parts: Sequence[str]) -> str | None:
        """Compose a root-to-leaf chain of same-grammar selectors into one.

        ``parts[0]`` is the absolute root; the rest are relative to their
        parent. ``None`` when this grammar cannot express the composition.
        """

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

    grammar: ClassVar[Grammar] = "css"
    label: ClassVar[str] = "CSS"
    playwright_engine: ClassVar[Grammar] = "css"

    @override
    def nth(self, position: int) -> Selector:
        # Playwright's :nth-match() picks the position-th match document-wide,
        # mirroring how the parse enumerated the CSS matches.
        return CSS(f":nth-match({self.value}, {position})")

    @override
    def query(self, element: ElementQuery) -> QueryResult:
        return element.cssselect(self.value)

    @override
    def can_playwright_wait(self) -> bool:
        """False for cssselect's ``:contains()``, which Playwright lacks.

        lxml's cssselect evaluates it, so a parse can use it; Playwright
        hands a pseudo-class it does not define to the browser, which
        rejects the selector. Its equivalent there is ``:has-text()``.
        Constructing such a selector stays legal — only waiting on one is
        refused (see :meth:`Selector.can_playwright_wait`).

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

    grammar: ClassVar[Grammar] = "xpath"
    label: ClassVar[str] = "XPath"
    playwright_engine: ClassVar[Grammar] = "xpath"

    @override
    def nth(self, position: int) -> Selector:
        # Parenthesize first so the positional predicate applies to the whole
        # node-set rather than only the last location step.
        return XPath(f"({self.value})[{position}]")

    @override
    def query(self, element: ElementQuery) -> QueryResult:
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
