"""How a request was produced — the ``via`` models.

A via records the browser action that yielded a
:class:`~jkent.data_types.Request` (following a link, submitting a form)
so the Playwright transport can replay it; the HTTP transport reads only the
request itself. ``via_json`` is the stored form and :func:`via_from_json` its
one reader.

A leaf: imports only :mod:`jkent.common.selectors`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from jkent.common.selectors import Selector


@dataclass(frozen=True)
class ViaLink:
    """Describes a request produced by following a link.

    Enables the Playwright driver to replay the browser action that corresponds
    to the request (clicking the link). The HTTP driver ignores this field.

    Attributes:
        selector: The :class:`Selector` that found the <a> element. Its grammar
            lets the driver route it to Playwright's engine without re-running
            the prefix heuristic.
        description: Human-readable description of the link.
    """

    selector: Selector
    description: str

    def to_json(self) -> str:
        """This via as the ``via_json`` wire format (see :func:`via_from_json`)."""
        return json.dumps(
            {
                "type": "link",
                "selector": self.selector.value,
                "selector_type": self.selector.grammar,
                "description": self.description,
            }
        )


# A form field's submitted value, or an async resolver for one. A resolver
# is a zero-arg async callable the driver awaits when the yielded request is
# enqueued — use it to fetch the value from an external service (e.g. an
# image-captcha solver), closing over whatever the call needs. Only concrete
# values ever reach the DB or the transports.
FieldValue = str | list[str]
FieldResolver = Callable[[], Awaitable[FieldValue]]


@dataclass(frozen=True)
class ViaFormSubmit:
    """Describes a request produced by submitting a form.

    Enables the Http and Browser based transports to have a unified interface for
    submitting a form.

    Attributes:
        form_selector: The :class:`Selector` that found the <form> element. Its
            grammar lets the driver route it to Playwright's engine without
            re-running the prefix heuristic.
        submit_selector: Selector relative to the form for the submit element.
        field_data: Merged field values (defaults + overrides). A list value
            means repeated keys (checkbox groups, multi-selects). A
            :data:`FieldResolver` value is awaited at enqueue time (see
            :meth:`Request.resolve_deferred_fields`).
        description: Human-readable description of the form.
    """

    form_selector: Selector
    submit_selector: str | None
    field_data: dict[str, FieldValue | FieldResolver]
    description: str

    def to_json(self) -> str:
        """This via as the ``via_json`` wire format (see :func:`via_from_json`).

        Requires concrete ``field_data`` values — a still-unresolved
        :data:`FieldResolver` is not JSON-serializable (the driver resolves
        them at enqueue time, before serialization).
        """
        return json.dumps(
            {
                "type": "form_submit",
                "form_selector": self.form_selector.value,
                "selector_type": self.form_selector.grammar,
                "submit_selector": self.submit_selector,
                "field_data": self.field_data,
                "description": self.description,
            }
        )


def _selector_grammar(selector: str) -> str:
    """Best-effort selector grammar for legacy via rows lacking selector_type.

    Mirrors ``find_form``/``find_links``: unambiguous XPath prefixes are
    "xpath", everything else "css". Only used as a fallback — rows written
    after selector_type was added carry the real value.
    """
    return "xpath" if selector.startswith(("//", "./", "(")) else "css"


def via_from_json(raw: str) -> ViaLink | ViaFormSubmit:
    """Rebuild a via from the ``via_json`` wire format.

    The inverse of :meth:`ViaLink.to_json` / :meth:`ViaFormSubmit.to_json` —
    the single reader for the ``via_json`` column consumers store and inspect
    (the driver's queue, jent's reconstruction). Shape:
    link → ``{type, selector, selector_type, description}``; form_submit →
    ``{type, form_selector, selector_type, submit_selector, field_data,
    description}``. Rows written before ``selector_type`` existed fall back
    to the prefix heuristic.

    Raises:
        ValueError: On an unknown ``type``.
    """
    data = json.loads(raw)
    kind = data.get("type")
    if kind == "form_submit":
        return ViaFormSubmit(
            form_selector=Selector.of(
                data["form_selector"],
                data.get(
                    "selector_type",
                    _selector_grammar(data["form_selector"]),
                ),
            ),
            submit_selector=data.get("submit_selector"),
            field_data=data["field_data"],
            description=data["description"],
        )
    if kind == "link":
        return ViaLink(
            selector=Selector.of(
                data["selector"],
                data.get("selector_type", _selector_grammar(data["selector"])),
            ),
            description=data["description"],
        )
    raise ValueError(f"unknown via type {kind!r} in via_json")
