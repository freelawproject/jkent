"""How a request was produced — the ``via`` models.

A via records the browser action that yielded a
:class:`~jkent.common.request.Request` (following a link, submitting a form)
so the Playwright transport can replay it; the HTTP transport reads only the
request itself. ``via_json`` is the stored form and :func:`via_from_json` its
one reader.

A leaf: imports only :mod:`jkent.common.selectors`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, TypeAdapter

from jkent.common.selectors import Selector

__all__ = [
    "FieldResolver",
    "FieldValue",
    "Via",
    "ViaFormSubmit",
    "ViaLink",
    "via_from_json",
]


class _ViaBase(PydanticBaseModel):
    """Shared config for the via models.

    Frozen: a via describes an action already taken, so nothing should
    rewrite one after the fact.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    def to_json(self) -> str:
        """This via as the ``via_json`` wire format (see :func:`via_from_json`)."""
        return self.model_dump_json()


class ViaLink(_ViaBase):
    """Describes a request produced by following a link.

    Enables the Playwright driver to replay the browser action that corresponds
    to the request (clicking the link). The HTTP driver ignores this field.

    Attributes:
        type: Discriminator for the ``via_json`` union.
        selector: The :class:`Selector` that found the <a> element. Its grammar
            lets the driver route it to Playwright's engine without re-running
            the prefix heuristic.
        description: Human-readable description of the link.
    """

    type: Literal["link"] = "link"
    selector: Selector
    description: str


# A form field's submitted value, or an async resolver for one. A resolver
# is a zero-arg async callable the driver awaits when the yielded request is
# enqueued — use it to fetch the value from an external service (e.g. an
# image-captcha solver), closing over whatever the call needs. Only concrete
# values ever reach the DB or the transports.
FieldValue = str | list[str]
FieldResolver = Callable[[], Awaitable[FieldValue]]


class ViaFormSubmit(_ViaBase):
    """Describes a request produced by submitting a form.

    Enables the Http and Browser based transports to have a unified interface for
    submitting a form.

    Attributes:
        type: Discriminator for the ``via_json`` union.
        form_selector: The :class:`Selector` that found the <form> element. Its
            grammar lets the driver route it to Playwright's engine without
            re-running the prefix heuristic.
        submit_selector: Selector relative to the form for the submit element.
        field_data: Merged field values (defaults + overrides). A list value
            means repeated keys (checkbox groups, multi-selects). A
            :data:`FieldResolver` value is awaited at enqueue time (see
            :meth:`Request.resolve_deferred_fields`) — serializing one is an
            error, which is why the driver resolves them before storing.
        description: Human-readable description of the form.
    """

    type: Literal["form_submit"] = "form_submit"
    form_selector: Selector
    submit_selector: str | None
    field_data: dict[str, FieldValue | FieldResolver]
    description: str


#: The ``via_json`` column's content: one of the via models, told apart by
#: their ``type`` field. A third via is one more class and one more literal —
#: the writer and the reader both follow from the union.
Via = Annotated[ViaLink | ViaFormSubmit, Field(discriminator="type")]

_VIA_ADAPTER: TypeAdapter[Via] = TypeAdapter(Via)


def via_from_json(raw: str) -> ViaLink | ViaFormSubmit:
    """Rebuild a via from the ``via_json`` wire format.

    The inverse of :meth:`ViaLink.to_json` / :meth:`ViaFormSubmit.to_json` —
    the single reader for the ``via_json`` column consumers store and inspect
    (the driver's queue, a host's request reconstruction). Shape:
    link → ``{type, selector, description}``; form_submit →
    ``{type, form_selector, submit_selector, field_data, description}``, where
    each selector is ``{value, grammar}``.

    The reader is strict about that shape: a flat
    ``selector``/``selector_type`` pair is rejected rather than parsed by
    guessing the grammar from the selector's prefix.

    Raises:
        ValueError: The JSON does not match either via shape.
    """
    return _VIA_ADAPTER.validate_json(raw)
