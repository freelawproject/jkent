"""The generated form model: controls, a whole ``FormCase``, and rendering.

A :class:`FormCase` is the single source a Hypothesis example produces. From it
the rig derives everything the transports and the oracle consume:

* ``rendered_html`` — the form exactly as a server would send it. This is what
  the Playwright transports stage into a tab and what the HTTP transport parses
  via ``find_form`` to reconstruct its request. Extra (non-rendered) fields are
  *not* here — that's the whole point of them.
* ``oracle_html`` — the same form, but with the extra fields materialised as
  real ``<input type=hidden>`` elements appended before ``</form>``. A vanilla
  browser submitting this form is the ground truth: a hidden input with a
  name/value *is* submitted, so this is what "the server should have received"
  means for an extra field.
* ``overrides`` / ``submit_selector`` — the arguments a scraper would pass to
  ``Form.submit(...)``: value edits for the filled text fields, the activated
  submit button's name/value, and the extra fields.
* ``native_fills`` / ``click_selector`` — the high-level actions the oracle
  browser performs (type into a field, click the chosen submit) so its
  submission expresses the *same intent* as the transports' request.

Equivalence is kept airtight by construction (see the per-kind notes below):
checkbox/radio/select state lives in the *rendered* markup (document order,
no override) so ordering matches a browser; only genuinely-empty text fields
are "filled" via an override that replaces an existing key in place. Where a
divergence is expected by design (extra fields, non-first submit buttons) the
differential tests target it deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from jkent.data_types import Selector

# Control kinds.
TEXT = "text"
TEXTAREA = "textarea"
HIDDEN = "hidden"
SELECT = "select"
MULTISELECT = "multiselect"
CHECKBOXES = "checkboxes"
RADIOS = "radios"
SUBMIT = "submit"
# Disabled variants — rendered with a submit-worthy value/checked state, but a
# browser never submits a disabled control. ``find_form``'s union XPath filters
# them out (``not(@disabled)``); these assert it stays in lockstep with the
# oracle, which omits them too.
DISABLED_TEXT = "disabled_text"
DISABLED_CHECKBOX = "disabled_checkbox"


@dataclass(frozen=True)
class Control:
    """One generated form control.

    Attributes:
        kind: One of the ``*`` constants above.
        name: The control's ``name`` (constrained to a selector-safe charset).
        elem_id: Stable DOM id (``c0``, ``c1`` …) for submit_selector targeting.
        value: Rendered initial value (text/hidden/textarea) or, when ``fill``
            is set, the value to type into an otherwise-empty field.
        fill: Text/textarea only — render empty and type ``value`` (a real
            "fill" action mirrored by both the transport override and the
            oracle's native typing).
        options: All option/checkbox/radio values for group kinds.
        chosen: Which of ``options`` are rendered selected/checked. Group state
            is encoded in the markup (not via overrides) so it submits in
            document order, matching a browser.
        label: Submit button value.
        activated: Whether this submit button is the one clicked/submitted.
        submit_as_input: SUBMIT only — render as ``<input type="submit">``
            instead of ``<button type="submit">``. An input submit only
            captures an id for submit_selector matching if ``find_form``
            records it, so this exercises that path (``<button>`` always did).
    """

    kind: str
    name: str
    elem_id: str
    value: str = ""
    fill: bool = False
    options: tuple[str, ...] = ()
    chosen: tuple[str, ...] = ()
    label: str = ""
    activated: bool = False
    submit_as_input: bool = False

    # --- rendering --------------------------------------------------------

    def render(self) -> str:
        """Render this control to an HTML fragment (rendered == oracle here)."""
        i = self.elem_id
        n = _attr(self.name)
        if self.kind in (TEXT, HIDDEN, DISABLED_TEXT):
            typ = "hidden" if self.kind == HIDDEN else "text"
            dis = " disabled" if self.kind == DISABLED_TEXT else ""
            val = "" if self.fill else self.value
            return (
                f'<input id="{i}" type="{typ}" name="{n}" '
                f'value="{_attr(val)}"{dis}>'
            )
        if self.kind == TEXTAREA:
            body = "" if self.fill else escape(self.value)
            return f'<textarea id="{i}" name="{n}">{body}</textarea>'
        if self.kind in (SELECT, MULTISELECT):
            multiple = " multiple" if self.kind == MULTISELECT else ""
            opts = "".join(
                f'<option value="{_attr(o)}"'
                f"{' selected' if o in self.chosen else ''}>"
                f"{escape(o)}</option>"
                for o in self.options
            )
            return f'<select id="{i}" name="{n}"{multiple}>{opts}</select>'
        if self.kind in (CHECKBOXES, RADIOS):
            typ = "checkbox" if self.kind == CHECKBOXES else "radio"
            boxes = "".join(
                f'<input id="{i}_{j}" type="{typ}" name="{n}" '
                f'value="{_attr(o)}"'
                f"{' checked' if o in self.chosen else ''}>"
                for j, o in enumerate(self.options)
            )
            return boxes
        if self.kind == DISABLED_CHECKBOX:
            # Rendered checked + disabled: a browser submits nothing.
            return (
                f'<input id="{i}" type="checkbox" name="{n}" '
                f'value="{_attr(self.value)}" checked disabled>'
            )
        if self.kind == SUBMIT:
            if self.submit_as_input:
                return (
                    f'<input id="{i}" type="submit" name="{n}" '
                    f'value="{_attr(self.label)}">'
                )
            return (
                f'<button id="{i}" type="submit" name="{n}" '
                f'value="{_attr(self.label)}">{escape(self.label) or "go"}'
                "</button>"
            )
        raise AssertionError(f"unknown control kind: {self.kind}")

    # --- request overrides + oracle actions -------------------------------

    def override(self) -> dict[str, str]:
        """This control's contribution to ``Form.submit(data=...)``.

        Only filled text/textarea fields and the activated submit button add an
        override; group state and disabled controls contribute nothing here.
        """
        if self.kind in (TEXT, TEXTAREA) and self.fill:
            return {self.name: self.value}
        if self.kind == SUBMIT and self.activated and self.name:
            return {self.name: self.label}
        return {}

    def native_fill(self) -> tuple[str, str] | None:
        """A ``(selector, value)`` the oracle types natively, or ``None``.

        Targeted by ``#id`` (not ``[name=]``) so a control sharing its name with
        another is still addressed unambiguously.
        """
        if self.kind in (TEXT, TEXTAREA) and self.fill:
            return (f"#{self.elem_id}", self.value)
        return None


@dataclass(frozen=True)
class FormCase:
    """A whole generated form plus the chosen submission."""

    method: str  # "GET" | "POST"
    controls: tuple[Control, ...]
    extras: tuple[
        tuple[str, str], ...
    ] = ()  # non-rendered fields (name,value)
    # Positional ``Form.submit(data=)`` overrides for a repeated (same-named)
    # control group: ``name -> (value-or-None, …)`` aligned to the same-named
    # controls in document order. A ``None`` keeps that control's rendered
    # default (the oracle leaves it untouched); a value replaces it (the oracle
    # types it into that control by id). Exercises the list-fill path where one
    # name maps to several distinct elements.
    list_overrides: tuple[tuple[str, tuple[str | None, ...]], ...] = ()

    FORM_ID = "f"

    @property
    def activated_submit(self) -> Control | None:
        for c in self.controls:
            if c.kind == SUBMIT and c.activated:
                return c
        return None

    @property
    def submit_selector(self) -> str:
        sub = self.activated_submit
        if sub is not None:
            return f"#{sub.elem_id}"
        # Fall back to the first submit button.
        for c in self.controls:
            if c.kind == SUBMIT:
                return f"#{c.elem_id}"
        raise AssertionError("FormCase has no submit button")

    def _form_open(self, action: str) -> str:
        return (
            f'<form id="{self.FORM_ID}" method="{self.method.lower()}" '
            f'action="{_attr(action)}">'
        )

    def rendered_html(self, action: str) -> str:
        """The form as a server sends it (no extra fields)."""
        body = "".join(c.render() for c in self.controls)
        return _page(self._form_open(action) + body + "</form>")

    def oracle_html(self, action: str) -> str:
        """The form a vanilla browser submits: extras as real hidden inputs.

        Extras are appended last so a browser submits them after the rendered
        controls — matching where ``Form.submit`` places new override keys.
        """
        body = "".join(c.render() for c in self.controls)
        body += "".join(
            f'<input type="hidden" name="{_attr(n)}" value="{_attr(v)}">'
            for n, v in self.extras
        )
        return _page(self._form_open(action) + body + "</form>")

    def overrides(self) -> dict[str | Selector, str | list[str | None]]:
        """The ``data=`` dict for ``Form.submit`` (fills, submit, extras)."""
        data: dict[str | Selector, str | list[str | None]] = {}
        for c in self.controls:
            for name, value in c.override().items():
                data[name] = value
        # Positional list overrides for a same-named group (None entries kept so
        # Form.submit resolves them to that control's rendered default).
        for name, values in self.list_overrides:
            data[name] = list(values)
        # Extras last so new keys append after the rendered defaults, matching
        # the oracle's trailing hidden inputs.
        for n, v in self.extras:
            data[n] = v
        return data

    def native_fills(self) -> list[tuple[str, str]]:
        """The ``(selector, value)`` pairs the oracle types before submitting."""
        out: list[tuple[str, str]] = []
        for c in self.controls:
            nf = c.native_fill()
            if nf is not None:
                out.append(nf)
        # Each non-None list override is typed into its same-named control by id,
        # in document order; None entries are left at their rendered default.
        for name, values in self.list_overrides:
            members = [c for c in self.controls if c.name == name]
            for member, value in zip(members, values):
                if value is not None:
                    out.append((f"#{member.elem_id}", value))
        return out


# --- helpers --------------------------------------------------------------


def _attr(value: str) -> str:
    """Escape a string for use inside a double-quoted HTML attribute."""
    return escape(value, quote=True)


def _page(form_html: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'></head>"
        f"<body>{form_html}</body></html>"
    )
