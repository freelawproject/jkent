"""Hypothesis strategies producing :class:`FormCase` variants.

Each entry point isolates one comparison dimension so a failure attributes to a
single behaviour:

* :func:`happy_cases` — POST forms built only from input/button controls
  (``find_form`` preserves their document order) with single-valued or
  repeated-key (checkbox/radio) fields. The well-behaved baseline: every
  transport is expected to match the browser, order included.
* :func:`cross_type_cases` — forms containing a ``<select>`` or ``<textarea>``
  *before* the trailing submit button. ``find_form`` emits inputs/buttons,
  then selects, then textareas — so document order is not preserved across
  types. Probed with an order-sensitive comparison.
* :func:`get_multivalue_cases` — GET forms with a checkbox group / multi-select
  that submits a key more than once. Probes repeated-key query encoding.
* :func:`disabled_cases` — POST forms with at least one disabled control.
* :func:`extra_cases` — POST forms with at least one field added only via
  ``Form.submit(data=)`` (absent from the rendered markup).
* :func:`multi_submit_cases` — POST forms with two or three submit buttons and a
  *non-first* one activated.
* :func:`duplicate_name_cases` — POST forms where several *distinct* controls
  (text/hidden/textarea) share one ``name``, so a key repeats from independent
  elements rather than from a single checkbox group / multi-select. Probed with
  an order-sensitive comparison.
* :func:`duplicate_fill_cases` — same same-named cluster, but some controls are
  overridden via a positional ``data={name: [...]}`` list (with ``None`` entries
  keeping a control's rendered default). Probes the repeated-field fill path.

Field *names* are synthetic and selector-safe (``n0``, ``s0``, ``x0`` …) on
purpose — the rig stresses value/structure fidelity, not name escaping, and the
Playwright fill path builds ``[name="…"]`` / ``[value="…"]`` selectors that a
quote in a name would break. Option/checkbox/radio *values* and submit labels
are likewise selector-safe tokens (they flow through those selectors); text,
textarea and extra-field *values* draw from a wide set (spaces, ``&``, ``=``,
``+``, ``%``, ``<``, unicode) where encoding fidelity actually matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from hypothesis import strategies as st

from tests.form_conformance.model import (
    CHECKBOXES,
    DISABLED_CHECKBOX,
    DISABLED_TEXT,
    HIDDEN,
    MULTISELECT,
    RADIOS,
    SELECT,
    SUBMIT,
    TEXT,
    TEXTAREA,
    Control,
    FormCase,
)

if TYPE_CHECKING:
    from hypothesis.strategies import DrawFn

# Selector-safe token (option/checkbox/radio values + submit labels).
_TOKEN = st.from_regex(r"[a-zA-Z0-9_-]{1,6}", fullmatch=True)
# Free-form value: printable ASCII including spaces and the punctuation where
# encoding fidelity matters (&, =, +, %, <, >, quotes). Control chars and line
# breaks sit below codepoint 32 and are excluded; the codepoint-bounded form
# matches the other generative rigs (test_replay_*).
_WILD = st.text(
    st.characters(min_codepoint=32, max_codepoint=126),
    max_size=8,
)

# Input/button kinds: ``find_form`` keeps these in document order, so they form
# the order-safe baseline. Selects/textareas are added only by cross_type_cases.
_INPUT_KINDS = (TEXT, HIDDEN, CHECKBOXES, RADIOS)
_CROSS_KINDS = (SELECT, MULTISELECT, TEXTAREA)
# Kinds whose single submitted value lives entirely in the rendered markup
# (no fill, no override). Several can share one name and a browser submits each
# value exactly once, in document order — so they can repeat a key from
# *distinct* elements without the Form.submit override dict (keyed by name)
# having to express the repeat.
_DUP_KINDS = (TEXT, HIDDEN, TEXTAREA)


def _tokens(draw: DrawFn, lo: int, hi: int) -> tuple[str, ...]:
    return tuple(draw(st.lists(_TOKEN, min_size=lo, max_size=hi, unique=True)))


def _subset(draw: DrawFn, options: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(o for o in options if draw(st.booleans()))


def _plain_control(draw: DrawFn, kind: str, idx: int) -> Control:
    """Build one non-submit control (synthetic name ``n{idx}`` / id ``c{idx}``)."""
    name, elem_id = f"n{idx}", f"c{idx}"
    if kind in (TEXT, TEXTAREA):
        return Control(
            kind=kind,
            name=name,
            elem_id=elem_id,
            value=draw(_WILD),
            fill=draw(st.booleans()),
        )
    if kind == HIDDEN:
        return Control(
            kind=kind, name=name, elem_id=elem_id, value=draw(_WILD)
        )
    if kind in (SELECT, RADIOS):
        options = _tokens(draw, 1, 4)
        # Single-valued: render zero or one chosen. A single <select> with none
        # chosen still submits its first option (browser + find_form agree).
        chosen = draw(st.sampled_from([(), *[(o,) for o in options]]))
        return Control(
            kind=kind,
            name=name,
            elem_id=elem_id,
            options=options,
            chosen=chosen,
        )
    if kind in (MULTISELECT, CHECKBOXES):
        options = _tokens(draw, 1, 4)
        return Control(
            kind=kind,
            name=name,
            elem_id=elem_id,
            options=options,
            chosen=_subset(draw, options),
        )
    raise AssertionError(f"not a plain kind: {kind}")


def _disabled_control(draw: DrawFn, idx: int) -> Control:
    name, elem_id = f"n{idx}", f"c{idx}"
    if draw(st.booleans()):
        return Control(
            kind=DISABLED_TEXT,
            name=name,
            elem_id=elem_id,
            value=draw(_WILD.filter(bool)),
        )
    return Control(
        kind=DISABLED_CHECKBOX, name=name, elem_id=elem_id, value=draw(_TOKEN)
    )


def _submit(
    idx: int,
    *,
    name: str,
    label: str,
    activated: bool,
    as_input: bool = False,
) -> Control:
    return Control(
        kind=SUBMIT,
        name=name,
        elem_id=f"c{idx}",
        label=label,
        activated=activated,
        submit_as_input=as_input,
    )


def _plain_controls(
    draw: DrawFn, kinds: tuple[str, ...], start: int, lo: int, hi: int
) -> tuple[list[Control], int]:
    """Draw ``lo..hi`` plain controls from ``kinds``; return them and next idx."""
    out: list[Control] = []
    idx = start
    for _ in range(draw(st.integers(min_value=lo, max_value=hi))):
        out.append(_plain_control(draw, draw(st.sampled_from(kinds)), idx))
        idx += 1
    return out, idx


def _single_submit(draw: DrawFn, idx: int) -> Control:
    has_name = draw(st.booleans())
    return _submit(
        idx,
        name=("s0" if has_name else ""),
        label=draw(_TOKEN),
        activated=True,
        as_input=draw(st.booleans()),
    )


@st.composite
def happy_cases(draw: DrawFn) -> FormCase:
    controls, idx = _plain_controls(draw, _INPUT_KINDS, 0, 0, 4)
    controls.append(_single_submit(draw, idx))
    return FormCase(method="POST", controls=tuple(controls))


@st.composite
def cross_type_cases(draw: DrawFn) -> FormCase:
    # Select/textarea(s) FIRST, then input/button control(s), then a NAMED
    # submit. The submit + inputs (which the old find_form pulls ahead of
    # selects/textareas) sit *after* the cross controls in the document, so any
    # reorder is observable. POST so repeated-key encoding (finding E) can't
    # contaminate the ordering signal; find_form ordering is method-independent.
    controls: list[Control] = []
    idx = 0
    for _ in range(draw(st.integers(min_value=1, max_value=2))):
        controls.append(
            _plain_control(draw, draw(st.sampled_from(_CROSS_KINDS)), idx)
        )
        idx += 1
    extra_inputs, idx = _plain_controls(draw, _INPUT_KINDS, idx, 0, 2)
    controls.extend(extra_inputs)
    controls.append(
        _submit(
            idx,
            name="s0",
            label=draw(_TOKEN),
            activated=True,
            as_input=draw(st.booleans()),
        )
    )
    return FormCase(method="POST", controls=tuple(controls))


@st.composite
def get_multivalue_cases(draw: DrawFn) -> FormCase:
    controls, idx = _plain_controls(draw, _INPUT_KINDS, 0, 0, 2)
    # A checkbox group (or multi-select) that submits its key more than once.
    options = _tokens(draw, 2, 4)
    kind = draw(st.sampled_from([CHECKBOXES, MULTISELECT]))
    controls.append(
        Control(
            kind=kind,
            name=f"n{idx}",
            elem_id=f"c{idx}",
            options=options,
            chosen=options,  # all chosen -> repeated key on submit
        )
    )
    idx += 1
    controls.append(_single_submit(draw, idx))
    return FormCase(method="GET", controls=tuple(controls))


@st.composite
def duplicate_name_cases(draw: DrawFn) -> FormCase:
    """Several *distinct* controls that share one ``name``.

    A repeated key produced by independent elements (``<input>`` / ``<input>`` /
    ``<textarea>`` all named ``d0``) rather than by a single checkbox group or
    multi-select. Each duplicate carries its value in the markup — text/hidden
    ``value=``, textarea body — never via a fill/override: ``Form.submit``'s
    ``data=`` dict is keyed by name, so an override would collapse the whole
    accumulated list to one scalar. The airtight-vs-browser intent is therefore
    "submit exactly what the markup says", and a browser submits each such
    control once in document order.

    Order-sensitive: ``find_form`` must accumulate *every* same-named control
    (finding A's single document-order pass) and the repeated key must survive
    POST body encoding just as a checkbox group's does. Leading/trailing
    unique-named controls let the duplicate cluster sit non-contiguously, so
    interleaved repeated and distinct keys are checked for order too.
    """
    shared = "d0"
    controls, idx = _plain_controls(draw, (TEXT, HIDDEN), 0, 0, 2)
    for _ in range(draw(st.integers(min_value=2, max_value=4))):
        controls.append(
            Control(
                kind=draw(st.sampled_from(_DUP_KINDS)),
                name=shared,
                elem_id=f"c{idx}",
                value=draw(_WILD),
            )
        )
        idx += 1
    tail, idx = _plain_controls(draw, (TEXT, HIDDEN), idx, 0, 1)
    controls.extend(tail)
    controls.append(_single_submit(draw, idx))
    return FormCase(method="POST", controls=tuple(controls))


@st.composite
def duplicate_fill_cases(draw: DrawFn) -> FormCase:
    """Override *some* of several same-named controls via a positional list.

    A cluster of 2..4 fillable controls (text/textarea) share one ``name``. The
    scraper passes ``data={name: [...]}`` where each entry either replaces that
    control's value or is ``None`` to keep its rendered default. Guaranteed to
    draw at least one of each, so both the fill path (``Form.submit`` fills the
    matching control positionally) and the ``None``-keeps-default path are
    exercised every example. A browser submits the cluster in document order
    (overridden values where given, rendered defaults elsewhere); the rig
    asserts every transport matches, order included.
    """
    shared = "d0"
    controls, idx = _plain_controls(draw, (TEXT, HIDDEN), 0, 0, 2)
    count = draw(st.integers(min_value=2, max_value=4))
    for _ in range(count):
        controls.append(
            Control(
                kind=draw(st.sampled_from((TEXT, TEXTAREA))),
                name=shared,
                elem_id=f"c{idx}",
                value=draw(
                    _WILD
                ),  # rendered default (kept where override None)
            )
        )
        idx += 1
    values: list[str | None] = []
    for _ in range(count):
        keep_default = draw(st.booleans())
        values.append(None if keep_default else draw(_WILD))
    # Force a mix so each example covers both a positional fill and a kept None.
    if all(v is not None for v in values):
        values[draw(st.integers(0, count - 1))] = None
    if all(v is None for v in values):
        values[draw(st.integers(0, count - 1))] = draw(_WILD)
    controls.append(_single_submit(draw, idx))
    return FormCase(
        method="POST",
        controls=tuple(controls),
        list_overrides=((shared, tuple(values)),),
    )


@st.composite
def disabled_cases(draw: DrawFn) -> FormCase:
    controls, idx = _plain_controls(draw, _INPUT_KINDS, 0, 0, 3)
    for _ in range(draw(st.integers(min_value=1, max_value=2))):
        controls.append(_disabled_control(draw, idx))
        idx += 1
    controls.append(_single_submit(draw, idx))
    return FormCase(method="POST", controls=tuple(controls))


@st.composite
def extra_cases(draw: DrawFn) -> FormCase:
    controls, idx = _plain_controls(draw, _INPUT_KINDS, 0, 0, 3)
    controls.append(_single_submit(draw, idx))
    n = draw(st.integers(min_value=1, max_value=3))
    extras = tuple((f"x{k}", draw(_WILD)) for k in range(n))
    return FormCase(method="POST", controls=tuple(controls), extras=extras)


@st.composite
def multi_submit_cases(draw: DrawFn) -> FormCase:
    controls, idx = _plain_controls(draw, _INPUT_KINDS, 0, 0, 3)
    count = draw(st.integers(min_value=2, max_value=3))
    names = draw(
        st.lists(
            st.sampled_from(["btn", "act", "go"]),
            min_size=count,
            max_size=count,
        )
    )
    activated_pos = draw(st.integers(min_value=1, max_value=count - 1))
    for j in range(count):
        controls.append(
            _submit(
                idx,
                name=names[j],
                label=draw(_TOKEN),
                activated=(j == activated_pos),
                as_input=draw(st.booleans()),
            )
        )
        idx += 1
    return FormCase(method="POST", controls=tuple(controls))
