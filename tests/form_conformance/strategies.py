"""Hypothesis strategies producing :class:`FormCase` variants.

Each entry point isolates one comparison dimension so a failure attributes to a
single behaviour:

* :func:`happy_cases` — POST forms built only from input/button controls
  (``find_form`` preserves their document order) with single-valued or
  repeated-key (checkbox/radio) fields. The well-behaved baseline: every
  transport is expected to match the browser, order included.
* :func:`cross_type_cases` — forms containing a ``<select>`` or ``<textarea>``
  *before* the input/button controls and the trailing submit button, so any
  cross-type reorder by ``find_form`` is observable. Probed with an
  order-sensitive comparison.
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
* :func:`group_override_cases` — POST forms whose radio/checkbox groups are
  overridden via ``data=``: a checkbox subset (empty included) or single
  value, a radio value, and values no box carries. Probes that the group
  submits exactly the override — rendered boxes it left out unchecked.
* :func:`select_override_cases` — POST forms whose ``<select>`` /
  ``<select multiple>`` are overridden via ``data=``, sometimes with a value
  no option carries, which the browser transports must inject.

Field *names* are synthetic and selector-safe (``n0``, ``s0``, ``x0`` …) on
purpose — the rig stresses value/structure fidelity, not name escaping, and the
Playwright fill path builds ``[name="…"]`` / ``[value="…"]`` selectors that a
quote in a name would break. Option/checkbox/radio *values* and submit labels
are likewise selector-safe tokens (they flow through those selectors); text,
textarea and extra-field *values* draw from a wide set (spaces, ``&``, ``=``,
``+``, ``%``, ``<``, unicode) where encoding fidelity actually matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
# Free-form value: spaces, the punctuation where encoding fidelity matters
# (&, =, +, %, <, >, quotes), and any non-ASCII codepoint, astral included.
# Excluded: control characters and line/paragraph separators, which a text
# input strips, and surrogates, which are not text.
_EXCLUDED_CATEGORIES: tuple[Literal["Cc", "Cs", "Zl", "Zp"], ...] = (
    "Cc",
    "Cs",
    "Zl",
    "Zp",
)
_WILD = st.text(
    st.characters(
        min_codepoint=32,
        exclude_categories=_EXCLUDED_CATEGORIES,
    ),
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
    # submit. The submit + inputs sit *after* the cross controls in the
    # document, so any reorder is observable. POST so GET repeated-key encoding
    # can't contaminate the ordering signal; find_form ordering is
    # method-independent.
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
    (in its single document-order pass) and the repeated key must survive
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


def _group_override(draw: DrawFn, group: Control) -> str | tuple[str, ...]:
    """An override for *group*: what ``Form.submit(data=)`` would carry.

    A checkbox group takes any subset of its options (``()`` unchecks all)
    as a list, or a lone value as a scalar; a radio group takes one value.
    Either may carry a value no box has, which only a hidden input submits.
    """
    unmatched = draw(_TOKEN.filter(lambda t: t not in group.options))
    if group.kind == RADIOS:
        return draw(st.sampled_from([*group.options, unmatched]))
    wanted = _subset(draw, group.options)
    if draw(st.booleans()):
        wanted = (*wanted, unmatched)
    if len(wanted) == 1 and draw(st.booleans()):
        return wanted[0]
    return wanted


@st.composite
def group_override_cases(draw: DrawFn) -> FormCase:
    """Radio/checkbox groups overridden through ``Form.submit(data=)``.

    Each group renders its own checked state, then an override says what it
    must submit instead. The browser transports must uncheck the boxes the
    override dropped and inject a hidden input for a value no box carries,
    so they submit what the HTTP transport posts. Unordered: a group with
    nothing rendered checked has no default key, so ``Form.submit`` appends
    its override after the rendered fields, as it does for an extra.
    """
    controls, idx = _plain_controls(draw, (TEXT, HIDDEN), 0, 0, 2)
    overrides: list[tuple[str, str | tuple[str, ...]]] = []
    for _ in range(draw(st.integers(min_value=1, max_value=2))):
        group = _plain_control(
            draw, draw(st.sampled_from((CHECKBOXES, RADIOS))), idx
        )
        controls.append(group)
        overrides.append((group.name, _group_override(draw, group)))
        idx += 1
    controls.append(_single_submit(draw, idx))
    return FormCase(
        method="POST",
        controls=tuple(controls),
        group_overrides=tuple(overrides),
    )


def _select_override(draw: DrawFn, select: Control) -> str | tuple[str, ...]:
    """An override for *select*: one value, or for a multi-select any subset
    of its options (``()`` deselects all) as a list or a lone scalar. Either
    may carry a value no option has."""
    unmatched = draw(_TOKEN.filter(lambda t: t not in select.options))
    if select.kind == SELECT:
        return draw(st.sampled_from([*select.options, unmatched]))
    wanted = _subset(draw, select.options)
    if draw(st.booleans()):
        wanted = (*wanted, unmatched)
    if len(wanted) == 1 and draw(st.booleans()):
        return wanted[0]
    return wanted


@st.composite
def select_override_cases(draw: DrawFn) -> FormCase:
    """``<select>`` / ``<select multiple>`` overridden through ``data=``.

    Each transport must submit exactly the override. A value no option
    carries is posted as-is by the HTTP transport, so the browser
    transports inject an ``<option>`` for it; the oracle renders that
    option for real. Unordered for the same reason as
    :func:`group_override_cases` (a multi-select with nothing selected has
    no default key).
    """
    controls, idx = _plain_controls(draw, (TEXT, HIDDEN), 0, 0, 2)
    overrides: list[tuple[str, str | tuple[str, ...]]] = []
    for _ in range(draw(st.integers(min_value=1, max_value=2))):
        select = _plain_control(
            draw, draw(st.sampled_from((SELECT, MULTISELECT))), idx
        )
        controls.append(select)
        overrides.append((select.name, _select_override(draw, select)))
        idx += 1
    controls.append(_single_submit(draw, idx))
    return FormCase(
        method="POST",
        controls=tuple(controls),
        select_overrides=tuple(overrides),
    )
