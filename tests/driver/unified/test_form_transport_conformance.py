"""Generative form-submission conformance across every transport.

Hypothesis generates HTML forms + fill plans; each example is submitted through
the HTTP transport, the chromium Playwright transport and the camoufox/firefox
transport, and compared against an *independent vanilla-browser oracle* (raw
Playwright, no jkent form code in the path).

A baseline plus isolated probes, each pinning a single behaviour (so a
failure attributes cleanly):

* :func:`test_happy_path_matches_browser` — POST forms of input/button controls.
  Order-sensitive: every transport reconstructs a browser-identical submission
  (values, repeated keys, and order).
* :func:`test_extra_fields_reach_server` — fields added via
  ``Form.submit(data=)`` with no rendered control (the Playwright fill path
  injects a hidden input for them).
* :func:`test_field_order_matches_browser` — submitted-field document order
  across control types (``find_form`` collects controls in one document-order
  pass).
* :func:`test_disabled_fields_not_submitted` — disabled controls (``find_form``
  skips them).
* :func:`test_multiple_submit_buttons` — only the activated submit button's
  name/value is submitted (``Form.submit`` emits the submit control matched by
  ``submit_selector``).
* :func:`test_repeated_get_params_match_browser` — a key submitted more than
  once on a GET form (the queue folds GET params with
  ``urlencode(..., doseq=True)``, so repeated keys repeat).
* :func:`test_duplicate_names_match_browser` — a key repeated by several
  *distinct* controls sharing one name (not a checkbox group / multi-select).
  Order-sensitive: ``find_form`` must accumulate every same-named control and
  the repeats must survive POST body encoding in document order.
* :func:`test_duplicate_fill_matches_browser` — overriding *some* of those
  same-named controls via a positional ``data={name: [...]}`` list, with
  ``None`` entries keeping a control's rendered default. Exercises the
  repeated-field fill path and ``Form.submit``'s ``None`` resolution.
* :func:`test_group_overrides_match_browser` — radio/checkbox groups
  overridden via ``data=``: the browser transports uncheck what the override
  dropped and inject a hidden input for a value no box carries.
* :func:`test_select_overrides_match_browser` — ``<select>`` overrides via
  ``data=``: the browser transports select exactly the override, injecting
  an ``<option>`` for a value no option carries.

Everything runs on one persistent loop with the transports opened once
(:mod:`tests.form_conformance.harness`); the module skips when no browser can
launch (no oracle => no ground truth).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings

from tests.form_conformance import strategies as strat
from tests.form_conformance.echo_server import Canonical, diff_message
from tests.form_conformance.harness import Harness

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.form_conformance.model import FormCase

# Camoufox can only launch one Firefox at a time; share test_camoufox_transport's
# xdist group so the two modules never launch concurrently.
pytestmark = [pytest.mark.generative, pytest.mark.xdist_group("camoufox")]

# Browsers are slow; keep the per-test budget bounded regardless of profile.
_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.fixture(scope="module")
def harness(browser_launch_failure: str | None) -> Iterator[Harness]:
    """Open the echo server, DB, transports and oracle once for the module.

    Every browser transport and the oracle are required: where a browser
    launches at all, one that fails to launch fails the rig rather than
    shrinking the comparison to httpx.
    """
    # Module-scoped, so it can't take the function-scoped require_browser.
    if browser_launch_failure is not None:
        pytest.skip(f"no launchable browser engine: {browser_launch_failure}")
    loop = asyncio.new_event_loop()
    h = Harness(loop)
    try:
        loop.run_until_complete(h.open())
        yield h
    finally:
        loop.run_until_complete(h.aclose())
        loop.close()


def _key(value: Canonical | str, *, ordered: bool) -> object:
    """Comparable projection: ordered pairs, or a multiset of pairs.

    The method and content type always take part.
    """
    if isinstance(value, str):  # error marker: a str never equals a tuple
        return value
    method, content_type, pairs = value
    if ordered:
        return (method, content_type, pairs)
    return (method, content_type, frozenset(Counter(pairs).items()))


def _check(harness: Harness, case: FormCase, *, ordered: bool) -> None:
    """Drive one case and assert every live transport matches the oracle."""
    if not harness.oracle_available:
        pytest.skip("no launchable browser engine for the oracle")
    oracle, results = harness.loop.run_until_complete(harness.run_case(case))
    target = _key(oracle, ordered=ordered)
    mismatched = [
        (n, r) for n, r in results if _key(r, ordered=ordered) != target
    ]
    if mismatched:
        raise AssertionError(diff_message(f"case: {case!r}", oracle, results))


@given(case=strat.happy_cases())
@_SETTINGS
def test_happy_path_matches_browser(harness: Harness, case: FormCase) -> None:
    """Input/button POST forms reach the server identically to a browser."""
    _check(harness, case, ordered=True)


@given(case=strat.cross_type_cases())
@_SETTINGS
def test_field_order_matches_browser(harness: Harness, case: FormCase) -> None:
    """Submitted fields should be in document order, as a browser sends them.

    find_form collects all controls in one document-order pass, so a
    ``<select>``/``<textarea>`` among inputs keeps its position.
    """
    _check(harness, case, ordered=True)


@given(case=strat.get_multivalue_cases())
@_SETTINGS
def test_repeated_get_params_match_browser(
    harness: Harness, case: FormCase
) -> None:
    """A key submitted multiple times on a GET form must repeat, as in a browser.

    The queue folds GET params with
    ``urlencode(..., doseq=True)``, so a checkbox group / multi-select encodes
    as repeated names (``q=a&q=b``) rather than one repr of the list.
    """
    _check(harness, case, ordered=False)


@given(case=strat.duplicate_name_cases())
@_SETTINGS
def test_duplicate_names_match_browser(
    harness: Harness, case: FormCase
) -> None:
    """Several distinct controls sharing one name submit like a browser.

    Independent ``<input>``/``<textarea>`` elements that share a ``name`` repeat
    that key once per element, in document order — distinct from a checkbox
    group / multi-select (one control, many values). ``find_form`` must collect
    every same-named control (in its one document-order pass) and the
    repeats must survive POST body encoding intact.
    """
    _check(harness, case, ordered=True)


@given(case=strat.duplicate_fill_cases())
@_SETTINGS
def test_duplicate_fill_matches_browser(
    harness: Harness, case: FormCase
) -> None:
    """Positionally overriding some of several same-named controls matches.

    ``Form.submit(data={name: [...]})`` fills repeated same-named controls in
    order; a ``None`` entry keeps that control's rendered default. The HTTP
    transport must encode the resolved repeats and the Playwright fill path must
    type each override into the matching control — both identical to a browser.
    """
    _check(harness, case, ordered=True)


@given(case=strat.group_override_cases())
@_SETTINGS
def test_group_overrides_match_browser(
    harness: Harness, case: FormCase
) -> None:
    """A radio/checkbox group submits exactly its ``data=`` override.

    ``Form.submit`` puts the override in ``field_data``, which the HTTP
    transport posts verbatim; the Playwright fill path must set the group to
    those values — unchecking rendered boxes the override left out — and
    send any value no box carries as a hidden input. Unordered, as for
    extra fields: an override for a group with nothing rendered checked is a
    new key, appended after the rendered defaults.
    """
    _check(harness, case, ordered=False)


@given(case=strat.select_override_cases())
@_SETTINGS
def test_select_overrides_match_browser(
    harness: Harness, case: FormCase
) -> None:
    """A ``<select>`` override submits exactly its values, like a browser.

    ``Form.submit`` puts the override in ``field_data`` and the HTTP
    transport posts it, whether or not an option carries each value. The
    Playwright fill path must select exactly those options, injecting an
    ``<option>`` for a value the select lacks — the oracle's form has that
    option rendered. Unordered, as for group overrides.
    """
    _check(harness, case, ordered=False)


@given(case=strat.disabled_cases())
@_SETTINGS
def test_disabled_fields_not_submitted(
    harness: Harness, case: FormCase
) -> None:
    """Disabled controls must not be submitted by any transport.

    find_form skips disabled controls, so neither the HTTP
    transport submits them nor the Playwright fill path tries to type into them.
    """
    _check(harness, case, ordered=False)


@given(case=strat.extra_cases())
@_SETTINGS
def test_extra_fields_reach_server(harness: Harness, case: FormCase) -> None:
    """Fields added via ``Form.submit(data=)`` reach the server everywhere.

    The Playwright fill path injects a hidden input for any
    ``field_data`` name with no rendered control, so the browser transports
    submit extra fields just as the HTTP transport does.
    """
    _check(harness, case, ordered=False)


@given(case=strat.multi_submit_cases())
@_SETTINGS
def test_multiple_submit_buttons(harness: Harness, case: FormCase) -> None:
    """Only the activated submit button's name/value should be submitted.

    Form.submit emits only the submit control matched by an ``#id``
    ``submit_selector`` (or the first, for implicit submission), so
    activating a non-first button does not leak the first button's pair.
    """
    _check(harness, case, ordered=False)
