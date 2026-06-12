"""Generative form-submission conformance rig.

Generates HTML forms + fill plans with Hypothesis, then fans a single
submission across every transport (HTTP, Playwright/chromium,
Camoufox/firefox) and checks that each reaches the server identically to an
independent vanilla-browser oracle.

See ``tests/driver/unified/test_form_transport_conformance.py`` for the
``@given`` bindings and ``harness.py`` for the per-example execution.
"""
