"""Package import smoke test.

The v0.1.0 CI skeleton (PR 1.3) needs one collectable test so the test job
is green from the first PR; it rides along harmlessly on v0.0.0.
"""

import jkent


def test_package_imports():
    """The jkent package shall import and carry its docstring."""
    assert jkent.__doc__ is not None
    assert "Scraper-driver framework" in jkent.__doc__
