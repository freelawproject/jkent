"""Tests for the test utilities themselves."""

import logging

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperAssumptionException,
)
from tests.utils import log_structural_error_and_stop


def test_structural_error_log_extra_includes_url_and_selector(caplog):
    exception = HTMLStructuralAssumptionException(
        selector="//tr",
        selector_type="xpath",
        description="rows",
        expected_min=5,
        expected_max=None,
        actual_count=0,
        request_url="https://example.com/cases",
    )

    with caplog.at_level(logging.ERROR, logger="tests.utils"):
        result = log_structural_error_and_stop(exception)

    assert result is False
    # extra= keys land in the record's __dict__, not on the LogRecord type
    record = caplog.records[0]
    assert record.__dict__["url"] == "https://example.com/cases"
    assert record.__dict__["selector"] == "//tr"


def test_non_structural_error_log_extra_still_includes_url(caplog):
    """The URL must survive even when selector details don't apply."""
    exception = ScraperAssumptionException(
        "data format drifted", request_url="https://example.com/cases"
    )

    with caplog.at_level(logging.ERROR, logger="tests.utils"):
        result = log_structural_error_and_stop(exception)

    assert result is False
    record = caplog.records[0]
    assert record.__dict__["url"] == "https://example.com/cases"
