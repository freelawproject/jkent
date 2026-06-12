"""Invalid selectors are scraper bugs, not site changes.

A syntax error in a selector must raise ScraperConfigError — retrying or
re-verifying the court site won't fix a typo in the scraper. Only valid
selectors with unexpected match counts may raise
HTMLStructuralAssumptionException ("the website changed").
"""

import pytest
from lxml import html

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperConfigError,
)
from jkent.common.lxml_page_element import LxmlPageElement


@pytest.fixture
def tree():
    return LxmlPageElement(
        html.fromstring("<div><p>x</p></div>"),
        "https://example.com/page",
    )


def test_invalid_xpath_raises_scraper_config_error(tree):
    """A broken XPath surfaces as a config error naming the selector.

    It used to escape as a raw lxml XPathEvalError with no description
    or URL context.
    """
    with pytest.raises(ScraperConfigError) as exc_info:
        tree.checked_xpath("//p[", "broken xpath")
    assert "//p[" in str(exc_info.value)


def test_invalid_css_raises_scraper_config_error(tree):
    """A broken CSS selector surfaces as a config error.

    It used to raise HTMLStructuralAssumptionException with
    actual_count=0, which reads as "the website changed" in triage and
    sends whoever is on call to re-verify a court site for nothing.
    """
    with pytest.raises(ScraperConfigError) as exc_info:
        tree.checked_css("p:::bad", "broken css")
    assert "p:::bad" in str(exc_info.value)


def test_valid_xpath_count_mismatch_is_still_structural(tree):
    with pytest.raises(HTMLStructuralAssumptionException):
        tree.checked_xpath("//span", "missing spans")


def test_valid_css_count_mismatch_is_still_structural(tree):
    with pytest.raises(HTMLStructuralAssumptionException):
        tree.checked_css("span.missing", "missing spans")
