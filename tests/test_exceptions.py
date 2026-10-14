"""Tests for the exception hierarchy."""

import pytest

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    InterstitialUnresolved,
    RequestTimeoutException,
    ResolveTimeout,
    TransientException,
    TransientKind,
)
from jkent.driver.database_engine.errors import describe_error


def test_http_response_assumption_exception_is_transient():
    """Unexpected status codes raise the HTTP-named transient exception."""
    exc = HTTPResponseAssumptionException(
        status_code=503,
        expected_codes=[200],
        url="https://example.com/cases",
    )

    assert isinstance(exc, TransientException)
    assert exc.status_code == 503
    assert "503" in exc.message


def test_direct_transient_raise_requires_kind():
    with pytest.raises(TypeError, match="kind"):
        TransientException("x")  # type: ignore[call-arg]


def test_direct_transient_raise_carries_kind():
    exc = TransientException("x", kind=TransientKind.NETWORK)

    assert describe_error(exc).kind is TransientKind.NETWORK


@pytest.mark.parametrize(
    "exc",
    [
        HTTPResponseAssumptionException(503, [200], "https://e.test/"),
        RequestTimeoutException("https://e.test/", 5.0),
        ResolveTimeout("https://e.test/", 5.0),
        InterstitialUnresolved("challenge still present"),
    ],
    ids=lambda exc: type(exc).__name__,
)
def test_subclasses_construct_without_kind(exc: TransientException):
    assert isinstance(exc, TransientException)


def test_interstitial_unresolved_is_filed_as_interstitial():
    exc = InterstitialUnresolved("challenge still present")

    assert describe_error(exc).kind is TransientKind.INTERSTITIAL
