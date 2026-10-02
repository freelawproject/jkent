"""Retry-After parsing and its attachment at classification time.

``parse_retry_after`` is the single parse-and-clamp site: both consumers
(the retry scheduler's per-request floor, the adaptive rate limiter's
global pause) trust its output. ``classify_and_raise`` attaches the parsed
value to every classified HTTP failure — transient, persistent, and
speculative alike.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import pytest

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    SpeculationHTTPFailure,
)
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.unified_driver.transport import (
    MAX_RETRY_AFTER_S,
    parse_retry_after,
)
from tests.driver.unified.test_transport_conformance import FakeTransport

# --- parse_retry_after ---------------------------------------------------


def test_parses_delta_seconds() -> None:
    assert parse_retry_after({"Retry-After": "120"}) == 120.0


def test_header_lookup_is_case_insensitive() -> None:
    assert parse_retry_after({"retry-after": "7"}) == 7.0
    assert parse_retry_after({"RETRY-AFTER": " 7 "}) == 7.0


def test_parses_http_date() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=60)
    seconds = parse_retry_after({"Retry-After": format_datetime(when)})
    assert seconds is not None
    # format_datetime truncates to whole seconds; allow the drift.
    assert 58.0 <= seconds <= 61.0


def test_past_http_date_clamps_to_zero() -> None:
    when = datetime.now(timezone.utc) - timedelta(hours=1)
    assert parse_retry_after({"Retry-After": format_datetime(when)}) == 0.0


@pytest.mark.parametrize("skew", [timedelta(hours=1), -timedelta(hours=1)])
def test_http_date_is_measured_against_the_response_date(
    skew: timedelta,
) -> None:
    """A skewed server clock skews Date and Retry-After alike: the wait is
    their difference, not Retry-After against our clock."""
    server_now = datetime.now(timezone.utc) + skew
    seconds = parse_retry_after(
        {
            "Date": format_datetime(server_now),
            "Retry-After": format_datetime(server_now + timedelta(seconds=60)),
        }
    )
    assert seconds is not None
    assert 59.0 <= seconds <= 61.0


def test_unparseable_date_falls_back_to_our_clock() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=60)
    seconds = parse_retry_after(
        {"Date": "not a date", "Retry-After": format_datetime(when)}
    )
    assert seconds is not None
    assert 58.0 <= seconds <= 61.0


def test_clamps_hostile_values() -> None:
    # A buggy or hostile header must slow the run down, never stall it.
    assert parse_retry_after({"Retry-After": "86400"}) == MAX_RETRY_AFTER_S
    assert parse_retry_after({"Retry-After": "-5"}) == 0.0


def test_absent_or_malformed_is_no_signal() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after({}) is None
    assert parse_retry_after({"content-type": "text/html"}) is None
    assert parse_retry_after({"Retry-After": "soon"}) is None
    assert parse_retry_after({"Retry-After": ""}) is None


# --- classify_and_raise attachment ---------------------------------------


class _Scraper(BaseScraper[Any]):
    """Default classification: 429 transient, 403 persistent, 200 success."""


def _request(*, speculative: bool = False) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/p"
        ),
        step="parse",
        is_speculative=speculative,
    )


def _classify(
    *, status: int, headers: dict[str, str], speculative: bool = False
) -> None:
    FakeTransport().classify_and_raise(
        _Scraper,
        _request(speculative=speculative),
        status_code=status,
        headers=headers,
        body=b"<html>throttled</html>",
        url="https://example.com/p",
    )


def test_transient_failure_carries_retry_after() -> None:
    with pytest.raises(HTTPResponseAssumptionException) as exc_info:
        _classify(status=429, headers={"Retry-After": "17"})
    assert exc_info.value.retry_after == 17.0


def test_persistent_failure_carries_retry_after() -> None:
    with pytest.raises(PersistentHTTPResponseException) as exc_info:
        _classify(status=403, headers={"Retry-After": "17"})
    assert exc_info.value.retry_after == 17.0


def test_speculative_failure_carries_retry_after() -> None:
    with pytest.raises(SpeculationHTTPFailure) as exc_info:
        _classify(status=403, headers={"Retry-After": "17"}, speculative=True)
    assert exc_info.value.retry_after == 17.0


def test_no_header_means_none() -> None:
    with pytest.raises(HTTPResponseAssumptionException) as exc_info:
        _classify(status=429, headers={"content-type": "text/html"})
    assert exc_info.value.retry_after is None
