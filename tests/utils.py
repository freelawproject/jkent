"""Test utilities for design step tests.

This module provides reusable utilities for testing the scraper-driver
architecture.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    ScraperAssumptionException,
)

logger = logging.getLogger(__name__)


def collect_results() -> tuple[Callable[[Any], None], list[Any]]:
    """Create a callback that collects results in a list.

    This utility replaces the pattern of collecting results from driver.run()
    return value. Instead, pass the callback to the driver's on_data parameter
    and check the results list after running.

    Returns:
        A tuple of (callback_function, results_list).
        The callback appends data to the results list.
        The results list is shared and can be inspected after driver.run().

    Example:
        callback, results = collect_results()
        driver = SyncDriver(scraper, on_data=callback)
        driver.run()
        assert len(results) > 0
    """
    results: list[Any] = []

    def callback(data: Any) -> None:
        results.append(data)

    return callback, results


def collect_results_async() -> tuple[
    Callable[[Any], Awaitable[None]], list[Any]
]:
    """Create an async callback that collects results in a list.

    This utility is the async version of collect_results for use with
    AsyncDriver which requires async callbacks.

    Returns:
        A tuple of (async_callback_function, results_list).
        The callback appends data to the results list.
        The results list is shared and can be inspected after driver.run().

    Example:
        callback, results = collect_results_async()
        driver = AsyncDriver(scraper, on_data=callback)
        await driver.run()
        assert len(results) > 0
    """
    results: list[Any] = []

    async def callback(data: Any) -> None:
        results.append(data)

    return callback, results


def log_structural_error_and_stop(
    exception: ScraperAssumptionException,
) -> bool:
    """Log structural error and return False to stop scraping.

    Example callback for on_structural_error that logs the exception
    details and returns False to halt scraping.

    Args:
        exception: The HTMLStructuralAssumptionException that was raised.

    Returns:
        False to stop the scraper.

    Example:
        driver = SyncDriver(
            scraper,
            on_structural_error=log_structural_error_and_stop
        )
        driver.run()
    """
    extra = {"url": exception.request_url} | (
        {
            "selector": exception.selector,
            "selector_type": exception.selector_type,
            "expected_min": exception.expected_min,
            "expected_max": exception.expected_max,
            "actual_count": exception.actual_count,
        }
        if isinstance(exception, HTMLStructuralAssumptionException)
        else {}
    )
    logger.error(
        f"Structural assumption failed: {exception.message}",
        extra=extra,
    )
    return False
