"""Wait conditions for the Playwright driver.

These describe what the driver should wait for before snapshotting the DOM,
supplied via ``@step(await_list=[...])``.

This is a leaf module — it imports nothing from jkent — so both
``jkent.data_types`` (which re-exports these names) and
``jkent.common.decorator_metadata`` (which annotates ``await_list`` with
``WaitCondition``) can depend on it without forming an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaitForSelector:
    """Wait for a selector to appear in the DOM.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for an element before taking a DOM snapshot.

    Attributes:
        selector: CSS or XPath selector to wait for.
        state: Optional state to wait for ('attached', 'detached', 'visible', 'hidden').
               Defaults to 'visible'.
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    selector: str
    state: str = "visible"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForLoadState:
    """Wait for a specific load state.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for a load state before taking a DOM snapshot.

    Attributes:
        state: Load state to wait for ('load', 'domcontentloaded', 'networkidle').
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    state: str = "load"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForURL:
    """Wait for the URL to match a pattern.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for URL navigation before taking a DOM snapshot.

    Attributes:
        url: URL string or pattern to wait for. Can be a string, regex pattern, or callable.
        timeout: Optional timeout in milliseconds. If None, uses Playwright's default.
    """

    url: str
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForTimeout:
    """Wait for a specific amount of time.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait before taking a DOM snapshot.

    Attributes:
        timeout: Time to wait in milliseconds.
    """

    timeout: int


# A single entry in @step(await_list=[...]): one of the Playwright wait
# conditions the driver applies before snapshotting the DOM.
WaitCondition = (
    WaitForSelector | WaitForLoadState | WaitForURL | WaitForTimeout
)
