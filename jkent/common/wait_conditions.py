"""Wait conditions for the Playwright driver.

These describe what the driver should wait for before snapshotting the DOM,
supplied via ``@step(await_list=[...])``.

A leaf: it imports nothing from jkent. ``jkent.common.decorator_metadata``
annotates ``await_list`` with ``WaitCondition``, and the authoring facade
(``jkent.data_types``) re-exports these names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: The element states ``WaitForSelector`` can wait for (Playwright's own set).
SelectorState = Literal["attached", "detached", "hidden", "visible"]
#: The document load states ``WaitForLoadState`` can wait for.
LoadState = Literal["domcontentloaded", "load", "networkidle"]


@dataclass(frozen=True)
class WaitForSelector:
    """Wait for a selector to appear in the DOM.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for an element before taking a DOM snapshot.

    Attributes:
        selector: CSS or XPath selector to wait for. An XPath must select
               elements and use no namespace prefixes, which Playwright
               cannot bind (see :meth:`~jkent.common.selectors.XPath.can_playwright_wait`).
        state: Optional state to wait for ('attached', 'detached', 'visible', 'hidden').
               Defaults to 'visible'.
        timeout: Optional timeout in milliseconds. If None, uses the request's timeout.
    """

    selector: str
    state: SelectorState = "visible"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForLoadState:
    """Wait for a specific load state.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for a load state before taking a DOM snapshot.

    Attributes:
        state: Load state to wait for ('load', 'domcontentloaded', 'networkidle').
        timeout: Optional timeout in milliseconds. If None, uses the request's timeout.
    """

    state: LoadState = "load"
    timeout: int | None = None


@dataclass(frozen=True)
class WaitForURL:
    """Wait for the URL to match a pattern.

    Used in @step(await_list=[...]) to instruct Playwright driver
    to wait for URL navigation before taking a DOM snapshot.

    Attributes:
        url: URL string or pattern to wait for. Can be a string, regex pattern, or callable.
        timeout: Optional timeout in milliseconds. If None, uses the request's timeout.
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
