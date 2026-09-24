"""``serve_cached_parent``: the one-shot route that answers a staging goto.

The other half of this module's ``route`` story from
``test_worker_page_capture``: that one covers interception that *records*
what a page fetches, this one interception that *answers* a navigation from
bytes already in hand.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from jkent.driver.browser_engine.worker_page import serve_cached_parent

URL = "https://example.gov/target"


async def test_cached_parent_route_is_one_shot_and_removed_on_failure() -> (
    None
):
    """A failed staging goto leaves no intercept on the long-lived page.

    Left behind, it would serve the cached bytes to every later navigation
    to that URL, with no error.
    """
    page = MagicMock()
    page.route = AsyncMock()
    page.unroute = AsyncMock()
    page.goto = AsyncMock(side_effect=PlaywrightTimeoutError("staging"))

    with pytest.raises(PlaywrightTimeoutError):
        await serve_cached_parent(
            page, url=URL, body=b"<html></html>", headers={}, status=200
        )

    (route_call,) = page.route.await_args_list
    assert route_call.kwargs.get("times") == 1
    page.unroute.assert_awaited_once_with(URL, route_call.args[1])


async def test_content_type_defaults_to_html_when_absent() -> None:
    """A stored response with no ``content-type`` must still render as HTML.

    Playwright serves a fulfilled body with no content-type as a download
    rather than a document, so the tab would never be on the page a via
    needs to click.
    """
    page = MagicMock()
    page.route = AsyncMock()
    page.unroute = AsyncMock()
    page.goto = AsyncMock()
    route = MagicMock()
    route.fulfill = AsyncMock()

    await serve_cached_parent(
        page, url=URL, body=b"<html></html>", headers={}, status=200
    )

    (route_call,) = page.route.await_args_list
    await route_call.args[1](route)
    assert route.fulfill.await_args is not None
    headers = route.fulfill.await_args.kwargs["headers"]
    assert headers["content-type"] == "text/html; charset=utf-8"


async def test_a_supplied_content_type_is_not_overridden() -> None:
    """Case-insensitively: the stored header wins over the HTML default."""
    page = MagicMock()
    page.route = AsyncMock()
    page.unroute = AsyncMock()
    page.goto = AsyncMock()
    route = MagicMock()
    route.fulfill = AsyncMock()

    await serve_cached_parent(
        page,
        url=URL,
        body=b"{}",
        headers={"Content-Type": "application/json"},
        status=200,
    )

    (route_call,) = page.route.await_args_list
    await route_call.args[1](route)
    assert route.fulfill.await_args is not None
    headers = route.fulfill.await_args.kwargs["headers"]
    assert headers["Content-Type"] == "application/json"
    assert "content-type" not in headers
