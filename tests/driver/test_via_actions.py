"""Error mapping in ``execute_via_navigation``; ``serve_cached_parent``'s route.

The taxonomy is load-bearing: only a ``TransientException`` gets a retry. A
Playwright error that escapes this function's mapping travels all the way to
``PoolWorker._handle_one``'s generic ``except Exception``, which marks the
request permanently failed with its retry budget untouched. These pin the
abort codes that must map to transient so an aborted attempt is retried rather
than discarded.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.exceptions import TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.data_types import CSS
from jkent.driver.via_actions import (
    execute_via_navigation,
    prepare_form_submit,
    serve_cached_parent,
)

URL = "https://example.gov/target"


def _page_raising(exc: BaseException) -> MagicMock:
    """A page whose ``expect_navigation`` context raises ``exc`` on exit.

    Mirrors the real failure shape: Playwright surfaces the navigation error
    when the ``async with`` block exits, not when the click runs.
    """
    element = MagicMock()
    element.evaluate = AsyncMock()
    element.click = AsyncMock()

    page = MagicMock()
    page.wait_for_selector = AsyncMock(return_value=element)

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=MagicMock(value=AsyncMock()))

    async def _aexit(*_args: Any) -> None:
        raise exc

    ctx.__aexit__ = _aexit
    page.expect_navigation = MagicMock(return_value=ctx)
    return page


def _via() -> ViaLink:
    return ViaLink(selector=CSS("a#next"), description="next page")


@pytest.mark.parametrize(
    "message",
    [
        # Firefox's two distinct abort codes. NS_BINDING_ABORTED is what
        # necko reports when the load was cut short by window.stop() or a
        # competing navigation; the trailing clause is Playwright's own.
        "NS_ERROR_ABORT",
        "NS_BINDING_ABORTED; maybe frame was detached?",
    ],
)
async def test_abort_codes_map_to_transient(message: str) -> None:
    """Both Firefox abort codes are retryable, not terminal."""
    page = _page_raising(PlaywrightError(message))
    with pytest.raises(TransientException, match="Navigation aborted"):
        await execute_via_navigation(page, _via(), URL)


async def test_navigation_timeout_maps_to_transient() -> None:
    """A navigation timeout stays transient (unchanged behaviour)."""
    page = _page_raising(PlaywrightTimeoutError("Timeout 30000ms exceeded."))
    with pytest.raises(TransientException, match="Navigation timeout"):
        await execute_via_navigation(page, _via(), URL)


async def test_unrelated_playwright_error_is_not_swallowed() -> None:
    """A non-abort Playwright error still propagates as itself.

    The mapping must stay narrow: turning every ``PlaywrightError`` into a
    transient would retry genuine bugs (a bad selector grammar, a closed
    context) until the backoff budget ran out.
    """
    page = _page_raising(PlaywrightError("Unsupported pseudo-class"))
    with pytest.raises(PlaywrightError, match="Unsupported pseudo-class"):
        await execute_via_navigation(page, _via(), URL)


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


async def test_element_wait_uses_the_callers_timeout() -> None:
    """The via element wait is bounded by ``timeout_ms``, not a fixed 5 s.

    A slow page that renders its link after 5 s would otherwise fail with
    ``Selector timeout`` although the request allowed far longer.
    """
    page = _page_raising(PlaywrightTimeoutError("nav"))
    with pytest.raises(TransientException):
        await execute_via_navigation(page, _via(), URL, timeout_ms=45_000)

    (wait_call,) = page.wait_for_selector.await_args_list
    assert wait_call.kwargs.get("timeout") == 45_000


async def test_form_wait_uses_the_callers_timeout() -> None:
    """``prepare_form_submit`` bounds its form wait by ``timeout_ms``."""
    form = MagicMock()
    form.evaluate = AsyncMock()
    form.query_selector = AsyncMock(return_value=MagicMock())
    page = MagicMock()
    page.wait_for_selector = AsyncMock(return_value=form)
    via = ViaFormSubmit(
        form_selector=CSS("form#search"),
        field_data={},
        submit_selector=None,
        description="search form",
    )

    await prepare_form_submit(page, via, URL, timeout_ms=45_000)

    (wait_call,) = page.wait_for_selector.await_args_list
    assert wait_call.kwargs.get("timeout") == 45_000
