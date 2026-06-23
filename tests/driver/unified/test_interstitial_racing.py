"""Interstitial handler racing in ``PlaywrightTransport``.

Ports the old driver's racing tests (tests/playwright/test_race_await_lists.py,
which stays with the excluded playwright driver) onto the unified transport:
the async racing logic is verified without a real browser via a mock Page
whose ``wait_for_selector`` blocks on asyncio.Event objects.

Also pins handler selection: a scraper's ``*_HANDLER`` driver requirements
resolve to the matching handlers from ``INTERSTITIAL_HANDLERS``.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.data_types import BaseScraper, DriverRequirement, WaitForSelector
from jkent.driver.unified_driver.interstitials import (
    INTERSTITIAL_HANDLERS,
    CloudflareHandler,
    HCaptchaHandler,
    InterstitialHandler,
    ReCaptchaHandler,
    WaitCondition,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubHandler(InterstitialHandler):
    """InterstitialHandler whose waitlist uses a controllable selector."""

    def __init__(self, selector: str = "div.interstitial") -> None:
        self._selector = selector
        self.navigate_through_called = False

    def waitlist(self) -> list[WaitCondition]:
        return [WaitForSelector(self._selector)]

    async def navigate_through(self, page: Any) -> None:
        self.navigate_through_called = True


def _make_transport(
    handlers: list[InterstitialHandler],
) -> PlaywrightTransport:
    """A transport with the handlers under test; nothing is launched."""
    transport = PlaywrightTransport(BaseScraper())
    transport._interstitial_handlers = handlers
    return transport


def _make_page(
    events: dict[str, asyncio.Event] | None = None,
) -> AsyncMock:
    """Create a mock Page whose wait_for_selector blocks until signalled.

    Args:
        events: mapping from CSS selector string to an asyncio.Event.
            ``wait_for_selector(sel)`` will block until the corresponding
            event is set.  Selectors not in the dict resolve immediately.
    """
    selector_events = events or {}
    page = AsyncMock()

    async def _wait_for_selector(
        selector: str, /, **_kwargs: Any
    ) -> MagicMock:
        ev = selector_events.get(selector)
        if ev is not None:
            await ev.wait()
        return MagicMock()  # locator-like return

    page.wait_for_selector = AsyncMock(side_effect=_wait_for_selector)
    return page


# ---------------------------------------------------------------------------
# Racing semantics
# ---------------------------------------------------------------------------


class TestRaceAwaitLists:
    """Tests for PlaywrightTransport._race_await_lists."""

    async def test_scraper_wins_returns_none(self) -> None:
        """When the scraper await list resolves first, return None."""
        scraper_ready = asyncio.Event()
        interstitial_ready = asyncio.Event()

        page = _make_page(
            {
                "#content": scraper_ready,
                "div.interstitial": interstitial_ready,
            }
        )
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [WaitForSelector("#content")]

        # Let scraper resolve immediately, interstitial never resolves
        scraper_ready.set()

        result = await transport._race_await_lists(page, scraper_await)
        assert result is None
        assert not handler.navigate_through_called

    async def test_interstitial_wins_returns_handler(self) -> None:
        """When the interstitial handler resolves first, return it."""
        scraper_ready = asyncio.Event()
        interstitial_ready = asyncio.Event()

        page = _make_page(
            {
                "#content": scraper_ready,
                "div.interstitial": interstitial_ready,
            }
        )
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [WaitForSelector("#content")]

        # Let interstitial resolve immediately, scraper never resolves
        interstitial_ready.set()

        result = await transport._race_await_lists(page, scraper_await)
        assert result is handler

    async def test_loser_tasks_are_cancelled(self) -> None:
        """Pending tasks should be cancelled after the winner resolves."""
        scraper_ready = asyncio.Event()

        page = _make_page(
            {
                "#content": scraper_ready,
                "div.interstitial": asyncio.Event(),  # never set
            }
        )
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [WaitForSelector("#content")]
        scraper_ready.set()

        await transport._race_await_lists(page, scraper_await)

        # If we get here without hanging, the interstitial task was
        # successfully cancelled (it was waiting on an event that
        # would never be set).

    async def test_winner_exception_propagates(self) -> None:
        """If the winning task raises, the exception propagates."""
        # Gate the interstitial so it never resolves; the scraper
        # selector will raise immediately, making it the "winner".
        page = _make_page({"div.interstitial": asyncio.Event()})

        original_side_effect = page.wait_for_selector.side_effect

        async def _exploding_wait(selector: str, /, **kwargs: Any) -> None:
            if selector == "#content":
                raise PlaywrightTimeoutError("timed out")
            return await original_side_effect(selector, **kwargs)

        page.wait_for_selector = AsyncMock(side_effect=_exploding_wait)

        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [WaitForSelector("#content")]

        with pytest.raises(PlaywrightTimeoutError):
            await transport._race_await_lists(page, scraper_await)

    async def test_multiple_interstitial_handlers(self) -> None:
        """With multiple handlers, the first to resolve wins."""
        scraper_ready = asyncio.Event()  # never set
        handler_a_ready = asyncio.Event()
        handler_b_ready = asyncio.Event()

        page = _make_page(
            {
                "#content": scraper_ready,
                "div.captcha": handler_a_ready,
                "div.disclaimer": handler_b_ready,
            }
        )

        handler_a = _StubHandler("div.captcha")
        handler_b = _StubHandler("div.disclaimer")
        transport = _make_transport([handler_a, handler_b])

        scraper_await = [WaitForSelector("#content")]

        # Only handler_b resolves
        handler_b_ready.set()

        result = await transport._race_await_lists(page, scraper_await)
        assert result is handler_b


# ---------------------------------------------------------------------------
# Handler selection from driver_requirements
# ---------------------------------------------------------------------------


class TestHandlerSelection:
    def _handlers_for(
        self, *reqs: DriverRequirement
    ) -> list[InterstitialHandler]:
        class _Scraper(BaseScraper[dict]):
            driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

        return PlaywrightTransport(_Scraper())._interstitial_handlers

    def test_no_requirements_no_handlers(self) -> None:
        assert self._handlers_for() == []
        assert self._handlers_for(DriverRequirement.JS_EVAL) == []

    def test_hcap_selects_hcaptcha_handler(self) -> None:
        handlers = self._handlers_for(DriverRequirement.HCAP_HANDLER)
        assert len(handlers) == 1
        assert isinstance(handlers[0], HCaptchaHandler)

    def test_rcap_selects_recaptcha_handler(self) -> None:
        handlers = self._handlers_for(DriverRequirement.RCAP_HANDLER)
        assert len(handlers) == 1
        assert isinstance(handlers[0], ReCaptchaHandler)

    def test_cfcap_selects_cloudflare_handler(self) -> None:
        handlers = self._handlers_for(DriverRequirement.CFCAP_HANDLER)
        assert len(handlers) == 1
        assert isinstance(handlers[0], CloudflareHandler)

    def test_registry_covers_exactly_the_handler_requirements(self) -> None:
        assert set(INTERSTITIAL_HANDLERS) == {
            DriverRequirement.HCAP_HANDLER,
            DriverRequirement.RCAP_HANDLER,
            DriverRequirement.CFCAP_HANDLER,
        }

    def test_cloudflare_waitlist_matches_response_input(self) -> None:
        (condition,) = INTERSTITIAL_HANDLERS[
            DriverRequirement.CFCAP_HANDLER
        ].waitlist()
        assert isinstance(condition, WaitForSelector)
        assert condition.selector == "input[name='cf-turnstile-response']"
        assert condition.state == "attached"
