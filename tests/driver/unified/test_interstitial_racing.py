"""Interstitial handler racing in ``PlaywrightTransport``.

Ports the old driver's racing tests (tests/playwright/test_race_await_lists.py,
which stays with the excluded playwright driver) onto the unified transport:
the async racing logic is verified without a real browser via a mock Page
whose ``wait_for_selector`` blocks on asyncio.Event objects.

Also pins handler selection: a scraper's ``*_HANDLER`` driver requirements
resolve to the matching handlers from ``INTERSTITIAL_FACTORIES``.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.driver.unified_driver.interstitials import (
    INTERSTITIAL_FACTORIES,
    CloudflareHandler,
    InterstitialHandler,
    ReCaptchaHandler,
    WaitCondition,
    handlers_for,
)
from jkent.driver.unified_driver.requirements import (
    INTERSTITIAL_REQUIREMENTS,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
    _asserts_real_content,
)
from tests.driver.unified.test_interstitials import (
    _make_page,
    no_display,  # noqa: F401 — the fixture, used by name in pytestmark
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.usefixtures("no_display")


#: The CA calctapp_1st await list: both pass against a Cloudflare challenge,
#: which is how six requests stored the challenge as a persistent 403 body.
_CA_INCIDENT_LIST: list[WaitCondition] = [
    WaitForLoadState("networkidle"),
    WaitForSelector("button[disabled]", state="hidden"),
]


@pytest.mark.parametrize(
    ("conditions", "competes"),
    [
        pytest.param([], False, id="empty"),
        pytest.param([WaitForSelector("#c")], True, id="selector-default"),
        pytest.param(
            [WaitForSelector("#c", state="visible")], True, id="visible"
        ),
        pytest.param(
            [WaitForSelector("#c", state="attached")], True, id="attached"
        ),
        pytest.param(
            [WaitForSelector("#c", state="hidden")], False, id="hidden"
        ),
        pytest.param(
            [WaitForSelector("#c", state="detached")], False, id="detached"
        ),
        pytest.param([WaitForLoadState("load")], False, id="load-state"),
        pytest.param([WaitForURL("**/case/*")], True, id="url"),
        pytest.param([WaitForTimeout(100)], False, id="timeout"),
        pytest.param(_CA_INCIDENT_LIST, False, id="ca-incident"),
        pytest.param(
            [*_CA_INCIDENT_LIST, WaitForSelector("#centerColumn")],
            True,
            id="ca-incident-plus-positive",
        ),
    ],
)
def test_asserts_real_content(
    conditions: list[WaitCondition], competes: bool
) -> None:
    """Only a positive selector or a URL wait proves the content arrived."""
    assert _asserts_real_content(conditions) is competes


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

        result = await transport._race_await_lists(
            page, scraper_await, timeout_ms=5000.0
        )
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

        result = await transport._race_await_lists(
            page, scraper_await, timeout_ms=5000.0
        )
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
        original_side_effect = page.wait_for_selector.side_effect
        cancelled: list[str] = []

        async def _recording_wait(selector: str, /, **kwargs: Any) -> Any:
            try:
                return await original_side_effect(selector, **kwargs)
            except asyncio.CancelledError:
                cancelled.append(selector)
                raise

        page.wait_for_selector = AsyncMock(side_effect=_recording_wait)
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [WaitForSelector("#content")]
        scraper_ready.set()

        # An uncancelled loser would block the race's cleanup forever.
        await asyncio.wait_for(
            transport._race_await_lists(
                page, scraper_await, timeout_ms=5000.0
            ),
            timeout=5.0,
        )
        assert cancelled == ["div.interstitial"]

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
            await transport._race_await_lists(
                page, scraper_await, timeout_ms=5000.0
            )

    async def test_vacuous_scraper_list_does_not_beat_a_handler(self) -> None:
        """A vacuously-satisfiable await list must not win the race.

        This is the bug that silently poisoned the CA corpus. The scraper's
        list was ``[WaitForLoadState("networkidle"), WaitForSelector(
        "button[disabled]", state="hidden")]``: networkidle settles on a
        Cloudflare challenge, and a "hidden" wait passes when the element is
        merely absent. So the scraper group resolved on the first tick, the
        race returned None, and the challenge HTML was snapshotted as content
        and classified as a persistent 403 — retry_count 0, handler never run.
        """
        interstitial_ready = asyncio.Event()
        page = _make_page({"div.interstitial": interstitial_ready})
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        # Both conditions pass instantly against a challenge page.
        scraper_await: list[WaitCondition] = [
            WaitForLoadState("networkidle"),
            WaitForSelector("button[disabled]", state="hidden"),
        ]
        # The handler's marker takes a moment, as a real selector wait does.
        asyncio.get_running_loop().call_later(0.01, interstitial_ready.set)

        result = await asyncio.wait_for(
            transport._race_await_lists(
                page, scraper_await, timeout_ms=5000.0
            ),
            timeout=5.0,
        )
        assert result is handler

    async def test_positive_condition_still_competes(self) -> None:
        """A list that asserts real content keeps racing as before."""
        scraper_ready = asyncio.Event()
        page = _make_page(
            {
                "#centerColumn": scraper_ready,
                "div.interstitial": asyncio.Event(),  # never set
            }
        )
        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await: list[WaitCondition] = [
            WaitForSelector("#centerColumn"),
            WaitForLoadState("networkidle"),
            WaitForSelector("button[disabled]", state="hidden"),
        ]
        scraper_ready.set()

        result = await asyncio.wait_for(
            transport._race_await_lists(
                page, scraper_await, timeout_ms=5000.0
            ),
            timeout=5.0,
        )
        assert result is None
        assert not handler.navigate_through_called

    async def test_held_out_list_is_still_applied(self) -> None:
        """A non-competing list must not be silently dropped.

        ``None`` means "conditions satisfied, snapshot as-is" — the caller
        does not re-apply them — so the race has to honour a list it kept out
        of the competition once the handlers concede.
        """
        page = _make_page()

        # The handler's marker times out, i.e. no interstitial is present, so
        # the handlers all concede and the held-out list is what remains.
        async def _wait(selector: str, /, **_kwargs: Any) -> MagicMock:
            if selector == "div.interstitial":
                raise PlaywrightTimeoutError("no interstitial")
            return MagicMock()

        page.wait_for_selector = AsyncMock(side_effect=_wait)

        handler = _StubHandler("div.interstitial")
        transport = _make_transport([handler])

        scraper_await = [
            WaitForSelector("button[disabled]", state="hidden"),
        ]
        result = await asyncio.wait_for(
            transport._race_await_lists(
                page, scraper_await, timeout_ms=5000.0
            ),
            timeout=5.0,
        )
        assert result is None
        waited = [c.args[0] for c in page.wait_for_selector.await_args_list]
        assert "button[disabled]" in waited

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

        result = await transport._race_await_lists(
            page, scraper_await, timeout_ms=5000.0
        )
        assert result is handler_b

    # ``done`` is a set, so when both sides finish in one tick its iteration
    # order — not the race's rules — would pick the winner. Repeat each race
    # so an order-dependent verdict shows up as a mix.
    _SAME_TICK_TRIALS = 50

    async def test_same_tick_scraper_success_beats_handler_success(
        self,
    ) -> None:
        """Both succeed in one tick: the content arrived, so no interstitial."""
        results = set()
        for _ in range(self._SAME_TICK_TRIALS):
            transport = _make_transport([_StubHandler("div.interstitial")])
            results.add(
                await transport._race_await_lists(
                    _make_page(),
                    [WaitForSelector("#content")],
                    timeout_ms=5000.0,
                )
            )
        assert results == {None}

    async def test_same_tick_handler_success_beats_scraper_failure(
        self,
    ) -> None:
        """Scraper fails as a handler succeeds: the interstitial is why."""
        page = _make_page()

        async def _wait_for_selector(selector: str, /, **_kw: Any) -> None:
            if selector == "#content":
                raise PlaywrightTimeoutError("content never appeared")

        page.wait_for_selector = AsyncMock(side_effect=_wait_for_selector)
        for _ in range(self._SAME_TICK_TRIALS):
            handler = _StubHandler("div.interstitial")
            transport = _make_transport([handler])
            result = await transport._race_await_lists(
                page, [WaitForSelector("#content")], timeout_ms=5000.0
            )
            assert result is handler


# ---------------------------------------------------------------------------
# Handler selection from driver_requirements
# ---------------------------------------------------------------------------


class TestHandlerSelection:
    def _handlers_for(
        self, *reqs: DriverRequirement
    ) -> list[InterstitialHandler]:
        class _Scraper(BaseScraper[dict[str, Any]]):
            driver_requirements: ClassVar[list[DriverRequirement]] = list(reqs)

        return PlaywrightTransport(_Scraper())._interstitial_handlers

    def test_no_requirements_no_handlers(self) -> None:
        assert self._handlers_for() == []
        assert self._handlers_for(DriverRequirement.JS_EVAL) == []

    def test_rcap_selects_recaptcha_handler(self) -> None:
        handlers = self._handlers_for(DriverRequirement.RCAP_HANDLER)
        assert len(handlers) == 1
        assert isinstance(handlers[0], ReCaptchaHandler)

    def test_cfcap_selects_cloudflare_handler(self) -> None:
        handlers = self._handlers_for(DriverRequirement.CFCAP_HANDLER)
        assert len(handlers) == 1
        assert isinstance(handlers[0], CloudflareHandler)

    def test_registry_covers_exactly_the_handler_requirements(self) -> None:
        assert set(INTERSTITIAL_FACTORIES) == {
            DriverRequirement.RCAP_HANDLER,
            DriverRequirement.CFCAP_HANDLER,
        }
        # The policy table and the factory table are two halves of one
        # decision; interstitials._verify_factories pins them at import.
        assert set(INTERSTITIAL_FACTORIES) == INTERSTITIAL_REQUIREMENTS

    def test_handlers_are_per_transport_not_shared(self) -> None:
        """Each transport builds its own handlers.

        CloudflareHandler carries the OS-click lock that serialises window
        raise->click across the workers of *one* browser; module-level
        singletons would have made two concurrent runs contend on one lock.
        """
        first = self._handlers_for(DriverRequirement.CFCAP_HANDLER)
        second = self._handlers_for(DriverRequirement.CFCAP_HANDLER)
        assert first[0] is not second[0]

    def test_cloudflare_waitlist_matches_challenge_shell(self) -> None:
        (condition,) = handlers_for([DriverRequirement.CFCAP_HANDLER])[
            0
        ].waitlist()
        assert isinstance(condition, WaitForSelector)
        assert condition.selector == CloudflareHandler._CHALLENGE_SHELL
        assert condition.state == "attached"
        # Detection must stay bounded well under Playwright's 30s default:
        # this is what every no-challenge navigation pays before the handler
        # concedes the race. "Well under" is held to a third of it.
        assert condition.timeout == CloudflareHandler._DETECT_TIMEOUT_MS
        assert condition.timeout is not None
        assert condition.timeout <= 30_000 // 3

    def test_cloudflare_waitlist_is_not_the_widget_input(self) -> None:
        """The widget input is the signal this handler must NOT wait on.

        It is absent from the challenge wire HTML entirely, lands up to a
        second after the shell, and never appears at all on the click-gated
        variant — see TestCloudflareChallengeShapes.
        """
        (condition,) = handlers_for([DriverRequirement.CFCAP_HANDLER])[
            0
        ].waitlist()
        assert isinstance(condition, WaitForSelector)
        assert condition.selector != CloudflareHandler._RESPONSE_INPUT
