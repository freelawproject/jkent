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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from playwright.sync_api import sync_playwright

from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    WaitForLoadState,
    WaitForSelector,
)
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


def _permissive_locator() -> MagicMock:
    """A locator mock whose every await resolves immediately.

    ``.first`` and ``.locator(...)`` return the same object so chained
    locator expressions (``page.locator(x).first.locator("xpath=..")``)
    stay awaitable without each test re-stubbing the chain.
    """
    locator = MagicMock()
    locator.first = locator
    locator.locator = MagicMock(return_value=locator)
    locator.wait_for = AsyncMock()
    locator.click = AsyncMock()
    locator.bounding_box = AsyncMock(
        return_value={"x": 0.0, "y": 0.0, "width": 300.0, "height": 65.0}
    )
    return locator


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
            transport._race_await_lists(page, scraper_await), timeout=5.0
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
            transport._race_await_lists(page, scraper_await), timeout=5.0
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
            transport._race_await_lists(page, scraper_await), timeout=5.0
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

    def test_cloudflare_waitlist_matches_challenge_shell(self) -> None:
        (condition,) = INTERSTITIAL_HANDLERS[
            DriverRequirement.CFCAP_HANDLER
        ].waitlist()
        assert isinstance(condition, WaitForSelector)
        assert condition.selector == CloudflareHandler._CHALLENGE_SHELL
        assert condition.state == "attached"
        # Detection must stay bounded well under Playwright's 30s default:
        # this is what every no-challenge navigation pays before the handler
        # concedes the race.
        assert condition.timeout == CloudflareHandler._DETECT_TIMEOUT_MS

    def test_cloudflare_waitlist_is_not_the_widget_input(self) -> None:
        """The widget input is the signal this handler must NOT wait on.

        It is absent from the challenge wire HTML entirely, lands up to a
        second after the shell, and never appears at all on the click-gated
        variant — see TestCloudflareChallengeShapes.
        """
        (condition,) = INTERSTITIAL_HANDLERS[
            DriverRequirement.CFCAP_HANDLER
        ].waitlist()
        assert isinstance(condition, WaitForSelector)
        assert condition.selector != CloudflareHandler._RESPONSE_INPUT


# ---------------------------------------------------------------------------
# Cloudflare challenge DOM shapes
# ---------------------------------------------------------------------------


#: Reduced from real captures: the four challenge shapes Cloudflare served to
#: appellatecases.courtinfo.ca.gov / ma-appellatecourts.org / courts.mo.gov,
#: all ``cvId:'3'`` ``cTplV:5`` ``cType:'managed'``. ``<style>`` blocks and
#: token blobs are stripped; every element the handler keys on is verbatim.
_SHAPE_BARE_SHELL = """
<html><head><title>Just a moment...</title>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1?ray=a2"></script>
</head><body><div class="main-wrapper" role="main"><div class="main-content">
<noscript><div class="h2"><span id="challenge-error-text">Enable JavaScript
and cookies to continue</span></div></noscript></div></div></body></html>
"""

_SHAPE_EMPTY_MOUNT = """
<html><head><title>Just a moment...</title>
<script src="/cdn-cgi/challenge-platform/h/g/orchestrate/chl_page/v1?ray=a0"></script>
<script src="https://challenges.cloudflare.com/turnstile/v0/g/8f/api.js"></script>
</head><body><div class="main-wrapper"><div class="main-content">
<h2 class="ch-title">Performing security verification</h2>
<div id="BbLB6"></div>
<div id="ROlTq4" class="spacer loading-verifying"><div class="lds-ring"></div></div>
</div></div></body></html>
"""

_SHAPE_WIDGET = """
<html><head><title>Just a moment...</title>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1?ray=a1"></script>
</head><body><div class="main-wrapper gAFgs0"><div class="main-content">
<div id="KSUV2" style="display: grid;"><div><div>
<input type="hidden" name="cf-turnstile-response" id="cf-chl-widget-0btk5_response">
</div></div></div></div></div></body></html>
"""

_SHAPE_CLICK_GATED = """
<html><head><title>Just a moment...</title>
<script src="/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1?ray=a2"></script>
<script src="https://challenges.cloudflare.com/turnstile/v0/b/8e/api.js"></script>
</head><body><div class="main-wrapper ZUSR4"><div class="main-content">
<h2 id="OcZj4" class="RBft2">Performing security verification</h2>
<div id="Untl7" style="display: grid;"><div id="qvUc8" style="display: flex;">
<input type="button" value="Verify you are human" class="YNXn4"></div></div>
</div></div></body></html>
"""

#: A real content page from the same site. Note it carries no
#: challenge-platform script in the main document.
_SHAPE_REAL_CONTENT = """
<html><head><title>California Courts - Appellate Court Case Information</title>
<script src="/includes/js/common.js?matcher"></script></head>
<body><div id="pageWrapper"><div id="mainContent"><div id="centerColumn"
role="main"><h2>Search Results - 1st Appellate District</h2></div></div></div>
</body></html>
"""

#: A clean 200 page from a CF-fronted zone that serves no challenge
#: (courtpass.nycourts.gov). Cloudflare injects its ``jsd`` bot-score beacon
#: into an inline bootstrap here — which is why the shell selector is scoped
#: to the ``orchestrate`` leaf rather than matching ``challenge-platform``.
_SHAPE_JSD_BEACON_ONLY = """
<html><head><title>NYS Court-PASS Home Page</title></head><body>
<script>window.__CF$cv$params={r:'a28',t:'MTc'};var a=document.createElement('script');
a.src='/cdn-cgi/challenge-platform/scripts/jsd/main.js';
document.getElementsByTagName('head')[0].appendChild(a);</script>
<div id="content">real content</div></body></html>
"""


class TestCloudflareChallengeShapes:
    """The detection selector against every challenge shape actually observed.

    Cloudflare serves at least four different DOM shapes from one template
    generation, and the widget input the handler used to wait on is present in
    exactly one of them. A shape the selector misses is not a loud failure: the
    handler never fires, the scraper's own await list wins the race, and the
    challenge HTML is stored as though it were case data.
    """

    ALL_CHALLENGES: ClassVar[dict[str, str]] = {
        "bare_shell": _SHAPE_BARE_SHELL,
        "empty_mount": _SHAPE_EMPTY_MOUNT,
        "widget": _SHAPE_WIDGET,
        "click_gated": _SHAPE_CLICK_GATED,
    }
    NON_CHALLENGES: ClassVar[dict[str, str]] = {
        "real_content": _SHAPE_REAL_CONTENT,
        "jsd_beacon_only": _SHAPE_JSD_BEACON_ONLY,
    }

    @staticmethod
    def _matches(html: str, selector: str) -> int:
        lxml_html = pytest.importorskip("lxml.html")
        return len(lxml_html.fromstring(html).cssselect(selector))

    @pytest.mark.parametrize("shape", sorted(ALL_CHALLENGES))
    def test_shell_matches_every_challenge_shape(self, shape: str) -> None:
        assert (
            self._matches(
                self.ALL_CHALLENGES[shape],
                CloudflareHandler._CHALLENGE_SHELL,
            )
            == 1
        )

    @pytest.mark.parametrize("shape", sorted(NON_CHALLENGES))
    def test_shell_ignores_pages_without_a_challenge(self, shape: str) -> None:
        assert (
            self._matches(
                self.NON_CHALLENGES[shape],
                CloudflareHandler._CHALLENGE_SHELL,
            )
            == 0
        )

    def test_widget_input_misses_three_of_four_shapes(self) -> None:
        """Why the waitlist moved off the response input.

        Pins the actual coverage gap rather than the conclusion, so if
        Cloudflare converges on one shape this test says so.
        """
        matched = {
            shape
            for shape, html in self.ALL_CHALLENGES.items()
            if self._matches(html, CloudflareHandler._RESPONSE_INPUT)
        }
        assert matched == {"widget"}

    def test_verify_button_matches_only_the_click_gated_shape(self) -> None:
        matched = {
            shape
            for shape, html in self.ALL_CHALLENGES.items()
            if self._matches(html, CloudflareHandler._VERIFY_BUTTON)
        }
        assert matched == {"click_gated"}


# ---------------------------------------------------------------------------
# Cloudflare readiness signal
# ---------------------------------------------------------------------------


class TestCloudflareFlowPath:
    """The flow-POST matcher that gates the Tab+Space press.

    A stale literal path is invisible at runtime: the readiness event just
    never fires and every challenge silently burns the full readiness timeout
    before pressing. These pin the URL shapes actually observed against the CA
    appellate deployment (captured in a run db's incidental_requests), so the
    next Cloudflare path rotation fails here instead of in production.
    """

    @pytest.mark.parametrize(
        "url",
        [
            # Current shape, both branch letters seen within one run.
            "https://appellatecases.courtinfo.ca.gov"
            "/cdn-cgi/challenge-platform/h/b/fo/3126285109:1785938414:_1QHC8",
            "https://appellatecases.courtinfo.ca.gov"
            "/cdn-cgi/challenge-platform/h/g/fo/1366341194:1785938414:xYz",
            # The widget iframe's own origin posts to the same path shape.
            "https://challenges.cloudflare.com"
            "/cdn-cgi/challenge-platform/h/b/fo/1345784007:1785938414:abc",
            # The historical shape must keep matching: a deployment that has
            # not rotated yet still needs its readiness signal.
            "https://example.gov"
            "/cdn-cgi/challenge-platform/h/b/flow/ov1/0.123:1785938414:tok",
        ],
    )
    def test_matches_observed_flow_post_urls(self, url: str) -> None:
        assert CloudflareHandler._FLOW_RE.search(url) is not None

    @pytest.mark.parametrize(
        "url",
        [
            # Sibling challenge-platform endpoints that are NOT the flow POST;
            # counting these would fire readiness early.
            "https://appellatecases.courtinfo.ca.gov"
            "/cdn-cgi/challenge-platform/h/b/orchestrate/chl_page/v1?ray=a266",
            "https://challenges.cloudflare.com"
            "/cdn-cgi/challenge-platform/h/b/turnstile/f/av0/nope",
            "https://challenges.cloudflare.com"
            "/cdn-cgi/challenge-platform/h/b/ci/a266d5269b13e168",
            "https://challenges.cloudflare.com/turnstile/v0/b/8eb6d5cd/api.js",
            "https://appellatecases.courtinfo.ca.gov/search/searchResults.cfm",
            # The jsd bot-score beacon. This is the reason the leaf stays an
            # explicit whitelist: it is a POST returning 200 under
            # h/<branch>/, and it fires on ordinary 200 pages of zones that
            # serve no challenge at all (observed on courtpass.nycourts.gov),
            # so a wildcard leaf would trip readiness with no widget present.
            "https://courtpass.nycourts.gov"
            "/cdn-cgi/challenge-platform/h/b/jsd/oneshot/8eb6d5cd556e/0.88:17",
            "https://courtpass.nycourts.gov"
            "/cdn-cgi/challenge-platform/h/b/scripts/jsd/8eb6d5cd556e/main.js",
        ],
    )
    def test_rejects_non_flow_urls(self, url: str) -> None:
        assert CloudflareHandler._FLOW_RE.search(url) is None

    async def test_readiness_fires_on_second_flow_post(self) -> None:
        """Two POST-200s to the flow path release the wait before its timeout.

        Drives ``navigate_through`` with a page that replays a realistic
        response sequence, and asserts the press happens without the handler
        having to fall through the readiness timeout. Regression guard for the
        stale-literal bug: with a non-matching path this only completes after
        ``_READY_TIMEOUT_MS``.
        """
        handler = CloudflareHandler()
        page = _make_page()
        listeners: list[Any] = []
        page.on = MagicMock(
            side_effect=lambda event, fn: (
                listeners.append(fn) if event == "response" else None
            )
        )
        page.remove_listener = MagicMock()
        page.frames = []

        def _response(url: str, method: str, status: int) -> MagicMock:
            resp = MagicMock()
            resp.url = url
            resp.status = status
            resp.request.method = method
            return resp

        base = "/cdn-cgi/challenge-platform/h/g/fo/123:456:tok"
        sequence = [
            # Not a flow POST: must not count.
            _response(
                "https://ex.gov/cdn-cgi/challenge-platform/h/g/orchestrate/c",
                "GET",
                200,
            ),
            # A flow GET and a failed flow POST: neither counts.
            _response(f"https://ex.gov{base}", "GET", 200),
            _response(f"https://ex.gov{base}", "POST", 403),
            _response(f"https://ex.gov{base}", "POST", 200),  # 1
            _response(f"https://challenges.cloudflare.com{base}", "POST", 200),
        ]

        page.keyboard.press = AsyncMock()
        page.keyboard.down = AsyncMock()
        page.keyboard.up = AsyncMock()
        # The focus guard is satisfied, so this test stays about readiness.
        page.evaluate = AsyncMock(return_value="widget")
        # Every locator wait resolves immediately, so navigate_through reaches
        # the press and then reports cleared, rather than exercising the
        # widget-click fallback.
        page.locator = MagicMock(return_value=_permissive_locator())

        # The listener has to be fed before the readiness wait would expire,
        # so drive the whole thing under a deadline far below the 20s timeout.
        async def _drive() -> None:
            fed = asyncio.get_running_loop().call_later(
                0.01,
                lambda: [fn(resp) for resp in sequence for fn in listeners],
            )
            try:
                await handler.navigate_through(page)
            finally:
                fed.cancel()

        await asyncio.wait_for(_drive(), timeout=5.0)
        # Space goes through the low-level down/dwell/up in
        # ``_press_space_humanlike``, not ``press`` — only Tab is a press.
        page.keyboard.press.assert_any_await("Tab")
        page.keyboard.down.assert_any_await("Space")
        page.keyboard.up.assert_any_await("Space")


# ---------------------------------------------------------------------------
# Cloudflare focus guard
# ---------------------------------------------------------------------------


def _focus_page(states: list[str]) -> AsyncMock:
    """A page whose focus probe returns ``states`` in order, one per Tab."""
    page = _make_page()
    page.keyboard.press = AsyncMock()
    page.keyboard.down = AsyncMock()
    page.keyboard.up = AsyncMock()
    page.evaluate = AsyncMock(side_effect=list(states))
    page.locator = MagicMock(return_value=_permissive_locator())
    return page


class TestCloudflareFocusProbeJs:
    """The probe expression's accept/reject boundary.

    Exercised against a real DOM rather than a mock, because the thing under
    test is a DOM traversal: the depth bound is only meaningful relative to
    the mount's actual ancestry. The fixture reproduces the chain measured on
    the live CA challenge (2026-08-13) — input -> div -> div -> div ->
    .main-content -> .main-wrapper -> body — so a future widening of
    ``_FOCUS_ANCESTOR_DEPTH`` that would start accepting page-level focus
    fails here.
    """

    # tabindex="-1" mirrors the live mount, which reports tabIndex -1 while
    # still becoming activeElement — and is what makes the divs focusable
    # enough for the fixture to place focus on them at all.
    HTML = """
    <body>
      <div class="main-wrapper"><div class="main-content">
        <div id="outer" tabindex="-1">
          <div id="mid" tabindex="-1">
            <div id="mount" tabindex="-1">
              <input name="cf-turnstile-response" type="hidden">
            </div>
          </div>
        </div>
      </div></div>
      <a id="footer-link" href="https://example.gov">Cloudflare</a>
    </body>
    """

    @pytest.fixture
    def probe(self):
        """Evaluate the real probe JS against the fixture DOM."""
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(self.HTML)

            def _run(focus_id: str | None) -> str:
                if focus_id is None:
                    page.evaluate("() => document.activeElement.blur()")
                else:
                    page.evaluate(
                        "(id) => document.getElementById(id).focus()",
                        focus_id,
                    )
                return page.evaluate(
                    CloudflareHandler._FOCUS_PROBE_JS,
                    CloudflareHandler._FOCUS_ANCESTOR_DEPTH,
                )

            yield _run
            browser.close()

    def test_accepts_the_mount_that_hosts_the_shadow_root(self, probe) -> None:
        """Focus lands on the input's parent, never the input itself."""
        assert probe("mount") == "widget"

    @pytest.mark.parametrize("focus_id", ["footer-link", None])
    def test_rejects_focus_away_from_the_widget(
        self, probe, focus_id: str | None
    ) -> None:
        assert probe(focus_id) == "elsewhere"

    def test_rejects_page_level_containers(self, probe) -> None:
        """The depth bound must not let ``.main-content`` count as the widget.

        This is the failure the guard exists to catch: a Tab that skipped the
        widget entirely must not be reported as a hit just because focus is
        *somewhere above* the input in the tree.
        """
        probe("mount")  # sanity: the same DOM does accept the real mount
        assert probe("outer") == "elsewhere"


class TestCloudflareFocusGuard:
    """``_focus_widget_via_tab``'s control flow over probe results."""

    async def test_returns_widget_on_first_tab(self) -> None:
        handler = CloudflareHandler()
        page = _focus_page(["widget"])
        assert await handler._focus_widget_via_tab(page) == "widget"
        assert page.keyboard.press.await_count == 1

    async def test_retries_until_focus_lands(self) -> None:
        handler = CloudflareHandler()
        page = _focus_page(["elsewhere", "widget"])
        assert await handler._focus_widget_via_tab(page) == "widget"
        assert page.keyboard.press.await_count == 2

    async def test_gives_up_after_max_attempts(self) -> None:
        handler = CloudflareHandler()
        page = _focus_page(["elsewhere"] * CloudflareHandler._MAX_TAB_ATTEMPTS)
        assert await handler._focus_widget_via_tab(page) == "elsewhere"
        assert (
            page.keyboard.press.await_count
            == CloudflareHandler._MAX_TAB_ATTEMPTS
        )

    async def test_stops_tabbing_when_unverifiable(self) -> None:
        """No response input to check against: stop, don't walk focus away.

        Extra Tabs past an unknown widget can only move focus onto the footer
        links, making things strictly worse than pressing Space where we are.
        """
        handler = CloudflareHandler()
        page = _focus_page(["unverifiable", "widget"])
        assert await handler._focus_widget_via_tab(page) == "unverifiable"
        assert page.keyboard.press.await_count == 1

    async def test_probe_failure_is_unverifiable_not_fatal(self) -> None:
        """A throwing probe must not cost us the Space attempt."""
        handler = CloudflareHandler()
        page = _focus_page([])
        page.evaluate = AsyncMock(side_effect=RuntimeError("execution ctx"))
        assert await handler._focus_state(page) == "unverifiable"


class TestCloudflareSkipsSpaceOnBadFocus:
    """``navigate_through`` presses Space only when focus is plausible."""

    @staticmethod
    def _page(focus_state: str) -> AsyncMock:
        page = _focus_page([focus_state] * CloudflareHandler._MAX_TAB_ATTEMPTS)
        page.on = MagicMock()
        page.remove_listener = MagicMock()
        page.frames = []
        return page

    @pytest.fixture(autouse=True)
    def _fast_readiness(self):
        """No response listener is fed here, so let readiness time out fast.

        ``_await_flow_readiness`` continues on timeout by design; these tests
        are about what happens *after* it, not about the wait itself.
        """
        with patch.object(CloudflareHandler, "_READY_TIMEOUT_MS", 10):
            yield

    async def test_presses_space_when_focus_verified(self) -> None:
        handler = CloudflareHandler()
        page = self._page("widget")
        await asyncio.wait_for(handler.navigate_through(page), timeout=5.0)
        page.keyboard.down.assert_any_await("Space")

    async def test_skips_space_when_focus_missed(self) -> None:
        """Focus off-target: Space would only scroll, so don't spend the wait.

        The handler still has to reach its fallback and then raise, rather
        than reporting a challenge it never pressed anything at as cleared.
        """
        handler = CloudflareHandler()
        page = self._page("elsewhere")
        # _permissive_locator resolves the "shell detached" wait, so the
        # widget-click fallback reports success; the point here is only that
        # Space never happened.
        await asyncio.wait_for(handler.navigate_through(page), timeout=5.0)
        page.keyboard.down.assert_not_awaited()
        page.keyboard.up.assert_not_awaited()
