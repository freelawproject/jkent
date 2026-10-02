"""Interstitial handlers against a mock page, no transport and no browser.

Cloudflare challenge detection over the challenge shapes actually observed,
the flow path, ``_first_success`` (the race inside both captcha handlers),
and the focus / OS-click guards.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from aiohttp import web
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from jkent.driver import xvfb
from jkent.driver.unified_driver.interstitials import (
    CloudflareHandler,
    LocalStenoTranscriber,
    ReCaptchaHandler,
    _first_success,
)
from tests.servers import StartedServer, start_app

if TYPE_CHECKING:
    from playwright.async_api import Page

    from jkent.driver.unified_driver.interstitials import _FlowWatch

_Listener = Callable[[Any], object]
_Probe = Callable[[str | None], str]


def _no_xdotool(_cmd: str) -> None:
    return None


def _has_xdotool(_cmd: str) -> str:
    return "/usr/bin/xdotool"


@pytest.fixture
def no_display(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the developer's X display, so OS input is off unless a test opts in.

    The Cloudflare handler takes the xdotool path whenever ``$DISPLAY`` is set
    and xdotool is on ``$PATH``; tests written for the synthetic path would
    otherwise pass or fail by machine. A test that wants OS input sets
    ``DISPLAY`` itself.
    """
    monkeypatch.delenv("DISPLAY", raising=False)


pytestmark = pytest.mark.usefixtures("no_display")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
#: The ``<script src>`` in ``<head>`` is the element the bootstrap appends,
#: written out statically as the live DOM holds it: lxml runs no script, and
#: the bootstrap text alone gives a ``src`` selector nothing to match.
_SHAPE_JSD_BEACON_ONLY = """
<html><head><title>NYS Court-PASS Home Page</title>
<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script>
</head><body>
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

    _FLOW = "/cdn-cgi/challenge-platform/h/g/fo/123:456:tok"

    @staticmethod
    def _flow_response(url: str, method: str, status: int) -> MagicMock:
        resp = MagicMock()
        resp.url = url
        resp.status = status
        resp.request.method = method
        return resp

    @pytest.mark.parametrize(
        ("url", "method", "status"),
        [
            pytest.param(
                "https://ex.gov/cdn-cgi/challenge-platform/h/g/orchestrate/c",
                "POST",
                200,
                id="not-the-flow-path",
            ),
            pytest.param(f"https://ex.gov{_FLOW}", "GET", 200, id="flow-get"),
            pytest.param(
                f"https://ex.gov{_FLOW}", "POST", 403, id="failed-flow-post"
            ),
        ],
    )
    async def test_non_flow_responses_do_not_count(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        url: str,
        method: str,
        status: int,
    ) -> None:
        """Beside one real flow POST, the response still leaves readiness at 1.

        If it counted, the pair would reach the two readiness needs and the
        wait would end early — so this fails exactly when it counts.
        """
        monkeypatch.setattr(CloudflareHandler, "_READY_TIMEOUT_MS", 50)
        handler = CloudflareHandler()
        page = _make_page()

        def _on(event: str, fn: _Listener) -> None:
            if event == "response":
                fn(self._flow_response(url, method, status))
                fn(
                    self._flow_response(
                        f"https://ex.gov{self._FLOW}", "POST", 200
                    )
                )

        page.on = MagicMock(side_effect=_on)
        page.remove_listener = MagicMock()

        with caplog.at_level("WARNING"):
            await handler._await_flow_readiness(page)
        assert "saw 1 matching response(s)" in caplog.text

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
        listeners: list[_Listener] = []

        def _on(event: str, fn: _Listener) -> None:
            if event == "response":
                listeners.append(fn)

        page.on = MagicMock(side_effect=_on)
        page.remove_listener = MagicMock()
        page.frames = list[Any]()

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


class TestFirstSuccess:
    """``_first_success`` — the race inside both captcha handlers."""

    async def test_first_to_return_wins_and_the_loser_is_cancelled(
        self,
    ) -> None:
        cancelled = asyncio.Event()

        async def slow() -> str:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "slow"

        async def fast() -> str:
            return "fast"

        assert await _first_success(slow(), fast()) == "fast"
        assert cancelled.is_set()

    async def test_a_raising_waiter_drops_out_of_the_race(self) -> None:
        async def fails() -> str:
            raise PlaywrightTimeoutError("gone")

        async def later() -> str:
            await asyncio.sleep(0.01)
            return "later"

        assert await _first_success(fails(), later()) == "later"

    async def test_none_when_every_waiter_raises(self) -> None:
        async def fails() -> str:
            raise PlaywrightTimeoutError("gone")

        assert await _first_success(fails(), fails()) is None

    async def test_a_dropped_waiter_is_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A waiter that raised leaves a trace, not just a missing result.

        Otherwise a selector typo and a genuine timeout look the same: None.
        """

        async def fails() -> str:
            raise ValueError("bad selector")

        async def later() -> str:
            await asyncio.sleep(0.01)
            return "later"

        with caplog.at_level("DEBUG", logger=_first_success.__module__):
            assert await _first_success(fails(), later()) == "later"
        (record,) = [r for r in caplog.records if r.exc_info]
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], ValueError)


async def test_flow_readiness_timeout_continues_and_unhooks() -> None:
    """No flow POST in time: the handler continues, and its listener goes."""
    page = MagicMock()
    with patch.object(CloudflareHandler, "_READY_TIMEOUT_MS", 10):
        await CloudflareHandler()._await_flow_readiness(page)
    (listener,) = [c.args[1] for c in page.on.call_args_list]
    page.remove_listener.assert_called_once_with("response", listener)


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
    def probe(self) -> Iterator[_Probe]:
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

    def test_accepts_the_mount_that_hosts_the_shadow_root(
        self, probe: _Probe
    ) -> None:
        """Focus lands on the input's parent, never the input itself."""
        assert probe("mount") == "widget"

    @pytest.mark.parametrize("focus_id", ["footer-link", None])
    def test_rejects_focus_away_from_the_widget(
        self, probe: _Probe, focus_id: str | None
    ) -> None:
        assert probe(focus_id) == "elsewhere"

    def test_rejects_page_level_containers(self, probe: _Probe) -> None:
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
        page.frames = list[Any]()
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


class TestCloudflareOSClick:
    """The OS-level click path: availability gating and coordinate mapping.

    The click itself needs a real X display, so what is testable without one is
    everything around it — and that is where the failure modes live. Each test
    here corresponds to a bug found by running the handler end-to-end in the
    container rather than to a hypothetical.
    """

    def test_unavailable_without_display(self) -> None:
        reason = CloudflareHandler()._os_input_unavailable_reason(None)
        assert reason is not None
        assert "display" in reason

    def test_unavailable_without_xdotool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "jkent.driver.unified_driver.interstitials.shutil.which",
            _no_xdotool,
        )
        reason = CloudflareHandler()._os_input_unavailable_reason(":99")
        assert reason is not None
        assert "xdotool" in reason

    def test_available_with_display_and_xdotool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "jkent.driver.unified_driver.interstitials.shutil.which",
            _has_xdotool,
        )
        assert CloudflareHandler()._os_input_unavailable_reason(":99") is None

    def test_private_display_preferred_over_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A registered per-browser display wins, and is reported as private.

        os.environ cannot describe more than one browser (workers are asyncio
        tasks in one process), and camoufox's own virtual_display= mutates it —
        so the registry is the only trustworthy source.
        """
        monkeypatch.setenv("DISPLAY", ":99")
        page = _make_page()
        context = MagicMock()
        page.context = context
        xvfb.register(context, ":137")
        display, is_private = CloudflareHandler._resolve_display(page)
        assert (display, is_private) == (":137", True)

    def test_falls_back_to_environment_as_shared(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DISPLAY", ":99")
        page = _make_page()
        page.context = MagicMock()  # never registered
        display, is_private = CloudflareHandler._resolve_display(page)
        assert (display, is_private) == (":99", False)

    @staticmethod
    def _page_with_box(dpr: float) -> AsyncMock:
        """A page whose widget mount has a layout box and a known screen origin."""
        page = _make_page()
        locator = _permissive_locator()
        locator.bounding_box = AsyncMock(
            return_value={
                "x": 100.0,
                "y": 200.0,
                "width": 896.0,
                "height": 70.0,
            }
        )
        page.locator = MagicMock(return_value=locator)
        page.evaluate = AsyncMock(return_value={"sx": 4, "sy": 57, "dpr": dpr})
        return page

    @pytest.mark.asyncio
    async def test_screen_point_offsets_by_window_origin(self) -> None:
        """CSS box + window origin, with the checkbox inset applied.

        Turnstile draws the checkbox near the mount's left edge, so the inset is
        what makes this land on the checkbox rather than in the middle of a
        896px-wide container.
        """
        handler = CloudflareHandler()
        point = await handler._checkbox_screen_point(self._page_with_box(1))
        assert point == (
            int(4 + 100.0 + CloudflareHandler._CHECKBOX_INSET_PX),
            int(57 + 200.0 + 35.0),
        )

    @pytest.mark.asyncio
    async def test_screen_point_declines_non_unit_dpr(self) -> None:
        """dpr != 1 is declined, not guessed at.

        The viewport->screen mapping is only verified at dpr 1 (what Xvfb
        gives). Guessing between `origin + css` and `(origin + css) * dpr` on an
        untested display would click at arbitrary coordinates.
        """
        handler = CloudflareHandler()
        assert (
            await handler._checkbox_screen_point(self._page_with_box(2))
            is None
        )

    @pytest.mark.asyncio
    async def test_no_layout_box_yields_no_point(self) -> None:
        handler = CloudflareHandler()
        page = _make_page()
        locator = _permissive_locator()
        locator.bounding_box = AsyncMock(return_value=None)
        page.locator = MagicMock(return_value=locator)
        assert await handler._checkbox_screen_point(page) is None

    @pytest.mark.asyncio
    async def test_moves_away_before_targeting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The approach hop must precede the --sync move.

        ``xdotool mousemove --sync`` waits for a motion event, so moving to
        where the pointer already sits blocks until it is killed — which is
        exactly what the second attempt of a same-pixel retry does. Stepping
        away first guarantees the final hop actually moves.
        """
        monkeypatch.setenv("DISPLAY", ":99")
        monkeypatch.setattr(
            "jkent.driver.unified_driver.interstitials.shutil.which",
            _has_xdotool,
        )
        handler = CloudflareHandler()
        calls: list[tuple[str, ...]] = []

        async def _fake_xdotool(
            self: CloudflareHandler, display: str, *args: str
        ) -> bool:
            assert display == ":99"  # threaded through, not read from environ
            calls.append(args)
            return True

        monkeypatch.setattr(CloudflareHandler, "_run_xdotool", _fake_xdotool)
        page = self._page_with_box(1)
        # Never clears, so both attempts run and the retry path is exercised.
        monkeypatch.setattr(
            CloudflareHandler,
            "_challenge_cleared",
            AsyncMock(return_value=False),
        )

        assert await handler._os_click_attempts(page, ":99") is False

        moves = [c for c in calls if c[0] == "mousemove"]
        assert len(moves) == 2 * CloudflareHandler._OS_CLICK_ATTEMPTS
        # Within each attempt: a plain approach hop, then the --sync target hop.
        assert "--sync" not in moves[0]
        assert "--sync" in moves[1]
        assert moves[0][1:] != moves[1][1:]
        assert [c[0] for c in calls].count("click") == (
            CloudflareHandler._OS_CLICK_ATTEMPTS
        )


class TestCloudflareConcurrentOSClick:
    """Concurrent workers: raise-then-click, and recovering a queued-stale widget.

    The transport runs ONE browser and leases each worker its own page, and every
    Playwright page is a separate OS window — measured: 3 pages, 3 X windows, all
    at 0,0, all the same size, only the topmost taking pointer input. So the
    handler must raise its own window and serialise, and a worker that queued
    behind a sibling must not click a widget that went stale while it waited.
    """

    @staticmethod
    def _clickable_page() -> AsyncMock:
        page = _make_page()
        locator = _permissive_locator()
        locator.bounding_box = AsyncMock(
            return_value={"x": 10.0, "y": 20.0, "width": 896.0, "height": 70.0}
        )
        page.locator = MagicMock(return_value=locator)
        page.evaluate = AsyncMock(return_value={"sx": 0, "sy": 57, "dpr": 1})
        page.bring_to_front = AsyncMock()
        page.reload = AsyncMock()
        page.context = MagicMock()
        # Listener registration is synchronous on a real Page.
        page.on = MagicMock()
        page.remove_listener = MagicMock()
        return page

    @pytest.fixture(autouse=True)
    def _os_input_available(
        self, monkeypatch: pytest.MonkeyPatch, no_display: None
    ) -> None:
        # After no_display (the module's usefixtures would otherwise run
        # second and delete this).
        monkeypatch.setenv("DISPLAY", ":99")
        monkeypatch.setattr(
            "jkent.driver.unified_driver.interstitials.shutil.which",
            _has_xdotool,
        )
        monkeypatch.setattr(
            CloudflareHandler, "_RAISE_SETTLE_S", 0.0, raising=False
        )
        monkeypatch.setattr(
            CloudflareHandler, "_run_xdotool", AsyncMock(return_value=True)
        )

    @pytest.mark.asyncio
    async def test_foregrounds_its_window_before_solving(self) -> None:
        """Without this the click lands in whichever window is on top."""
        handler = CloudflareHandler()
        page = self._clickable_page()
        with patch.object(CloudflareHandler, "_solve_challenge", AsyncMock()):
            await handler.navigate_through(page)
        page.bring_to_front.assert_awaited()

    @pytest.mark.asyncio
    async def test_uncontended_solve_does_not_reload(self) -> None:
        """A worker that never queued has a fresh widget — leave it alone."""
        handler = CloudflareHandler()
        page = self._clickable_page()
        with patch.object(CloudflareHandler, "_solve_challenge", AsyncMock()):
            await handler.navigate_through(page)
        page.reload.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_queued_worker_reloads_before_clicking(self) -> None:
        """Contended: the widget went stale while waiting, so refresh it first.

        Measured 1/2 without this — the queued worker clicked a widget that had
        sat ~15s and cleared nothing, while a reload gets a fresh one (and often
        no challenge at all, the cookie jar being shared).
        """
        handler = CloudflareHandler()
        page = self._clickable_page()

        # Hold the lock so the call under test observes contention, exactly as a
        # sibling mid-solve would. Via the accessor: the lock is per-handler,
        # created on first use so it binds to the running loop.
        lock = handler._click_lock()
        await lock.acquire()

        async def _release_soon() -> None:
            await asyncio.sleep(0.05)
            lock.release()

        releaser = asyncio.create_task(_release_soon())
        solve = AsyncMock()
        try:
            with (
                patch.object(CloudflareHandler, "_solve_challenge", solve),
                patch.object(
                    CloudflareHandler,
                    "_challenge_cleared",
                    AsyncMock(return_value=True),
                ),
            ):
                await handler.navigate_through(page)
            page.reload.assert_awaited()
            # Cleared after the reload -> a sibling's clearance covered us, so
            # the solve was skipped entirely.
            solve.assert_not_awaited()
        finally:
            await releaser

    @pytest.mark.asyncio
    async def test_queued_worker_sees_the_flow_posts_of_its_reload(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Readiness is armed before the refresh, not after it.

        The reload is what fires the fresh challenge's flow POSTs. A listener
        attached only once the reload returned misses them, and the queued
        worker then sits out the whole readiness timeout holding the click
        lock every sibling is waiting on.
        """
        monkeypatch.setattr(CloudflareHandler, "_READY_TIMEOUT_MS", 2_000)
        handler = CloudflareHandler()
        page = self._clickable_page()
        listeners: list[_Listener] = []

        def _on(event: str, fn: _Listener) -> None:
            if event == "response":
                listeners.append(fn)

        page.on = MagicMock(side_effect=_on)

        def _off(_event: str, fn: _Listener) -> None:
            listeners.remove(fn)

        page.remove_listener = MagicMock(side_effect=_off)

        def _flow_post() -> MagicMock:
            resp = MagicMock()
            resp.url = "https://ex.gov/cdn-cgi/challenge-platform/h/g/fo/1:2"
            resp.status = 200
            resp.request.method = "POST"
            return resp

        async def _reload(**_kw: Any) -> None:
            for _ in range(2):
                for fn in list(listeners):
                    fn(_flow_post())

        page.reload = AsyncMock(side_effect=_reload)
        # Still challenged after the reload, cleared by the OS click.
        cleared = AsyncMock(side_effect=[False, True])

        lock = handler._click_lock()
        await lock.acquire()
        asyncio.get_running_loop().call_later(0.01, lock.release)
        with (
            patch.object(CloudflareHandler, "_challenge_cleared", cleared),
            caplog.at_level("INFO"),
        ):
            await asyncio.wait_for(handler.navigate_through(page), 1.0)
        assert "orchestrator ready" in caplog.text
        assert not listeners

    @pytest.mark.asyncio
    async def test_clicks_are_serialised(self) -> None:
        """Two workers must never interleave raise/move/click.

        No clearance ever lands, so every queued worker still has to solve:
        all four solves run, and none of them overlap.
        """
        handler = CloudflareHandler()
        active = 0
        overlaps = 0
        solve_calls = 0

        async def _solve(
            self: CloudflareHandler,
            page: Page,
            display: str | None,
            flow: _FlowWatch | None = None,
        ) -> None:
            nonlocal active, overlaps, solve_calls
            solve_calls += 1
            active += 1
            if active > 1:
                overlaps += 1
            await asyncio.sleep(0.02)
            active -= 1

        with (
            patch.object(CloudflareHandler, "_solve_challenge", _solve),
            patch.object(
                CloudflareHandler,
                "_challenge_cleared",
                AsyncMock(return_value=False),
            ),
        ):
            await asyncio.gather(
                *(
                    handler.navigate_through(self._clickable_page())
                    for _ in range(4)
                )
            )
        assert (solve_calls, overlaps) == (4, 0)


class TestReCaptchaAutoSolve:
    """The auto-solve path, against a stand-in widget in a real browser."""

    # Parent page: what api.js does on a solve is write the token into the
    # page's g-recaptcha-response textarea and then call the page's callback,
    # in one task. The token lands a beat after the anchor iframe checks its
    # box, which is the gap the handler must not return into.
    PARENT = """
    <html><body>
      <div class="g-recaptcha"></div>
      <iframe src="https://www.google.com/recaptcha/api2/anchor?k=x"></iframe>
      <textarea name="g-recaptcha-response" style="display:none"></textarea>
      <div id="out">pending</div>
      <script>
        window.addEventListener("message", (e) => {
          document.querySelector("[name=g-recaptcha-response]").value = e.data;
          document.getElementById("out").textContent = "callback ran";
        });
      </script>
    </body></html>
    """
    ANCHOR = """
    <html><body>
      <div id="recaptcha-anchor" style="width:30px;height:30px"
           onclick="this.classList.add('recaptcha-checkbox-checked');
                    setTimeout(() => parent.postMessage('tok', '*'), 300);">
      </div>
    </body></html>
    """

    async def test_returns_after_the_page_callback_not_the_checkmark(
        self,
    ) -> None:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch()
            except PlaywrightError as exc:
                pytest.skip(f"no launchable chromium: {exc}".splitlines()[0])
            try:
                page = await browser.new_page()

                async def _serve(route: Any) -> None:
                    anchor = "recaptcha" in route.request.url
                    await route.fulfill(
                        status=200,
                        content_type="text/html",
                        body=self.ANCHOR if anchor else self.PARENT,
                    )

                await page.route("**/*", _serve)
                await page.goto("https://court.example/")
                await ReCaptchaHandler(AsyncMock()).navigate_through(page)
                assert await page.text_content("#out") == "callback ran"
            finally:
                await browser.close()


class TestLocalStenoTranscriber:
    """POSTs the clip to ``/transcribe?format=text`` and returns its text."""

    @staticmethod
    async def _serve(
        status: int, text: str, seen: list[dict[str, Any]]
    ) -> StartedServer:
        async def transcribe(request: web.Request) -> web.Response:
            form = await request.post()
            upload = form["file"]
            assert isinstance(upload, web.FileField)
            seen.append(
                {
                    "query": dict(request.query),
                    "filename": upload.filename,
                    "content_type": upload.content_type,
                    "data": upload.file.read(),
                }
            )
            return web.Response(status=status, text=text)

        app = web.Application()
        app.router.add_post("/transcribe", transcribe)
        return await start_app(app)

    async def test_posts_audio_and_returns_stripped_text(self) -> None:
        seen: list[dict[str, Any]] = []
        server = await self._serve(200, "  seven three nine \n", seen)
        try:
            text = await LocalStenoTranscriber(
                server.base_url + "/"
            ).transcribe(b"ID3-audio")
        finally:
            await server.aclose()
        assert text == "seven three nine"
        assert seen == [
            {
                "query": {"format": "text"},
                "filename": "audio.mp3",
                "content_type": "audio/mpeg",
                "data": b"ID3-audio",
            }
        ]

    async def test_error_status_raises(self) -> None:
        server = await self._serve(503, "busy", [])
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await LocalStenoTranscriber(server.base_url).transcribe(b"x")
        finally:
            await server.aclose()
