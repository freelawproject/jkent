"""Interstitial page handlers for the unified browser transports.

Interstitial handlers run on the live Playwright page after navigation
but before the DOM snapshot is taken.  When a scraper declares a
``*_HANDLER`` driver requirement, :class:`PlaywrightTransport` races each
handler's waitlist against the scraper step's own await conditions; if a
handler's conditions match first, it gets to interact with the page (e.g.
solve a captcha) before the scraper ever sees the HTML.

The handler classes depend only on a live Playwright ``Page`` and the
``WaitFor*`` condition types.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import os
import random
import re
import shutil
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import httpx
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from jkent.common.exceptions import InterstitialUnresolved
from jkent.data_types import (
    DriverRequirement,
    WaitCondition,
    WaitForSelector,
)
from jkent.driver.unified_driver.requirements import INTERSTITIAL_REQUIREMENTS
from jkent.driver.xvfb import display_for
from jkent.driver.xvfb import env_for as xvfb_env_for

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable, Iterator
    from typing import Any

    from playwright.async_api import (
        FloatRect,
        FrameLocator,
        Locator,
        Page,
        Response,
    )

logger = logging.getLogger(__name__)

T = TypeVar("T")
_FactoryT = TypeVar("_FactoryT", bound=Callable[[], "InterstitialHandler"])


async def _first_success(*waiters: Coroutine[Any, Any, T]) -> T | None:
    """The result of the first waiter to finish without raising.

    Races the waiters; one that raises (typically a timeout) drops out
    rather than ending the race. None if every waiter raised. The losers
    are cancelled and awaited before returning.
    """
    tasks = [asyncio.create_task(waiter) for waiter in waiters]
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                exc = task.exception()
                if exc is None:
                    return task.result()
                # Usually a timeout, but a bad selector drops out the same
                # way; without this both read as "neither matched".
                logger.debug(
                    "Waiter dropped out of the race: %r", exc, exc_info=exc
                )
        return None
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class AudioTranscriber(abc.ABC):
    """Abstract base for audio-to-text transcription services.

    Implementations receive raw audio bytes and return the transcribed
    text.  Used by :class:`ReCaptchaHandler` to solve audio challenges.
    """

    @abc.abstractmethod
    async def transcribe(self, audio_data: bytes) -> str: ...


class LocalStenoTranscriber(AudioTranscriber):
    """AudioTranscriber backed by a local steno transcription server.

    Posts audio bytes to the steno server's ``/transcribe`` endpoint
    and returns the plain-text transcription.

    Args:
        server_url: Base URL of the steno server.
            Defaults to ``http://127.0.0.1:8000``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._timeout = timeout

    async def transcribe(self, audio_data: bytes) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._server_url}/transcribe",
                params={"format": "text"},
                files={
                    "file": ("audio.mp3", audio_data, "audio/mpeg"),
                },
            )
            response.raise_for_status()
            return response.text.strip()


class _FlowWatch:
    """Counts Cloudflare's flow-POST 200s on one page; ready at the second."""

    def __init__(self, flow_re: re.Pattern[str]) -> None:
        self._flow_re = flow_re
        self.count = 0
        self.ready = asyncio.Event()

    def on_response(self, resp: Response) -> None:
        # Only the orchestrator's flow POSTs returning 200 signal readiness;
        # GET preflights and 3xx/4xx/5xx responses to the same path don't
        # count.
        if (
            self._flow_re.search(resp.url)
            and resp.request.method == "POST"
            and resp.status == 200
        ):
            self.count += 1
            if self.count >= 2:
                self.ready.set()


class InterstitialHandler(abc.ABC):
    """Handles interstitial pages (captchas, disclaimers, etc.) on the live
    Playwright page, after navigation but before DOM snapshot."""

    @abc.abstractmethod
    def waitlist(self) -> list[WaitCondition]:
        """Conditions that indicate this interstitial is present.

        All conditions must match (conjunction) for the handler to fire.
        """

    @abc.abstractmethod
    async def navigate_through(self, page: Page) -> None:
        """Interact with the live page to get past the interstitial.

        When this returns, the page should be showing the real content
        (or another interstitial that a subsequent handler can deal with).
        """


class ReCaptchaHandler(InterstitialHandler):
    """Handles reCAPTCHA v2 interstitials via the audio challenge.

    Detects a ``div.g-recaptcha`` widget (even when hidden behind
    invisible parents), switches to the audio challenge, downloads
    the audio clip, transcribes it via the provided
    :class:`AudioTranscriber`, and submits the answer.

    Args:
        transcriber: An :class:`AudioTranscriber` implementation for
            converting audio challenge clips to text.
    """

    def __init__(self, transcriber: AudioTranscriber) -> None:
        self._transcriber = transcriber

    def waitlist(self) -> list[WaitCondition]:
        return [WaitForSelector("div.g-recaptcha", state="attached")]

    async def navigate_through(self, page: Page) -> None:
        logger.info("reCAPTCHA interstitial detected — solving via audio")

        # 1. Reveal hidden parent elements of .g-recaptcha
        await page.evaluate("""() => {
            const el = document.querySelector('.g-recaptcha');
            if (!el) return;
            let node = el.parentElement;
            while (node && node !== document.body) {
                node.style.display = '';
                node.style.visibility = '';
                const cs = window.getComputedStyle(node);
                if (cs.display === 'none') node.style.display = 'block';
                if (cs.visibility === 'hidden')
                    node.style.visibility = 'visible';
                node = node.parentElement;
            }
        }""")

        # 2. Click the reCAPTCHA checkbox inside the anchor iframe
        anchor = page.frame_locator(
            "iframe[src*='google.com/recaptcha'][src*='anchor']"
        )
        await anchor.locator("#recaptcha-anchor").click(timeout=10_000)

        # 3. Race: auto-solve (checkmark appears) vs challenge (bframe)
        async def _wait_for_checkmark() -> str:
            await anchor.locator(".recaptcha-checkbox-checked").wait_for(
                state="attached", timeout=10_000
            )
            return "solved"

        async def _wait_for_bframe() -> str:
            await page.locator(
                "iframe[src*='google.com/recaptcha'][src*='bframe']"
            ).wait_for(state="attached", timeout=10_000)
            return "challenge"

        # A challenge (or neither, e.g. both timed out) falls through to the
        # audio-challenge path rather than crashing.
        outcome = await _first_success(
            _wait_for_checkmark(), _wait_for_bframe()
        )
        if outcome == "solved":
            logger.info("reCAPTCHA auto-solved (no challenge)")
            await self._await_page_callback(page)
            return

        # 4. Click the audio challenge button inside the bframe
        bframe = page.frame_locator(
            "iframe[src*='google.com/recaptcha'][src*='bframe']"
        )
        await bframe.locator("#recaptcha-audio-button").click(timeout=10_000)

        # 5. Click PLAY and intercept the audio response
        await bframe.locator(".rc-audiochallenge-tdownload-link").wait_for(
            state="attached", timeout=15_000
        )

        audio_response = await self._intercept_audio(page, bframe)
        audio_data = await audio_response.body()

        # 6. Transcribe the audio
        logger.debug(
            "Transcribing reCAPTCHA audio (%d bytes)",
            len(audio_data),
        )
        transcription = await self._transcriber.transcribe(audio_data)
        logger.info("reCAPTCHA audio transcription: %r", transcription)

        # 7. Fill the response field and submit
        await bframe.locator("#audio-response").fill(transcription)
        await bframe.locator("#recaptcha-verify-button").click(timeout=10_000)

        # 8. Verify success — checkmark visible in the anchor iframe
        await anchor.locator(".recaptcha-checkbox-checkmark").wait_for(
            state="visible", timeout=15_000
        )
        await self._await_page_callback(page)
        logger.info("reCAPTCHA solved successfully")

    # Non-empty once the widget has handed the page its token. api.js writes
    # the token and calls the page's data-callback in the same task, so a set
    # value means the callback has run too.
    _TOKEN_SET_JS = """() => Array.from(
        document.querySelectorAll("[name='g-recaptcha-response']")
    ).some((t) => t.value)"""

    @classmethod
    async def _await_page_callback(cls, page: Page) -> None:
        """Wait for the page to receive the token, not just the checkmark.

        The checkmark is drawn inside the anchor iframe; the token reaches the
        page by postMessage a beat later, and the page's callback (enable the
        submit button, post the form) only runs then. Returning on the
        checkmark hands the scraper a page whose callback has not run.
        """
        try:
            await page.wait_for_function(cls._TOKEN_SET_JS, timeout=10_000)
        except PlaywrightTimeoutError:
            # Solved as far as the widget is concerned; snapshot what is there
            # rather than fail a request the checkmark says went through.
            logger.warning(
                "reCAPTCHA checked but no g-recaptcha-response token reached "
                "the page in 10s; continuing"
            )

    @staticmethod
    async def _intercept_audio(
        page: Page,
        bframe: FrameLocator,
    ) -> Response:
        """Click PLAY and intercept the audio payload response.

        Uses ``page.expect_response`` to capture the audio file as
        the browser fetches it, avoiding a separate HTTP request.
        """
        async with page.expect_response(
            lambda r: "recaptcha" in r.url and "payload" in r.url,
            timeout=15_000,
        ) as response_info:
            await bframe.locator(".rc-audiochallenge-play-button").click(
                timeout=10_000
            )
        return await response_info.value


class CloudflareHandler(InterstitialHandler):
    """Handles Cloudflare "Just a moment..." interstitials.

    Detection keys on the *challenge shell* — the orchestrator bootstrap
    ``<script>`` Cloudflare ships inside the challenge document — not on the
    Turnstile widget the orchestrator later injects. The widget input lands
    up to a second later, is absent from the wire HTML, and never renders on
    the click-gated variant.

    ``navigate_through`` then:

    1. Waits for the orchestrator's flow-POST readiness signal
       (:attr:`_FLOW_RE`, second POST returning 200).
    2. Waits for something interactive to exist; readiness lands while the
       widget iframe is still loading.
    3. Clicks "Verify you are human" on the click-gated variant, where
       Turnstile only mounts once that button is activated.
    4. Where OS input is possible — a headed browser on a real display
       (``$DISPLAY`` with ``xdotool`` on ``$PATH``, i.e. the Xvfb container)
       — clicks the checkbox with a real X11 pointer event
       (:meth:`_os_click_attempts`). This is the path that clears it.
    5. Otherwise, or if that fails, tabs onto the Turnstile mount, verifying
       focus landed there (:meth:`_focus_widget_via_tab`), presses Space as a
       keydown/dwell/keyup, then tries :meth:`_click_widget_checkbox`.
       Turnstile currently ignores synthetic input, so this rarely clears it.
    6. Confirms the *challenge is gone* rather than that the widget input
       detached — an input that never attached satisfies ``detached``
       immediately.

    A challenge that survives every step raises rather than handing back
    challenge HTML the classifier would store as content.
    """

    # The challenge shell: Cloudflare's orchestrator bootstrap, present in the
    # challenge document's own HTML for every template shape observed (bare
    # ~6KB shell, empty-mount + spinner, pre-rendered widget, click-gated
    # button). Scoped to the ``orchestrate`` leaf deliberately: a bare
    # ``challenge-platform`` match also hits the invisible ``jsd`` bot-score
    # beacon CF injects into ordinary 200 pages — observed on
    # courtpass.nycourts.gov, whose zone serves no challenge at all — which
    # would fire this handler on real content.
    _CHALLENGE_SHELL = "script[src*='challenge-platform'][src*='orchestrate']"
    _RESPONSE_INPUT = "input[name='cf-turnstile-response']"
    # The click-gated variant's button. Matched on its value, not class or id:
    # this template's class/id names are per-build obfuscated (``YNXn4``,
    # ``#Untl7``) while the copy is stable. Case-sensitive on purpose — the
    # CSS4 ``i`` flag is not portable across the selector engines this repo
    # tests with, and if Cloudflare ever re-cases the copy this simply falls
    # through to the widget path instead of breaking.
    _VERIFY_BUTTON = "input[type='button'][value*='Verify']"
    # The orchestrator's flow-POST path. Deliberately a regex, not a literal:
    # Cloudflare versions this path and rotates the branch segment. It was
    # ``/cdn-cgi/challenge-platform/h/b/flow/ov1/``; the CA appellate
    # deployment now posts to ``/cdn-cgi/challenge-platform/h/{b,g}/fo/<id>``,
    # and both branch letters appear within a single run. A literal match went
    # stale silently — the readiness event simply never fired, so every
    # challenge burned the full _READY_TIMEOUT_MS before pressing Tab+Space.
    #
    # The ``(?:fo|flow/ov1)`` leaf is load-bearing, so keep any future
    # rotation as an explicit alternative rather than widening to ``h/[^/]+/``:
    # CF's ``jsd/oneshot`` bot-score beacon is *also* a POST returning 200
    # under ``h/<branch>/``, and it fires on pages carrying no challenge
    # whatsoever, so a wildcard leaf would trip readiness spuriously. ``ci``
    # needs no entry either — it is a GET, excluded by the method check below.
    _FLOW_RE = re.compile(
        r"/cdn-cgi/challenge-platform/h/[^/]+/(?:fo|flow/ov1)/"
    )
    # Detection is bounded well under Playwright's 30s default: the shell
    # matches in under 0.4s whenever it is there at all, and this timeout is
    # what a no-challenge navigation pays before the handler concedes the
    # race.
    _DETECT_TIMEOUT_MS = 5_000
    # Empirical: the second flow POST returns ~2s after navigation
    # in camoufox.  Pad heavily in case the orchestrator is slow.
    _READY_TIMEOUT_MS = 20_000
    # Gap between flow readiness and the widget/button actually existing;
    # observed under 1s, padded.
    _INTERACTIVE_TIMEOUT_MS = 15_000
    # After we press Space, the page navigates to a token URL then back
    # to the original; clear time observed at ~1s, allow 30s headroom.
    _CLEAR_TIMEOUT_MS = 30_000
    # The fallback click only needs to cover that same ~1s clear, so it gets a
    # much shorter leash: two full 30s waits put an unsolvable challenge at
    # ~61s per attempt, and with retries a doomed request burned minutes before
    # the queue gave up on it.
    _FALLBACK_CLEAR_TIMEOUT_MS = 10_000
    # Turnstile draws its checkbox at the left edge of the mount container,
    # not its centre — the container spans the full content column (896px
    # observed) while the widget itself is ~300px.
    _CHECKBOX_INSET_PX = 30
    # OS clicks get more than one shot: the widget sometimes needs a second
    # click, and the sampled-behaviour scoring behind it is not deterministic.
    # Two is deliberate — each attempt pays _FALLBACK_CLEAR_TIMEOUT_MS, and an
    # unsolvable challenge must not sit here burning the queue's time.
    _OS_CLICK_ATTEMPTS = 2
    # How long to wait after bring_to_front() before clicking. The call returns
    # when Firefox has been asked to raise; the X restack lands after that, and
    # clicking into the gap hits whichever window is still on top.
    _RAISE_SETTLE_S = 0.4
    # xdotool is a local, non-network binary; if it has not returned in a
    # second something is wrong with the display, not with the click.
    _OS_CLICK_SUBPROCESS_TIMEOUT_S = 5.0
    # Viewport -> screen mapping for the OS click. mozInnerScreenX/Y are CSS px
    # relative to the screen origin, so under dpr=1 they map 1:1 onto X11
    # device px. Xvfb is always dpr=1, which is the only environment this path
    # runs in; anything else is logged and skipped rather than clicked blind at
    # coordinates that would land somewhere arbitrary.
    _SCREEN_ORIGIN_JS = """() => ({
        sx: window.mozInnerScreenX,
        sy: window.mozInnerScreenY,
        dpr: window.devicePixelRatio,
    })"""

    def __init__(self) -> None:
        self._os_click_lock: asyncio.Lock | None = None

    def _click_lock(self) -> asyncio.Lock:
        """The OS-click lock, created on first use.

        Serialises raise → locate → click across every worker sharing this
        handler — which, since handlers are built per transport
        (:data:`INTERSTITIAL_FACTORIES`), is every worker sharing one browser.

        Required, not belt-and-braces. The transport runs one browser and
        gives each worker its own page, and every Playwright page is a
        separate OS window — 3 pages, 3 X windows, all at 0,0, all the same
        size, only the topmost receiving pointer input. So the sequence has to
        be atomic: if a sibling worker calls ``bring_to_front()`` between our
        raise and our click, our click lands in its window instead.

        Per-display allocation (see :mod:`jkent.driver.xvfb`) does not replace
        this. It isolates one *browser* from another; the pages inside a
        browser still share its display and window stack.

        Created lazily rather than in ``__init__``: an ``asyncio.Lock`` binds
        to the loop that first awaits it, and a handler may be constructed
        outside the loop it will run on.
        """
        if self._os_click_lock is None:
            self._os_click_lock = asyncio.Lock()
        return self._os_click_lock

    # How many Tabs to spend looking for the widget. One is enough whenever
    # the widget is mounted — measured tab order on the CA challenge is
    # [widget mount] -> footer "Cloudflare" -> footer "Privacy" — so the extra
    # attempts only cover a variant that puts something ahead of it. Kept
    # small deliberately: past the widget there is nothing but those two
    # footer links, and tabbing further just walks focus away from the target.
    _MAX_TAB_ATTEMPTS = 3
    # How far up from ``_RESPONSE_INPUT`` the focus probe will accept a match.
    # Focus lands on the input's *parent* (the mount that hosts the closed
    # shadow root), never on the input itself, so 0 would never match. The
    # bound matters in the other direction too: the measured chain is
    # input -> div -> div -> div -> div.main-content -> div.main-wrapper ->
    # body, so 2 stops three levels short of anything page-level and cannot
    # mistake "focus is somewhere on the document" for "focus is on the
    # widget".
    _FOCUS_ANCESTOR_DEPTH = 2
    # Returns "widget" | "elsewhere" | "unverifiable". The last is not a
    # failure: on the click-gated variant the response input may never
    # attach, and the widget can still be present and functional, so the
    # caller presses Space anyway rather than skipping it on a technicality.
    _FOCUS_PROBE_JS = """(depth) => {
        const input = document.querySelector(
            "input[name='cf-turnstile-response']");
        if (!input) return "unverifiable";
        const active = document.activeElement;
        if (!active || active === document.body) return "elsewhere";
        let node = input;
        for (let i = 0; i <= depth && node; i++) {
            if (node === active) return "widget";
            node = node.parentElement;
        }
        return "elsewhere";
    }"""

    def waitlist(self) -> list[WaitCondition]:
        return [
            WaitForSelector(
                self._CHALLENGE_SHELL,
                state="attached",
                timeout=self._DETECT_TIMEOUT_MS,
            )
        ]

    async def navigate_through(self, page: Page) -> None:
        """Get past the challenge, holding the display while doing it.

        Where OS input is possible the *whole* solve is serialised and run with
        this page's window in the foreground — not merely the click. Measured:
        with three concurrent workers and no serialisation, the two background
        pages never rendered a widget at all ("no layout box", zero flow POSTs
        seen in 20s), because Firefox does not lay out or run occluded windows.
        Every page here is its own window (3 pages, 3 X windows), so only the
        foreground one has a live challenge to solve.

        Serialising is cheap: the pages share one cookie jar, so the first
        worker's ``cf_clearance`` covers the rest — a queued worker usually just
        reloads into clear content instead of solving anything.
        """
        display, _ = self._resolve_display(page)
        if self._os_input_unavailable_reason(display) is not None:
            # No display: nothing to foreground or contend over, and only the
            # synthetic path is available.
            await self._solve_challenge(page, None)
            return

        lock = self._click_lock()
        queued = lock.locked()
        async with lock:
            try:
                await page.bring_to_front()
                await asyncio.sleep(self._RAISE_SETTLE_S)
            except Exception:  # a raise failure is not fatal
                logger.debug("bring_to_front failed", exc_info=True)

            # Get a fresh, foreground challenge before solving if either:
            #  * we queued behind a sibling (our widget went stale waiting, and
            #    their clearance may already cover us — shared cookie jar), or
            #  * nothing is rendered at all, which is what a page that loaded
            #    while occluded looks like. Firefox does not lay out background
            #    windows, so its challenge never ran: no flow POSTs, no widget,
            #    no layout box. Foregrounding it now does not revive it; only a
            #    reload does.
            if not queued:
                await self._solve_challenge(page, display)
                return
            # Armed before the reload, because the reload is what fires the
            # fresh challenge's flow POSTs: a listener attached afterwards
            # misses them and sits out _READY_TIMEOUT_MS holding the lock.
            with self._flow_watch(page) as flow:
                if not await self._refresh_stale_challenge(page):
                    return  # a sibling's clearance already covers us
                await self._solve_challenge(page, display, flow)

    async def _widget_has_layout(self, page: Page) -> bool:
        """Whether a widget is actually rendered with a non-zero box.

        Distinguishes "challenge present and live" from "page never rendered" —
        the latter reports a challenge shell in the DOM but no laid-out widget,
        because an occluded window is not laid out at all.
        """
        try:
            if not await self._widget_container(page).count():
                return False
            return await self._widget_box(page, timeout_ms=2_000) is not None
        except Exception:  # a probe must not break the solve
            logger.debug("Widget layout probe failed", exc_info=True)
            return False

    def _widget_container(self, page: Page) -> Locator:
        """Turnstile's mount container: in the light DOM, unlike its iframe."""
        return page.locator(self._RESPONSE_INPUT).first.locator("xpath=..")

    async def _widget_box(
        self, page: Page, *, timeout_ms: int = 5_000
    ) -> FloatRect | None:
        """The mount container's layout box, or None if it has no area.

        None covers a container that never attached in ``timeout_ms`` and
        one that is present but not laid out (an occluded window).
        """
        try:
            box = await self._widget_container(page).bounding_box(
                timeout=timeout_ms
            )
        except PlaywrightTimeoutError:
            return None
        return box if box and box["width"] and box["height"] else None

    async def _solve_challenge(
        self,
        page: Page,
        display: str | None,
        flow: _FlowWatch | None = None,
    ) -> None:
        """The solve itself.

        With a ``display``, the caller holds the click lock and has
        foregrounded ``page``; with ``None`` it has done neither. ``flow`` is
        a readiness watch the caller armed earlier (see
        :meth:`_await_flow_readiness`).
        """
        await self._await_flow_readiness(page, flow)

        # Readiness only means the orchestrator finished its setup POSTs; the
        # widget iframe can still be loading. Acting here is what made the old
        # Tab+Space land on a spinner.
        target = await self._await_interactive(page)

        # Nothing rendered even after the interactive wait: the page loaded while
        # its window was occluded, so its challenge never ran at all (Firefox
        # does not lay out background windows — measured as zero flow POSTs and
        # no layout box). Foregrounding does not revive it; only a reload does,
        # and this handler is now foregrounded, so the reload gets a live one.
        if target is None and not await self._widget_has_layout(page):
            logger.warning(
                "Challenge never rendered (page was occluded while loading); "
                "reloading in the foreground and retrying"
            )
            if not await self._refresh_stale_challenge(page):
                return
            target = await self._await_interactive(page)

        if target == "button":
            logger.info(
                "Cloudflare click-gated variant — clicking verify button"
            )
            await page.locator(self._VERIFY_BUTTON).first.click(timeout=5_000)
            # Turnstile mounts only after that click on this variant; a miss
            # is not fatal, Tab+Space is still worth attempting.
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.locator(self._RESPONSE_INPUT).first.wait_for(
                    state="attached", timeout=self._INTERACTIVE_TIMEOUT_MS
                )

        # OS-level input goes FIRST where the environment can post it, because
        # it is the only step measured to clear a challenge since the
        # 2026-08-09 regression. Ordering is not cosmetic: the first end-to-end
        # container run had it running third, and it cleared nothing — by then
        # the widget had spent ~40s being pressed with Space and clicked
        # synthetically on the same pixel. Standalone, firing at ~15s on an
        # untouched widget, the same click is 6/6. So spend the real click while
        # the widget is fresh and keep the synthetic steps as the fallback for
        # hosts with no display.
        if display is not None:
            if await self._os_click_attempts(page, display):
                logger.info("Cloudflare challenge cleared via OS-level click")
                return
            logger.warning(
                "OS-level click did not clear CF; falling back to synthetic "
                "input (which has not worked since 2026-08-09)"
            )
        else:
            logger.info("No OS-level input available; using synthetic input")

        # Tab focuses the Turnstile widget; Space activates the checkbox
        # inside its closed shadow root, which no selector can address.
        focus = await self._focus_widget_via_tab(page)
        if focus == "elsewhere":
            # Space on a footer link or on <body> would only scroll the page,
            # so skip it and save the _CLEAR_TIMEOUT_MS wait on a press that
            # cannot work. The log line also keeps "focus never reached the
            # widget" distinguishable from "focus was right and Cloudflare
            # rejected the key" — the two have entirely different fixes.
            logger.warning(
                "Tab did not land on the Turnstile widget in %d attempt(s); "
                "skipping Space and going straight to the widget click",
                self._MAX_TAB_ATTEMPTS,
            )
        else:
            if focus == "unverifiable":
                logger.info(
                    "No %s to verify focus against; pressing Space blind",
                    self._RESPONSE_INPUT,
                )
            await self._press_space_humanlike(page)

            if await self._challenge_cleared(page):
                logger.info("Cloudflare challenge cleared via Tab+Space")
                return

            logger.warning(
                "Tab+Space did not clear CF in %ds; trying a widget click",
                self._CLEAR_TIMEOUT_MS // 1000,
            )
        if await self._click_widget_checkbox(page) and (
            await self._challenge_cleared(
                page, timeout_ms=self._FALLBACK_CLEAR_TIMEOUT_MS
            )
        ):
            logger.info("Cloudflare challenge cleared via widget click")
            return

        # Nothing worked. Raising is the point: the caller stores the failure
        # and retries, instead of snapshotting the challenge DOM as content.
        raise InterstitialUnresolved(
            "Cloudflare challenge still present after OS click, Tab+Space and "
            f"widget click ({self._CHALLENGE_SHELL} still attached)"
        )

    async def _focus_widget_via_tab(self, page: Page) -> str:
        """Tab until focus lands on the Turnstile mount.

        Returns ``"widget"`` once focus is verifiably on the mount,
        ``"unverifiable"`` if there is no ``_RESPONSE_INPUT`` to check against
        (the caller presses Space anyway — see :attr:`_FOCUS_PROBE_JS`), or
        ``"elsewhere"`` if focus never got there.

        The check is worth its cost because the thing it verifies is
        unobservable from selectors: the checkbox lives in a closed shadow
        root, so ``focus`` on it is only visible as the *mount* becoming
        ``document.activeElement``. Without this, a Tab that missed and a Tab
        that landed produce the same log line and the same 30s timeout, and
        the handler's one interesting distinction — did we fail to aim, or did
        Cloudflare reject a well-aimed key — is lost.
        """
        for attempt in range(1, self._MAX_TAB_ATTEMPTS + 1):
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.15)
            state = await self._focus_state(page)
            if state == "widget":
                if attempt > 1:
                    logger.info(
                        "Turnstile widget focused after %d Tabs", attempt
                    )
                return state
            # Nothing to check against, so more Tabs would only walk focus
            # further from a widget we cannot see; stop and let Space fly.
            if state == "unverifiable":
                return state
        return "elsewhere"

    async def _focus_state(self, page: Page) -> str:
        """Where focus currently is, relative to the Turnstile mount.

        A probe that throws (page navigating out from under us, evaluation
        disabled) reports ``"unverifiable"`` rather than propagating: this is
        a diagnostic aid, and it must never be the reason a solvable challenge
        goes unattempted.
        """
        try:
            return await page.evaluate(
                self._FOCUS_PROBE_JS, self._FOCUS_ANCESTOR_DEPTH
            )
        except Exception:  # see docstring
            logger.debug("Turnstile focus probe failed", exc_info=True)
            return "unverifiable"

    async def _press_space_humanlike(self, page: Page) -> None:
        """Press Space via low-level ``down``/dwell/``up`` instead of ``press``.

        ``page.keyboard.press("Space")`` releases the key ~1-2ms after pressing
        it; a genuine key hold measures ~45-130ms between keydown and keyup.
        Splitting the press into ``keyboard.down`` / sleep / ``keyboard.up``
        and holding for a value drawn from that band reproduces the human
        dwell, and the randomised hold avoids the fixed-interval signature a
        constant ``press(delay=...)`` would leave.

        This shapes *timing only*. The synthesized key still carries the same
        ``isTrusted`` / input-source level as any Juggler event, so it does not
        touch the input-trust gate documented above and is not expected to
        overturn the 2026-08-09 regression on its own — it is the cheap,
        human-shaped retry that docstring says to try before anything heavier.
        """
        await page.keyboard.down("Space")
        await asyncio.sleep(random.uniform(0.045, 0.130))
        await page.keyboard.up("Space")

    @contextlib.contextmanager
    def _flow_watch(self, page: Page) -> Iterator[_FlowWatch]:
        """Count ``page``'s flow-POST 200s for as long as the block runs."""
        watch = _FlowWatch(self._FLOW_RE)
        listener = watch.on_response
        page.on("response", listener)
        try:
            yield watch
        finally:
            page.remove_listener("response", listener)

    async def _await_flow_readiness(
        self, page: Page, flow: _FlowWatch | None = None
    ) -> None:
        """Wait for the orchestrator's second flow POST to return 200.

        ``flow`` is a watch armed before whatever fires those POSTs (a
        reload); without one, a watch is armed here, which only sees POSTs
        that have not happened yet.
        """
        if flow is None:
            with self._flow_watch(page) as flow:
                await self._await_flow_readiness(page, flow)
            return
        try:
            await asyncio.wait_for(
                flow.ready.wait(), timeout=self._READY_TIMEOUT_MS / 1000
            )
            logger.info("Cloudflare orchestrator ready (2 flow-POST 200s)")
        except asyncio.TimeoutError:
            logger.warning(
                "Cloudflare flow-POST readiness signal never fired in %ds "
                "(saw %d matching response(s)); continuing anyway",
                self._READY_TIMEOUT_MS // 1000,
                flow.count,
            )

    async def _await_interactive(self, page: Page) -> str | None:
        """Wait for whichever interactive shape this challenge rendered.

        Returns ``"button"`` for the click-gated variant, ``"widget"`` once
        the Turnstile response input attaches, or ``None`` if neither shows
        up — in which case Tab+Space is still attempted, since a widget can
        be present and functional without either marker resolving.
        """

        async def _button() -> str:
            await page.locator(self._VERIFY_BUTTON).first.wait_for(
                state="visible", timeout=self._INTERACTIVE_TIMEOUT_MS
            )
            return "button"

        async def _widget() -> str:
            await page.locator(self._RESPONSE_INPUT).first.wait_for(
                state="attached", timeout=self._INTERACTIVE_TIMEOUT_MS
            )
            return "widget"

        shape = await _first_success(_button(), _widget())
        if shape is None:
            logger.warning(
                "Cloudflare challenge rendered neither a verify button nor a "
                "Turnstile response input in %ds; attempting Tab+Space anyway",
                self._INTERACTIVE_TIMEOUT_MS // 1000,
            )
        return shape

    async def _challenge_cleared(
        self, page: Page, timeout_ms: int | None = None
    ) -> bool:
        """Whether the challenge shell is gone (i.e. real content loaded).

        Keyed on the shell rather than on ``_RESPONSE_INPUT`` detaching: an
        input that never attached is already ``detached``, so the old check
        reported success against challenges that were still up.
        """
        try:
            await page.locator(self._CHALLENGE_SHELL).first.wait_for(
                state="detached",
                timeout=timeout_ms or self._CLEAR_TIMEOUT_MS,
            )
            return True
        except PlaywrightTimeoutError:
            return False

    async def _click_widget_checkbox(self, page: Page) -> bool:
        """Pointer-click where Turnstile draws its checkbox.

        Replaces the old ``page.frames`` body-click, which could not run: the
        widget's iframe sits inside a closed shadow root, so it is absent from
        ``document.querySelectorAll('iframe')``, and Playwright reports its
        frame URL as the empty string — so filtering frames on
        ``challenges.cloudflare.com`` matched nothing and the fallback raised
        every time. The mount container *is* in the light DOM, so click that
        instead.

        Returns whether a click was actually dispatched.
        """
        box = await self._widget_box(page)
        if box is None:
            logger.warning(
                "Cloudflare widget container has no layout box; "
                "cannot place a click"
            )
            return False
        await page.mouse.click(
            box["x"] + self._CHECKBOX_INSET_PX,
            box["y"] + box["height"] / 2,
        )
        return True

    @staticmethod
    def _resolve_display(page: Page) -> tuple[str | None, bool]:
        """``(display, is_private)`` for this page's browser.

        A *private* display was allocated for this browser alone
        (:mod:`jkent.driver.xvfb`); the browser's own pages still share it, so
        clicks are serialised by :meth:`_click_lock` either way.

        Falling back to ``$DISPLAY`` is shared by definition — every browser in
        this process maps windows onto it, all at 0,0 with no window manager to
        separate them — so even under the lock the intended window may not be
        on top.
        """
        private = display_for(getattr(page, "context", None))
        if private:
            return private, True
        return os.environ.get("DISPLAY"), False

    @staticmethod
    def _os_input_unavailable_reason(display: str | None) -> str | None:
        """Why an OS click cannot be posted here, or ``None`` if it can.

        Neither reason is an error: no display (headless, or a host without
        Xvfb) and no ``xdotool`` (a macOS dev machine) are both ordinary.
        """
        if not display:
            return "no X display (needs Xvfb; headless runs get none)"
        if not shutil.which("xdotool"):
            return "xdotool not on $PATH"
        return None

    async def _checkbox_screen_point(
        self, page: Page
    ) -> tuple[int, int] | None:
        """Screen coordinates of Turnstile's checkbox, or ``None``.

        Uses the light-DOM mount box — the same source
        :meth:`_click_widget_checkbox` clicks — rather than locating the widget
        visually in a screenshot. The visual route (find the widget's uniform
        grey fill, take its bbox, offset in) was what the investigation used and
        it works, but it costs a screenshot plus numpy/PIL, and the mount box is
        already right here: the container's left edge and the widget's are 8px
        apart, so ``_CHECKBOX_INSET_PX`` lands inside a checkbox that measures
        ~21px across. Keeping this in the DOM also keeps numpy out of the
        driver's dependency set.
        """
        box = await self._widget_box(page)
        if box is None:
            logger.warning(
                "Cloudflare widget container has no layout box; cannot place "
                "an OS click"
            )
            return None

        try:
            origin = await page.evaluate(self._SCREEN_ORIGIN_JS)
        except Exception:  # a dead page is not our error to raise
            logger.debug("Screen-origin probe failed", exc_info=True)
            return None

        dpr = origin.get("dpr") or 1
        if dpr != 1:
            # Rather than guess between (css + origin) and (css + origin) * dpr
            # on an untested display, decline. Xvfb is dpr=1, so this only fires
            # somewhere this path was never measured.
            logger.warning(
                "Skipping OS click: devicePixelRatio is %s, and the "
                "viewport->screen mapping is only verified at 1",
                dpr,
            )
            return None

        css_x = box["x"] + self._CHECKBOX_INSET_PX
        css_y = box["y"] + box["height"] / 2
        return int(origin["sx"] + css_x), int(origin["sy"] + css_y)

    async def _run_xdotool(self, display: str, *args: str) -> bool:
        """Run ``xdotool`` against ``display``, returning whether it succeeded.

        The display is passed through the child's environment rather than an
        ``xdotool --display`` flag: ``env=`` is verified to work per-process
        (two browsers, two displays, clicks landing 1:1) and does not depend on
        which xdotool subcommands accept the flag.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "xdotool",
                *args,
                env=xvfb_env_for(display),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            logger.warning("Could not execute xdotool", exc_info=True)
            return False
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._OS_CLICK_SUBPROCESS_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            logger.warning("xdotool %s timed out", " ".join(args))
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return False
        if proc.returncode != 0:
            logger.warning(
                "xdotool %s failed (rc=%s): %s",
                " ".join(args),
                proc.returncode,
                (stderr or b"").decode("utf-8", "replace").strip(),
            )
            return False
        return True

    async def _refresh_stale_challenge(self, page: Page) -> bool:
        """Recover a challenge that went stale while queued. Returns whether one
        still needs clicking.

        Every page in this browser shares one cookie jar, so a sibling's
        ``cf_clearance`` applies to us too — reloading is enough, and is what
        should be tried before spending a click. If a challenge is still served
        after the reload it is at least a *fresh* one, which is the state where
        an OS click actually works.
        """
        logger.info(
            "Reloading to get a live foreground challenge (queued behind a "
            "sibling, or the page never rendered while occluded)"
        )
        try:
            await page.reload(wait_until="domcontentloaded", timeout=30_000)
        except PlaywrightTimeoutError:
            logger.warning("Reload timed out; clicking the widget as-is")
            return True

        if await self._challenge_cleared(
            page, timeout_ms=self._FALLBACK_CLEAR_TIMEOUT_MS
        ):
            logger.info(
                "Cloudflare challenge already cleared by a sibling worker's "
                "solve (shared cf_clearance)"
            )
            return False

        # Fresh challenge: let the widget mount before aiming at it.
        await self._await_interactive(page)
        return True

    async def _os_click_attempts(self, page: Page, display: str) -> bool:
        """The click loop itself; callers hold :meth:`_click_lock`.

        Measured 4/4 on the CA deployment, and every one of those four cleared
        on the **second** click, ~1s after it, having sat unmoved through the
        first. That shape is consistent enough to be a mechanism rather than
        noise — most likely X11 click-to-focus, where the first click activates
        the browser window and the second is the one the widget actually sees.
        It is why :attr:`_OS_CLICK_ATTEMPTS` is 2 and not 1; a single-click
        version of this method would have a 0/4 record.
        """
        for attempt in range(1, self._OS_CLICK_ATTEMPTS + 1):
            point = await self._checkbox_screen_point(page)
            if point is None:
                return False
            x, y = point

            # Approach hop first, then the target. Two reasons, both load-bearing:
            #
            # 1. ``--sync`` waits for a pointer-motion event, so moving to where
            #    the cursor already *is* blocks forever. Retry attempt 2 targets
            #    the same pixel as attempt 1, which hung until the subprocess
            #    timeout in the first end-to-end run. Stepping away guarantees
            #    the second hop actually moves.
            # 2. Turnstile samples pointer positions, so arriving from somewhere
            #    beats materialising on the checkbox.
            #
            # --sync matters on the final hop: a click posted before the cursor
            # arrives is a click somewhere else.
            if not await self._run_xdotool(
                display, "mousemove", str(x - 40), str(y + 25)
            ):
                return False
            await asyncio.sleep(random.uniform(0.05, 0.09))
            if not await self._run_xdotool(
                display, "mousemove", "--sync", str(x), str(y)
            ):
                return False
            await asyncio.sleep(random.uniform(0.08, 0.15))
            if not await self._run_xdotool(display, "click", "1"):
                return False
            logger.info(
                "Posted OS-level click %d/%d at screen (%d, %d)",
                attempt,
                self._OS_CLICK_ATTEMPTS,
                x,
                y,
            )

            if await self._challenge_cleared(
                page, timeout_ms=self._FALLBACK_CLEAR_TIMEOUT_MS
            ):
                return True
            logger.warning(
                "Challenge still up %ds after OS click %d/%d",
                self._FALLBACK_CLEAR_TIMEOUT_MS // 1000,
                attempt,
                self._OS_CLICK_ATTEMPTS,
            )
        return False


#: Environment override for the steno server :class:`ReCaptchaHandler`
#: transcribes audio challenges against. Previously hardcoded at the single
#: construction site, which made it unreachable from a deployment.
STENO_URL_ENV = "JKENT_STENO_URL"

#: Factory per interstitial-contributing :class:`DriverRequirement`.
#:
#: Factories, not instances: a handler carries per-browser state (the
#: Cloudflare OS-click lock), so each transport builds its own set rather
#: than sharing module-level singletons across every run in the process.
#: Rows are registered at import with :func:`interstitial_for`, which accepts
#: only requirements ``REQUIREMENTS`` marks ``interstitial=True``.
INTERSTITIAL_FACTORIES: dict[
    DriverRequirement, Callable[[], InterstitialHandler]
] = {}


def interstitial_for(
    requirement: DriverRequirement,
) -> Callable[[_FactoryT], _FactoryT]:
    """Register the factory that builds ``requirement``'s handler.

    Raises:
        ValueError: ``requirement`` is not marked ``interstitial=True`` in
            :data:`~jkent.driver.unified_driver.requirements.REQUIREMENTS`
            (its handler would never be wired, and — worse — the requirement
            would not force a browser transport), or it is already
            registered.
    """

    def register(factory: _FactoryT) -> _FactoryT:
        if requirement not in INTERSTITIAL_REQUIREMENTS:
            raise ValueError(
                f"{requirement.name} is not declared interstitial=True in "
                "requirements.REQUIREMENTS; add the row there first"
            )
        if requirement in INTERSTITIAL_FACTORIES:
            raise ValueError(
                f"{requirement.name} already has an interstitial factory: "
                f"{INTERSTITIAL_FACTORIES[requirement]!r}"
            )
        INTERSTITIAL_FACTORIES[requirement] = factory
        return factory

    return register


interstitial_for(DriverRequirement.CFCAP_HANDLER)(CloudflareHandler)


@interstitial_for(DriverRequirement.RCAP_HANDLER)
def _build_recaptcha_handler() -> ReCaptchaHandler:
    """reCAPTCHA handler against the configured steno server."""
    url = os.environ.get(STENO_URL_ENV)
    transcriber = (
        LocalStenoTranscriber(url) if url else LocalStenoTranscriber()
    )
    return ReCaptchaHandler(transcriber)


def _verify_factories() -> None:
    """Every interstitial requirement has a factory, and vice versa.

    Import-time: the policy table and this module are the two halves of one
    decision, and a half-declared handler is the silent failure this pairing
    exists to prevent.
    """
    missing = INTERSTITIAL_REQUIREMENTS - set(INTERSTITIAL_FACTORIES)
    if missing:
        names = ", ".join(sorted(r.name for r in missing))
        raise RuntimeError(
            f"Requirements declared interstitial=True with no factory: {names}"
        )


_verify_factories()


def handlers_for(
    requirements: Iterable[DriverRequirement],
) -> list[InterstitialHandler]:
    """Build a fresh handler per interstitial requirement, in order."""
    return [
        INTERSTITIAL_FACTORIES[req]()
        for req in requirements
        if req in INTERSTITIAL_FACTORIES
    ]
