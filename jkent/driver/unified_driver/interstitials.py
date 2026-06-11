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
import random
import re
from typing import TYPE_CHECKING

import httpx
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from jkent.data_types import (
    DriverRequirement,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)

if TYPE_CHECKING:
    from playwright.async_api import FrameLocator, Page, Response

logger = logging.getLogger(__name__)

WaitCondition = (
    WaitForSelector | WaitForLoadState | WaitForURL | WaitForTimeout
)


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


class HCaptchaHandler(InterstitialHandler):
    """Handles hCaptcha interstitial pages.

    Clicks the ``div.h-captcha`` element, which triggers the hCaptcha
    widget.  In headless Firefox with ``navigator.webdriver`` overridden,
    this auto-solves; the JS callback then submits the form, navigating
    to the real content page.
    """

    def waitlist(self) -> list[WaitCondition]:
        return [WaitForSelector("div.h-captcha")]

    async def navigate_through(self, page: Page) -> None:
        logger.info("hCaptcha interstitial detected — clicking to solve")
        captcha = page.locator("div.h-captcha")
        await captcha.click()
        await page.wait_for_load_state("networkidle")


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

        checkmark_task = asyncio.create_task(_wait_for_checkmark())
        bframe_task = asyncio.create_task(_wait_for_bframe())
        tasks = {checkmark_task, bframe_task}
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # Prefer the checkmark outcome: if auto-solve won the race we're
        # done.  A task that was cancelled (lost the race) or finished by
        # raising (e.g. both timed out) falls through to the audio-challenge
        # path rather than crashing.
        if (
            not checkmark_task.cancelled()
            and checkmark_task.exception() is None
        ):
            logger.info("reCAPTCHA auto-solved (no challenge)")
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
        logger.info("reCAPTCHA solved successfully")

    @staticmethod
    async def _intercept_audio(
        page: Page,
        bframe: FrameLocator,
    ) -> Response:
        """Click PLAY and intercept the audio payload response.

        Uses ``page.expect_response`` to capture the audio file as
        the browser fetches it, avoiding a separate HTTP request.
        """
        async with page.expect_response(  # type: ignore
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
    ``<script>`` Cloudflare ships inside the challenge document itself — not
    on the Turnstile widget the orchestrator later injects.  Measured against
    live challenges on the MA and MO deployments, both ``cvId:'3'``
    ``cTplV:5`` ``cType:'managed'``, the same template generation the CA
    appellate site serves:

    =======================================  ===========  ======
    first match                              MA           MO
    =======================================  ===========  ======
    ``_CHALLENGE_SHELL``                     0.10-0.18s   0.13s
    ``input[name='cf-turnstile-response']``  0.56-1.33s   0.65s
    =======================================  ===========  ======

    The widget input is the wrong signal three ways: it lands up to a second
    late, it is absent from the wire HTML entirely (the 403 body is ~6KB of
    bare shell), and on the click-gated variant Cloudflare never renders it
    at all.  A handler waiting on it simply never fires, and the challenge
    page gets snapshotted as though it were content.

    ``navigate_through`` then:

    1. Waits for the orchestrator's flow-POST readiness signal
       (:attr:`_FLOW_RE`, second POST returning 200).
    2. Waits for something interactive to actually exist.  Readiness is not
       the same as interactive — the second flow POST lands while the widget
       iframe is still loading.
    3. Clicks "Verify you are human" on the click-gated variant, where
       Turnstile only mounts once that button is activated.
    4. Tabs onto the Turnstile mount, *verifying* focus landed there
       (:meth:`_focus_widget_via_tab`), then presses Space as a low-level
       keydown/dwell/keyup — a human-plausible ~45-130ms hold rather than
       ``press()``'s ~1-2ms release.  Focus traversal crosses the widget's
       closed shadow root, where selectors cannot reach, so the guard keys on
       the light-DOM mount that hosts it rather than on the checkbox itself.
    5. Confirms the *challenge is gone*, rather than that the widget input
       detached — an input that never attached satisfies ``detached``
       immediately, which reported success on challenges still standing.

    **As of 2026-08-09 ~13:00 UTC, Turnstile's checkbox stopped accepting
    synthetic input, so step 4 no longer succeeds. This is a Cloudflare-side
    regression, not a property of the design — treat it as possibly
    temporary.** The evidence, in case it reverts or someone is tempted to
    rewrite this around it:

    * Step 4 worked 47 times between 2026-06-30 and 2026-08-09, several times
      a day for weeks. Each successful solve leaves a ``cf_clearance`` cookie
      in ``run_metadata.browser_cookies_json``, so the run databases carry the
      history. The last one issued was 2026-08-09 12:47:41 UTC.
    * The run that started 55 minutes later (13:42 UTC) got zero solves in
      twelve hours and logged seven "Cloudflare challenge stuck" failures,
      roughly hourly, before dying on a browser crash.
    * Nothing in this repo explains the cutover: the run that solved at
      12:47 was already running the current code.
    * Controlled A/B afterwards, within a single challenge on
      appellatecases.courtinfo.ca.gov (Ray ``a28bab3e0e95a9c6``): a synthetic
      ``page.keyboard.press("Space")`` on the focused checkbox did nothing
      over 8s; a human pressing Space on that same focused checkbox, in that
      same session, cleared it instantly to the real document. Four synthetic
      attempts across CA, MA and MO — headless and headed, camoufox and
      vanilla — failed identically, so the gate is on input trust, not on a
      per-zone or per-engine quirk.
    * Re-measured 2026-08-13 on the same CA deployment (Rays
      ``a2ab8f640e981855`` and successors), this time instrumenting focus
      rather than assuming it: after one Tab, ``document.activeElement`` is
      the widget mount (the ``<div>`` that parents
      ``input[name='cf-turnstile-response']``, 896x70 box) and Firefox draws
      its focus ring around the checkbox. Space on that verified-focused
      checkbox still cleared nothing — the shell was still attached 15s
      later. So mis-aimed focus is *not* what breaks step 4; the gate is
      where the A/B put it. :meth:`_focus_widget_via_tab` exists to keep that
      distinction observable, not because focus was ever the fault.

    ``page.mouse.click`` is synthetic too and cleared nothing on any of the
    three, which is why :meth:`_click_widget_checkbox` is a cheap second
    attempt rather than a fix.

    The synthetic-input ceiling covers the mouse too: ``page.mouse.click`` at
    the checkbox's own coordinates cleared nothing on CA, MA or MO, so
    :meth:`_click_widget_checkbox` is a cheap second attempt, not a fix.

    Step 3's button is the one interaction *not* measured — the click-gated
    variant has not recurred live since it was captured on 2026-08-09, so
    whether Cloudflare accepts a synthetic click there is untested. Note it
    would not be a way around this anyway: activating that button mounts
    Turnstile, which lands back on the trusted-input-gated checkbox. The
    branch exists so that variant reaches the same place as the others rather
    than stalling on a widget that never mounts.

    Because it is a regression rather than a wall, re-test step 4 before
    reaching for anything heavier: a ``cf_clearance`` appearing in a fresh
    run's metadata means synthetic input works again. If it stays broken, the
    options are a fingerprint/proxy good enough that Cloudflare stops issuing
    the challenge, OS-level input injection against a real display, or an
    external solver — none of which belong in a selector. Meanwhile the
    handler's job is to detect every shape and *fail loudly* rather than hand
    back challenge HTML that the classifier would store as case data.
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
        await self._await_flow_readiness(page)

        # Readiness only means the orchestrator finished its setup POSTs; the
        # widget iframe can still be loading. Acting here is what made the old
        # Tab+Space land on a spinner.
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
        raise PlaywrightTimeoutError(
            "Cloudflare challenge still present after Tab+Space and widget "
            f"click ({self._CHALLENGE_SHELL} still attached)"
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
        except Exception:  # noqa: BLE001 — see docstring
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

    async def _await_flow_readiness(self, page: Page) -> None:
        """Wait for the orchestrator's second flow POST to return 200."""
        ready = asyncio.Event()
        flow_count = 0

        def _on_response(resp):
            nonlocal flow_count
            # Only the orchestrator's flow POSTs returning 200 signal
            # readiness; GET preflights and 3xx/4xx/5xx responses to the
            # same path don't count.
            if (
                self._FLOW_RE.search(resp.url)
                and resp.request.method == "POST"
                and resp.status == 200
            ):
                flow_count += 1
                if flow_count >= 2:
                    ready.set()

        page.on("response", _on_response)
        try:
            await asyncio.wait_for(
                ready.wait(), timeout=self._READY_TIMEOUT_MS / 1000
            )
            logger.info("Cloudflare orchestrator ready (2 flow-POST 200s)")
        except asyncio.TimeoutError:
            logger.warning(
                "Cloudflare flow-POST readiness signal never fired in %ds "
                "(saw %d matching response(s)); continuing anyway",
                self._READY_TIMEOUT_MS // 1000,
                flow_count,
            )
        finally:
            page.remove_listener("response", _on_response)

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

        tasks = {
            asyncio.create_task(_button(), name="cf-button"),
            asyncio.create_task(_widget(), name="cf-widget"),
        }
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.exception() is None:
                        return task.result()
            logger.warning(
                "Cloudflare challenge rendered neither a verify button nor a "
                "Turnstile response input in %ds; attempting Tab+Space anyway",
                self._INTERACTIVE_TIMEOUT_MS // 1000,
            )
            return None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
        container = page.locator(self._RESPONSE_INPUT).first.locator(
            "xpath=.."
        )
        try:
            box = await container.bounding_box(timeout=5_000)
        except PlaywrightTimeoutError:
            box = None
        if not box or not box["width"] or not box["height"]:
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


INTERSTITIAL_HANDLERS: dict[DriverRequirement, InterstitialHandler] = {
    DriverRequirement.HCAP_HANDLER: HCaptchaHandler(),
    DriverRequirement.RCAP_HANDLER: ReCaptchaHandler(LocalStenoTranscriber()),
    DriverRequirement.CFCAP_HANDLER: CloudflareHandler(),
}
