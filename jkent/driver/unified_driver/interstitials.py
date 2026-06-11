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
import logging
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

    Strategy proven in ``try_camoufox.py`` against the CA appellate
    courts CF deployment:

    1. Waitlist matches when ``input[name='cf-turnstile-response']``
       attaches — that input ships in the initial HTML so the handler
       fires immediately on navigation.
    2. ``navigate_through`` subscribes to ``page.on("response")`` and
       counts hits on ``/cdn-cgi/challenge-platform/h/b/flow/ov1/``.
       The orchestrator's second flow-POST returning 200 is the
       readiness signal — at that moment the loading spinner inside
       the Turnstile iframe is swapped for the interactive checkbox.
    3. Tab + Space at the page level focuses the (in-closed-shadow-root)
       checkbox and activates it; CF accepts and the page navigates
       to the real content.
    4. If the flow signal never fires (CF rendered something else, or
       the deployment changed), fall back to clicking the iframe
       body's center — the older approach we've already verified
       works against Turnstile's default layout.

    Why no selector-targeted click: the Turnstile widget's contents
    live in a closed shadow root attached to the iframe's body, so
    ``input[type='checkbox']`` etc. never resolve via ``frame.locator``.
    Focus traversal *does* cross closed shadow roots, which is why
    Tab+Space works where selectors don't.
    """

    _RESPONSE_INPUT = "input[name='cf-turnstile-response']"
    _CF_FRAME_HOST = "challenges.cloudflare.com"
    _FLOW_PATH = "/cdn-cgi/challenge-platform/h/b/flow/ov1/"
    # Empirical: the second /flow/ POST returns ~2s after navigation
    # in camoufox.  Pad heavily in case the orchestrator is slow.
    _READY_TIMEOUT_MS = 20_000
    # After we press Space, the page navigates to a token URL then back
    # to the original; clear time observed at ~1s, allow 30s headroom.
    _CLEAR_TIMEOUT_MS = 30_000

    def waitlist(self) -> list[WaitCondition]:
        return [WaitForSelector(self._RESPONSE_INPUT, state="attached")]

    async def navigate_through(self, page: Page) -> None:
        # Count /flow/ov1/ responses.  The second one returning 200 is
        # the orchestrator-finished-setup signal.
        ready = asyncio.Event()
        flow_count = 0

        def _on_response(resp):
            nonlocal flow_count
            # Only the orchestrator's flow POSTs returning 200 signal
            # readiness; GET preflights and 3xx/4xx/5xx responses to the
            # same path don't count.
            if (
                self._FLOW_PATH in resp.url
                and resp.request.method == "POST"
                and resp.status == 200
            ):
                flow_count += 1
                if flow_count >= 2:
                    ready.set()

        page.on("response", _on_response)
        try:
            try:
                await asyncio.wait_for(
                    ready.wait(), timeout=self._READY_TIMEOUT_MS / 1000
                )
                logger.info(
                    "Cloudflare widget ready (2 /flow/ responses) — "
                    "pressing Tab+Space"
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Cloudflare /flow/ readiness signal never fired in %ds; "
                    "attempting Tab+Space anyway",
                    self._READY_TIMEOUT_MS // 1000,
                )
        finally:
            page.remove_listener("response", _on_response)

        # Tab focuses the Turnstile checkbox (the only tabbable element
        # the orchestrator has rendered into the iframe's closed shadow
        # root); Space activates it.
        await page.keyboard.press("Tab")
        await asyncio.sleep(0.15)
        await page.keyboard.press("Space")

        # Wait for CF to accept and the page to navigate to real content
        # (cf-turnstile-response input detaches once the new document
        # finishes loading).
        try:
            await page.locator(self._RESPONSE_INPUT).wait_for(
                state="detached", timeout=self._CLEAR_TIMEOUT_MS
            )
            logger.info("Cloudflare challenge cleared via Tab+Space")
            return
        except PlaywrightTimeoutError:
            logger.warning(
                "Tab+Space did not clear CF in %ds; "
                "falling back to body-click",
                self._CLEAR_TIMEOUT_MS // 1000,
            )

        # Fallback: click the iframe body's center.  Playwright's
        # default click target is the element's center, which (with
        # Turnstile's default layout) coincides with the checkbox.
        cf_frames = [fr for fr in page.frames if self._CF_FRAME_HOST in fr.url]
        if not cf_frames:
            raise PlaywrightTimeoutError(
                "Cloudflare challenge stuck: no challenges.cloudflare.com "
                "frame in page.frames after Tab+Space failed"
            )
        # Prefer the visible widget frame (URL contains "turnstile")
        # over orchestrator worker frames.
        cf_frames.sort(key=lambda fr: 0 if "turnstile" in fr.url else 1)
        await cf_frames[0].locator("body").click(timeout=5_000)
        await page.locator(self._RESPONSE_INPUT).wait_for(
            state="detached", timeout=self._CLEAR_TIMEOUT_MS
        )
        logger.info("Cloudflare challenge cleared via body-click fallback")


INTERSTITIAL_HANDLERS: dict[DriverRequirement, InterstitialHandler] = {
    DriverRequirement.HCAP_HANDLER: HCaptchaHandler(),
    DriverRequirement.RCAP_HANDLER: ReCaptchaHandler(LocalStenoTranscriber()),
    DriverRequirement.CFCAP_HANDLER: CloudflareHandler(),
}
