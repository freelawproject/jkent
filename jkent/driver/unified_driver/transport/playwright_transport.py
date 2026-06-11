"""Playwright transport — engine + per-worker page lifecycle (B1).

Owns browser launch/teardown and per-worker page acquisition. The engine and
browser context are built in ``open`` (wrapping ``engines/``) and torn down in
``aclose``; each worker gets a long-lived :class:`WorkerPage` from ``acquire``,
stable until ``release``.

``resolve`` (B2) handles the navigation path. Crash recovery (B3) implements
the transport-internal :class:`~jkent.driver.unified_driver.lifecycle.Recoverable`
surface (``generation`` / ``should_restart`` / ``restart``): a dead connection
noticed in ``resolve`` poisons the handle and re-maps to ``TransientException``,
and the next ``acquire`` rebuilds the handle, escalating to a single-flight
engine restart when the connection itself is dead. ``resolve_archive`` (B4)
triggers the download via the request's ``via`` (link click / form submit),
stages the file Playwright hands back, and streams it; ``finish_archiving``
deletes the staged file.

This transport reuses the driver's :class:`SQLManager` for its
execution-time DB needs: reading a parent's cached response to stage a
forked tab, and persisting captured incidental sub-requests against the
navigating request's row id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any

from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.exceptions import TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.data_types import (
    Response,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.driver.browser_engine.engines import (
    BrowserEngine,
    CamoufoxEngine,
    PlaywrightEngine,
)
from jkent.driver.browser_engine.worker_page import WorkerPage
from jkent.driver.database_engine.compression import decompress
from jkent.driver.unified_driver.interstitials import (
    INTERSTITIAL_HANDLERS,
    InterstitialHandler,
)
from jkent.driver.unified_driver.lifecycle import Recoverable
from jkent.driver.unified_driver.requirements import select_browser
from jkent.driver.unified_driver.transport import (
    FileArchiveStream,
    Transport,
)
from jkent.driver.via_actions import (
    execute_via_navigation,
    fill_form_fields,
    selector_for_playwright,
    serve_cached_parent,
    wait_for_required_element,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from playwright.async_api import (
        BrowserContext,
        Download,
        Page,
    )
    from playwright.async_api import (
        Response as PlaywrightResponse,
    )

    from jkent.data_types import BaseScraper, Request, TimeoutType
    from jkent.driver.browser_engine.browser_profile import BrowserProfile
    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.unified_driver.transport import (
        ArchiveStream,
        AwaitCondition,
        QueuedRequest,
    )

logger = logging.getLogger(__name__)

# Bounded fallback deadline for an archive download whose request carries no
# explicit timeout, so download.path() can't stall forever.
_DEFAULT_DOWNLOAD_TIMEOUT_S = 120.0


class ResolveTimeout(TransientException):
    """A resolve navigation/await timeout that carries the partial DOM snapshot.

    A timeout is retryable (so this is a :class:`TransientException`), but the
    page was still snapshotted before giving up. ``debug_response`` carries that
    partial DOM so the worker can persist it for debugging (e.g. inspecting a
    Cloudflare interstitial that never cleared) before the retry — the DOM is
    stored even on timeout.
    """

    def __init__(self, message: str, *, debug_response: Response) -> None:
        super().__init__(message)
        self.debug_response = debug_response


class PlaywrightTransport(Transport[WorkerPage], Recoverable):
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over a browser.

    Also implements :class:`Recoverable` (``generation`` / ``should_restart`` /
    ``restart``) for the transport-internal crash recovery its ``acquire``
    drives; an archive download is staged to a temp file and streamed via the
    shared :class:`FileArchiveStream`, whose temp file ``finish_archiving``
    deletes.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        *,
        browser_type: str | None = None,
        headless: bool = True,
        viewport: dict[str, int] | None = None,
        user_agent: str | None = None,
        locale: str = "en-US",
        timezone_id: str = "America/New_York",
        browser_profile: BrowserProfile | None = None,
        proxy: str | None = None,
        excluded_resource_types: set[str] | None = None,
        db: SQLManager | None = None,
    ) -> None:
        self._scraper = scraper
        # Execution-time DB handle (parent-response reads + incidental writes).
        # Optional so B1's lifecycle tests construct without a DB; resolve
        # raises if invoked without one.
        self._db = db
        self._browser_type = browser_type
        self._headless = headless
        self._viewport = viewport or {"width": 1280, "height": 720}
        self._user_agent = user_agent
        self._locale = locale
        self._timezone_id = timezone_id
        self._browser_profile = browser_profile
        self._proxy = proxy
        self._excluded_resource_types = excluded_resource_types or {
            "image",
            "media",
            "font",
        }
        # Interstitial handlers resolved from scraper's driver_requirements
        self._interstitial_handlers: list[InterstitialHandler] = [
            INTERSTITIAL_HANDLERS[req]
            for req in getattr(scraper, "driver_requirements", [])
            if req in INTERSTITIAL_HANDLERS
        ]
        # Set by open(); cleared by aclose().
        self._engine: BrowserEngine | None = None
        self._engine_cm: Any | None = None
        self._context: BrowserContext | None = None
        self._handles: dict[int, WorkerPage] = {}
        # Crash recovery (B3): single-flight engine restart guarded by a
        # generation. The lock serializes racing restarts; the generation
        # lets losers of the race detect a rebuild already happened.
        self._generation = 0
        self._restart_lock = asyncio.Lock()

    async def open(self) -> None:
        """Select + launch the engine and bring up a live browser context."""
        engine = self._build_engine()
        # The engine exposes its lifecycle as an async context manager;
        # drive it imperatively so open/aclose own enter/exit.
        cm = engine.acquire()
        context = await cm.__aenter__()
        self._engine = engine
        self._engine_cm = cm
        self._context = context

    async def aclose(self) -> None:
        """Close every worker page, then tear the context + engine down."""
        for handle in self._handles.values():
            # A page may already be dead at shutdown (browser crash, Ctrl-C);
            # swallow per-handle close errors so engine teardown below always
            # runs and the browser process isn't leaked.
            with contextlib.suppress(Exception):
                await handle.close()
        self._handles.clear()
        if self._engine_cm is not None:
            # Context + browser + playwright teardown is owned by acquire().
            await self._engine_cm.__aexit__(None, None, None)
        self._engine_cm = None
        self._engine = None
        self._context = None

    async def acquire(self, worker_id: int) -> WorkerPage:
        """Get-or-create the worker's long-lived page, stable until release."""
        handle = self._handles.get(worker_id)
        if handle is not None and handle.page.is_closed():
            self._handles.pop(worker_id)
            handle = None
        if handle is not None:
            try:
                await handle.reset_for_reuse()
                return handle
            except Exception as exc:
                # A reused page that won't reset is worthless whatever the
                # cause — a dead connection, or a navigation race left by the
                # prior request (a slow/timed-out goto keeps navigating in the
                # browser after raising, so the reset's about:blank goto gets
                # "interrupted by another navigation" / NS_BINDING_ABORTED).
                # Discard it and build a fresh page instead of failing the
                # request; _new_page() escalates to a single-flight engine
                # restart when the connection itself is dead.
                logger.warning(
                    "Worker %d page failed reset_for_reuse (%s); "
                    "rebuilding page",
                    worker_id,
                    exc,
                )
                await self._poison_handle(handle)
        page = await self._new_page()
        handle = WorkerPage(page, self._excluded_resource_types)
        self._handles[worker_id] = handle
        return handle

    async def _new_page(self) -> Page:
        """Open a page; escalate a dead-connection to a single-flight restart.

        A live engine but closed page just builds a fresh page. A dead
        connection escalates to :meth:`restart` (one engine rebuild across
        racing workers, guarded by the generation) and retries ``new_page``
        once. Any failure in the restart path surfaces as
        ``TransientException`` so the worker retries instead of failing hard.
        """
        try:
            return await self._require_context().new_page()
        except Exception as exc:
            if not self.should_restart(exc):
                raise
        # The connection is dead. Rebuild the engine once (single-flight),
        # then retry on the freshly-restarted context.
        await self.restart(self.generation)
        try:
            return await self._require_context().new_page()
        except TransientException:
            raise
        except Exception as exc:
            raise TransientException(f"Browser restart failed: {exc}") from exc

    async def release(self, worker_id: int) -> None:
        """Close + drop the worker's page; the next acquire makes a fresh one."""
        handle = self._handles.pop(worker_id, None)
        if handle is not None:
            await handle.close()

    async def resolve(
        self,
        handle: WorkerPage,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        """Navigate, await conditions, snapshot, persist incidentals."""
        if self._db is None:
            raise RuntimeError(
                "PlaywrightTransport.resolve requires a DB reference; "
                "construct with db=..."
            )
        try:
            return await self._resolve(handle, queued, await_conditions)
        except PlaywrightTimeoutError as exc:
            # A slow page load or an await_list selector that never appears is
            # retryable, not a hard failure — re-map Playwright timeouts to a
            # transient so the worker retries with backoff (the page/handle is
            # still alive, so don't poison it).
            raise TransientException(
                f"Playwright timeout during resolve: {exc}"
            ) from exc
        except Exception as exc:
            if not self.should_restart(exc):
                raise
            # Dead connection: poison this worker's handle so the next
            # acquire rebuilds it (and escalates to a restart if the engine
            # is dead), then re-map to a transient so the worker re-queues.
            # The restart itself does NOT happen here.
            await self._poison_handle(handle)
            raise TransientException(
                f"Browser connection lost during resolve: {exc}"
            ) from exc

    async def _poison_handle(self, handle: WorkerPage) -> None:
        """Drop a dead handle from the cache and close it best-effort."""
        for worker_id, cached in list(self._handles.items()):
            if cached is handle:
                self._handles.pop(worker_id, None)
        with contextlib.suppress(Exception):
            await handle.close()

    async def _resolve(
        self,
        handle: WorkerPage,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition],
    ) -> Response:
        """The raw navigation path.

        On a navigation/await timeout the DOM is *still* snapshotted and the
        incidentals persisted (for debugging), then a :class:`ResolveTimeout`
        carrying that partial response is raised so the worker can store it and
        retry (store-then-re-raise on timeout).
        """
        assert self._db is not None  # guarded by resolve()
        request = queued.request
        page = handle.page
        handle.clear_request_state()

        # The request's timeout governs every navigation in this resolve;
        # unset falls through to Playwright's default.
        timeout_ms = self._timeout_ms(request.request.timeout)
        goto_kwargs: dict[str, Any] = {}
        if timeout_ms is not None:
            goto_kwargs["timeout"] = timeout_ms

        # Headers from a prior request on this reused page must not leak.
        await page.set_extra_http_headers(request.request.headers or {})

        # Capture a navigation/await timeout but keep going to snapshot the
        # (partial) DOM below, for debugging.
        timeout_error: PlaywrightTimeoutError | None = None
        # The HTTP status of the navigation we end up snapshotting, when
        # Playwright surfaces it. None falls back to 200 (e.g. same-document
        # navigations expose no response).
        nav_status: int | None = None
        try:
            # Parent-tab staging is only for via (click/form) requests reached
            # FROM a parent page; a plain child request that merely records a
            # parent for lineage must navigate to its OWN url (matches the old
            # driver's `parent_request_id and request.via is not None` guard).
            via = getattr(request, "via", None)
            if queued.parent_request_id is not None and via is not None:
                staged = await self._stage_parent_tab(
                    page, queued.parent_request_id, timeout_ms=timeout_ms
                )
                if staged:
                    # Parent page is loaded from cache; click/submit the via to
                    # navigate through to the child, then snapshot the child DOM.
                    nav_status = await execute_via_navigation(
                        page,
                        via,
                        request.request.url,
                        timeout_ms=timeout_ms,
                    )
                else:
                    # Parent has no stored response — navigate to the child url.
                    nav_response = await page.goto(
                        request.request.url,
                        wait_until="domcontentloaded",
                        **goto_kwargs,
                    )
                    nav_status = nav_response.status if nav_response else None
            else:
                nav_response = await page.goto(
                    request.request.url,
                    wait_until="domcontentloaded",
                    **goto_kwargs,
                )
                nav_status = nav_response.status if nav_response else None

            if self._interstitial_handlers:
                # Race the handlers' waitlists against the scraper's await
                # conditions; an interstitial win means the handler interacts
                # with the page first, then the scraper's own conditions are
                # processed.
                winner = await self._race_await_lists(page, await_conditions)
                if winner is not None:
                    # navigate_through replaces the document; the initial
                    # navigation status now describes the (gone) interstitial,
                    # not the real content, so don't claim it.
                    nav_status = None
                    await winner.navigate_through(page)
                    for condition in await_conditions:
                        await self._apply_await_condition(page, condition)
            else:
                for condition in await_conditions:
                    await self._apply_await_condition(page, condition)
        except PlaywrightTimeoutError as exc:
            timeout_error = exc
            # Playwright's timeout only stops the wait — the browser keeps
            # loading. window.stop() is the programmatic stop button (both
            # engines): abort pending fetches and any uncommitted navigation
            # so the page isn't left mid-navigation for the next reuse, and
            # so the content() snapshot below doesn't race the load.
            # Best-effort: the execution context may die if the navigation
            # commits mid-call; reset/acquire self-heal whatever remains.
            with contextlib.suppress(Exception):
                await page.evaluate("window.stop()")

        # Snapshot the DOM (best-effort) — always, even on timeout. A dead page
        # may refuse content(); fall back to a plain transient then.
        try:
            html_content = await page.content()
            page_url = page.url
        except Exception as exc:
            if timeout_error is not None:
                raise TransientException(
                    f"Playwright timeout during resolve: {timeout_error}"
                ) from timeout_error
            raise TransientException(
                f"Failed to snapshot page during resolve: {exc}"
            ) from exc

        response = Response(
            status_code=nav_status if nav_status is not None else 200,
            url=page_url,
            content=html_content.encode("utf-8"),
            text=html_content,
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )

        # Await outstanding response-body captures so every incidental row has
        # its body before we persist. Without this the capture tasks Playwright
        # fires for the response event may not have finished their body() read,
        # leaving content_compressed NULL for a row that a downstream
        # incidental= request then tries to promote.
        await handle.drain_captures()

        # Persist incidentals against this request's row id.
        for incidental in list(handle.incidental_requests):
            await self._db.insert_incidental_request(  # type: ignore
                parent_request_id=queued.request_id, **incidental
            )

        if timeout_error is not None:
            # Carry the partial DOM to the worker, which stores it before the
            # retry so the failed attempt is inspectable. The partial snapshot
            # is not classified: whatever its status, a timed-out resolve is
            # transient by definition (ResolveTimeout is a TransientException).
            raise ResolveTimeout(
                f"Playwright timeout during resolve: {timeout_error}",
                debug_response=response,
            ) from timeout_error

        # Same contract as the HTTP transport: the scraper's classifier
        # decides transient/persistent/pass-through. The browser can't hand
        # over raw wire bytes, so the classifier sees the DOM snapshot and
        # the synthesized headers; when nav_status is None the fallback 200
        # is classified successful by default, so only content-based
        # overrides can act there.
        self.classify_and_raise(
            self._scraper,
            request,
            status_code=response.status_code,
            headers=response.headers,
            body=response.content,
            url=page_url,
        )
        return response

    async def _stage_parent_tab(
        self,
        page: Page,
        parent_request_id: int,
        *,
        timeout_ms: float | None = None,
    ) -> bool:
        """Serve the parent's cached response into the tab via route intercept.

        ``timeout_ms`` (the navigating request's timeout) bounds the staging
        goto; ``None`` leaves Playwright's default in place.
        """
        assert self._db is not None
        parent_data = await self._db.get_parent_response_for_tab(
            parent_request_id
        )
        if parent_data is None:
            return False

        (
            response_url,
            content_compressed,
            compression_dict_id,
            response_headers_json,
            response_status_code,
        ) = parent_data
        if not response_url or not content_compressed:
            return False

        dictionary = None
        if compression_dict_id is not None:
            dictionary = await self._db.get_compression_dict(  # type: ignore
                compression_dict_id
            )
        body = decompress(content_compressed, dictionary=dictionary)

        headers: dict[str, str] = {}
        if response_headers_json:
            headers = json.loads(response_headers_json)

        await serve_cached_parent(
            page,
            url=response_url,
            body=body,
            headers=headers,
            status=response_status_code or 200,
            timeout_ms=timeout_ms,
        )
        return True

    async def _race_await_lists(
        self,
        page: Page,
        scraper_await_list: Sequence[AwaitCondition],
    ) -> InterstitialHandler | None:
        """Race scraper waitlist against interstitial handler waitlists.

        Each group's conditions are awaited sequentially (conjunction). The two
        sides are not symmetric:

        * The scraper group is *terminal*: if it succeeds, the real content is
          ready (no interstitial → ``None``); if it raises (its selector never
          appeared), that is a genuine resolve timeout and propagates at once.
          We do not wait on the handlers past it.
        * A handler group only ends the race by *succeeding* — that means its
          interstitial is present and it wins. A handler that raises (its marker
          never attached → timeout) merely lost; it isn't present, so the race
          continues on whatever is still pending.

        Losing/pending tasks are cancelled on the way out.

        Returns:
            The winning ``InterstitialHandler``, or ``None`` if the scraper's
            own await conditions completed first (or there is no interstitial
            to handle).
        """

        async def _run_group(conditions: Sequence[AwaitCondition]) -> None:
            for condition in conditions:
                await self._apply_await_condition(page, condition)

        # An empty scraper await list carries no readiness signal, so it must
        # NOT compete: a zero-condition group resolves in the first event-loop
        # tick and would always "win" the race, snapshotting an interstitial
        # that is actually present (e.g. a CFCAP scraper that just navigates).
        # With conditions it is a real racer — real-content vs interstitial-
        # marker, whichever appears first.
        tasks: dict[asyncio.Task[None], InterstitialHandler | None] = {}
        scraper_task: asyncio.Task[None] | None = None
        if scraper_await_list:
            scraper_task = asyncio.create_task(
                _run_group(scraper_await_list), name="scraper"
            )
            tasks[scraper_task] = None
        for handler in self._interstitial_handlers:
            task = asyncio.create_task(
                _run_group(handler.waitlist()),
                name=type(handler).__name__,
            )
            tasks[task] = handler

        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task is scraper_task:
                        # Scraper finished: success → no interstitial (None);
                        # failure → a real resolve timeout, propagate now
                        # rather than waiting for handlers to also time out.
                        task.result()  # re-raises on the scraper's failure
                        return None
                    # A handler finished. A success means its interstitial is
                    # present and it wins; a failure means that interstitial
                    # isn't here — drop it and keep racing the rest.
                    if task.exception() is None:
                        return tasks[task]
            # No scraper task (empty scraper list) and every handler lost:
            # no interstitial was detected, so the caller snapshots as-is.
            return None
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _apply_await_condition(
        page: Page, condition: AwaitCondition
    ) -> None:
        """Apply one await_list directive before snapshotting."""
        if isinstance(condition, WaitForSelector):
            await page.wait_for_selector(
                condition.selector,
                state=condition.state,  # type: ignore[arg-type]
                timeout=condition.timeout,
            )
        elif isinstance(condition, WaitForLoadState):
            await page.wait_for_load_state(
                condition.state,  # type: ignore[arg-type]
                timeout=condition.timeout,
            )
        elif isinstance(condition, WaitForURL):
            await page.wait_for_url(condition.url, timeout=condition.timeout)
        elif isinstance(condition, WaitForTimeout):
            await asyncio.sleep(condition.timeout / 1000.0)

    async def resolve_archive(
        self,
        handle: WorkerPage,
        queued: QueuedRequest,
        decision: object | None = None,
    ) -> ArchiveStream:
        """Trigger a browser download, stage it to a temp file, stream from it.

        Stages the parent tab (if any), triggers the download via the request's
        ``via``, then waits for ``download.path()`` under the request's timeout.
        The worker owns the download decision + save; ``decision`` is accepted
        only for signature parity (a skip never reaches here). A dead connection
        mid-download follows the same poison + transient re-map as ``resolve``.
        """
        if queued.parent_request_id is not None and self._db is None:
            raise RuntimeError(
                "PlaywrightTransport.resolve_archive needs a DB reference to "
                "stage a parent tab; construct with db=..."
            )
        try:
            return await self._resolve_archive(handle, queued)
        except PlaywrightTimeoutError as exc:
            # A download that never triggers (expect_download/click timeout) is
            # retryable, not a hard failure — re-map to a transient so the
            # worker retries with backoff (mirrors resolve(); the handle is
            # still alive, so don't poison it).
            raise TransientException(
                f"Playwright timeout during resolve_archive: {exc}"
            ) from exc
        except Exception as exc:
            if not self.should_restart(exc):
                raise
            # Dead connection mid-download: poison the handle so the next
            # acquire rebuilds it, and re-map to a transient so the worker
            # re-queues (mirrors resolve()).
            await self._poison_handle(handle)
            raise TransientException(
                f"Browser connection lost during resolve_archive: {exc}"
            ) from exc

    async def _resolve_archive(
        self, handle: WorkerPage, queued: QueuedRequest
    ) -> FileArchiveStream:
        """The raw archive-download path."""
        request = queued.request
        page = handle.page
        handle.clear_request_state()

        timeout_ms = self._timeout_ms(request.request.timeout)
        if queued.parent_request_id is not None:
            staged = await self._stage_parent_tab(
                page, queued.parent_request_id, timeout_ms=timeout_ms
            )
            if not staged:
                raise TransientException(
                    "Archive download: parent has no stored response to stage"
                )

        result = await self._execute_via_download(request, page)

        # The browser rendered the archive inline (Firefox pdf.js) instead of
        # downloading it; the stream is already staged over a temp file, so
        # the rest of the pipeline (including finish_archiving's unlink) is
        # identical to a real download.
        if isinstance(result, FileArchiveStream):
            return result

        download = result

        # Download.path() has no native timeout: a server that starts the
        # response then trickles/stalls would hang forever. Honor the
        # request's timeout as a hard deadline, re-mapping an overrun to a
        # transient. The request timeout defaults to None, so fall back to a
        # bounded deadline rather than waiting forever (which would defeat
        # this guard).
        download_timeout = (
            timeout_ms / 1000.0
            if timeout_ms is not None
            else _DEFAULT_DOWNLOAD_TIMEOUT_S
        )
        try:
            download_path = await asyncio.wait_for(
                download.path(), timeout=download_timeout
            )
        except asyncio.TimeoutError as exc:
            raise TransientException(
                f"Archive download exceeded timeout of {download_timeout}s"
            ) from exc
        if download_path is None:
            raise TransientException("Archive download produced no file")

        return FileArchiveStream(
            status_code=200,
            headers={},
            url=download.url or request.request.url,
            file_path=str(download_path),
        )

    async def _execute_via_download(
        self, request: Request, page: Page
    ) -> Download | FileArchiveStream:
        """Click the request's ``via`` target and return the resulting archive.

        Normally the click triggers a browser ``download`` event and a
        :class:`Download` is returned. But Firefox/camoufox opens some files
        (PDFs) in its built-in viewer and *navigates* the page to the file
        instead of downloading it — no ``download`` event ever fires. In that
        case the navigation's response bytes are staged to a temp file and
        returned as a ready :class:`FileArchiveStream` instead. See
        :meth:`_await_download_or_inline`.

        Honors ``request.request.timeout`` (seconds; tuple -> read element) as
        a millisecond deadline on both the download wait and the click — the
        click's "wait for scheduled navigations" phase otherwise uses
        Playwright's 30s default, ignoring a longer user timeout.
        """
        params = request.request
        click_kwargs: dict[str, Any] = {}
        timeout_ms = self._timeout_ms(params.timeout)
        if timeout_ms is not None:
            click_kwargs["timeout"] = timeout_ms

        if isinstance(request.via, ViaLink):
            element = await wait_for_required_element(
                page,
                selector_for_playwright(request.via.selector),
                request.request.url,
            )
            # Drop target=_blank so the click can't open the file in a new tab
            # (engine-agnostic complement to the Firefox open_newwindow pref) —
            # an orphan tab would leak since we reuse one page per worker.
            await element.evaluate("el => el.removeAttribute('target')")

            async def _click_link() -> None:
                await element.click(**click_kwargs)

            return await self._await_download_or_inline(
                page, _click_link, timeout_ms=timeout_ms
            )

        if isinstance(request.via, ViaFormSubmit):
            form_via = request.via
            form = await wait_for_required_element(
                page,
                selector_for_playwright(form_via.form_selector),
                request.request.url,
            )
            await fill_form_fields(form, form_via.field_data)
            # Same tab-normalization as the ViaLink path: a form with
            # target=_blank would submit into a new tab we never close.
            await form.evaluate("el => el.removeAttribute('target')")
            submit_selector = (
                form_via.submit_selector
                or 'button[type="submit"], input[type="submit"]'
            )
            submit = await form.query_selector(submit_selector)
            if submit is None:
                raise TransientException(
                    f"Submit selector not found: {submit_selector}"
                )

            async def _submit_form() -> None:
                if await submit.is_visible():
                    await submit.click(**click_kwargs)
                else:
                    # The real submit control is non-interactable — e.g. its
                    # element was swallowed by malformed HTML (an unclosed
                    # <style> turning the button into raw text) so
                    # fill_form_fields synthesized a hidden input carrying its
                    # name/value in place. A hidden element can't be clicked,
                    # but its name=value is already a field on the form, so a
                    # bare form.submit() POSTs the same data the click would —
                    # mirroring execute_via_navigation's __EVENTTARGET path
                    # (via_actions).
                    await form.evaluate("(f) => f.submit()")

            return await self._await_download_or_inline(
                page, _submit_form, timeout_ms=timeout_ms
            )

        raise ValueError(
            f"Archive download requires ViaLink or ViaFormSubmit, "
            f"got {type(request.via)}"
        )

    async def _await_download_or_inline(
        self,
        page: Page,
        trigger: Callable[[], Awaitable[None]],
        *,
        timeout_ms: float | None,
    ) -> Download | FileArchiveStream:
        """Fire ``trigger`` and return the download — or inline-rendered bytes.

        A real download fires a ``download`` event while the page stays put; an
        inline render (Firefox opening a PDF in pdf.js) instead *navigates* the
        main frame to the file and fires no download event. We register both a
        ``download`` waiter and a main-frame ``framenavigated`` waiter before
        the trigger, then race them:

        * download event first  -> return the :class:`Download` (normal path);
        * navigation first       -> the browser rendered the file inline, so we
          stage the captured response's bytes to a temp file and return them
          as a ready :class:`FileArchiveStream`.

        A response listener runs throughout so the inline branch has the file's
        own response body to hand. If neither a download nor a usable inline
        body materializes, a :class:`TransientException` is raised so the worker
        retries (mirroring the old ``expect_download`` timeout).
        """
        wait_kwargs: dict[str, Any] = {}
        if timeout_ms is not None:
            wait_kwargs["timeout"] = timeout_ms

        # Capture file-ish responses so an inline render can be reconstructed
        # from the navigation's own response body (the last one wins — a
        # redirect chain ends on the real file).
        captured: list[PlaywrightResponse] = []

        def _on_response(resp: PlaywrightResponse) -> None:
            ctype = (resp.headers or {}).get("content-type", "").lower()
            path = resp.url.split("?", 1)[0].lower()
            if "pdf" in ctype or path.endswith((".pdf", ".doc", ".docx")):
                captured.append(resp)

        # The race itself is one future: the first of `download` /
        # main-frame `framenavigated` resolves it (a Download for the normal
        # path, None for an inline render). Listeners attach synchronously,
        # so they're in place before the trigger can produce either event.
        outcome: asyncio.Future[Download | None] = (
            asyncio.get_running_loop().create_future()
        )

        def _on_download(download: Download) -> None:
            if not outcome.done():
                outcome.set_result(download)

        def _on_framenavigated(frame: Any) -> None:
            if frame == page.main_frame and not outcome.done():
                outcome.set_result(None)

        page.on("response", _on_response)
        page.on("download", _on_download)
        page.on("framenavigated", _on_framenavigated)
        try:
            try:
                await trigger()
            except PlaywrightTimeoutError:
                # An inline navigation can trip the click's own auto-wait for
                # "scheduled navigations"; the outcome future is the real
                # arbiter of what actually happened, so don't fail here.
                pass

            # Match the old waiters' deadline: the request timeout when set,
            # else Playwright's 30s default.
            timeout_s = timeout_ms / 1000.0 if timeout_ms is not None else 30.0
            try:
                result = await asyncio.wait_for(outcome, timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                raise TransientException(
                    "Archive trigger produced neither a download nor an "
                    "inline render before the timeout"
                ) from exc

            if result is not None:
                return result

            # The page navigated instead of downloading — an inline render.
            # Let the file's response settle, then take its bytes.
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("load", **wait_kwargs)
            inline = await self._stage_inline_stream(captured)
            if inline is not None:
                return inline

            raise TransientException(
                "Archive trigger navigated inline but no readable file "
                "response was captured"
            )
        finally:
            for event, listener in (
                ("response", _on_response),
                ("download", _on_download),
                ("framenavigated", _on_framenavigated),
            ):
                with contextlib.suppress(Exception):
                    page.remove_listener(event, listener)

    @staticmethod
    async def _stage_inline_stream(
        captured: list[PlaywrightResponse],
    ) -> FileArchiveStream | None:
        """Stage an inline-rendered archive from a captured response.

        Prefers the most recent captured file response; skips any whose body is
        unavailable (e.g. Playwright can't surface it). The bytes go to a temp
        file mirroring the one a real Playwright download stages to, so
        ``finish_archiving`` unlinks it exactly like a download. Returns
        ``None`` when no captured response yields bytes.
        """
        for resp in reversed(captured):
            try:
                body = await resp.body()
            except PlaywrightError:
                continue
            if body:
                break
        else:
            return None

        def _write_temp() -> str:
            fd, path = tempfile.mkstemp(prefix="jkent-inline-archive-")
            with os.fdopen(fd, "wb") as handle:
                handle.write(body)
            return path

        return FileArchiveStream(
            status_code=resp.status,
            headers={},
            url=resp.url,
            file_path=await asyncio.to_thread(_write_temp),
        )

    @staticmethod
    def _timeout_ms(timeout: TimeoutType) -> float | None:
        """The request's timeout (requests-style seconds; a (connect, read)
        tuple uses the read element) as Playwright milliseconds, or ``None``
        when unset so callers fall through to Playwright's default."""
        if timeout is None:
            return None
        if isinstance(timeout, tuple):
            timeout = timeout[1]
        return float(timeout) * 1000.0

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Delete the staged temp file backing a Playwright download stream."""
        if isinstance(stream, FileArchiveStream):
            with contextlib.suppress(FileNotFoundError, OSError):
                await asyncio.to_thread(os.unlink, stream.file_path)

    def _build_engine(self) -> BrowserEngine:
        """Pick the engine per the scraper's requirements.

        :func:`~jkent.driver.unified_driver.requirements.select_browser` is
        the one selection-precedence site: camoufox → :class:`CamoufoxEngine`,
        else a :class:`PlaywrightEngine` whose flavor is the explicit
        ``browser_type`` constructor arg if given, else the selected browser
        (defaulting to chromium when the scraper has no preference). A
        ``browser_profile.browser_type`` still overrides either inside the
        engine.
        """
        choice = select_browser(self._scraper)
        if choice == "camoufox":
            return CamoufoxEngine(
                scraper=self._scraper,
                browser_profile=self._browser_profile,
                headless=self._headless,
                locale=self._locale,
                proxy=self._proxy,
                # Disable mouse humanization — it can stall clicks
                # indefinitely.
                humanize=False,
            )
        browser_type = self._browser_type
        if browser_type is None:
            browser_type = choice if choice is not None else "chromium"
        return PlaywrightEngine(
            scraper=self._scraper,
            browser_profile=self._browser_profile,
            browser_type=browser_type,
            headless=self._headless,
            viewport=self._viewport,
            user_agent=self._user_agent,
            locale=self._locale,
            timezone_id=self._timezone_id,
            proxy=self._proxy,
        )

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("PlaywrightTransport used before open()")
        return self._context

    # --- Cookie persistence ---------------------------------------------

    async def export_cookies(self) -> str | None:
        """Dump the live context's cookies as JSON, or None if unavailable.

        Returns None (not raises) when the context is gone or already closed —
        e.g. after a Ctrl-C tore the browser down before this best-effort save.
        """
        if self._context is None:
            return None
        try:
            return json.dumps(await self._context.cookies())
        except PlaywrightError:
            return None

    async def import_cookies(self, cookies_json: str) -> None:
        """Apply previously-exported cookies to the live context."""
        cookies = json.loads(cookies_json)
        if cookies:
            await self._require_context().add_cookies(cookies)

    # --- Recoverable (transport-internal crash recovery) -----------------

    @property
    def generation(self) -> int:
        """Monotonic count of how many times the engine was (re)built."""
        return self._generation

    def should_restart(self, exc: BaseException) -> bool:
        """Whether ``exc`` means the browser connection died (ported predicate).

        Pure and side-effect-free. Matches on message because Playwright
        rewraps the channel-layer transport error as a bare ``Exception``.
        """
        msg = str(exc)
        return (
            "Connection closed" in msg
            or "Browser has been closed" in msg
            or "Target page, context or browser has been closed" in msg
        )

    async def restart(self, seen_generation: int) -> None:
        """Rebuild the engine once, single-flight under the generation guard.

        If ``seen_generation`` no longer matches the current generation a
        racing caller already rebuilt — this is a no-op. Otherwise the
        poisoned handle cache is cleared, the context rebuilt, and the
        generation advanced.
        """
        async with self._restart_lock:
            if seen_generation != self._generation:
                return  # another caller already rebuilt this generation
            # Drop every handle: they all reference the dead browser.
            self._handles.clear()
            await self._rebuild_context()
            self._generation += 1

    async def _rebuild_context(self) -> None:
        """The browser-touching rebuild step (overridable for tests).

        Default: drive the engine's restart and reassign ``self._context``
        (the single ref ``acquire`` reads). Engines that can't restart raise
        ``TransientException`` from ``restart_context``.
        """
        if self._engine is None:
            raise TransientException(
                "Browser connection lost; no engine attached"
            )
        self._context = await self._engine.restart_context()
