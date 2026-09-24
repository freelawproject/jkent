"""Playwright transport — engine + per-worker page lifecycle.

Owns browser launch/teardown and per-worker page acquisition. The engine and
browser context are built in ``open`` (wrapping ``engines/``) and torn down in
``aclose``; each worker gets a long-lived :class:`WorkerPage` from ``acquire``,
stable until ``release``.

``resolve`` handles the navigation path. Crash recovery is the
transport-internal ``generation`` / ``should_restart`` / ``restart``
surface: a dead connection
noticed in ``resolve`` poisons the handle and re-maps to ``TransientException``,
and the next ``acquire`` rebuilds the handle, escalating to a single-flight
engine restart when the connection itself is dead. ``resolve_archive``
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
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)
from typing_extensions import override

from jkent.common.exceptions import (
    InterstitialUnresolved,
    ResolveTimeout,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.common.headers import merge_headers
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.common.request import DEFAULT_TIMEOUT_S
from jkent.common.response import utf8_document
from jkent.common.serialization import dump_json
from jkent.data_types import (
    HttpMethod,
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
from jkent.driver.database_engine.compression import (
    decompress,
    get_dict_by_id,
)
from jkent.driver.unified_driver.interstitials import (
    InterstitialHandler,
    handlers_for,
)
from jkent.driver.unified_driver.requirements import ResolvedRequirements
from jkent.driver.unified_driver.transport import (
    FileArchiveStream,
    Transport,
)
from jkent.driver.via_actions import (
    execute_via_navigation,
    prepare_form_submit,
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

T = TypeVar("T")

logger = logging.getLogger(__name__)

#: Incidental resource types fetched but never stored.
_EXCLUDED_RESOURCE_TYPES = frozenset({"image", "media", "font"})

#: Plain Playwright only; Camoufox runs with ``no_viewport`` so its
#: fingerprinted screen size stands.
_VIEWPORT = {"width": 1280, "height": 720}


#: Selector states that pass when the element is simply absent, so a wait on
#: them proves nothing about the current document having loaded.
_VACUOUS_SELECTOR_STATES = frozenset({"hidden", "detached"})


def _asserts_real_content(conditions: Sequence[AwaitCondition]) -> bool:
    """Whether an await list actually proves the scraper's content arrived.

    Only a *positive* condition does: a selector that must be attached or
    visible, or a URL the page must reach. A ``hidden``/``detached`` selector
    wait passes against a document that never had the element, and every load
    state a real page reaches an interstitial reaches too — so a list made
    only of those is satisfied by a Cloudflare challenge and cannot be raced
    against an interstitial handler.

    Pure and side-effect-free; see :meth:`PlaywrightTransport._race_await_lists`
    for what a false verdict changes.
    """
    return any(
        (
            isinstance(c, WaitForSelector)
            and c.state not in _VACUOUS_SELECTOR_STATES
        )
        or isinstance(c, WaitForURL)
        for c in conditions
    )


#: Content types a main-frame navigation answers with when it lands on a
#: page rather than a file the browser renders inline.
_PAGE_CONTENT_TYPES = ("text/html", "application/xhtml+xml")


def _is_inline_file(resp: PlaywrightResponse, main_frame: Any) -> bool:
    """Whether ``resp`` may be the file an inline render is showing.

    Any main-frame navigation that is not answered with a page: a browser
    renders PDF, text, JSON, XML and images in place, at whatever URL. Also,
    anywhere on the page, a PDF content type or a ``.pdf``/``.doc``/``.docx``
    path, which catches a file served mislabelled as HTML. An HTML page is
    never the file by type alone: a login or error page stays a failed
    archive rather than being saved as one.
    """
    ctype = (resp.headers or {}).get("content-type", "").lower()
    path = resp.url.split("?", 1)[0].lower()
    if "pdf" in ctype or path.endswith((".pdf", ".doc", ".docx")):
        return True
    return (
        resp.frame == main_frame
        and resp.request.is_navigation_request()
        and not ctype.startswith(_PAGE_CONTENT_TYPES)
    )


def _is_download_abort(exc: PlaywrightError) -> bool:
    """Whether ``exc`` is a goto the browser abandoned for a download."""
    message = str(exc)
    return any(
        marker in message
        for marker in (
            "Download is starting",
            "net::ERR_ABORTED",
            "NS_BINDING_ABORTED",
        )
    )


class PlaywrightTransport(Transport[WorkerPage]):
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over a browser.

    Owns its crash recovery (``generation`` / ``should_restart`` /
    ``restart``), the transport-internal surface its ``acquire`` drives; an archive download is staged to a temp file and streamed via the
    shared :class:`FileArchiveStream`, whose temp file ``finish_archiving``
    deletes.
    """

    #: A live page fires sub-requests, and this transport records them
    #: against the navigation's request id — so ``incidental=`` steps work.
    captures_incidentals: ClassVar[bool] = True

    def __init__(
        self,
        scraper: BaseScraper[Any],
        *,
        browser_type: str | None = None,
        headless: bool = True,
        browser_profile: BrowserProfile | None = None,
        proxy: str | None = None,
        blocked_resource_types: set[str] | None = None,
        page_recycle_after: int | None = 500,
        db: SQLManager | None = None,
        timeout: float | None = None,
    ) -> None:
        self._scraper = scraper
        # What a request without its own timeout waits, on every wait this
        # transport performs (navigation, via click, download, await).
        self._timeout = DEFAULT_TIMEOUT_S if timeout is None else timeout
        # Baseline headers for every request — the same merge httpx does, so
        # a browser scraper's wire headers match its HTTP twin's.
        self._default_headers = dict(
            getattr(scraper, "default_headers", None) or {}
        )
        # Execution-time DB handle (parent-response reads + incidental writes).
        # Optional so B1's lifecycle tests construct without a DB; resolve
        # raises if invoked without one.
        self._db: SQLManager | None = db
        self._browser_type = browser_type
        self._headless = headless
        self._browser_profile = browser_profile
        self._proxy = proxy
        # Request-time blocking. These types are already never stored, so on
        # a content scraper the fetch buys nothing but latency: the page's
        # sub-resources are re-fetched for every cached-parent re-stage, which
        # on a WebForms site is most of the per-request network time. Pass an
        # empty set to fetch them again — that is the whole switch. Scrapers
        # that need pixels (an image CAPTCHA challenge, a rendered document)
        # must do so, since blocking is otherwise invisible to them.
        self._blocked_resource_types = (
            {"image", "media", "font"}
            if blocked_resource_types is None
            else blocked_resource_types
        )
        # Rebuild a worker's page once it has been reused N times (so after
        # N + 1 requests). A page reused for a whole run accumulates
        # per-navigation network state that never drains
        # (reset_for_reuse only navigates to about:blank), and per-request
        # latency climbs with it — measurably, and identically across runs.
        # Cookies live on the *context*, so a recycled page keeps the session;
        # only the accumulated state is dropped. None disables recycling.
        self._page_recycle_after = page_recycle_after
        # The one resolved answer for this scraper: engine choice and the
        # interstitial handlers to wire. Built once — the handlers are this
        # transport's own (they carry per-browser state), not shared
        # module-level singletons.
        self._requirements = ResolvedRequirements.of(scraper)
        self._interstitial_handlers: list[InterstitialHandler] = handlers_for(
            self._requirements.interstitials
        )
        # Set by open(); cleared by aclose().
        self._engine: BrowserEngine | None = None
        self._engine_cm: Any | None = None
        self._context: BrowserContext | None = None
        self._handles: dict[int, WorkerPage] = {}
        # Crash recovery: single-flight engine restart guarded by a
        # generation. The lock serializes racing restarts; the generation
        # lets losers of the race detect a rebuild already happened.
        self._generation = 0
        self._restart_lock = asyncio.Lock()

    @property
    @override
    def timeout(self) -> float:
        """Seconds a request without its own ``timeout`` waits."""
        return self._timeout

    @override
    def bind_run_db(self, db: SQLManager) -> None:
        """Adopt the run's database unless one was supplied at construction.

        The bootstrapper passes the run's own manager as ``db=`` (every
        SQLite writer must share one manager), so this normally finds ``_db``
        already set and leaves it alone. It covers the caller that builds
        this transport without one.
        """
        if self._db is None:
            self._db = db

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
        # Forgotten before the exit runs: a teardown that raises has still
        # consumed the context manager, and a second aclose must not re-exit
        # it.
        cm, self._engine_cm = self._engine_cm, None
        self._engine = None
        self._context = None
        if cm is not None:
            # Context + browser + playwright teardown is owned by acquire().
            await cm.__aexit__(None, None, None)

    async def acquire(self, worker_id: int) -> WorkerPage:
        """Get-or-create the worker's long-lived page, stable until release."""
        handle = self._handles.get(worker_id)
        if handle is not None and handle.page.is_closed():
            self._handles.pop(worker_id)
            handle = None
        if (
            handle is not None
            and self._page_recycle_after is not None
            and handle.uses >= self._page_recycle_after
        ):
            # ``uses`` counts reuses; the page's first request came before any.
            logger.info(
                "Worker %d page recycled after %d requests (%d reuses)",
                worker_id,
                handle.uses + 1,
                handle.uses,
            )
            await self._poison_handle(handle)
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
        handle = WorkerPage(
            page,
            _EXCLUDED_RESOURCE_TYPES,
            self._blocked_resource_types,
        )
        await handle.install_request_blocking()
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
        # Sampled before the attempt: a racer may restart while this
        # new_page is still failing, and the generation read afterwards
        # would name the fresh browser as the dead one.
        seen_generation = self.generation
        try:
            return await self._require_context().new_page()
        except Exception as exc:
            if not self.should_restart(exc):
                raise
        # The connection is dead. Rebuild the engine once (single-flight),
        # then retry on the freshly-restarted context.
        await self.restart(seen_generation)
        try:
            return await self._require_context().new_page()
        except TransientException:
            raise
        except Exception as exc:
            raise TransientException(
                f"Browser restart failed: {exc}",
                kind=TransientKind.BROWSER_CRASH,
            ) from exc

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
        return await self._crash_guard(
            handle,
            "resolve",
            self._resolve(handle, queued, await_conditions),
            url=queued.request.request.url,
            timeout_kind=TransientKind.NAVIGATION,
        )

    async def _crash_guard(
        self,
        handle: WorkerPage,
        operation: str,
        work: Awaitable[T],
        *,
        url: str,
        timeout_kind: TransientKind,
    ) -> T:
        """Await *work*, re-mapping its browser failures to transients.

        A Playwright timeout (a slow load, an await_list selector or download
        that never appears) is retryable, and the page is still alive, so the
        handle is kept. A dead connection poisons the handle so the next
        ``acquire`` rebuilds it (escalating to an engine restart if needed);
        the restart itself does NOT happen here. Anything else propagates.
        """
        try:
            return await work
        except PlaywrightTimeoutError as exc:
            raise TransientException(
                f"Playwright timeout during {operation}: {exc}",
                url=url,
                kind=timeout_kind,
            ) from exc
        except Exception as exc:
            if not self.should_restart(exc):
                raise
            await self._poison_handle(handle)
            raise TransientException(
                f"Browser connection lost during {operation}: {exc}",
                url=url,
                kind=TransientKind.BROWSER_CRASH,
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

        # The request's timeout (else the transport's) governs every
        # navigation in this resolve.
        timeout_ms = self._timeout_ms(request.request.timeout)
        goto_kwargs: dict[str, Any] = {"timeout": timeout_ms}

        await self._apply_request_headers(page, request)

        # Capture a navigation/await timeout but keep going to snapshot the
        # (partial) DOM below, for debugging.
        # A Playwright wait that ran out, or an interstitial handler that
        # gave up — both mean "we never reached the real content", and both
        # take the snapshot-then-re-raise path below.
        timeout_error: PlaywrightTimeoutError | InterstitialUnresolved | None
        timeout_error = None
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
                winner = await self._race_await_lists(
                    page, await_conditions, timeout_ms=timeout_ms
                )
                if winner is not None:
                    # navigate_through replaces the document; the initial
                    # navigation status now describes the (gone) interstitial,
                    # not the real content, so don't claim it.
                    nav_status = None
                    await winner.navigate_through(page)
                    for condition in await_conditions:
                        await self._apply_await_condition(
                            page, condition, timeout_ms=timeout_ms
                        )
            else:
                for condition in await_conditions:
                    await self._apply_await_condition(
                        page, condition, timeout_ms=timeout_ms
                    )
        except (PlaywrightTimeoutError, InterstitialUnresolved) as exc:
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
                # The page died before it could be snapshotted, so there is
                # no partial DOM to carry and this cannot be a ResolveTimeout.
                # The timeout is still the cause worth naming.
                raise TransientException(
                    f"Resolve did not reach content: {timeout_error}",
                    url=request.request.url,
                    kind=TransientKind.NAVIGATION,
                ) from timeout_error
            raise TransientException(
                f"Failed to snapshot page during resolve: {exc}",
                url=request.request.url,
                kind=TransientKind.SNAPSHOT,
            ) from exc

        response = Response(
            status_code=nav_status if nav_status is not None else 200,
            url=page_url,
            # The DOM keeps the page's own <meta charset>, which would
            # outrank the header when these UTF-8 bytes are decoded, so the
            # bytes declare UTF-8 themselves; ``text`` derives from them the
            # same way a stored row or a replay will.
            content=utf8_document(html_content),
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )

        # Await outstanding response-body captures so every incidental row has
        # its body before we persist. Without this the capture tasks Playwright
        # fires for the response event may not have finished their body() read,
        # leaving content_compressed NULL for a row that a downstream
        # incidental= request then tries to promote.
        await handle.drain_captures()

        # Persist incidentals against this request's row id — one transaction
        # for the whole navigation, not one commit per subresource. Always,
        # even with nothing captured: the batch replaces an earlier attempt's.
        await self._db.replace_incidental_requests(
            queued.request_id, list(handle.incidental_requests)
        )

        if timeout_error is not None:
            # Carry the partial DOM to the worker, which stores it before the
            # retry so the failed attempt is inspectable. The partial snapshot
            # is not classified: whatever its status, a timed-out resolve is
            # transient by definition (ResolveTimeout is a TransientException).
            raise ResolveTimeout(
                url=request.request.url,
                timeout_seconds=timeout_ms / 1000.0,
                message=f"Resolve did not reach content: {timeout_error}",
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
        parent = await self._db.get_stored_response(parent_request_id)
        content_compressed = parent.content_compressed if parent else None
        if parent is None or not content_compressed:
            return False
        dictionary = None
        if parent.compression_dict_id is not None:
            # The cached lookup: tab staging runs per navigating child, so
            # re-reading the dictionary blob per stage would be pure waste.
            dictionary = await get_dict_by_id(
                self._db, parent.compression_dict_id
            )
        body = decompress(content_compressed, dictionary=dictionary)

        headers: dict[str, str] = {}
        if parent.response_headers_json:
            headers = json.loads(parent.response_headers_json)

        await serve_cached_parent(
            page,
            url=parent.response_url,
            body=body,
            headers=headers,
            status=parent.response_status_code,
            timeout_ms=timeout_ms,
        )
        return True

    async def _race_await_lists(
        self,
        page: Page,
        scraper_await_list: Sequence[AwaitCondition],
        *,
        timeout_ms: float,
    ) -> InterstitialHandler | None:
        """Race scraper waitlist against interstitial handler waitlists.

        ``timeout_ms`` bounds every condition, on either side, that sets no
        timeout of its own (see :meth:`_apply_await_condition`).

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
            to handle). When ``None`` is returned the scraper's conditions have
            already been applied, so the caller must not re-apply them.
        """

        async def _run_group(conditions: Sequence[AwaitCondition]) -> None:
            for condition in conditions:
                await self._apply_await_condition(
                    page, condition, timeout_ms=timeout_ms
                )

        # A scraper await list only gets to compete if it actually asserts that
        # real content arrived. Two kinds don't, and both resolve on the first
        # event-loop tick against an interstitial:
        #   * the empty list (a CFCAP scraper that just navigates)
        #   * a list whose every condition is vacuously satisfiable — a
        #     ``state="hidden"``/``"detached"`` selector wait passes when the
        #     element is simply absent, and a load-state wait passes because a
        #     challenge page reaches load/networkidle like any other document.
        # Letting either race meant the scraper "won" instantly and we
        # snapshotted the challenge as content: six requests in the CA
        # calctapp_1st run died as persistent HTTP 403 with the challenge HTML
        # stored as their body, retry_count 0, handler never consulted.
        # Non-competing lists are applied after the handlers concede instead.
        tasks: dict[asyncio.Task[None], InterstitialHandler | None] = {}
        scraper_task: asyncio.Task[None] | None = None
        scraper_competes = _asserts_real_content(scraper_await_list)
        if scraper_competes:
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
                # ``done`` may hold both sides at once; read it in a fixed
                # order rather than the set's. Scraper success → no
                # interstitial (None): its content is there.
                scraper_done = scraper_task if scraper_task in done else None
                if (
                    scraper_done is not None
                    and scraper_done.exception() is None
                ):
                    return None
                # A handler finished. A success means its interstitial is
                # present and it wins — even over a scraper failure in the
                # same tick, which it explains; a failure means that
                # interstitial isn't here — drop it and keep racing the rest.
                for task in done:
                    if task is not scraper_task and task.exception() is None:
                        return tasks[task]
                if scraper_done is not None:
                    # A real resolve timeout: propagate now rather than
                    # waiting for handlers to also time out.
                    scraper_done.result()
            # Every handler lost, so no interstitial is present. A scraper list
            # that was held out of the race still has to be honoured before the
            # caller snapshots — apply it here so the ``None`` return means
            # "conditions satisfied, snapshot as-is" either way.
            if not scraper_competes and scraper_await_list:
                await _run_group(scraper_await_list)
            return None
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _apply_await_condition(
        page: Page, condition: AwaitCondition, *, timeout_ms: float
    ) -> None:
        """Apply one await_list directive before snapshotting.

        A condition's own ``timeout`` wins; one without waits ``timeout_ms``
        (the request's, else the transport's) rather than Playwright's
        default, like every other wait in a resolve.
        """
        if isinstance(condition, WaitForTimeout):
            await asyncio.sleep(condition.timeout / 1000.0)
            return
        timeout = (
            timeout_ms if condition.timeout is None else condition.timeout
        )
        if isinstance(condition, WaitForSelector):
            await page.wait_for_selector(
                condition.selector,
                state=condition.state,
                timeout=timeout,
            )
        elif isinstance(condition, WaitForLoadState):
            await page.wait_for_load_state(
                condition.state,
                timeout=timeout,
            )
        elif isinstance(condition, WaitForURL):
            await page.wait_for_url(condition.url, timeout=timeout)

    async def resolve_archive(
        self,
        handle: WorkerPage,
        queued: QueuedRequest,
    ) -> ArchiveStream:
        """Trigger a browser download, stage it to a temp file, stream from it.

        Stages the parent tab (if any), triggers the download via the request's
        ``via``, then waits for ``download.path()`` under the request's timeout.
        The worker owns the download decision + save (a skip never reaches
        here). A dead connection
        mid-download follows the same poison + transient re-map as ``resolve``.
        """
        if queued.parent_request_id is not None and self._db is None:
            raise RuntimeError(
                "PlaywrightTransport.resolve_archive needs a DB reference to "
                "stage a parent tab; construct with db=..."
            )
        return await self._crash_guard(
            handle,
            "resolve_archive",
            self._resolve_archive(handle, queued),
            url=queued.request.request.url,
            timeout_kind=TransientKind.ARCHIVE,
        )

    async def _resolve_archive(
        self, handle: WorkerPage, queued: QueuedRequest
    ) -> FileArchiveStream:
        """The raw archive-download path.

        Records no incidentals: nothing here persists them, and the file's
        own response would be one.
        """
        with handle.capture_paused():
            return await self._download_archive(handle.page, queued)

    async def _download_archive(
        self, page: Page, queued: QueuedRequest
    ) -> FileArchiveStream:
        """Stage the parent, trigger the download, classify the result."""
        request = queued.request

        timeout_ms = self._timeout_ms(request.request.timeout)
        if queued.parent_request_id is not None:
            staged = await self._stage_parent_tab(
                page, queued.parent_request_id, timeout_ms=timeout_ms
            )
            if not staged:
                raise TransientException(
                    "Archive download: parent has no stored response to stage",
                    url=request.request.url,
                    kind=TransientKind.ARCHIVE,
                )

        await self._apply_request_headers(page, request)
        result = await self._execute_via_download(request, page)

        if isinstance(result, FileArchiveStream):
            # The browser rendered the archive inline (Firefox pdf.js)
            # instead of downloading it; the stream is already staged over a
            # temp file, so the rest of the pipeline (including
            # finish_archiving's unlink) is identical to a real download.
            stream = result
        else:
            download, observed = result
            # Download.path() has no native timeout: a server that starts
            # the response then trickles/stalls would hang forever. The
            # request's (else the transport's) timeout is the hard deadline.
            download_timeout = timeout_ms / 1000.0
            try:
                download_path = await asyncio.wait_for(
                    download.path(), timeout=download_timeout
                )
            except asyncio.TimeoutError as exc:
                raise TransientException(
                    f"Archive download exceeded timeout of {download_timeout}s",
                    url=request.request.url,
                    kind=TransientKind.ARCHIVE,
                ) from exc
            if download_path is None:
                raise TransientException(
                    "Archive download produced no file",
                    url=request.request.url,
                    kind=TransientKind.ARCHIVE,
                )
            # The download's own response, when the browser reported one:
            # the row records what the server actually sent, not a
            # placeholder 200. A download with no observed response (a
            # same-document trigger) keeps the placeholder.
            stream = FileArchiveStream(
                status_code=observed.status if observed is not None else 200,
                headers=dict(observed.headers) if observed is not None else {},
                url=download.url or request.request.url,
                file_path=str(download_path),
            )

        # Same contract as the HTTP stream path: the scraper's classifier
        # sees the status + headers before a byte is saved. The staged file
        # is released on a verdict that raises, exactly as finish_archiving
        # would after a save.
        try:
            self.classify_and_raise(
                self._scraper,
                request,
                status_code=stream.status_code,
                headers=stream.headers,
                body=None,
                url=stream.url,
            )
        except BaseException:
            await self.finish_archiving(stream)
            raise
        return stream

    async def _apply_request_headers(
        self, page: Page, request: Request
    ) -> None:
        """Put the scraper's defaults, the request's headers, and its cookies on the wire.

        Extra headers are set afresh for every request, so nothing from a
        prior request on this reused page leaks into this one. Per-request
        cookies join the context's jar for the request's URL, so they
        outlive the request: every worker sends them to that site from then
        on, and ``export_cookies`` saves them for the next run. That diverges
        from the HTTP transport, which scopes them to the one request (see
        ``HTTPRequestParams.cookies``). A browser has no per-request
        ``Cookie`` header, and Playwright ignores an override of it.
        """
        params = request.request
        await page.set_extra_http_headers(
            merge_headers(self._default_headers, params.headers)
        )
        if params.cookies:
            # The dict literals are valid SetCookieParams; typed loosely
            # because playwright's public API does not export the TypedDict.
            cookies: list[Any] = [
                {"name": name, "value": value, "url": params.url}
                for name, value in params.cookies.items()
            ]
            await page.context.add_cookies(cookies)

    async def _execute_via_download(
        self, request: Request, page: Page
    ) -> tuple[Download, PlaywrightResponse | None] | FileArchiveStream:
        """Click the request's ``via`` target and return the resulting archive.

        A request with no ``via`` navigates straight to its URL instead, which
        must then be a plain GET. Normally the click (or navigation) triggers a browser ``download`` event and the
        :class:`Download` is returned with the response the browser reported
        for it (``None`` if it reported none). But Firefox/camoufox opens some files
        (PDFs) in its built-in viewer and *navigates* the page to the file
        instead of downloading it — no ``download`` event ever fires. In that
        case the navigation's response bytes are staged to a temp file and
        returned as a ready :class:`FileArchiveStream` instead. See
        :meth:`_await_download_or_inline`.

        The request's (else the transport's) timeout is the millisecond
        deadline on the element wait, the download wait and the click — the
        click's "wait for scheduled navigations" phase would otherwise use
        Playwright's own default, ignoring a longer user timeout.
        """
        params = request.request
        timeout_ms = self._timeout_ms(params.timeout)
        click_kwargs: dict[str, Any] = {"timeout": timeout_ms}

        if isinstance(request.via, ViaLink):
            element = await wait_for_required_element(
                page,
                selector_for_playwright(request.via.selector),
                request.request.url,
                timeout_ms=timeout_ms,
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
            submit = await prepare_form_submit(
                page, request.via, request.request.url, timeout_ms=timeout_ms
            )
            return await self._await_download_or_inline(
                page, submit, timeout_ms=timeout_ms
            )

        if request.via is None:
            if (
                params.method is not HttpMethod.GET
                or params.data
                or params.json
            ):
                raise ScraperConfigError(
                    f"archive request for {params.url} has no via, so the "
                    "browser can only navigate to it: it must be a GET "
                    f"with no body (got {params.method.value})"
                )

            async def _navigate() -> None:
                try:
                    await page.goto(
                        params.url, wait_until="commit", timeout=timeout_ms
                    )
                except PlaywrightError as exc:
                    # A navigation that turns into a download is aborted by
                    # the browser; the download event carries on without it.
                    if not _is_download_abort(exc):
                        raise

            return await self._await_download_or_inline(
                page, _navigate, timeout_ms=timeout_ms
            )

        raise ValueError(
            f"Archive download requires ViaLink, ViaFormSubmit or no via, "
            f"got {type(request.via)}"
        )

    async def _await_download_or_inline(
        self,
        page: Page,
        trigger: Callable[[], Awaitable[None]],
        *,
        timeout_ms: float,
    ) -> tuple[Download, PlaywrightResponse | None] | FileArchiveStream:
        """Fire ``trigger`` and return the download — or inline-rendered bytes.

        A real download fires a ``download`` event while the page stays put; an
        inline render (Firefox opening a PDF in pdf.js) instead *navigates* the
        main frame to the file and fires no download event. We register both a
        ``download`` waiter and a main-frame ``framenavigated`` waiter before
        the trigger, then race them:

        * download event first  -> return the :class:`Download` plus the
          response observed for its URL (normal path);
        * navigation first       -> the browser rendered the file inline, so we
          stage the captured response's bytes to a temp file and return them
          as a ready :class:`FileArchiveStream`.

        A response listener runs throughout so the inline branch has the file's
        own response body to hand. If neither a download nor a usable inline
        body materializes, a :class:`TransientException` is raised so the worker
        retries (mirroring the old ``expect_download`` timeout).
        """
        wait_kwargs: dict[str, Any] = {"timeout": timeout_ms}

        # Every response, by URL, so a download's own status + headers can be
        # looked up (Playwright's Download carries neither); and the
        # file-ish ones in order, so an inline render can be reconstructed
        # from the navigation's own response body (the last one wins — a
        # redirect chain ends on the real file).
        observed: dict[str, PlaywrightResponse] = {}
        captured: list[PlaywrightResponse] = []

        def _on_response(resp: PlaywrightResponse) -> None:
            observed[resp.url] = resp
            if _is_inline_file(resp, page.main_frame):
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

            timeout_s = timeout_ms / 1000.0
            try:
                result = await asyncio.wait_for(outcome, timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                raise TransientException(
                    "Archive trigger produced neither a download nor an "
                    "inline render before the timeout",
                    url=page.url,
                    kind=TransientKind.ARCHIVE,
                ) from exc

            if result is not None:
                return result, observed.get(result.url)

            # The page navigated instead of downloading — an inline render.
            # Let the file's response settle, then take its bytes.
            with contextlib.suppress(PlaywrightTimeoutError):
                await page.wait_for_load_state("load", **wait_kwargs)
            inline = await self._stage_inline_stream(captured)
            if inline is not None:
                return inline

            raise TransientException(
                "Archive trigger navigated inline but no readable file "
                "response was captured",
                url=page.url,
                kind=TransientKind.ARCHIVE,
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
            headers=dict(resp.headers),
            url=resp.url,
            file_path=await asyncio.to_thread(_write_temp),
        )

    def _timeout_ms(self, timeout: TimeoutType) -> float:
        """A request's timeout (requests-style seconds; a (connect, read)
        tuple uses the read element), else this transport's, as Playwright
        milliseconds. Never ``None``: every wait gets an explicit deadline
        rather than falling through to Playwright's own default."""
        if timeout is None:
            return self._timeout * 1000.0
        if isinstance(timeout, tuple):
            timeout = timeout[1]
        return float(timeout) * 1000.0

    @override
    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Close ``stream``; delete the temp file a Playwright download staged."""
        await stream.aclose()
        if isinstance(stream, FileArchiveStream):
            with contextlib.suppress(FileNotFoundError, OSError):
                await asyncio.to_thread(os.unlink, stream.file_path)

    def _browser_data_root(self) -> Path | None:
        """This run's browser-data directory: ``<run db>.browser-data``.

        Next to the run database, so a resume reuses the profile and it goes
        away with the database; two runs never share one. ``None`` (the
        engine's scratch directory) when there is no file-backed run db.
        """
        if self._db is None:
            return None
        database = self._db.engine.url.database
        if not database or database == ":memory:":
            return None
        return Path(f"{database}.browser-data")

    def _build_camoufox_engine(
        self: PlaywrightTransport, _browser_type: str
    ) -> BrowserEngine:
        return CamoufoxEngine(
            scraper=self._scraper,
            browser_profile=self._browser_profile,
            headless=self._headless,
            proxy=self._proxy,
            # Disable mouse humanization — it can stall clicks indefinitely.
            humanize=False,
            user_data_root=self._browser_data_root(),
        )

    def _build_playwright_engine(
        self: PlaywrightTransport, browser_type: str
    ) -> BrowserEngine:
        return PlaywrightEngine(
            scraper=self._scraper,
            browser_profile=self._browser_profile,
            browser_type=browser_type,
            headless=self._headless,
            viewport=_VIEWPORT,
            proxy=self._proxy,
            user_data_root=self._browser_data_root(),
        )

    #: Engine key (from
    #: :data:`~jkent.driver.unified_driver.requirements.BROWSERS`) → builder.
    _ENGINE_BUILDERS: ClassVar[
        dict[str, Callable[[PlaywrightTransport, str], BrowserEngine]]
    ] = {
        "camoufox": _build_camoufox_engine,
        "playwright": _build_playwright_engine,
    }

    def _build_engine(self) -> BrowserEngine:
        """Pick the engine per the scraper's requirements.

        :meth:`~jkent.driver.unified_driver.requirements.ResolvedRequirements.engine_for`
        makes the choice — the one selection-precedence site — and
        :attr:`_ENGINE_BUILDERS` maps its answer to a constructor. A
        ``browser_profile.browser_type`` still overrides the flavor inside
        the engine.
        """
        engine, browser_type = self._requirements.engine_for(
            self._browser_type
        )
        return self._ENGINE_BUILDERS[engine](self, browser_type)

    def _require_context(self) -> BrowserContext:
        if self._context is None:
            raise RuntimeError("PlaywrightTransport used before open()")
        return self._context

    # --- Cookie persistence ---------------------------------------------

    @override
    async def export_cookies(self) -> str | None:
        """Dump the live context's cookies as JSON, or None if unavailable.

        Returns None (not raises) when the context is gone or already closed —
        e.g. after a Ctrl-C tore the browser down before this best-effort save.
        """
        if self._context is None:
            return None
        try:
            return dump_json(await self._context.cookies())
        except PlaywrightError:
            return None

    @override
    async def import_cookies(self, cookies_json: str) -> None:
        """Apply previously-exported cookies to the live context."""
        cookies = json.loads(cookies_json)
        if cookies:
            await self._require_context().add_cookies(cookies)

    # --- Crash recovery (transport-internal) ------------------------------

    @property
    def generation(self) -> int:
        """Monotonic count of how many times the engine was (re)built."""
        return self._generation

    def should_restart(self, exc: BaseException) -> bool:
        """Whether ``exc`` means the browser connection died.

        The engine's predicate (:meth:`BrowserEngine.should_restart`); the
        base one before ``open`` has picked an engine.
        """
        return (self._engine or BrowserEngine).should_restart(exc)

    async def restart(self, seen_generation: int) -> None:
        """Rebuild the engine once, single-flight under the generation guard.

        If ``seen_generation`` no longer matches the current generation a
        racing caller already rebuilt — this is a no-op. Otherwise the
        poisoned handles are closed and dropped, the context rebuilt, and the
        generation advanced.
        """
        async with self._restart_lock:
            if seen_generation != self._generation:
                return  # another caller already rebuilt this generation
            # Drop every handle: they all reference the dead browser. Close
            # them too, best-effort — a rebuild that then fails (a persistent
            # context cannot restart) leaves that browser up, pages and all.
            handles = list(self._handles.values())
            self._handles.clear()
            for handle in handles:
                with contextlib.suppress(Exception):
                    await handle.close()
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
                "Browser connection lost; no engine attached",
                kind=TransientKind.BROWSER_CRASH,
            )
        self._context = await self._engine.restart_context()
