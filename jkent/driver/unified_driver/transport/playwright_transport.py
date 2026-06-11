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

Like ``ReplayTransport`` reuses ``SourceIndex``, this reuses the persistent
driver's :class:`SQLManager` for its execution-time DB needs: reading a
parent's cached response to stage a forked tab, and persisting captured
incidental sub-requests against the navigating request's row id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import TYPE_CHECKING, Any

from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.exceptions import ScraperConfigError, TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink
from jkent.data_types import (
    DriverRequirement,
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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from playwright.async_api import (
        BrowserContext,
        Download,
        ElementHandle,
        Page,
    )

    from jkent.data_types import BaseScraper, Request
    from jkent.driver.browser_engine.browser_profile import BrowserProfile
    from jkent.driver.database_engine.sql_manager import SQLManager
    from jkent.driver.unified_driver.transport import (
        ArchiveStream,
        AwaitCondition,
        QueuedRequest,
    )


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


class _PlaywrightArchiveStream:
    """An ``ArchiveStream`` over a Playwright download staged to a temp file.

    Playwright only surfaces a download as a local file, so ``resolve_archive``
    stages it and this streams the staged file in chunks; ``finish_archiving``
    deletes ``file_path``.
    """

    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        url: str,
        file_path: str,
        chunk_size: int = 65536,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self.file_path = file_path
        self._chunk_size = chunk_size

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._chunks()

    async def _chunks(self) -> AsyncIterator[bytes]:
        with open(self.file_path, "rb") as handle:
            while True:
                chunk = await asyncio.to_thread(handle.read, self._chunk_size)
                if not chunk:
                    break
                yield chunk


class PlaywrightTransport:
    """A :class:`~jkent.driver.unified_driver.transport.Transport` over a browser."""

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
        if handle is None:
            page = await self._new_page()
            handle = WorkerPage(page, self._excluded_resource_types)
            self._handles[worker_id] = handle
        else:
            await handle.reset_for_reuse()
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
        handle.current_parent_request_id = queued.request_id

        # Headers from a prior request on this reused page must not leak.
        await page.set_extra_http_headers(request.request.headers or {})

        # Capture a navigation/await timeout but keep going to snapshot the
        # (partial) DOM below, for debugging.
        timeout_error: PlaywrightTimeoutError | None = None
        try:
            # Parent-tab staging is only for via (click/form) requests reached
            # FROM a parent page; a plain child request that merely records a
            # parent for lineage must navigate to its OWN url (matches the old
            # driver's `parent_request_id and request.via is not None` guard).
            via = getattr(request, "via", None)
            if queued.parent_request_id is not None and via is not None:
                staged = await self._stage_parent_tab(
                    page, queued.parent_request_id
                )
                if staged:
                    # Parent page is loaded from cache; click/submit the via to
                    # navigate through to the child, then snapshot the child DOM.
                    await self._execute_via_navigation(request, page)
                else:
                    # Parent has no stored response — navigate to the child url.
                    await page.goto(
                        request.request.url, wait_until="domcontentloaded"
                    )
            else:
                await page.goto(
                    request.request.url, wait_until="domcontentloaded"
                )

            if self._interstitial_handlers:
                # Race the handlers' waitlists against the scraper's await
                # conditions; an interstitial win means the handler interacts
                # with the page first, then the scraper's own conditions are
                # processed.
                winner = await self._race_await_lists(page, await_conditions)
                if winner is not None:
                    await winner.navigate_through(page)
                    for condition in await_conditions:
                        await self._apply_await_condition(page, condition)
            else:
                for condition in await_conditions:
                    await self._apply_await_condition(page, condition)
        except PlaywrightTimeoutError as exc:
            timeout_error = exc

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
            status_code=200,
            url=page_url,
            content=html_content.encode("utf-8"),
            text=html_content,
            headers={"content-type": "text/html; charset=utf-8"},
            request=request,
        )

        # Persist incidentals against this request's row id. Snapshot the
        # list since on_response callbacks may still be appending.
        for incidental in list(handle.incidental_requests):
            await self._db.insert_incidental_request(  # type: ignore
                parent_request_id=queued.request_id, **incidental
            )

        if timeout_error is not None:
            # Carry the partial DOM to the worker, which stores it before the
            # retry so the failed attempt is inspectable.
            raise ResolveTimeout(
                f"Playwright timeout during resolve: {timeout_error}",
                debug_response=response,
            ) from timeout_error

        return response

    async def _stage_parent_tab(
        self, page: Page, parent_request_id: int
    ) -> bool:
        """Serve the parent's cached response into the tab via route intercept."""
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
        if "content-type" not in {k.lower() for k in headers}:
            headers["content-type"] = "text/html; charset=utf-8"
        status = response_status_code or 200

        async def _intercept_handler(route: Any) -> None:
            await route.fulfill(status=status, headers=headers, body=body)

        await page.route(response_url, _intercept_handler)
        await page.goto(response_url, wait_until="domcontentloaded")
        await page.unroute(response_url, _intercept_handler)
        return True

    async def _race_await_lists(
        self,
        page: Page,
        scraper_await_list: Sequence[AwaitCondition],
    ) -> InterstitialHandler | None:
        """Race scraper waitlist against interstitial handler waitlists.

        Each group's conditions are awaited sequentially (conjunction).
        The first group to fully resolve wins; losing tasks are cancelled.

        Returns:
            The winning ``InterstitialHandler``, or ``None`` if the
            scraper's own await conditions completed first.
        """

        async def _run_group(conditions: Sequence[AwaitCondition]) -> None:
            for condition in conditions:
                await self._apply_await_condition(page, condition)

        scraper_task = asyncio.create_task(
            _run_group(scraper_await_list), name="scraper"
        )
        handler_tasks: dict[asyncio.Task[None], InterstitialHandler] = {}
        for handler in self._interstitial_handlers:
            task = asyncio.create_task(
                _run_group(handler.waitlist()),
                name=type(handler).__name__,
            )
            handler_tasks[task] = handler

        all_tasks = {scraper_task} | set(handler_tasks.keys())
        try:
            done, _pending = await asyncio.wait(
                all_tasks, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in all_tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*all_tasks, return_exceptions=True)

        winner = next(iter(done))
        winner.result()  # re-raise if the winner raised

        if winner is scraper_task:
            return None
        return handler_tasks[winner]

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
    ) -> _PlaywrightArchiveStream:
        """The raw archive-download path."""
        request = queued.request
        page = handle.page
        handle.clear_request_state()
        handle.current_parent_request_id = queued.request_id

        if queued.parent_request_id is not None:
            staged = await self._stage_parent_tab(
                page, queued.parent_request_id
            )
            if not staged:
                raise TransientException(
                    "Archive download: parent has no stored response to stage"
                )

        download = await self._execute_via_download(request, page)

        # Download.path() has no native timeout: a server that starts the
        # response then trickles/stalls would hang forever. Honor the
        # request's timeout (requests-style seconds; tuple -> read element)
        # as a hard deadline, re-mapping an overrun to a transient.
        download_timeout = request.request.timeout
        if isinstance(download_timeout, tuple):
            download_timeout = download_timeout[1]
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

        return _PlaywrightArchiveStream(
            status_code=200,
            headers={},
            url=download.url or request.request.url,
            file_path=str(download_path),
        )

    async def _execute_via_download(
        self, request: Request, page: Page
    ) -> Download:
        """Click the request's ``via`` target, expecting a download (ported).

        Honors ``request.request.timeout`` (seconds; tuple -> read element) as
        a millisecond deadline on both ``expect_download`` and the click — the
        click's "wait for scheduled navigations" phase otherwise uses
        Playwright's 30s default, ignoring a longer user timeout.
        """
        params = request.request
        expect_kwargs: dict[str, Any] = {}
        click_kwargs: dict[str, Any] = {}
        if params.timeout is not None:
            timeout_ms = (
                float(params.timeout[1]) * 1000.0
                if isinstance(params.timeout, tuple)
                else float(params.timeout) * 1000.0
            )
            expect_kwargs["timeout"] = timeout_ms
            click_kwargs["timeout"] = timeout_ms

        if isinstance(request.via, ViaLink):
            element = await self._wait_for_required_element(
                page, request.via.selector.value, request.request.url
            )
            async with page.expect_download(**expect_kwargs) as download_info:  # type: ignore
                await element.click(**click_kwargs)
            return await download_info.value

        if isinstance(request.via, ViaFormSubmit):
            form_via = request.via
            form = await self._wait_for_required_element(
                page, form_via.form_selector.value, request.request.url
            )
            await self._fill_form_fields(form, form_via.field_data)
            submit_selector = (
                form_via.submit_selector
                or 'button[type="submit"], input[type="submit"]'
            )
            submit = await form.query_selector(submit_selector)
            if submit is None:
                raise TransientException(
                    f"Submit selector not found: {submit_selector}"
                )
            async with page.expect_download(**expect_kwargs) as download_info:
                if await submit.is_visible():
                    await submit.click(**click_kwargs)
                else:
                    # The real submit control is non-interactable — e.g. its
                    # element was swallowed by malformed HTML (an unclosed
                    # <style> turning the button into raw text) so
                    # _fill_form_fields synthesized a hidden input carrying its
                    # name/value in place. A hidden element can't be clicked,
                    # but its name=value is already a field on the form, so a
                    # bare form.submit() POSTs the same data the click would —
                    # mirroring the __EVENTTARGET path in _execute_via_navigation.
                    await form.evaluate("(f) => f.submit()")
            return await download_info.value

        raise ValueError(
            f"Archive download requires ViaLink or ViaFormSubmit, "
            f"got {type(request.via)}"
        )

    async def _execute_via_navigation(
        self, request: Request, page: Page
    ) -> None:
        """Click/submit the via element and wait for the resulting navigation.

        The navigation counterpart of :meth:`_execute_via_download` (it expects
        a navigation, not a download), used after the parent tab is staged.
        A missing element or a navigation timeout/abort is transient (the
        request retries); honors ``HTTPRequestParams.timeout`` as a deadline.
        """
        params = request.request
        via = request.via
        expect_kwargs: dict[str, Any] = {}
        click_kwargs: dict[str, Any] = {}
        if params.timeout is not None:
            timeout_ms = (
                float(params.timeout[1]) * 1000.0
                if isinstance(params.timeout, tuple)
                else float(params.timeout) * 1000.0
            )
            expect_kwargs["timeout"] = timeout_ms
            click_kwargs["timeout"] = timeout_ms

        try:
            if isinstance(via, ViaLink):
                element = await self._wait_for_required_element(
                    page, via.selector.value, params.url
                )
                async with page.expect_navigation(**expect_kwargs):  # type: ignore
                    await element.click(**click_kwargs)
            elif isinstance(via, ViaFormSubmit):
                form = await self._wait_for_required_element(
                    page, via.form_selector.value, params.url
                )
                await self._fill_form_fields(form, via.field_data)
                # Branch priority is load-bearing: an explicit submit_selector
                # wins over __EVENTTARGET. The page's hidden
                # __EVENTTARGET input is harvested into field_data (empty)
                # during form-field collection, so keying on its mere presence
                # wrongly routes a button submit (e.g. a grid-row Select) to a
                # bare form.submit() with an empty event target — the server
                # then re-renders the same page instead of navigating.
                if via.submit_selector:
                    submit = await form.query_selector(via.submit_selector)
                    if submit is None:
                        raise TransientException(
                            f"Submit selector not found: {via.submit_selector}"
                        )
                    # requestSubmit(submitter) (not a bare click) so the
                    # button's name/value is reliably in the POST — ASP.NET
                    # uses it to identify the event source (which row's Select).
                    async with page.expect_navigation(**expect_kwargs):
                        await submit.evaluate(
                            "(btn) => btn.form.requestSubmit(btn)"
                        )
                elif "__EVENTTARGET" in via.field_data:
                    # ASP.NET __doPostBack: __EVENTTARGET is set as a hidden
                    # field, so a raw form.submit() fires that postback.
                    async with page.expect_navigation(**expect_kwargs):
                        await form.evaluate("(form) => form.submit()")
                else:
                    submit = await form.query_selector(
                        'button[type="submit"], input[type="submit"]'
                    )
                    if submit is None:
                        raise TransientException(
                            f"No submit element in form {via.form_selector}"
                        )
                    async with page.expect_navigation(**expect_kwargs):
                        await submit.evaluate(
                            "(btn) => btn.form.requestSubmit(btn)"
                        )
            else:
                raise ValueError(
                    f"via-navigation requires ViaLink or ViaFormSubmit, "
                    f"got {type(via)}"
                )
        except PlaywrightTimeoutError as exc:
            raise TransientException(
                f"Navigation timeout: {params.url}"
            ) from exc
        except PlaywrightError as exc:
            if "NS_ERROR_ABORT" in str(exc):
                raise TransientException(
                    f"Navigation aborted: {params.url}"
                ) from exc
            raise

    @staticmethod
    async def _wait_for_required_element(
        page: Page, selector: str, request_url: str
    ) -> ElementHandle:
        """Wait for a required selector; a miss/timeout is a transient."""
        try:
            element = await page.wait_for_selector(selector, timeout=5000)
        except PlaywrightTimeoutError as exc:
            raise TransientException(
                f"Selector timeout: {selector} ({request_url})"
            ) from exc
        if element is None:
            raise TransientException(
                f"Selector not found: {selector} ({request_url})"
            )
        return element

    @staticmethod
    async def _fill_form_fields(
        form: ElementHandle, field_data: dict[str, str | list[str]]
    ) -> None:
        """Populate form fields by name based on tag/type/visibility (ported).

        ``fill`` only works on visible, editable inputs, so selects use
        ``select_option``, radios/checkboxes set ``checked`` on the matching
        value, and hidden/invisible inputs (e.g. ASP.NET ``__VIEWSTATE`` or
        Telerik 1px parents) assign ``.value`` directly via JS.

        A list value means repeated keys (checkbox groups, multi-selects): each
        member selects/checks the option with the matching value, mirroring what
        the browser POSTs as repeated names.
        """
        for name, value in field_data.items():
            if isinstance(value, list):
                await PlaywrightTransport._fill_repeated_field(
                    form, name, value
                )
                continue
            field = await form.query_selector(f'[name="{name}"]')
            if field is None:
                # No rendered control for this name. ViaFormSubmit can carry
                # fields the form never showed (the merged overrides a scraper
                # passed to ``Form.submit``); inject a hidden input so the
                # browser submits them too, instead of silently dropping them.
                # The HTTP transport sends these verbatim, so this keeps the
                # browser path in step with it.
                await PlaywrightTransport._append_hidden_input(
                    form, name, str(value)
                )
                continue
            tag = await field.evaluate("el => el.tagName.toLowerCase()")
            input_type = await field.get_attribute("type")
            str_value = str(value)

            if tag == "select":
                await field.select_option(value=str_value)
            elif input_type == "radio":
                radio = await form.query_selector(
                    f'[name="{name}"][value="{str_value}"]'
                )
                if radio is not None:
                    await radio.evaluate("(el) => el.checked = true")
            elif input_type == "checkbox":
                checkbox = await form.query_selector(
                    f'[name="{name}"][value="{str_value}"]'
                )
                if checkbox is not None:
                    await checkbox.evaluate("(el) => el.checked = true")
                else:
                    await field.evaluate(
                        "(el, val) => el.checked = !!val", str_value
                    )
            elif input_type in ("hidden", "submit") or not (
                await field.is_visible()
            ):
                await field.evaluate("(el, val) => el.value = val", str_value)
            else:
                await field.fill(str_value)

    @staticmethod
    async def _fill_repeated_field(
        form: ElementHandle, name: str, values: list[str]
    ) -> None:
        """Replay a repeated-key field (checkbox group or multi-select).

        A ``<select multiple>`` selects all matching options at once; a checkbox
        group checks each box whose value is in ``values``. The fallback covers
        repeated text/hidden inputs (rare), assigning each value positionally to
        the matching ``name=`` elements in document order.
        """
        str_values = [str(v) for v in values]
        field = await form.query_selector(f'[name="{name}"]')
        if field is None:
            # No rendered control: inject one hidden input per value so a
            # repeated key absent from the DOM still reaches the server as
            # repeated names (see ``_fill_form_fields`` for the rationale).
            for str_value in str_values:
                await PlaywrightTransport._append_hidden_input(
                    form, name, str_value
                )
            return
        tag = await field.evaluate("el => el.tagName.toLowerCase()")
        input_type = await field.get_attribute("type")

        if tag == "select":
            await field.select_option(value=str_values)
        elif input_type in ("radio", "checkbox"):
            for str_value in str_values:
                box = await form.query_selector(
                    f'[name="{name}"][value="{str_value}"]'
                )
                if box is not None:
                    await box.evaluate("(el) => el.checked = true")
        else:
            elements = await form.query_selector_all(f'[name="{name}"]')
            for element, str_value in zip(elements, str_values):
                await element.evaluate(
                    "(el, val) => el.value = val", str_value
                )

    @staticmethod
    async def _append_hidden_input(
        form: ElementHandle, name: str, value: str
    ) -> None:
        """Append a ``<input type=hidden name=value>`` to ``form``.

        Used when ``field_data`` carries a name the rendered form never showed,
        so the submitted request matches what the HTTP transport sends. Name and
        value are passed as JS arguments (not interpolated), so any characters
        are safe.
        """
        await form.evaluate(
            """(form, args) => {
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = args.name;
                input.value = args.value;
                form.appendChild(input);
            }""",
            {"name": name, "value": value},
        )

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Delete the staged temp file backing a Playwright download stream."""
        if isinstance(stream, _PlaywrightArchiveStream):
            with contextlib.suppress(FileNotFoundError, OSError):
                await asyncio.to_thread(os.unlink, stream.file_path)

    def _build_engine(self) -> BrowserEngine:
        """Pick the engine: camoufox for CFCAP scrapers, playwright otherwise.

        The playwright browser flavor is the explicit ``browser_type``
        constructor arg if given, else derived from the scraper's
        requirements: ``FF_ALIKE`` → firefox, otherwise chromium. A
        ``browser_profile.browser_type`` still overrides either inside the
        engine.
        """
        reqs = getattr(self._scraper, "driver_requirements", [])
        if (
            DriverRequirement.FF_ALIKE in reqs
            and DriverRequirement.CHROME_ALIKE in reqs
        ):
            raise ScraperConfigError(
                f"Scraper '{type(self._scraper).__name__}' declares both "
                "FF_ALIKE and CHROME_ALIKE driver requirements. These are "
                "mutually exclusive."
            )
        if DriverRequirement.CFCAP_HANDLER in reqs:
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
            if DriverRequirement.FF_ALIKE in reqs:
                browser_type = "firefox"
            else:
                browser_type = "chromium"
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
