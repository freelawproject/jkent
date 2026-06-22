"""The per-run harness: one event loop, one set of transports, the oracle.

Launching a browser per Hypothesis example is too slow (the repo's other
generative rigs avoid browsers for exactly this reason). So this harness opens
everything *once* — the echo server, an in-process DB, the HTTP transport, the
chromium + camoufox transports, and an independent vanilla-browser oracle — and
reuses them across every example, driven from a single persistent loop via
``loop.run_until_complete`` (the ``test_transport_registry_machine`` pattern,
since Hypothesis does not compose with pytest-asyncio).

:meth:`Harness.run_case` is the heart: it turns one :class:`FormCase` into a
single ``Form.submit(...)`` request (the production path: parse the rendered
HTML with ``find_form``, then ``submit`` with the case's overrides), then drives
that one request through every transport plus the browser oracle and returns
each side's canonical submission for comparison.

Each transport that can't launch in the environment is simply skipped (recorded
in :attr:`available`); if the oracle browser itself can't launch there is no
ground truth and the bound tests skip entirely.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode, urlparse

import sqlalchemy as sa
from aiohttp import web
from lxml import html as lxml_html

from jkent.common.lxml_page_element import LxmlPageElement
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
)
from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.transport.camoufox_transport import (
    CamoufoxTransport,
)
from jkent.driver.unified_driver.transport.httpx_transport import (
    HttpxTransport,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    PlaywrightTransport,
)
from tests.form_conformance.echo_server import (
    ORACLE,
    Canonical,
    create_app,
    extract_echo,
)

if TYPE_CHECKING:
    import asyncio

    from tests.form_conformance.model import FormCase

# Base URL used only to resolve the parsed form's base; the form action is
# absolute (the echo endpoint), so this is cosmetic.
_PARSE_BASE = "https://staged.example"


class _Scraper(BaseScraper[None]):
    """Minimal scraper: no requirements -> plain chromium/camoufox engines."""

    def get_entry(self):  # type: ignore[no-untyped-def]
        return iter(())


class Harness:
    """Owns the loop-bound resources and runs one :class:`FormCase` at a time."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._counter = itertools.count(1)
        self._runner: web.AppRunner | None = None
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self._engine: Any = None
        self._oracle_pw: Any = None
        self._oracle_browser: Any = None
        self._oracle_context: Any = None

        self.base_url: str = ""
        self.app: web.Application | None = None
        self.sql: SQLManager | None = None
        self.sf: Any = None
        self.httpx: HttpxTransport | None = None
        self.chromium: PlaywrightTransport | None = None
        self.firefox: CamoufoxTransport | None = None
        # Transport name -> whether it launched here.
        self.available: dict[str, bool] = {}

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def oracle_available(self) -> bool:
        return self._oracle_context is not None

    @property
    def transport_names(self) -> list[str]:
        """Live transports to compare against the oracle, in display order."""
        names = ["httpx"]
        if self.available.get("chromium"):
            names.append("chromium")
        if self.available.get("firefox"):
            names.append("firefox")
        return names

    # --- lifecycle --------------------------------------------------------

    async def open(self) -> None:
        # Echo + oracle-form server. Use locals so the runner is non-optional
        # where the runner API needs it; the attributes are for teardown.
        self.app = create_app()
        runner = web.AppRunner(self.app)
        self._runner = runner
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][0], runner.addresses[0][1]
        self.base_url = f"http://{host}:{port}"

        # DB (needed by the Playwright transports to stage the parent form).
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "rig.db"
        self._engine, self.sf = await init_database(db_path)
        self.sql = SQLManager(self._engine, self.sf)

        # HTTP transport — always available.
        self.httpx = HttpxTransport()
        await self.httpx.open()
        self.available["httpx"] = True

        # Browser transports — best-effort.
        self.chromium = PlaywrightTransport(
            _Scraper(), headless=True, browser_type="chromium", db=self.sql
        )
        self.available["chromium"] = await _try_open(self.chromium)

        self.firefox = CamoufoxTransport(
            _Scraper(), headless=True, db=self.sql
        )
        self.available["firefox"] = await _try_open(self.firefox)

        # Independent vanilla-browser oracle (raw Playwright chromium, no jkent
        # form code in the path).
        await self._open_oracle()

    async def _open_oracle(self) -> None:
        try:
            from playwright.async_api import async_playwright

            self._oracle_pw = await async_playwright().start()
            self._oracle_browser = await self._oracle_pw.chromium.launch(
                headless=True
            )
            self._oracle_context = await self._oracle_browser.new_context()
        except Exception:
            await self._close_oracle()

    async def _close_oracle(self) -> None:
        for closer in (
            getattr(self._oracle_context, "close", None),
            getattr(self._oracle_browser, "close", None),
            getattr(self._oracle_pw, "stop", None),
        ):
            if closer is not None:
                try:
                    await closer()
                except Exception:
                    pass
        self._oracle_context = None
        self._oracle_browser = None
        self._oracle_pw = None

    async def aclose(self) -> None:
        await self._close_oracle()
        for t in (self.httpx, self.chromium, self.firefox):
            if t is not None:
                try:
                    await t.aclose()
                except Exception:
                    pass
        if self._engine is not None:
            await self._engine.dispose()
        if self._runner is not None:
            await self._runner.cleanup()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    # --- execution --------------------------------------------------------

    async def run_case(
        self, case: FormCase
    ) -> tuple[Canonical, list[tuple[str, Canonical | str]]]:
        """Fan one case across the oracle + every live transport.

        Returns the oracle's canonical submission and a list of
        ``(transport_name, canonical-or-error-string)`` for each live transport.
        """
        action = f"{self.base_url}/echo"
        rendered = case.rendered_html(action)
        request = self._build_request(case, rendered)

        oracle = await self._submit_oracle(case, action)

        results: list[tuple[str, Canonical | str]] = []
        results.append(
            ("httpx", await self._guard(self._submit_httpx(request)))
        )
        if self.available.get("chromium"):
            results.append(
                (
                    "chromium",
                    await self._guard(
                        self._submit_playwright(
                            self.chromium, rendered, request
                        )
                    ),
                )
            )
        if self.available.get("firefox"):
            results.append(
                (
                    "firefox",
                    await self._guard(
                        self._submit_playwright(
                            self.firefox, rendered, request
                        )
                    ),
                )
            )
        return oracle, results

    @staticmethod
    async def _guard(coro: Any) -> Canonical | str:
        """Run a submission, capturing failures as a readable marker string."""
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 - surfaced in the diff
            return f"<error: {type(exc).__name__}: {exc}>"

    def _build_request(self, case: FormCase, rendered: str) -> Request:
        """The production path: parse the rendered form, then ``Form.submit``."""
        doc = lxml_html.fromstring(rendered)
        page = LxmlPageElement(doc, _PARSE_BASE)
        form = page.find_form(Selector.CSS("#f"), "generated form")
        return form.submit(
            data=case.overrides(),
            submit_selector=case.submit_selector,
            continuation="parse",
        )

    async def _submit_oracle(self, case: FormCase, action: str) -> Canonical:
        # Bind to locals: attribute narrowing is invalidated across awaits.
        app = self.app
        context = self._oracle_context
        assert app is not None
        assert context is not None
        app[ORACLE]["html"] = case.oracle_html(action)
        page = await context.new_page()
        try:
            await page.goto(
                f"{self.base_url}/oracle", wait_until="domcontentloaded"
            )
            for selector, value in case.native_fills():
                await page.fill(selector, value)
            async with page.expect_navigation():
                await page.click(case.submit_selector)
            content = await page.content()
        finally:
            await page.close()
        return extract_echo(content)

    async def _submit_httpx(self, request: Request) -> Canonical:
        httpx = self.httpx
        assert httpx is not None
        folded = _fold_get_params(request)
        handle = await httpx.acquire(0)
        from jkent.driver.unified_driver.transport import QueuedRequest

        resp = await httpx.resolve(
            handle,
            QueuedRequest(request=folded, request_id=next(self._counter)),
        )
        return extract_echo(resp.text)

    async def _submit_playwright(
        self, transport: Any, rendered: str, request: Request
    ) -> Canonical:
        from jkent.driver.unified_driver.transport import QueuedRequest

        n = next(self._counter)
        parent_url = f"{_PARSE_BASE}/form/{n}"
        parent_id = await self._insert_parent(parent_url, rendered.encode())
        child_id = await self._insert_child(request.request.url)
        handle = await transport.acquire(0)
        queued = QueuedRequest(
            request=request, request_id=child_id, parent_request_id=parent_id
        )
        resp = await transport.resolve(handle, queued)
        return extract_echo(resp.text)

    # --- DB row helpers (mirror the Playwright transport tests) -----------

    async def _insert_parent(self, url: str, body: bytes) -> int:
        compressed = compress(body)
        qc = next(self._counter)
        async with self.sf() as session:
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location, response_status_code,
                        response_url, response_headers_json, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES ('completed', 9, :qc, 'GET', :url, 'parse', '', 200,
                        :url, NULL, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "url": url,
                    "qc": qc,
                    "compressed": compressed,
                    "osize": len(body),
                    "csize": len(compressed),
                },
            )
            await session.commit()
            return (
                await session.execute(
                    sa.text(
                        "SELECT id FROM requests WHERE queue_counter = :qc"
                    ),
                    {"qc": qc},
                )
            ).scalar_one()

    async def _insert_child(self, url: str) -> int:
        qc = next(self._counter)
        async with self.sf() as session:
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location)
                    VALUES ('in_progress', 9, :qc, 'GET', :url, 'parse', '')
                    """
                ),
                {"qc": qc, "url": url},
            )
            await session.commit()
            return (
                await session.execute(
                    sa.text(
                        "SELECT id FROM requests WHERE queue_counter = :qc"
                    ),
                    {"qc": qc},
                )
            ).scalar_one()


async def _try_open(transport: Any) -> bool:
    """Open a transport, returning whether it launched (closing on failure)."""
    try:
        await transport.open()
    except Exception:
        try:
            await transport.aclose()
        except Exception:
            pass
        return False
    return True


def _fold_get_params(request: Request) -> Request:
    """Fold GET ``params`` into the URL exactly as ``DBQueue`` does on enqueue.

    ``HttpxTransport`` deliberately ignores ``HTTPRequestParams.params`` — the
    queue is expected to have baked the query into the URL upstream. So to test
    the HTTP transport faithfully we reproduce that fold here verbatim (see
    ``jkent/driver/database_engine/queue.py`` ~line 130). POST forms keep their
    ``data`` body untouched.
    """
    p = request.request
    if p.method == HttpMethod.GET and p.params:
        url = p.url
        parsed = urlparse(url)
        if isinstance(p.params, bytes):
            query = p.params.decode()
        else:
            query = urlencode(p.params, doseq=True)
        if parsed.query:
            query = parsed.query + "&" + query
        # geturl() (vs urlunparse) is typed -> str, avoiding an AnyStr ambiguity.
        url = parsed._replace(query=query).geturl()
        return Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=url),
            continuation="parse",
        )
    return Request(
        request=HTTPRequestParams(
            method=p.method, url=p.url, data=p.data, headers=p.headers
        ),
        continuation="parse",
    )
