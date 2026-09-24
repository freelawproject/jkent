"""Browser-engine abstract base class plus shared helpers.

A :class:`BrowserEngine` owns the engine-specific bits of launching a
browser and producing a Playwright :class:`BrowserContext`.  The driver
only ever sees the context; engine internals stay encapsulated."""

from __future__ import annotations

import abc
import logging
import shutil
import tempfile
from contextlib import (
    AbstractAsyncContextManager,
    asynccontextmanager,
    nullcontext,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from playwright.async_api import BrowserContext

    from jkent.data_types import BaseScraper
    from jkent.driver.browser_engine.browser_profile import BrowserProfile

logger = logging.getLogger(__name__)

#: Substrings of the error Playwright raises once the browser, or the page's
#: content process, is gone. See :meth:`BrowserEngine.should_restart`.
DEAD_CONNECTION_MESSAGES: Final = (
    "Connection closed",
    "Browser has been closed",
    "Target page, context or browser has been closed",
    "Page crashed",
    "Target crashed",
)


class BrowserEngine(abc.ABC):
    """Engine-specific browser factory + lifecycle manager.

    The lifecycle is a template shared by every engine: :meth:`acquire`
    enters the engine's run-scoped :meth:`_session` (a Playwright driver, a
    virtual display), opens a context, and tears both down on exit;
    :meth:`restart_context` closes the dead browser and opens a fresh one
    inside the same session. Subclasses supply only :meth:`_open_context`
    and :meth:`_close_context` (plus :meth:`_session` and
    :meth:`_check_restartable` where they differ).

    The driver receives a ``BrowserContext`` from :meth:`acquire` and uses
    it for the rest of its life; on a connection-dead event
    (:meth:`should_restart`) it asks the engine to rebuild via
    :meth:`restart_context`.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        browser_profile: BrowserProfile | None = None,
        user_data_root: Path | None = None,
    ) -> None:
        """Set up an engine; nothing launches until :meth:`acquire`.

        Args:
            scraper: The scraper the browser serves.
            browser_profile: Launch/context options; ``None`` is anonymous.
            user_data_root: This run's browser-data directory, under which a
                persistent context keeps its profile. ``None`` uses a scratch
                directory that is removed when :meth:`acquire` exits.
        """
        self._scraper = scraper
        self._browser_profile = browser_profile
        self._user_data_root = user_data_root
        self._scratch_root: Path | None = None
        self._acquired = False

    def _user_data_dir(self, profile_name: str) -> Path:
        """Where a persistent context for ``profile_name`` keeps its data."""
        root = self._user_data_root
        if root is None:
            if self._scratch_root is None:
                self._scratch_root = Path(
                    tempfile.mkdtemp(prefix="jkent-browser-data-")
                )
            root = self._scratch_root
        return resolve_user_data_dir(root, profile_name)

    @staticmethod
    def should_restart(exc: BaseException) -> bool:
        """Whether ``exc`` means the browser connection died.

        Pure and side-effect-free. Matches on message because Playwright
        rewraps the channel-layer transport error as a bare ``Exception``.

        A crashed *content process* ("Page crashed" / "Target crashed")
        counts too. Such a page is not ``is_closed()``, so a transport would
        otherwise hand it back and only discover the damage when resetting it
        fails — logging a generic reset warning that reads like a navigation
        race rather than a crash.
        """
        msg = str(exc)
        return any(dead in msg for dead in DEAD_CONNECTION_MESSAGES)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[BrowserContext]:
        """Yield a live ``BrowserContext``; tear it down on exit."""
        async with self._session():
            try:
                context = await self._open()
                self._acquired = True
                yield context
            finally:
                # Also after a failed open: a launched browser whose context
                # never came up must not leak.
                self._acquired = False
                await self._close_context()
                if self._scratch_root is not None:
                    shutil.rmtree(self._scratch_root, ignore_errors=True)
                    self._scratch_root = None

    async def restart_context(self) -> BrowserContext:
        """Tear down + rebuild the context after a crash.

        Only valid while :meth:`acquire` is active; calling it outside is a
        programming error (``RuntimeError``). Raises ``TransientException``
        when the engine cannot restart.
        """
        self._check_restartable()
        name = type(self).__name__
        logger.warning("%s: browser connection lost — restarting", name)
        await self._close_context()
        context = await self._open()
        logger.info("%s: browser restarted", name)
        return context

    async def _open(self) -> BrowserContext:
        context = await self._open_context()
        await apply_init_scripts(context, self._browser_profile)
        return context

    def _session(self) -> AbstractAsyncContextManager[None]:
        """Run-scoped resources that outlive restarts. Default: none."""
        return nullcontext()

    def _check_restartable(self) -> None:
        """Raise if :meth:`restart_context` can't run.

        Outside :meth:`acquire` that is a caller bug, not a crash: nothing
        died, so it must not be retried as a ``BROWSER_CRASH``.
        """
        if not self._acquired:
            raise RuntimeError(
                f"{type(self).__name__}.restart_context called outside "
                "acquire()"
            )

    @abc.abstractmethod
    async def _open_context(self) -> BrowserContext:
        """Launch the browser and return its context (init scripts excluded)."""

    @abc.abstractmethod
    async def _close_context(self) -> None:
        """Close the context and its browser, best-effort (see
        :func:`close_quietly`); the browser may already be dead."""


async def close_quietly(
    what: str, close: Callable[[], Awaitable[Any]]
) -> None:
    """Await ``close()``, logging rather than raising a failure.

    Teardown of a browser that may already be dead: the error is expected,
    but a swallowed one is also the only trace of a leaked process, so it is
    logged instead of dropped.
    """
    try:
        await close()
    except Exception:
        logger.warning("closing %s failed", what, exc_info=True)


def parse_proxy_for_playwright(proxy_url: str) -> dict[str, str]:
    """Convert a proxy URL into Playwright's ``proxy=`` dict.

    Playwright expects ``{"server": "<scheme>://<host>:<port>"}`` with
    credentials in separate ``username`` / ``password`` fields — not
    embedded in the URL.  Accepts any scheme Playwright supports
    (``http``, ``https``, ``socks4``, ``socks5``).
    """
    parts = urlsplit(proxy_url)
    if not parts.scheme or not parts.hostname:
        raise ValueError(f"Invalid proxy URL: {proxy_url!r}")

    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port is not None:
        server += f":{parts.port}"

    result: dict[str, str] = {"server": server}
    if parts.username:
        result["username"] = unquote(parts.username)
    if parts.password:
        result["password"] = unquote(parts.password)
    return result


def resolve_user_data_dir(root: Path, profile_name: str) -> Path:
    """The persistent-context user data dir for ``profile_name`` under ``root``.

    ``root`` is the run's own browser-data directory
    (:meth:`BrowserEngine.__init__`'s ``user_data_root``), so two runs never
    share a profile. Pure: the engine creates the directory at launch.
    """
    return root / profile_name


async def apply_init_scripts(
    context: BrowserContext,
    profile: BrowserProfile | None,
) -> None:
    """Load the profile's init scripts onto a context via ``add_init_script``.

    No-op when ``profile`` is ``None``.  Centralises the read-then-inject
    loop shared by every engine launch + restart path.
    """
    if profile is None:
        return
    for script_path in profile.init_scripts:
        js = script_path.read_text(encoding="utf-8")
        await context.add_init_script(js)
