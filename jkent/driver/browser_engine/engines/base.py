"""Browser-engine abstract base class plus shared helpers.

A :class:`BrowserEngine` owns the engine-specific bits of launching a
browser and producing a Playwright :class:`BrowserContext`.  The driver
only ever sees the context; engine internals stay encapsulated."""

from __future__ import annotations

import abc
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import unquote, urlsplit

if TYPE_CHECKING:
    from playwright.async_api import BrowserContext

    from jkent.data_types import BaseScraper


class BrowserEngine(abc.ABC):
    """Engine-specific browser factory + lifecycle manager.

    Subclasses own:

    - Playwright start/stop lifecycle.
    - Browser process launch + close.
    - ``BrowserContext`` creation + close.
    - Crash-recovery context rebuilding (where supported).

    The driver receives a ``BrowserContext`` from :meth:`acquire` and uses
    it for the rest of its life; on a connection-dead event it asks the
    engine to rebuild via :meth:`restart_context`.
    """

    engine_name: ClassVar[str]

    @property
    @abc.abstractmethod
    def supports_restart(self) -> bool:
        """Whether :meth:`restart_context` can rebuild the context.

        Subclasses that don't support restart should raise
        ``TransientException`` from ``restart_context`` and return
        ``False`` here so callers can fail fast.
        """

    @abc.abstractmethod
    def acquire(
        self,
    ) -> AbstractAsyncContextManager[BrowserContext]:  # type: ignore
        """Yield a live ``BrowserContext``; tear it down on exit."""

    @abc.abstractmethod
    async def restart_context(self) -> BrowserContext:
        """Tear down + rebuild the context after a crash.

        Only valid while ``acquire()`` is active.  Implementations that
        cannot restart raise ``TransientException``.
        """


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


def resolve_user_data_dir(
    scraper: BaseScraper[Any],
    profile_name: str,
) -> Path:
    """Determine the user_data_dir for a persistent browser context.

    Returns ``~/.cache/jkent/<scraper_module>/<profile_name>/browser-data/``,
    creating the directory if needed.
    """
    scraper_module = scraper.__class__.__module__.replace(".", "_")
    cache_dir = (
        Path.home()
        / ".cache"
        / "jkent"
        / scraper_module
        / profile_name
        / "browser-data"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
