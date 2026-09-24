"""An ``HttpxTransport`` over ``httpx.MockTransport``: no server, no socket."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx

from jkent.data_types import BaseScraper
from jkent.driver.unified_driver import HttpxTransport


def mocked_httpx_transport(
    handler: Callable[[httpx.Request], httpx.Response],
    scraper: type[BaseScraper[Any]] | BaseScraper[Any] = BaseScraper,
    **transport_kwargs: Any,
) -> HttpxTransport:
    """An ``HttpxTransport`` whose main client answers from ``handler``.

    No socket: for tests that need to see what reaches the wire, or to
    script what comes back, without a server. The client carries the
    transport's timeout, like the one ``open`` builds. The caller owns
    teardown via ``await transport.aclose()``.
    """
    transport = HttpxTransport(scraper=scraper, **transport_kwargs)
    transport._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=transport.timeout
    )
    return transport
