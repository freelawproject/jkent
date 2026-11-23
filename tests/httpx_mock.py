"""An ``HttpxTransport`` over ``httpx2.MockTransport``: no server, no socket."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx2

from jkent.data_types import BaseScraper, ProxySetting
from jkent.driver.unified_driver import HttpxTransport


def mocked_httpx_transport(
    handler: Callable[[httpx2.Request], httpx2.Response],
    scraper: type[BaseScraper[Any]] | BaseScraper[Any] = BaseScraper,
    **transport_kwargs: Any,
) -> HttpxTransport:
    """An ``HttpxTransport`` whose main client answers from ``handler``.

    No socket: for tests that need to see what reaches the wire, or to
    script what comes back, without a server. The client is built from the
    same kwargs as the one ``open`` builds — timeout, the shared cookie
    jar — so session behavior matches production. The connection is direct
    unless ``proxy`` says otherwise: an environment proxy would mount a
    real transport over the mock. The caller owns teardown via
    ``await transport.aclose()``.
    """
    transport_kwargs.setdefault("proxy", ProxySetting.DIRECT)
    transport = HttpxTransport(scraper=scraper, **transport_kwargs)
    transport._client = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        **transport._client_kwargs(True),
    )
    return transport
