"""The per-worker handle a transport hands each worker.

A leaf module, outside the ``unified_driver`` package, because
``jkent.driver.browser_engine.worker_page`` subclasses :class:`WorkerHandle`:
importing it from ``unified_driver.transport`` would run
``unified_driver/__init__``, which imports the browser transport, which
imports ``worker_page`` — a cycle. ``unified_driver.transport`` re-exports
both names.
"""

from __future__ import annotations

import abc
from typing import Any


class WorkerHandle(abc.ABC):
    """A transport's per-worker resource (a browser page, or nothing).

    Acquired once per worker via :meth:`Transport.acquire`, reused across
    requests, and reset between them. HTTP and replay hand back a no-op
    handle (:class:`NoopHandle`) since they hold no per-worker state.
    """

    #: The live browser page this handle wraps, or ``None`` where there is
    #: no page. The worker passes it to the step for autowait, so a
    #: pageless transport (HTTP, replay) simply leaves it ``None`` rather
    #: than the caller probing for the attribute.
    page: Any = None

    @abc.abstractmethod
    async def reset_for_reuse(self) -> None:
        """Clear per-request state so the handle is ready for the next request."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release the underlying per-worker resource."""


class NoopHandle(WorkerHandle):
    """The handle for transports that hold no per-worker resource.

    Shared by HTTP (httpx pools internally) and replay (reads from a
    source DB); both hand one of these back from ``acquire`` so the worker
    loop has a uniform handle to reset and close.
    """

    async def reset_for_reuse(self) -> None:
        return None

    async def close(self) -> None:
        return None
