"""Lifecycle role shared by unified-driver components.

:class:`AsyncLifecycle` splits setup/teardown from construction so a
component can be built cheaply and acquire (and release) its resources at
a well-defined point in the run. ``Transport`` subclasses it explicitly.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib


class AsyncLifecycle(abc.ABC):
    """A component with explicit setup and teardown.

    Setup/teardown are split from construction so a component can be
    built cheaply and its resources acquired (and released) at a
    well-defined point in the run.
    """

    @abc.abstractmethod
    async def open(self) -> None:
        """Acquire the component's resources. Safe to call once per lifecycle."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release resources after a *clean* shutdown.

        The forceful teardown path (resource already dead) is a transport's
        own affair — see ``PlaywrightTransport.restart`` — not this one.
        """


async def sleep_unless_stopped(
    stop_event: asyncio.Event, delay: float
) -> bool:
    """Sleep ``delay`` seconds, waking early if ``stop_event`` is set.

    Returns whether the event is set, so a caller can bail out of its loop.
    """
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop_event.wait(), timeout=delay)
    return stop_event.is_set()
