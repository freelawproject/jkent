"""Cross-cutting role protocols shared by unified-driver components.

These factor the two responsibilities every long-lived, recreatable
component has — being set up/torn down, and recovering from an
out-of-band failure — into small protocols so each can be read and
verified on its own. ``Transport`` composes both; future components
(a browser engine, a run, a monitor) can compose them too.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class AsyncLifecycle(Protocol):
    """A component with explicit setup and teardown.

    Setup/teardown are split from construction so a component can be
    built cheaply and its resources acquired (and released) at a
    well-defined point in the run.
    """

    async def open(self) -> None:
        """Acquire the component's resources. Safe to call once per lifecycle."""
        ...

    async def aclose(self) -> None:
        """Release resources after a *clean* shutdown.

        The forceful teardown path (resource already dead) lives in
        :meth:`Recoverable.restart`, not here.
        """
        ...


@runtime_checkable
class Recoverable(Protocol):
    """A component that can rebuild a shared resource after it fails.

    Detection is distributed (any consumer may hit the dead resource);
    recovery is centralized and happens exactly once via the generation
    guard in :meth:`restart`. Consumers that miss the race observe the
    bumped :attr:`generation` and simply renew their lease.

    Transport-internal: a transport (e.g. Playwright) uses this inside its
    ``acquire`` to rebuild a crashed engine. It is **not** part of the public
    ``Transport`` surface — callers never invoke restart; they retry the
    ``TransientException`` a dead resource raises and re-``acquire``.
    """

    @property
    def generation(self) -> int:
        """Monotonic count of how many times the shared resource was (re)built."""
        ...

    def should_restart(self, exc: BaseException) -> bool:
        """Whether ``exc`` means the shared resource died and must be rebuilt.

        A predicate rather than an exception-type tuple: Playwright
        rewraps transport errors as bare ``Exception`` with only a
        message, so callers match on content, not type.
        """
        ...

    async def restart(self, seen_generation: int) -> None:
        """Rebuild the shared resource, once, under single-flight.

        ``seen_generation`` is the generation the caller last held. If it
        no longer matches the current generation, another caller already
        rebuilt and this is a no-op.
        """
        ...
