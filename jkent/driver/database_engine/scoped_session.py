"""Scoped session factory for per-worker connection isolation.

Provides a drop-in replacement for ``async_sessionmaker`` that caches
sessions per scope (worker, monitor, etc.) using a ``ContextVar``.
When no scope is set, sessions behave normally (close on ``__aexit__``).

The ``ScopedSessionFactory`` is the main entry point.  Workers call
:func:`set_scope` at startup and the factory returns the same
persistent session for every ``async with factory() as session:`` block
within that scope.  The session is only closed when explicitly removed
via :meth:`ScopedSessionFactory.remove` or
:meth:`ScopedSessionFactory.remove_all`.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scope key context variable
# ---------------------------------------------------------------------------

_scope_key: ContextVar[str | None] = ContextVar[str | None](
    "_scope_key", default=None
)


def set_scope(key: str) -> None:
    """Set the scope key for the current task/coroutine."""
    _scope_key.set(key)


def clear_scope() -> None:
    """Clear the scope key, returning to unscoped (one-shot) mode."""
    _scope_key.set(None)


def get_scope() -> str | None:
    """Return the current scope key, or ``None`` if unscoped."""
    return _scope_key.get()


# ---------------------------------------------------------------------------
# No-close context manager
# ---------------------------------------------------------------------------


class _NoCloseSessionContext:
    """Async context manager that yields a session without closing on exit.

    Used for scoped sessions where the session persists beyond a single
    ``async with`` block.  On exception the session is rolled back so
    it remains usable for subsequent operations.
    """

    __slots__ = ("_session",)

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if exc_type is not None:
            await self._session.rollback()


# ---------------------------------------------------------------------------
# Scoped session factory
# ---------------------------------------------------------------------------


class ScopedSessionFactory:
    """Session factory with optional per-scope caching.

    Drop-in replacement for ``async_sessionmaker``.  Calling an instance
    returns a context manager compatible with::

        async with factory() as session:
            ...

    **Scoped** (``set_scope()`` was called): returns a
    ``_NoCloseSessionContext`` wrapping a cached ``AsyncSession``.  The
    same session is reused across all calls within the scope and is NOT
    closed when the ``async with`` block exits.

    **Unscoped** (default): returns a fresh ``AsyncSession`` that closes
    normally on ``__aexit__`` — identical to ``async_sessionmaker``.
    """

    def __init__(self, underlying_factory: async_sessionmaker) -> None:
        self._factory = underlying_factory
        self._registry: dict[str, AsyncSession] = {}

    # -- callable interface (drop-in for async_sessionmaker) ----------------

    def __call__(self) -> _NoCloseSessionContext | AsyncSession:
        key = _scope_key.get()
        if key is not None:
            session = self._registry.get(key)
            if session is None:
                session = self._factory()
                self._registry[key] = session
            return _NoCloseSessionContext(session)
        # Unscoped — return a normal session (closes on __aexit__)
        return self._factory()

    # -- lifecycle management -----------------------------------------------

    async def remove(self, key: str) -> None:
        """Close and remove a scoped session by *key*."""
        session = self._registry.pop(key, None)
        if session is not None:
            try:
                await session.close()
            except Exception:
                logger.warning(
                    "Error closing session for scope '%s'",
                    key,
                    exc_info=True,
                )

    async def remove_all(self) -> None:
        """Close and remove every scoped session.

        Intended for shutdown — called by the monitor on exit and as a
        safety net in ``PersistentDriver.close()``.
        """
        sessions = list(self._registry.items())
        self._registry.clear()
        for key, session in sessions:
            try:
                await session.close()
            except Exception:
                logger.warning(
                    "Error closing session for scope '%s'",
                    key,
                    exc_info=True,
                )
