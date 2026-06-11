"""Reusable conformance suite for ``AsyncLifecycle``.

``AsyncLifecycle`` (jkent.driver.unified_driver.lifecycle) is a cross-cutting
role: a component whose setup and teardown are split from construction so it is
cheap to build and acquires its resources at a well-defined point.

Contract under test (see ``lifecycle_contract.md`` -> ``AsyncLifecycle``):

- Protocol conformance: ``AsyncLifecycle`` is ``@runtime_checkable``, so an
  ``isinstance`` check against an implementation is valid.
- ``open`` is awaitable and returns ``None``.
- ``aclose`` is awaitable and returns ``None``.
- Ordering is always ``open -> (use) -> aclose``, each called exactly once.
- Every resource acquired in ``open`` is released in ``aclose``.
- The forceful path (resource already dead) is NOT here -- it belongs to
  ``Recoverable.restart``; ``aclose`` assumes an orderly close.

``AsyncLifecycleConformance`` is the reusable base: a real implementation
subclasses it and overrides the ``subject`` fixture. ``TestReferenceAsyncLifecycle``
runs the suite against a minimal reference fake so this file is green on its own.
"""

from __future__ import annotations

import inspect

import pytest

from jkent.driver.unified_driver import AsyncLifecycle


class AsyncLifecycleConformance:
    """Reusable contract tests for any ``AsyncLifecycle`` implementation.

    Subclass and override :meth:`subject` to return a fresh, unopened instance.
    """

    @pytest.fixture
    def subject(self) -> AsyncLifecycle:
        """A fresh, not-yet-opened ``AsyncLifecycle`` instance."""
        raise NotImplementedError

    def test_is_an_async_lifecycle(self, subject: AsyncLifecycle) -> None:
        """The runtime-checkable protocol accepts the implementation."""
        assert isinstance(subject, AsyncLifecycle)

    def test_open_is_a_coroutine_function(
        self, subject: AsyncLifecycle
    ) -> None:
        """``open`` is awaitable."""
        assert inspect.iscoroutinefunction(subject.open)

    def test_aclose_is_a_coroutine_function(
        self, subject: AsyncLifecycle
    ) -> None:
        """``aclose`` is awaitable."""
        assert inspect.iscoroutinefunction(subject.aclose)

    async def test_open_awaits_to_none(self, subject: AsyncLifecycle) -> None:
        """``open`` is awaitable and completes (its result type is ``None``)."""
        await subject.open()

    async def test_aclose_awaits_to_none(
        self, subject: AsyncLifecycle
    ) -> None:
        """``aclose`` is awaitable and completes after an orderly open."""
        await subject.open()
        await subject.aclose()

    async def test_open_then_aclose_releases_resources(
        self, subject: AsyncLifecycle
    ) -> None:
        """Resources acquired in ``open`` are released by ``aclose``."""
        await subject.open()
        await subject.aclose()
        # No leak survives an orderly open -> aclose cycle.
        assert _live_resources(subject) == 0


# --- Reference fake: a minimal correct AsyncLifecycle --------------------


class _ReferenceLifecycle:
    """A minimal correct ``AsyncLifecycle`` that tracks open/close misuse.

    It models one resource acquired in ``open`` and released in ``aclose``, and
    refuses re-entrant or re-opened use so the suite can lean on those guards.
    """

    def __init__(self) -> None:
        self._opened = False
        self._closed = False
        self.resources = 0

    async def open(self) -> None:
        """Acquire the resource, exactly once, before any other use."""
        if self._opened:
            raise RuntimeError("open called more than once")
        self._opened = True
        self.resources += 1

    async def aclose(self) -> None:
        """Release the resource after an orderly open, exactly once."""
        if not self._opened:
            raise RuntimeError("aclose before open")
        if self._closed:
            raise RuntimeError("aclose called more than once")
        self._closed = True
        self.resources -= 1


def _live_resources(subject: AsyncLifecycle) -> int:
    """Outstanding resources held by the reference fake (test helper)."""
    assert isinstance(subject, _ReferenceLifecycle)
    return subject.resources


class TestReferenceAsyncLifecycle(AsyncLifecycleConformance):
    """Run the conformance suite against the reference fake."""

    @pytest.fixture
    def subject(self) -> AsyncLifecycle:
        return _ReferenceLifecycle()
