"""Reusable conformance suite for ``AsyncLifecycle``.

``AsyncLifecycle`` (jkent.driver.unified_driver.lifecycle) is a cross-cutting
role: a component whose setup and teardown are split from construction so it is
cheap to build and acquires its resources at a well-defined point.

Contract under test (see ``lifecycle_contract.md`` -> ``AsyncLifecycle``):

- ABC conformance: ``AsyncLifecycle`` is an ``abc.ABC`` implementations
  subclass, so an ``isinstance`` check against an implementation holds.
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

    Subclass and override :meth:`subject` (and :meth:`live_resources`) to bind
    the suite to a concrete implementation.
    """

    @pytest.fixture
    def subject(self) -> AsyncLifecycle:
        """A fresh, not-yet-opened ``AsyncLifecycle`` instance."""
        raise NotImplementedError

    def live_resources(self, subject: AsyncLifecycle) -> int:
        """Count of resources the subject currently holds open.

        Override per implementation to report outstanding resources (open
        clients, engines, contexts, handles...). The suite asserts this is
        ``> 0`` after ``open`` and ``0`` after ``aclose``.
        """
        raise NotImplementedError

    def test_is_an_async_lifecycle(self, subject: AsyncLifecycle) -> None:
        """The implementation subclasses the ``AsyncLifecycle`` ABC."""
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
        # ``open`` actually acquired something...
        assert self.live_resources(subject) > 0
        await subject.aclose()
        # ...and no leak survives an orderly open -> aclose cycle.
        assert self.live_resources(subject) == 0


# --- Reference fake: a minimal correct AsyncLifecycle --------------------


class _ReferenceLifecycle(AsyncLifecycle):
    """A minimal correct ``AsyncLifecycle``.

    It models one resource acquired in ``open`` and released in ``aclose``.
    The contract assumes an orderly ``open -> (use) -> aclose`` cycle, so the
    fake does not guard against misuse.
    """

    def __init__(self) -> None:
        self.resources = 0

    async def open(self) -> None:
        """Acquire the resource."""
        self.resources += 1

    async def aclose(self) -> None:
        """Release the resource after an orderly open."""
        self.resources -= 1


class TestReferenceAsyncLifecycle(AsyncLifecycleConformance):
    """Run the conformance suite against the reference fake."""

    @pytest.fixture
    def subject(self) -> AsyncLifecycle:
        return _ReferenceLifecycle()

    def live_resources(self, subject: AsyncLifecycle) -> int:
        assert isinstance(subject, _ReferenceLifecycle)
        return subject.resources
