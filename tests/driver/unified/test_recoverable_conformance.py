"""Conformance suite for transport-internal crash recovery.

``PlaywrightTransport`` rebuilds its shared browser engine after it dies via
``generation`` / ``should_restart`` / ``restart``. There is no shared ABC for
that surface (it has one implementation), so the contract lives here: a
reusable base class (``RecoverableConformance``) that the implementation's
test subclasses, plus a reference fake exercised here so the file runs green
on its own. ``Recoverable`` below is the structural type the suite is written
against.

Contract under test:

- ``generation`` is monotonic and non-decreasing, starts at ``0``, and
  increments by exactly one per successful rebuild.
- ``should_restart(exc)`` is a pure, side-effect-free bool predicate: it does
  not perturb ``generation`` and returns the same answer on repeated calls.
- ``restart(seen_generation)`` is single-flight: a no-op when
  ``seen_generation != generation``; otherwise it rebuilds exactly once and
  increments ``generation``. Postcondition: ``generation > seen_generation``.
- Concurrency: N concurrent ``restart(g)`` calls at the same seen generation
  ``g`` cause exactly one rebuild.

The sequential and concurrent single-flight properties are exercised with
hypothesis; async property tests are driven via ``asyncio.run`` inside a sync
test because ``@given`` does not compose with ``async def`` under
pytest-asyncio.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typing_extensions import override


class Recoverable(Protocol):
    """The crash-recovery surface the suite drives (structural, test-only)."""

    @property
    def generation(self) -> int: ...

    def should_restart(self, exc: BaseException) -> bool: ...

    async def restart(self, seen_generation: int) -> None: ...


class CountingRecoverable(Recoverable, Protocol):
    """A test subject that also counts the rebuilds it actually performed.

    ``generation`` alone cannot tell "rebuilt once" from "rebuilt N times and
    advanced once"; the property tests read both.
    """

    rebuild_count: int


class RecoverableConformance:
    """Reusable contract tests for any ``Recoverable`` implementation.

    Subclass and override :meth:`subject` (and :meth:`dead_exc` if the
    implementation's ``should_restart`` recognizes a specific exception).
    """

    @pytest.fixture
    def subject(self) -> Recoverable:
        """The implementation under test."""
        raise NotImplementedError

    def dead_exc(self) -> BaseException:
        """An exception the subject's ``should_restart`` recognizes as death."""
        raise NotImplementedError

    # --- generation ------------------------------------------------------

    def test_generation_starts_at_zero(self, subject: Recoverable) -> None:
        """A freshly built subject reports generation 0."""
        assert subject.generation == 0

    async def test_single_restart_increments_by_one(
        self, subject: Recoverable
    ) -> None:
        """One in-band restart bumps generation by exactly one."""
        await subject.restart(subject.generation)
        assert subject.generation == 1

    # --- should_restart is a pure predicate ------------------------------

    def test_should_restart_recognizes_death(
        self, subject: Recoverable
    ) -> None:
        """The death exception is recognized as restartable."""
        assert subject.should_restart(self.dead_exc()) is True

    def test_should_restart_rejects_unrelated(
        self, subject: Recoverable
    ) -> None:
        """An unrelated exception is not treated as death."""
        assert subject.should_restart(ValueError("unrelated")) is False

    def test_should_restart_has_no_side_effects(
        self, subject: Recoverable
    ) -> None:
        """Calling the predicate never perturbs generation, and is stable."""
        before = subject.generation
        exc = self.dead_exc()
        first = subject.should_restart(exc)
        second = subject.should_restart(exc)
        assert first == second
        assert subject.generation == before

    # --- restart single-flight -------------------------------------------

    async def test_restart_postcondition_generation_advanced(
        self, subject: Recoverable
    ) -> None:
        """After an in-band restart, generation strictly exceeds the seen one."""
        seen = subject.generation
        await subject.restart(seen)
        assert subject.generation > seen

    async def test_stale_restart_is_noop(self, subject: Recoverable) -> None:
        """A restart at a stale seen generation does not rebuild."""
        await subject.restart(subject.generation)  # advance to gen 1
        current = subject.generation
        await subject.restart(0)  # stale: someone already rebuilt
        assert subject.generation == current

    # More than one subclass is collected per session, so hypothesis sees
    # these called from "differing executors"; each class still exercises the
    # property correctly on its own subject.
    @pytest.mark.generative
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(k=st.integers(min_value=1, max_value=50))
    def test_k_sequential_restarts_advance_by_k(self, k: int) -> None:
        """K sequential in-band restarts advance generation by exactly K."""

        async def drive() -> tuple[int, int]:
            subject = self.make_subject()
            for _ in range(k):
                await subject.restart(subject.generation)
            return subject.generation, subject.rebuild_count

        assert asyncio.run(drive()) == (k, k)

    @pytest.mark.generative
    @settings(suppress_health_check=[HealthCheck.differing_executors])
    @given(n=st.integers(min_value=1, max_value=50))
    def test_concurrent_restarts_rebuild_once(self, n: int) -> None:
        """N concurrent restarts at the same seen generation rebuild once."""

        async def drive() -> tuple[int, int]:
            subject = self.make_subject()
            seen = subject.generation
            await asyncio.gather(*(subject.restart(seen) for _ in range(n)))
            return subject.generation, subject.rebuild_count

        assert asyncio.run(drive()) == (1, 1)

    def make_subject(self) -> CountingRecoverable:
        """Build a fresh subject for property tests that need many instances.

        Property tests construct their own subjects (a fixture yields one
        instance per test, but ``@given`` drives many examples), so an
        implementation that uses hypothesis must override this too. The
        subject's rebuild step must yield to the event loop (as a real
        rebuild does), or concurrent restarts never interleave and the
        single-flight property holds without any lock.
        """
        raise NotImplementedError


# --- Reference fake -------------------------------------------------------


class _DeadResource(Exception):
    """Sentinel: the shared resource died and must be rebuilt."""


class ReferenceRecoverable:
    """Minimal correct implementation with real single-flight semantics."""

    def __init__(self) -> None:
        self._generation = 0
        self._lock = asyncio.Lock()
        self.rebuild_count = 0

    @property
    def generation(self) -> int:
        return self._generation

    def should_restart(self, exc: BaseException) -> bool:
        """Recognize only the sentinel death exception."""
        return isinstance(exc, _DeadResource)

    async def restart(self, seen_generation: int) -> None:
        """Rebuild once under the lock, guarded by the generation."""
        async with self._lock:
            if seen_generation != self._generation:
                return  # someone already rebuilt this generation
            self.rebuild_count += 1
            await asyncio.sleep(0)  # the rebuild itself awaits
            self._generation += 1


class TestReferenceRecoverable(RecoverableConformance):
    """Run the conformance suite against the reference fake."""

    @pytest.fixture
    @override
    def subject(self) -> Recoverable:
        return ReferenceRecoverable()

    @override
    def make_subject(self) -> CountingRecoverable:
        return ReferenceRecoverable()

    @override
    def dead_exc(self) -> BaseException:
        return _DeadResource("engine crashed")
