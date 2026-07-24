"""Conformance suite for ``Recoverable`` (jkent.driver.unified_driver.lifecycle).

Any component that rebuilds a shared resource after it dies satisfies this
protocol. The suite is packaged as a reusable base class
(``RecoverableConformance``) that an implementation's test subclasses, plus a
reference fake exercised here so the file runs green on its own.

Contract under test (see ``lifecycle_contract.md`` -> ``Recoverable``):

- ABC conformance: ``Recoverable`` is an ``abc.ABC``; a conforming object
  subclasses it and so is a ``Recoverable`` instance.
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

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jkent.driver.unified_driver.lifecycle import Recoverable


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

    # --- ABC conformance -------------------------------------------------

    def test_is_a_recoverable(self, subject: Recoverable) -> None:
        """A conforming object subclasses the ``Recoverable`` ABC."""
        assert isinstance(subject, Recoverable)

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

    @pytest.mark.generative
    @given(k=st.integers(min_value=1, max_value=50))
    def test_k_sequential_restarts_advance_by_k(self, k: int) -> None:
        """K sequential in-band restarts advance generation by exactly K."""

        async def drive() -> int:
            subject = self.make_subject()
            for _ in range(k):
                await subject.restart(subject.generation)
            return subject.generation

        assert asyncio.run(drive()) == k

    @pytest.mark.generative
    @given(n=st.integers(min_value=1, max_value=50))
    def test_concurrent_restarts_rebuild_once(self, n: int) -> None:
        """N concurrent restarts at the same seen generation rebuild once."""

        async def drive() -> int:
            subject = self.make_subject()
            seen = subject.generation
            await asyncio.gather(*(subject.restart(seen) for _ in range(n)))
            return subject.generation

        assert asyncio.run(drive()) == 1

    def make_subject(self) -> Recoverable:
        """Build a fresh subject for property tests that need many instances.

        Property tests construct their own subjects (a fixture yields one
        instance per test, but ``@given`` drives many examples), so an
        implementation that uses hypothesis must override this too.
        """
        raise NotImplementedError


# --- Reference fake -------------------------------------------------------


class _DeadResource(Exception):
    """Sentinel: the shared resource died and must be rebuilt."""


class ReferenceRecoverable(Recoverable):
    """Minimal correct ``Recoverable`` with real single-flight semantics."""

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
            self._generation += 1


class TestReferenceRecoverable(RecoverableConformance):
    """Run the conformance suite against the reference fake."""

    @pytest.fixture
    def subject(self) -> Recoverable:
        return ReferenceRecoverable()

    def make_subject(self) -> Recoverable:
        return ReferenceRecoverable()

    def dead_exc(self) -> BaseException:
        return _DeadResource("engine crashed")

    # --- Fake-specific reinforcement of the single-rebuild invariant -----

    @pytest.mark.generative
    @given(k=st.integers(min_value=1, max_value=50))
    def test_rebuild_count_tracks_sequential_restarts(self, k: int) -> None:
        """The fake performs exactly K rebuilds for K sequential restarts."""

        async def drive() -> ReferenceRecoverable:
            subject = ReferenceRecoverable()
            for _ in range(k):
                await subject.restart(subject.generation)
            return subject

        subject = asyncio.run(drive())
        assert subject.rebuild_count == k
        assert subject.generation == k

    @pytest.mark.generative
    @given(n=st.integers(min_value=1, max_value=50))
    def test_concurrent_restarts_rebuild_exactly_once(self, n: int) -> None:
        """N concurrent restarts at one seen generation rebuild exactly once."""

        async def drive() -> ReferenceRecoverable:
            subject = ReferenceRecoverable()
            seen = subject.generation
            await asyncio.gather(*(subject.restart(seen) for _ in range(n)))
            return subject

        subject = asyncio.run(drive())
        assert subject.rebuild_count == 1
        assert subject.generation == 1
