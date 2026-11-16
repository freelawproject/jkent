"""Speculative base class for entry point parameter models.

Defines an abstract base class that Pydantic BaseModel classes can
inherit to power the speculation system. Instead of jkent owning the
speculation configuration classes, scraper authors define their own
Pydantic models with speculation semantics.

The base class has four members:

- ``should_advance``: attribute (field or property) controlling whether
  the driver attempts to push the speculation ceiling past ``seed_range``
  on success.
- ``seed_range()``: returns the integer IDs to seed immediately. Every
  resulting request is enqueued as ``is_speculative=True``, carrying its
  ``speculative_index`` and a reference to the template's tracking row.
- ``from_int(n)``: builds a new template for a specific integer ID,
  preserving all non-range configuration.
- ``max_gap()``: the consecutive-failure ceiling (and initial window
  size when ``should_advance`` is True).

Example::

    class DocketId(BaseModel, Speculative):
        year: int
        min: int
        soft_max: int = 0
        should_advance: bool = True
        gap: int = 3

        def seed_range(self) -> range:
            return range(self.min, self.soft_max)

        def from_int(self, n: int) -> DocketId:
            return DocketId(
                year=self.year, min=n, soft_max=self.soft_max,
                should_advance=self.should_advance, gap=self.gap,
            )

        def max_gap(self) -> int:
            return self.gap

A template can also be *persisted*: seeded from a value stored outside the
run, under a key, and handed back advanced once the run is done. See
:class:`PersistedSpeculative`; the store it reads from is the
:func:`persisted_cursor_store` context.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Self


class Speculative(ABC):
    """Abstract base for entry-point parameter models that support speculation.

    When an ``@entry`` function has a parameter whose type subclasses this
    ABC, the driver detects it automatically and runs the speculation
    loop: seeding ``seed_range()`` as speculative requests, optionally
    opening an adaptive advance window, tracking success/failure, and
    stopping once ``max_gap()`` consecutive failures accrue beyond the
    highest observed success.

    Attributes:
        should_advance: Whether the driver should extend the speculation
            ceiling past ``seed_range().stop`` on success. When False the
            driver enqueues ``seed_range()`` only and stops.

    Methods:
        seed_range: Returns the integer IDs to seed immediately. A
            ``range`` (``start`` inclusive, ``stop`` exclusive, step=1).
            An empty range is valid and means "no initial seeding — rely
            entirely on the advance window".
        from_int: Creates a new instance for integer ID *n*, preserving
            all other fields (year, config, etc.) from the template.
        max_gap: Maximum consecutive failures beyond the highest success
            before the speculation stops. Also controls the size of the
            initial advance window enqueued when ``should_advance`` is
            True. Returning 0 disables the advance window entirely.
    """

    should_advance: bool

    @abstractmethod
    def seed_range(self) -> range:
        """The integer IDs to seed immediately; see the class docstring."""
        ...

    @abstractmethod
    def from_int(self, n: int) -> Self:
        """A copy of this template re-based on ID *n*."""
        ...

    @abstractmethod
    def max_gap(self) -> int:
        """Consecutive failures tolerated past the highest success."""
        ...


#: A ``[key]`` reference: a single key wrapped in square brackets, the whole
#: value. Surrounding whitespace is tolerated; the key may not contain
#: brackets.
_KEY_REF_RE = re.compile(r"\s*\[([^\[\]]+)\]\s*")

#: The values ``[key]`` references resolve against, keyed by cursor key.
#: Unset by default; populate it with :func:`persisted_cursor_store`.
PERSISTED_CURSORS: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "persisted_cursors", default=None
)


@contextmanager
def persisted_cursor_store(store: Mapping[str, Any]) -> Iterator[None]:
    """Resolve ``[key]`` references against *store* for the block.

    Seeding validates ``seed_params`` synchronously, so a host whose store
    is remote or async prefetches the keys it needs
    (:func:`persisted_keys_in`) into a plain dict and passes that::

        with persisted_cursor_store({"cal-dockets": {"min": 1200}}):
            async with RunBootstrapper(scraper, db_path, config=...) as run:
                ...

    Args:
        store: Each cursor key's stored value, as the JSON-compatible object
            its template validates from.
    """
    token = PERSISTED_CURSORS.set(store)
    try:
        yield
    finally:
        PERSISTED_CURSORS.reset(token)


def persisted_key_ref(value: Any) -> str | None:
    """The cursor key *value* references, if it is a ``[key]`` string.

    Args:
        value: A raw seed value.

    Returns:
        The key inside the brackets, or None for anything else.
    """
    if not isinstance(value, str):
        return None
    match = _KEY_REF_RE.fullmatch(value)
    return match.group(1) if match is not None else None


def persisted_keys_in(value: Any) -> set[str]:
    """Every ``[key]`` reference among the leaves of *value*.

    For a host to prefetch the cursors a run's ``seed_params`` cite before
    entering :func:`persisted_cursor_store`.

    Args:
        value: Raw ``seed_params``, or any JSON-shaped value.

    Returns:
        The referenced keys; empty for ``None``.
    """
    if isinstance(value, Mapping):
        return set().union(*(persisted_keys_in(v) for v in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(persisted_keys_in(v) for v in value))
    key = persisted_key_ref(value)
    return {key} if key is not None else set()


def resolve_persisted_cursor(key: str) -> dict[str, Any]:
    """The stored value for *key* in the active cursor store.

    Args:
        key: The cursor key a ``[key]`` reference named.

    Returns:
        A copy of the stored mapping, for the template to validate from.

    Raises:
        ValueError: No store is active, it has no *key*, or the stored value
            is not a mapping. ValueError so it surfaces as a pydantic
            ValidationError from the validator that calls this.
    """
    store = PERSISTED_CURSORS.get()
    if store is None:
        raise ValueError(
            f"cannot resolve [{key}]: no persisted cursor store is active "
            "(enter a persisted_cursor_store(...) block first)"
        )
    if key not in store:
        raise ValueError(
            f"cannot resolve [{key}]: the persisted cursor store has no "
            "value for it"
        )
    value = store[key]
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError(
            f"cannot resolve [{key}]: stored value is "
            f"{type(value).__name__}, not a mapping"
        )
    return dict(value)


class PersistedSpeculative(Speculative):
    """A speculative template whose starting point is stored outside the run.

    Seeded with a ``[key]`` reference, the template resolves its fields from
    the active cursor store and remembers the key as ``persist_key``, which
    every probe built from it (and the template's stored JSON, across a
    resume) carries. A literal seed value leaves ``persist_key`` None: there
    is nothing to write back.

    When the run is done, the host asks the run for
    :meth:`~jkent.driver.unified_driver.ScrapeRun.persisted_cursors` — each
    keyed template's :meth:`resume_after` its highest success — and decides
    for itself whether and when to store each one's :meth:`stored_value`.
    The driver never writes to the store.

    Attributes:
        persist_key: The cursor key this template was seeded from, or None.
    """

    persist_key: str | None

    @abstractmethod
    def resume_after(self, highest_successful_id: int | None) -> Self:
        """The template the next run should start from.

        Args:
            highest_successful_id: The highest ID this run found, or None if
                it found none.

        Returns:
            The cursor to store under ``persist_key``.
        """
        ...

    @abstractmethod
    def stored_value(self) -> dict[str, Any]:
        """The value to store under ``persist_key``.

        What a ``[key]`` reference resolves to on the next run, so it must
        validate back into this template's type.

        Returns:
            A JSON-compatible mapping, without ``persist_key``.
        """
        ...
