"""Shared speculation logic for the driver's speculative dispatch.

A speculative request can be dispatched in different ways — synchronously or
via an async queue/DB insert — so the seed/extend/track machinery lives here
and a consumer subclasses the mixin and implements its two hooks,
``_enqueue_speculative`` and ``_persist``.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from jkent.common.exceptions import ScraperConfigError
from jkent.common.scraper import inherit_step_metadata
from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    Request,
    Response,
)
from jkent.driver.database_engine.enums import SpeculationOutcome

logger = logging.getLogger(__name__)


@dataclass
class SpeculationState:
    """Tracks speculation state for a single speculative template.

    Each template (one per param invocation of a speculative entry)
    gets its own SpeculationState, keyed by ``{func_name}:{param_index}``.

    Attributes:
        func_name: State key: ``{entry_name}:{param_index}``.
        template: The Speculative instance (template for from_int calls).
        param_index: Position of this invocation in the params list.
        base_func_name: The actual method name on the scraper.
        highest_successful_id: Highest ID that returned a successful response,
            or None before any has. Outcomes are counted only for IDs past it
            (see :meth:`is_past_highest_success`); None is below every ID.
        consecutive_failures: Consecutive non-success responses beyond
            highest_successful_id.
        current_ceiling: Highest ID currently seeded to the queue, or None
            before the state has been seeded.
        stopped: True when max_gap consecutive failures reached or max_gap == 0.
        seed_value: The raw (pre-validation) seed value the template was
            built from — e.g. a host's "[cursor_key]" reference string.
            Persisted with the state so hosts can map state rows back to
            their seed without re-deriving the index assignment.
        tracking_id: Row id of this state's ``speculation_tracking`` entry,
            which every probe built from the template points at. Set by
            ``_persist`` before the first probe is enqueued (and by
            the resume path when loading persisted state), so it is None
            only for a state that has never been persisted.
    """

    func_name: str
    template: Speculative
    param_index: int
    base_func_name: str = ""
    highest_successful_id: int | None = None
    consecutive_failures: int = 0
    current_ceiling: int | None = None
    stopped: bool = False
    seed_value: Any = None
    tracking_id: int | None = None

    def is_past_highest_success(self, speculative_id: int) -> bool:
        """Whether *speculative_id* lies beyond the highest success so far.

        Every ID is past a state with no success yet.
        """
        return (
            self.highest_successful_id is None
            or speculative_id > self.highest_successful_id
        )


def find_speculative_param(
    scraper: BaseScraper[Any], base_func_name: str
) -> str:
    """Return the speculative param name for the given entry function."""
    for entry_info in scraper.list_speculative_entries():
        if entry_info.func_name == base_func_name:
            assert entry_info.speculative_param is not None
            return entry_info.speculative_param
    raise AssertionError(
        f"No speculative entry registered for {base_func_name!r}"
    )


def discover_speculate_functions(
    scraper: BaseScraper[Any],
) -> dict[str, SpeculationState]:
    """Build initial SpeculationState for every discovered template.

    Looks up templates from ``scraper._speculation_templates`` (populated
    by ``initial_seed()`` as ``(template, raw_seed_value)`` pairs). Each
    template at index *i* becomes a ``SpeculationState`` keyed by
    ``{func_name}:{i}``.
    """
    state: dict[str, SpeculationState] = {}
    templates = getattr(scraper, "_speculation_templates", {})

    for entry_info in scraper.list_speculative_entries():
        func_templates = templates.get(entry_info.func_name, [])
        for i, (template, seed_value) in enumerate(func_templates):
            key = f"{entry_info.func_name}:{i}"
            state[key] = SpeculationState(
                func_name=key,
                template=template,
                param_index=i,
                base_func_name=entry_info.func_name,
                seed_value=seed_value,
            )
    return state


def get_entry_requests(
    scraper: BaseScraper[Any],
    seed_params: list[dict[str, dict[str, Any]]] | None,
) -> Generator[Request, None, None]:
    """Yield initial entry requests for queue initialization.

    If ``seed_params`` is set, dispatches those via ``initial_seed()``.
    Otherwise invokes every parameterless, non-speculative ``@entry``
    method. A scraper with neither cannot be started without
    ``seed_params``: its entries all need arguments (or are speculative
    templates), so this raises rather than seeding nothing.

    Raises:
        ScraperConfigError: No ``seed_params`` and no parameterless entry.
    """
    if seed_params is not None:
        yield from scraper.initial_seed(seed_params)
        return
    entries = scraper.list_entries()
    invocations: list[dict[str, dict[str, Any]]] = [
        {entry_info.func_name: {}}
        for entry_info in entries
        if not entry_info.speculative and not entry_info.param_types
    ]
    if not invocations:
        names = [e.func_name for e in entries]
        raise ScraperConfigError(
            f"{type(scraper).__name__} has no parameterless @entry method to "
            f"seed from (entries: {names or 'none'}); pass seed_params or "
            "add an @entry that takes no arguments"
        )
    yield from scraper.initial_seed(invocations)


def build_speculative_request(
    scraper: BaseScraper[Any],
    spec_state: SpeculationState,
    n: int,
) -> Request:
    """Construct the speculative request for a given template ID.

    ``spec_state.tracking_id`` must already be set — a probe carries a
    foreign key to its tracking row, so the row is upserted before any
    request that references it is built (see ``AsyncSpeculationSupport.seed``).
    """
    tracking_id = spec_state.tracking_id
    if tracking_id is None:
        raise AssertionError(
            f"speculation state {spec_state.func_name!r} has no tracking row; "
            "it must be persisted before probes are built"
        )
    func = getattr(scraper, spec_state.base_func_name)
    speculative_param = find_speculative_param(
        scraper, spec_state.base_func_name
    )
    concrete = spec_state.template.from_int(n)
    request = inherit_step_metadata(
        scraper, func(**{speculative_param: concrete})
    )
    return request.speculative(tracking_id, n)


def compute_seed_plan(
    spec_state: SpeculationState,
) -> tuple[list[int], int, bool]:
    """Return (ids_to_seed, new_ceiling, stopped) for one state.

    The advance window is the ``max_gap`` IDs past the frontier: the end of
    the seed range, or the highest success if that is further. Resume-aware:
    when ``current_ceiling`` is None (a state never seeded) this reduces to
    plain seeding; otherwise (a persistent resume) it skips IDs at or below
    the ceiling, which are already queued or answered. The window is not
    re-anchored at the ceiling, so reopening a run with no new outcome plans
    nothing.
    """
    template = spec_state.template
    seed_ids = template.seed_range()
    gap = template.max_gap()

    resume_floor = (
        seed_ids.start
        if spec_state.current_ceiling is None
        else spec_state.current_ceiling + 1
    )
    seed_ids_to_run = [n for n in seed_ids if n >= resume_floor]

    window_start = max(seed_ids.start, seed_ids.stop)
    if spec_state.highest_successful_id is not None:
        window_start = max(window_start, spec_state.highest_successful_id + 1)
    window = (
        range(max(window_start, resume_floor), window_start + gap)
        if template.should_advance and gap > 0
        else None
    )

    if window is not None:
        ids = seed_ids_to_run + list(window)
        new_ceiling = max(window_start + gap, resume_floor) - 1
        stopped = False
    else:
        ids = seed_ids_to_run
        new_ceiling = max(window_start, resume_floor) - 1
        stopped = True

    return ids, new_ceiling, stopped


class AsyncSpeculationSupport(abc.ABC):
    """Async mixin: seed/extend/track over two hooks.

    ``_enqueue_speculative`` dispatches a probe; ``_persist`` saves a state
    and sets its ``tracking_id``. Outcome tracking is serialised under
    ``_speculation_lock``, and the state is persisted inside that lock.
    """

    scraper: BaseScraper[Any]
    _speculation_state: dict[str, SpeculationState]
    _speculation_lock: asyncio.Lock

    @abc.abstractmethod
    async def _enqueue_speculative(self, request: Request) -> None:
        """Dispatch one speculative probe."""

    @abc.abstractmethod
    async def _persist(self, spec_state: SpeculationState) -> None:
        """Save *spec_state* and set its ``tracking_id``."""

    def _state_by_tracking_id(
        self, tracking_id: int
    ) -> SpeculationState | None:
        """The state whose tracking row is *tracking_id*, if still tracked."""
        for spec_state in self._speculation_state.values():
            if spec_state.tracking_id == tracking_id:
                return spec_state
        return None

    async def seed(self) -> None:
        """Seed the queue with the initial speculative probe window."""
        for spec_state in self._speculation_state.values():
            if spec_state.stopped:
                continue
            # A probe row carries a foreign key to the tracking row, so the
            # state is persisted (and its tracking_id set) before any probe.
            await self._persist(spec_state)
            ids, new_ceiling, stopped = compute_seed_plan(spec_state)
            for n in ids:
                request = build_speculative_request(
                    self.scraper, spec_state, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling
            spec_state.stopped = stopped
            # Record the ceiling now, so a resume before any outcome arrives
            # plans past it instead of probing the same IDs again.
            await self._persist(spec_state)

    async def _extend_speculation(self, spec_state: SpeculationState) -> None:
        if spec_state.stopped:
            return

        gap = spec_state.template.max_gap()

        # An unseeded state has no window to extend.
        ceiling = spec_state.current_ceiling
        if ceiling is None:
            return

        if not spec_state.is_past_highest_success(ceiling - gap):
            new_ceiling = ceiling + gap
            for n in range(ceiling + 1, new_ceiling + 1):
                request = build_speculative_request(
                    self.scraper, spec_state, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling

    def _tracked_state(self, request: Request) -> SpeculationState | None:
        """The state a probe reports to, or None (logged) if it has none."""
        if (
            request.speculation_tracking_id is None
            or request.speculative_index is None
        ):
            logger.warning(
                "Speculative probe %s has no tracking row or index; "
                "outcome not tracked",
                request.request.url,
            )
            return None
        spec_state = self._state_by_tracking_id(
            request.speculation_tracking_id
        )
        if spec_state is None:
            logger.warning(
                "No speculation state owns tracking row %d (probe %s); "
                "outcome not tracked",
                request.speculation_tracking_id,
                request.request.url,
            )
        return spec_state

    def has_stopped(self, request: Request) -> bool:
        """Whether ``request``'s template stopped probing before it ran.

        The worker asks before it fetches a dequeued probe: a probe queued
        when its template stopped is completed unfetched as
        :attr:`~SpeculationOutcome.TERMINATED_EARLY`. An untracked probe has
        no template to have stopped.
        """
        if not request.is_speculative:
            return False
        tracking_id = request.speculation_tracking_id
        if tracking_id is None:
            return False
        spec_state = self._state_by_tracking_id(tracking_id)
        return spec_state is not None and spec_state.stopped

    async def track_outcome(
        self, request: Request, response: Response
    ) -> SpeculationOutcome | None:
        """Record a speculative probe outcome and extend/stop as needed.

        A 2xx that ``actually_successful`` accepts is a
        :attr:`~SpeculationOutcome.HIT`; anything else is a
        :attr:`~SpeculationOutcome.MISS`, or
        :attr:`~SpeculationOutcome.STOPPED` for the miss that stops the
        template. A probe that cannot be tracked (no tracking fields, or a
        tracking row no loaded state owns) is logged and records nothing.

        Returns:
            The outcome recorded, for the worker to store with the probe;
            None for a probe that is not tracked.
        """
        if not request.is_speculative:
            return None
        spec_state = self._tracked_state(request)
        if spec_state is None:
            return None
        speculative_id = request.speculative_index
        assert speculative_id is not None  # checked by _tracked_state

        is_success = 200 <= response.status_code < 300
        if is_success and not self.scraper.actually_successful(response):
            is_success = False

        async with self._speculation_lock:
            if is_success:
                outcome = SpeculationOutcome.HIT
                if spec_state.is_past_highest_success(speculative_id):
                    spec_state.highest_successful_id = speculative_id
                spec_state.consecutive_failures = 0
                await self._extend_speculation(spec_state)
            else:
                outcome = SpeculationOutcome.MISS
                if spec_state.is_past_highest_success(speculative_id):
                    spec_state.consecutive_failures += 1
                    gap = spec_state.template.max_gap()
                    if (
                        spec_state.consecutive_failures >= gap
                        and not spec_state.stopped
                    ):
                        spec_state.stopped = True
                        outcome = SpeculationOutcome.STOPPED

            await self._persist(spec_state)
        return outcome
