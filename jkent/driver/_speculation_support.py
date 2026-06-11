"""Shared speculation logic for the driver's speculative dispatch.

A speculative request can be dispatched in different ways — synchronously or
via an async queue/DB insert — so the seed/extend/track machinery lives here
and each consumer subclasses the matching mixin and implements the single
``_enqueue_speculative`` hook.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    Request,
    Response,
)

if TYPE_CHECKING:
    pass

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
        highest_successful_id: Highest ID that returned a successful response.
            Defaults to 0 and doubles as the "no success yet" sentinel. The
            tracking logic compares with ``speculative_id > highest_successful_id``,
            so a speculative ID of exactly 0 is invisible to both the watermark
            and the failure counter. In practice speculative ID spaces start at
            1 (``seed_range()`` begins at the scraper's ``min``, conventionally
            ≥ 1), so this boundary is never hit; a 0-based ID space would need a
            distinct sentinel.
        consecutive_failures: Consecutive non-success responses beyond highest_successful_id.
        current_ceiling: Highest ID currently seeded to the queue.
        stopped: True when max_gap consecutive failures reached or max_gap == 0.
        seed_value: The raw (pre-validation) seed value the template was
            built from — e.g. a host's "[cursor_key]" reference string.
            Persisted with the state so hosts can map state rows back to
            their seed without re-deriving the index assignment.
        tracking_id: Row id of this state's ``speculation_tracking`` entry,
            which every probe built from the template points at. Set by
            ``_ensure_tracked`` before the first probe is enqueued (and by
            the resume path when loading persisted state), so it is None
            only for a state that has never been persisted.
    """

    func_name: str
    template: Speculative
    param_index: int
    base_func_name: str = ""
    highest_successful_id: int = 0
    consecutive_failures: int = 0
    current_ceiling: int = 0
    stopped: bool = False
    seed_value: Any = None
    tracking_id: int | None = None


def find_speculative_param(scraper: BaseScraper, base_func_name: str) -> str:
    """Return the speculative param name for the given entry function."""
    for entry_info in scraper.list_speculative_entries():
        if entry_info.func_name == base_func_name:
            assert entry_info.speculative_param is not None
            return entry_info.speculative_param
    raise AssertionError(
        f"No speculative entry registered for {base_func_name!r}"
    )


def discover_speculate_functions(
    scraper: BaseScraper,
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
    scraper: BaseScraper,
    seed_params: list[dict[str, dict[str, Any]]] | None,
) -> Generator[Request, None, None]:
    """Yield initial entry requests for queue initialization.

    If ``seed_params`` is set, dispatches those via ``initial_seed()``.
    Otherwise builds default invocations from @entry-decorated methods.
    Falls back to ``get_entry()`` for scrapers without @entry decorators.
    """
    if seed_params is not None:
        yield from scraper.initial_seed(seed_params)
        return
    entries = scraper.list_entries()
    if entries:
        invocations: list[dict[str, dict[str, Any]]] = []
        for entry_info in entries:
            if not entry_info.speculative and not entry_info.param_types:
                invocations.append({entry_info.func_name: {}})
        if invocations:
            yield from scraper.initial_seed(invocations)
            return
    yield from scraper.get_entry()


def build_speculative_request(
    scraper: BaseScraper,
    spec_state: SpeculationState,
    n: int,
) -> Request:
    """Construct the speculative request for a given template ID.

    ``spec_state.tracking_id`` must already be set — a probe carries a
    foreign key to its tracking row, so the row is upserted before any
    request that references it is built (see ``_ensure_tracked``).
    """
    if spec_state.tracking_id is None:
        raise AssertionError(
            f"speculation state {spec_state.func_name!r} has no tracking row; "
            "_ensure_tracked must run before probes are built"
        )
    func = getattr(scraper, spec_state.base_func_name)
    speculative_param = find_speculative_param(
        scraper, spec_state.base_func_name
    )
    concrete = spec_state.template.from_int(n)
    request = func(**{speculative_param: concrete})
    return request.speculative(spec_state.tracking_id, n)


def compute_seed_plan(
    spec_state: SpeculationState,
) -> tuple[list[int], int, bool]:
    """Return (ids_to_seed, new_ceiling, stopped) for one state.

    Uses resume-aware semantics: when ``current_ceiling`` is 0 (fresh
    state) this reduces to plain seeding; when non-zero (persistent
    resume), skips IDs at or below the ceiling.
    """
    template = spec_state.template
    seed_ids = template.seed_range()

    resume_floor = (
        spec_state.current_ceiling + 1
        if spec_state.current_ceiling > 0
        else seed_ids.start
    )
    seed_ids_to_run = [n for n in seed_ids if n >= resume_floor]

    advance_floor = max(seed_ids.start, seed_ids.stop, resume_floor)
    window = (
        range(advance_floor, advance_floor + template.max_gap())
        if template.should_advance and template.max_gap() > 0
        else range(0)
    )

    ids = seed_ids_to_run + list(window)

    if window:
        new_ceiling = advance_floor + template.max_gap() - 1
        stopped = False
    else:
        new_ceiling = advance_floor - 1
        stopped = True

    return ids, new_ceiling, stopped


class _SpeculationBase:
    """Shared attributes contract for both sync and async mixins."""

    scraper: BaseScraper  # type: ignore
    seed_params: list[dict[str, dict[str, Any]]] | None  # type: ignore
    _speculation_state: dict[str, SpeculationState]  # type: ignore

    def _discover_speculate_functions(self) -> dict[str, SpeculationState]:
        return discover_speculate_functions(self.scraper)


class AsyncSpeculationSupport(_SpeculationBase):
    """Async mixin: seed/extend/track with an async ``_enqueue_speculative`` hook.

    Outcome tracking is serialised under ``_speculation_lock``.  Subclasses
    that persist state override ``_after_outcome`` to run additional work
    inside that lock.
    """

    _speculation_lock: asyncio.Lock  # type: ignore

    async def _enqueue_speculative(self, request: Request) -> None:
        raise NotImplementedError

    async def _ensure_tracked(self, spec_state: SpeculationState) -> None:
        """Hook: persist *spec_state* and set its ``tracking_id``.

        Runs before any probe for the state is built, because a probe row
        carries a foreign key to the tracking row. Subclasses that do not
        persist state assign an id of their own choosing.
        """
        return None

    async def _after_outcome(self, spec_state: SpeculationState) -> None:
        """Hook run inside the speculation lock after outcome updates."""
        return None

    def _state_by_tracking_id(
        self, tracking_id: int
    ) -> SpeculationState | None:
        """The state whose tracking row is *tracking_id*, if still tracked."""
        for spec_state in self._speculation_state.values():
            if spec_state.tracking_id == tracking_id:
                return spec_state
        return None

    async def _seed_speculative_queue(self) -> None:
        for spec_state in self._speculation_state.values():
            if spec_state.stopped:
                continue
            await self._ensure_tracked(spec_state)
            ids, new_ceiling, stopped = compute_seed_plan(spec_state)
            for n in ids:
                request = build_speculative_request(
                    self.scraper, spec_state, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling
            spec_state.stopped = stopped

    async def _extend_speculation(self, spec_state: SpeculationState) -> None:
        if spec_state.stopped:
            return

        gap = spec_state.template.max_gap()
        if gap == 0:
            return

        if spec_state.consecutive_failures >= gap:
            spec_state.stopped = True
            return

        if (
            spec_state.highest_successful_id
            >= spec_state.current_ceiling - gap
        ):
            new_ceiling = spec_state.current_ceiling + gap
            for n in range(spec_state.current_ceiling + 1, new_ceiling + 1):
                request = build_speculative_request(
                    self.scraper, spec_state, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling

    async def _track_speculation_outcome(
        self, request: Request, response: Response
    ) -> None:
        if (
            not request.is_speculative
            or request.speculation_tracking_id is None
            or request.speculative_index is None
        ):
            return

        speculative_id = request.speculative_index
        spec_state = self._state_by_tracking_id(
            request.speculation_tracking_id
        )
        if spec_state is None:
            return

        is_success = 200 <= response.status_code < 300
        if is_success and not self.scraper.actually_successful(response):
            is_success = False

        async with self._speculation_lock:
            if is_success:
                if speculative_id > spec_state.highest_successful_id:
                    spec_state.highest_successful_id = speculative_id
                spec_state.consecutive_failures = 0
                await self._extend_speculation(spec_state)
            else:
                if speculative_id > spec_state.highest_successful_id:
                    spec_state.consecutive_failures += 1
                    gap = spec_state.template.max_gap()
                    if spec_state.consecutive_failures >= gap:
                        spec_state.stopped = True

            await self._after_outcome(spec_state)
