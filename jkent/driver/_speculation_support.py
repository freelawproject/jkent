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

from jkent.common.exceptions import ScraperConfigError
from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    HTTPRequestPrep,
    JSRequestPrep,
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
        consecutive_failures: Consecutive non-success responses beyond highest_successful_id.
        current_ceiling: Highest ID currently seeded to the queue.
        stopped: True when max_gap consecutive failures reached or max_gap == 0.
    """

    func_name: str
    template: Speculative
    param_index: int
    base_func_name: str = ""
    highest_successful_id: int = 0
    consecutive_failures: int = 0
    current_ceiling: int = 0
    stopped: bool = False


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
    by ``initial_seed()``). Each template at index *i* becomes a
    ``SpeculationState`` keyed by ``{func_name}:{i}``.
    """
    state: dict[str, SpeculationState] = {}
    templates = getattr(scraper, "_speculation_templates", {})

    for entry_info in scraper.list_speculative_entries():
        func_templates = templates.get(entry_info.func_name, [])
        for i, template in enumerate(func_templates):
            key = f"{entry_info.func_name}:{i}"
            state[key] = SpeculationState(
                func_name=key,
                template=template,
                param_index=i,
                base_func_name=entry_info.func_name,
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

    Entry steps cannot yield ``JSRequestPrep`` / ``HTTPRequestPrep`` —
    preps need a parent response/page that doesn't exist yet — so we
    validate each yield as it comes through.
    """

    def _check(item: Any) -> Any:
        if isinstance(item, JSRequestPrep | HTTPRequestPrep):
            raise ScraperConfigError(
                f"entry step yielded {type(item).__name__}; "
                f"prep wrappers require a parent response and cannot be "
                f"yielded from entry steps"
            )
        return item

    if seed_params is not None:
        for item in scraper.initial_seed(seed_params):
            yield _check(item)
        return
    entries = scraper.list_entries()
    if entries:
        invocations: list[dict[str, dict[str, Any]]] = []
        for entry_info in entries:
            if not entry_info.speculative and not entry_info.param_types:
                invocations.append({entry_info.func_name: {}})
        if invocations:
            for item in scraper.initial_seed(invocations):
                yield _check(item)
            return
    for item in scraper.get_entry():
        yield _check(item)


def build_speculative_request(
    scraper: BaseScraper,
    spec_state: SpeculationState,
    state_key: str,
    n: int,
) -> Request:
    """Construct the speculative request for a given template ID."""
    func = getattr(scraper, spec_state.base_func_name)
    speculative_param = find_speculative_param(
        scraper, spec_state.base_func_name
    )
    concrete = spec_state.template.from_int(n)
    request = func(**{speculative_param: concrete})
    if isinstance(request, JSRequestPrep | HTTPRequestPrep):
        raise ScraperConfigError(
            f"speculative step {spec_state.base_func_name!r} returned "
            f"{type(request).__name__}; prep wrappers cannot be used "
            f"with @speculate (no parent response/page exists)"
        )
    return request.speculative(state_key, spec_state.param_index, n)


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

    def _get_entry_requests(self) -> Generator[Request, None, None]:
        yield from get_entry_requests(self.scraper, self.seed_params)


class SyncSpeculationSupport(_SpeculationBase):
    """Sync mixin: seed/extend/track with a sync ``_enqueue_speculative`` hook."""

    def _enqueue_speculative(self, request: Request) -> None:
        raise NotImplementedError

    def _seed_speculative_queue(self) -> None:
        for state_key, spec_state in self._speculation_state.items():
            if spec_state.stopped:
                continue
            ids, new_ceiling, stopped = compute_seed_plan(spec_state)
            for n in ids:
                request = build_speculative_request(
                    self.scraper, spec_state, state_key, n
                )
                self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling
            spec_state.stopped = stopped

    def _extend_speculation(self, state_key: str) -> None:
        spec_state = self._speculation_state.get(state_key)
        if spec_state is None or spec_state.stopped:
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
                    self.scraper, spec_state, state_key, n
                )
                self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling

    def _track_speculation_outcome(
        self, request: Request, response: Response
    ) -> None:
        if not request.is_speculative or request.speculation_id is None:
            return

        state_key, _param_index, speculative_id = request.speculation_id
        spec_state = self._speculation_state.get(state_key)
        if spec_state is None:
            return

        is_success = 200 <= response.status_code < 300
        if is_success and not self.scraper.actually_successful(response):
            is_success = False

        if is_success:
            if speculative_id > spec_state.highest_successful_id:
                spec_state.highest_successful_id = speculative_id
            spec_state.consecutive_failures = 0
            self._extend_speculation(state_key)
        else:
            if speculative_id > spec_state.highest_successful_id:
                spec_state.consecutive_failures += 1
                gap = spec_state.template.max_gap()
                if spec_state.consecutive_failures >= gap:
                    spec_state.stopped = True


class AsyncSpeculationSupport(_SpeculationBase):
    """Async mixin: seed/extend/track with an async ``_enqueue_speculative`` hook.

    Outcome tracking is serialised under ``_speculation_lock``.  Subclasses
    that persist state override ``_after_outcome`` to run additional work
    inside that lock.
    """

    _speculation_lock: asyncio.Lock  # type: ignore

    async def _enqueue_speculative(self, request: Request) -> None:
        raise NotImplementedError

    async def _after_outcome(self, spec_state: SpeculationState) -> None:
        """Hook run inside the speculation lock after outcome updates."""
        return None

    async def _seed_speculative_queue(self) -> None:
        for state_key, spec_state in self._speculation_state.items():
            if spec_state.stopped:
                continue
            ids, new_ceiling, stopped = compute_seed_plan(spec_state)
            for n in ids:
                request = build_speculative_request(
                    self.scraper, spec_state, state_key, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling
            spec_state.stopped = stopped

    async def _extend_speculation(self, state_key: str) -> None:
        spec_state = self._speculation_state.get(state_key)
        if spec_state is None or spec_state.stopped:
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
                    self.scraper, spec_state, state_key, n
                )
                await self._enqueue_speculative(request)
            spec_state.current_ceiling = new_ceiling

    async def _track_speculation_outcome(
        self, request: Request, response: Response
    ) -> None:
        if not request.is_speculative or request.speculation_id is None:
            return

        state_key, _param_index, speculative_id = request.speculation_id
        spec_state = self._speculation_state.get(state_key)
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
                await self._extend_speculation(state_key)
            else:
                if speculative_id > spec_state.highest_successful_id:
                    spec_state.consecutive_failures += 1
                    gap = spec_state.template.max_gap()
                    if spec_state.consecutive_failures >= gap:
                        spec_state.stopped = True

            await self._after_outcome(spec_state)
