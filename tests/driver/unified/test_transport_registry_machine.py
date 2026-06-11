"""Generative registry rig for ``Transport``'s per-worker handle ledger.

A Hypothesis ``RuleBasedStateMachine`` drives random ``acquire`` /
``release`` / ``resolve`` (and, on the fake, crash) sequences against a
live transport and checks the ``WorkerHandle`` registry laws after every
step — the laws ``test_transport_conformance`` pins with single examples:

- get-or-create stability: ``acquire(w)`` while a lease is held returns
  the SAME handle object;
- freshness: after ``release(w)`` (or a crash poisons the handle), the
  next ``acquire(w)`` returns a different object;
- isolation: two live worker ids never share a handle object;
- ``resolve`` works through any currently-held handle.

One event loop and one open transport live for the whole machine run —
rules are sync and await via ``loop.run_until_complete`` because
Hypothesis does not compose with pytest-asyncio. Each binding's test
function passes a factory closure to ``run_state_machine_as_test`` so
machines can be built per Hypothesis run with fixture-derived config.

Bindings: the reference fake (plus an out-of-band crash rule), a real
``HttpxTransport`` over a live aiohttp server, and a real
``ReplayTransport`` over a one-row run DB. Browser transports are
covered by the example-based concurrency tests instead — a browser per
machine step is too slow for a generative rig.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from aiohttp import web
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    rule,
    run_state_machine_as_test,
)

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.unified_driver import (
    HttpxTransport,
    QueuedRequest,
    ReplayTransport,
)
from tests.driver.unified.test_transport_conformance import FakeTransport
from tests.driver.unified.test_transport_impl_conformance import (
    _materialize_dual,
    _ok_app,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.generative

_WORKER_IDS = st.integers(min_value=0, max_value=3)


class TransportRegistryMachine(RuleBasedStateMachine):
    """Registry-law machine over any live transport.

    Subclass per binding: provide :meth:`make_transport` and
    :meth:`make_queued`. The machine mirrors the registry it expects in
    ``self.held`` and remembers each worker's last surrendered handle in
    ``self.last_retired`` so freshness can be asserted on re-acquire.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.transport = self.make_transport()
        self.loop.run_until_complete(self.transport.open())
        self.held: dict[int, Any] = {}
        self.last_retired: dict[int, Any] = {}

    def make_transport(self) -> Any:
        raise NotImplementedError

    def make_queued(self) -> QueuedRequest:
        raise NotImplementedError

    @rule(w=_WORKER_IDS)
    def acquire(self, w: int) -> None:
        handle = self.loop.run_until_complete(self.transport.acquire(w))
        if w in self.held:
            assert handle is self.held[w], (
                f"acquire({w}) while leased returned a different handle "
                "(get-or-create stability violated)"
            )
        else:
            retired = self.last_retired.get(w)
            assert handle is not retired, (
                f"acquire({w}) after release returned the surrendered "
                "handle (freshness violated)"
            )
            self.held[w] = handle

    @rule(w=_WORKER_IDS)
    def release(self, w: int) -> None:
        self.loop.run_until_complete(self.transport.release(w))
        if w in self.held:
            self.last_retired[w] = self.held.pop(w)

    @rule(w=_WORKER_IDS)
    def resolve(self, w: int) -> None:
        if w not in self.held:
            return  # only resolve through a held lease
        response = self.loop.run_until_complete(
            self.transport.resolve(self.held[w], self.make_queued())
        )
        assert isinstance(response, Response)

    @invariant()
    def held_handles_are_isolated(self) -> None:
        """No two live worker ids ever share a handle object."""
        handles = list(self.held.values())
        assert len({id(h) for h in handles}) == len(handles)

    def teardown(self) -> None:
        try:
            self.loop.run_until_complete(self.transport.aclose())
        finally:
            self.loop.close()


# --- Binding 1: reference fake, with out-of-band crashes -------------------


class CrashableFakeTransport(FakeTransport):
    """The reference fake, plus handles that can die out-of-band.

    A poisoned handle is detected and discarded by the next ``acquire``,
    which rebuilds a fresh one — the registry-visible half of the
    recovery contract (the single-flight restart half lives in
    ``test_recoverable_conformance``).
    """

    async def acquire(self, worker_id: int) -> Any:
        handle = self._handles.get(worker_id)
        if handle is not None and getattr(handle, "dead", False):
            self._handles.pop(worker_id)  # poisoned: rebuild below
        return await super().acquire(worker_id)


class FakeRegistryMachine(TransportRegistryMachine):
    """The machine over the crashable reference fake."""

    def make_transport(self) -> CrashableFakeTransport:
        return CrashableFakeTransport()

    def make_queued(self) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url="https://example.com"
                ),
                continuation="parse",
            ),
            request_id=1,
        )

    @rule(w=_WORKER_IDS)
    def crash(self, w: int) -> None:
        """Kill a held handle out-of-band; the next acquire must rebuild."""
        if w not in self.held:
            return
        handle = self.held.pop(w)
        handle.dead = True
        self.last_retired[w] = handle  # acquire must hand back a new one


def test_fake_transport_registry_machine() -> None:
    run_state_machine_as_test(FakeRegistryMachine)


# --- Binding 2: real HttpxTransport over a live aiohttp server -------------


class HttpxRegistryMachine(TransportRegistryMachine):
    """The machine over a real ``HttpxTransport`` hitting a local server.

    ``make_transport`` runs inside the base ``__init__`` with ``self.loop``
    already created, so the backing server is brought up there too.
    """

    # Assigned by make_transport (called from the base __init__); declared
    # with defaults so checkers see them initialized.
    runner: web.AppRunner | None = None
    base_url: str = ""

    def make_transport(self) -> HttpxTransport:
        runner = web.AppRunner(_ok_app())
        self.loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        host, port = runner.addresses[0][0], runner.addresses[0][1]
        self.base_url = f"http://{host}:{port}"
        self.runner = runner
        return HttpxTransport()

    def make_queued(self) -> QueuedRequest:
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=f"{self.base_url}/r"
                ),
                continuation="parse",
            ),
            request_id=1,
        )

    def teardown(self) -> None:
        try:
            if self.runner is not None:
                self.loop.run_until_complete(self.runner.cleanup())
        finally:
            super().teardown()


def test_httpx_transport_registry_machine() -> None:
    run_state_machine_as_test(HttpxRegistryMachine)


# --- Binding 3: real ReplayTransport over a one-row run DB -----------------


def test_replay_transport_registry_machine(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://machine.test/r"
        ),
        continuation="parse",
    )
    workdir = tmp_path_factory.mktemp("replay_registry")
    db_path = workdir / "run.db"
    _materialize_dual(schema_template, db_path, workdir, request)

    class ReplayRegistryMachine(TransportRegistryMachine):
        """The machine over a real ``ReplayTransport`` (shared one-row DB)."""

        def make_transport(self) -> ReplayTransport:
            return ReplayTransport([db_path])

        def make_queued(self) -> QueuedRequest:
            return QueuedRequest(request=request, request_id=1)

    run_state_machine_as_test(ReplayRegistryMachine)
