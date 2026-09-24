"""Tests for the unified driver's concrete ``Run`` (:class:`ScrapeRun`).

Two layers:

* ``TestScrapeRunConformance`` binds the real ``ScrapeRun`` to the shared
  ``RunConformance`` suite over a temp-file DB, a trivial scraper, and a spy
  transport that is never hit while the queue is empty.
* ``Test*`` targeted cases pin the lifecycle wiring the conformance suite
  leaves out: open brings the transport up and aclose tears it down (and a
  failure mid-open unwinds what was acquired); ``spawn_worker`` registers and
  its on-done callback deregisters; ``status`` walks unstarted -> done across
  a trivial empty run; open resets interrupted rows; the error budget stops
  the run gracefully.

The worker pool and the compactor registry have their own suites
(``test_pool.py``, ``test_compactors.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine
from typing_extensions import override

from jkent.common.decorators import entry, step
from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    RequestTimeoutException,
)
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import (
    RequestStatus,
    RequestType,
    RunStatus,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport import NoopHandle, Transport
from jkent.driver.unified_driver.wiring import RunConfig
from tests.driver.unified.test_run_conformance import RunConformance

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from jkent.driver.unified_driver.transport import (
        ArchiveStream,
        AwaitCondition,
        QueuedRequest,
    )


class TrivialScraper(BaseScraper[dict[str, Any]]):
    """Minimal scraper: one @step, empty rate_limits.

    The entry yields no requests so a freshly opened run starts with an empty
    queue — the conformance invariant the spy transport relies on (a hit
    ``resolve`` is an assertion failure).
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def start(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse(
        self, response: Response
    ) -> Generator[ScraperYield[dict[str, Any]], None, None]:
        yield ParsedData({"ok": True})


def test_strictly_serial_scraper_caps_workers_to_one(tmp_path: Path) -> None:
    """A STRICTLY_SERIAL scraper forces num_workers to 1.

    Concurrent workers would interleave a stateful session and defeat the
    per-step priority ordering, so the contract is enforced when the run
    resolves its config, regardless of the requested worker count.
    """

    class _SerialScraper(TrivialScraper):
        driver_requirements = [DriverRequirement.STRICTLY_SERIAL]

    run = ScrapeRun(
        _SerialScraper(), tmp_path / "s.db", config=RunConfig(num_workers=4)
    )
    assert run.config.num_workers == 1

    # A non-serial scraper keeps its requested count.
    plain = ScrapeRun(
        TrivialScraper(), tmp_path / "p.db", config=RunConfig(num_workers=4)
    )
    assert plain.config.num_workers == 4


class SpyTransport(Transport[NoopHandle]):
    """A run-scoped transport peer that records open/close and is never hit."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self._handles: dict[int, NoopHandle] = {}

    async def open(self) -> None:
        self.opened = True

    async def aclose(self) -> None:
        self.closed = True

    async def acquire(self, worker_id: int) -> NoopHandle:
        return self._handles.setdefault(worker_id, NoopHandle())

    async def release(self, worker_id: int) -> None:
        self._handles.pop(worker_id, None)

    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        raise AssertionError("resolve must not be hit with an empty queue")

    async def resolve_archive(
        self, handle: NoopHandle, queued: QueuedRequest
    ) -> ArchiveStream:
        raise AssertionError("resolve_archive must not be hit")

    @override
    async def finish_archiving(self, stream: ArchiveStream) -> None:
        return None


def _make_run(
    db_path: Path,
    transport: SpyTransport | None = None,
    *,
    config: RunConfig | None = None,
    **run_kwargs: Any,
) -> ScrapeRun:
    """A ScrapeRun over a fresh temp DB + trivial scraper + spy transport.

    ``TrivialScraper``'s entry yields nothing, so a fresh queue stays empty
    and the spy transport is never hit; rate limiting is off.
    Signals are the bootstrapper's business, so a bare ``ScrapeRun`` never
    installs any.
    """
    return ScrapeRun(
        TrivialScraper(),
        db_path,
        transport=transport if transport is not None else SpyTransport(),
        config=replace(config or RunConfig(), rate_limited=False),
        **run_kwargs,
    )


# --- Conformance ---------------------------------------------------------


class TestScrapeRunConformance(RunConformance):
    """Runs the shared conformance suite against the real ``ScrapeRun``."""

    @pytest.fixture
    @override
    def subject(self, tmp_path: Path) -> ScrapeRun:
        return _make_run(tmp_path / "run.db")


# --- Targeted lifecycle cases -------------------------------------------


async def test_open_brings_transport_up_then_aclose_down(
    tmp_path: Path,
) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)

    await run.open()
    assert transport.opened is True
    assert transport.closed is False
    assert run.transport is transport

    await run.aclose()
    assert transport.closed is True


async def test_open_failure_unwinds_what_was_acquired(tmp_path: Path) -> None:
    """A failure partway through open() tears down the earlier steps.

    The transport is up before storage is built; if building storage raises,
    the transport must be closed by ``open`` itself — nobody else knows it
    was opened.
    """

    class _StorageBoomRun(ScrapeRun):
        @override
        def _make_storage(self):
            raise RuntimeError("storage boom")

    transport = SpyTransport()
    run = _StorageBoomRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=transport,
        config=RunConfig(rate_limited=False),
    )
    with pytest.raises(RuntimeError, match="storage boom"):
        await run.open()
    assert transport.opened is True
    assert transport.closed is True
    # Nothing left to unwind: a later aclose is a harmless no-op.
    await run.aclose()


async def test_open_twice_is_an_error(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run.db")
    await run.open()
    try:
        with pytest.raises(RuntimeError, match="already open"):
            await run.open()
    finally:
        await run.aclose()


async def test_run_twice_is_an_error(tmp_path: Path) -> None:
    """A run runs once: a second ``run()`` — say after an error-budget stop,
    whose stop event nothing clears — would spawn workers that exit at once,
    scrape nothing, and overwrite the first run's ``started_at``."""
    run = _make_run(tmp_path / "run.db")
    await run.open()
    try:
        run.stop()
        await run.run()
        metadata = await run.db.get_run_metadata()
        assert metadata is not None
        with pytest.raises(RuntimeError, match="already run"):
            await run.run()
        again = await run.db.get_run_metadata()
        assert again is not None
        assert again.started_at == metadata.started_at
    finally:
        await run.aclose()


async def test_aclose_forgets_what_open_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``aclose`` the accessors refuse like before ``open``, and a
    reopen builds (and on close disposes) a fresh engine rather than
    adopting the disposed one as if it were borrowed."""
    disposed: list[AsyncEngine] = []
    dispose = AsyncEngine.dispose

    async def spy(self: AsyncEngine, close: bool = True) -> None:
        disposed.append(self)
        await dispose(self, close)

    monkeypatch.setattr(AsyncEngine, "dispose", spy)

    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    first = run.db.engine
    await run.aclose()
    with pytest.raises(RuntimeError, match="not open"):
        await run.stats()
    with pytest.raises(AssertionError):
        run.collaborators  # noqa: B018
    # The injected transport is the caller's, so it stays.
    assert run.transport is transport

    await run.open()
    second = run.db.engine
    await run.aclose()
    assert second is not first
    assert disposed == [first, second]


async def test_public_accessors_expose_db_and_engine(tmp_path: Path) -> None:
    # The public surface hosts reach for
    # instead of run._db / run._db.engine.sync_engine.
    run = _make_run(tmp_path / "run.db")
    await run.open()
    try:
        assert run.db is run._db
        assert run._engine is not None
        assert run.sync_engine is run._engine.sync_engine
        assert run.collaborators.transport is run.transport
        assert run.collaborators.error_sink is run.error_budget
        assert run.collaborators.circuit_breaker is run.circuit_breaker
        assert run.compactors is run.collaborators.compactors
    finally:
        await run.aclose()


async def test_open_records_the_session_config(tmp_path: Path) -> None:
    """The configured pool, backoff cap and jitter land in run_metadata.

    Non-default values throughout, so a wiring that records a default (or a
    constant) instead of the config shows up as a mismatch.
    """
    config = RunConfig(
        num_workers=3, max_backoff_time=45.0, retry_jitter=0.137
    )
    run = _make_run(tmp_path / "run.db", config=config)
    await run.open()
    try:
        metadata = await run.db.get_run_metadata()
        assert metadata is not None
        assert (
            metadata.num_workers,
            metadata.max_backoff_time,
            metadata.jitter,
        ) == (3, 45.0, 0.137)
    finally:
        await run.aclose()


async def test_make_engine_seam_builds_the_run_engine(tmp_path: Path) -> None:
    # Subclasses (a host's replay run) override _make_engine to swap the
    # pool.
    class _SeamRun(ScrapeRun):
        make_engine_calls = 0

        @override
        async def _make_engine(self):
            type(self).make_engine_calls += 1
            return await super()._make_engine()

    run = _SeamRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        config=RunConfig(rate_limited=False),
    )
    await run.open()
    try:
        assert _SeamRun.make_engine_calls == 1
        assert run._engine is not None
    finally:
        await run.aclose()


async def test_make_worker_seam_builds_pool_workers(tmp_path: Path) -> None:
    # Subclasses (a host's replay run) override _make_worker to swap the
    # worker class; the pool must build through it.
    built: list[int] = []

    class _SeamRun(ScrapeRun):
        @override
        def _make_worker(self, worker_id: int) -> Any:
            built.append(worker_id)
            return super()._make_worker(worker_id)

    run = _SeamRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        config=RunConfig(rate_limited=False, num_workers=2),
    )
    await run.open()
    try:
        await run.run()
        assert built == [0, 1]
    finally:
        await run.aclose()


async def test_injected_db_is_used_and_left_open(tmp_path: Path) -> None:
    # Callers that must open the DB before the run exists (RunBootstrapper,
    # for a browser transport's handle) pass that manager in with db=. Two
    # managers on one file would mean two write locks and no serialization.
    db_path = tmp_path / "run.db"
    engine, session_factory = await init_database(db_path)
    manager = SQLManager(engine, session_factory)
    run = _make_run(db_path, db=manager)
    await run.open()
    try:
        assert run.db is manager
        assert run.sync_engine is engine.sync_engine
    finally:
        await run.aclose()

    # A borrowed engine is the caller's to dispose — aclose() left it usable.
    try:
        async with session_factory() as session:
            assert (
                await session.execute(sa.text("SELECT count(*) FROM requests"))
            ).scalar() == 0
    finally:
        await engine.dispose()


async def test_make_engine_is_skipped_when_a_db_is_injected(
    tmp_path: Path,
) -> None:
    class _SeamRun(ScrapeRun):
        make_engine_calls = 0

        @override
        async def _make_engine(self):
            type(
                self
            ).make_engine_calls += 1  # pragma: no cover - must not run
            return await super()._make_engine()

    db_path = tmp_path / "run.db"
    engine, session_factory = await init_database(db_path)
    run = _SeamRun(
        TrivialScraper(),
        db_path,
        db=SQLManager(engine, session_factory),
        transport=SpyTransport(),
        config=RunConfig(rate_limited=False),
    )
    await run.open()
    try:
        assert _SeamRun.make_engine_calls == 0
    finally:
        await run.aclose()
        await engine.dispose()


async def test_default_transport_is_an_httpx_transport(
    tmp_path: Path,
) -> None:
    # With no transport injected, open() builds and brings up an
    # HttpxTransport, and aclose() tears it down.
    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        config=RunConfig(rate_limited=False),
    )
    await run.open()
    assert run.transport is not None  # built an HttpxTransport
    await run.aclose()


async def test_spawn_worker_registers_and_deregisters(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    try:
        run.stop()  # so the spawned worker exits on its first idle check
        before = run.active_worker_count
        worker_id = run.spawn_worker()
        assert isinstance(worker_id, int)
        assert run.active_worker_count == before + 1

        # The on-done callback deregisters the worker once it exits.
        task = run._pool.task_for(worker_id)
        assert task is not None
        await task
        await asyncio.sleep(0)  # let the done-callback fire
        assert run.active_worker_count == before
    finally:
        await run.aclose()


async def test_status_transitions_unstarted_to_done(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    try:
        assert await run.status() == "unstarted"
        run.stop()
        await run.run()
        assert await run.status() == "done"
    finally:
        await run.aclose()


# --- Graceful resume ----------------------------------------------------


class _ServingTransport(SpyTransport):
    """A SpyTransport that resolves every request to a trivial 200 response."""

    @override
    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        return Response(
            status_code=200,
            headers={},
            content=b"<html></html>",
            text="<html></html>",
            url=queued.request.request.url,
            request=queued.request,
        )


async def _seed_in_progress(db: SQLManager) -> None:
    """Insert one ``in_progress`` row addressed to the ``parse`` step."""
    async with db.session_factory() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, request_type, method,
                    url, step, current_location)
                VALUES (:status, 5, :rtype, :method,
                    'http://127.0.0.1/page1', 'parse', '')
                """
            ),
            {
                "status": RequestStatus.IN_PROGRESS.code,
                "rtype": RequestType.NAVIGATING.code,
                "method": HttpMethod.GET.code,
            },
        )
        await session.commit()


async def _status_counts(db: SQLManager) -> dict[str, int]:
    """Group the requests table by status into a {status: count} dict."""
    async with db.session_factory() as session:
        result = await session.execute(
            sa.text("SELECT status, COUNT(*) FROM requests GROUP BY status")
        )
        # Raw SQL yields the stored integer code, so decode back to the label
        # the assertions read in.
        return {
            str(RequestStatus.from_code(row[0])): row[1]
            for row in result.all()
        }


async def test_open_resets_in_progress_to_pending(tmp_path: Path) -> None:
    # Seed an interrupted (in_progress) row on a temp-file DB, as a run that
    # died mid-request would leave it.
    db_path = tmp_path / "run.db"
    engine, session_factory = await init_database(db_path)
    try:
        seed_db = SQLManager(engine, session_factory)
        await _seed_in_progress(seed_db)
        assert await _status_counts(seed_db) == {"in_progress": 1}
    finally:
        await engine.dispose()

    # Reopen the SAME db: every open's restore_queue resets it to pending.
    resumed = _make_run(db_path, transport=_ServingTransport())
    await resumed.open()
    try:
        assert await _status_counts(resumed.db) == {"pending": 1}

        # Bonus: a subsequent run() drains the restored row to completion.
        await resumed.run()
        assert await _status_counts(resumed.db) == {"completed": 1}
    finally:
        await resumed.aclose()


# --- Error budget (max_persistent_errors) --------------------------------


def _persistent_exc() -> PersistentHTTPResponseException:
    return PersistentHTTPResponseException(404, "http://127.0.0.1/gone")


async def test_error_budget_stops_the_run_gracefully(tmp_path: Path) -> None:
    # Non-transient errors are charged via the ErrorSink funnel; hitting
    # the budget sets the stop event (graceful, resumable) exactly once.
    run = _make_run(
        tmp_path / "run.db", config=RunConfig(max_persistent_errors=2)
    )
    await run.open()
    try:
        budget = run.error_budget
        await budget.record(_persistent_exc(), request_url="u1")
        assert not run.stop_event.is_set()
        # Unclassified exceptions never retry either, so they count too.
        await budget.record(ValueError("parse bug"), request_url="u2")
        assert run.stop_event.is_set()
        assert budget.exhausted is True
        assert run.persistent_error_count == 2
        # Stragglers past the threshold still count, without re-stopping.
        await budget.record(_persistent_exc(), request_url="u3")
        assert run.persistent_error_count == 3
    finally:
        await run.aclose()


async def test_transient_errors_do_not_charge_the_budget(
    tmp_path: Path,
) -> None:
    # A transient that exhausted its retries stores an error row but is the
    # circuit breaker's signal, not scraper breakage: never budgeted.
    run = _make_run(
        tmp_path / "run.db", config=RunConfig(max_persistent_errors=1)
    )
    await run.open()
    try:
        for _ in range(3):
            await run.error_budget.record(
                RequestTimeoutException("http://127.0.0.1/slow", 30.0)
            )
        assert run.persistent_error_count == 0
        assert not run.stop_event.is_set()
    finally:
        await run.aclose()


async def test_unbudgeted_run_counts_but_never_stops(tmp_path: Path) -> None:
    # Default None = unlimited: the meter still runs for host visibility.
    run = _make_run(tmp_path / "run.db")
    await run.open()
    try:
        for _ in range(5):
            await run.error_budget.record(_persistent_exc())
        assert run.persistent_error_count == 5
        assert not run.stop_event.is_set()
    finally:
        await run.aclose()


# --- Config resolution ----------------------------------------------------


def test_run_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="num_workers"):
        RunConfig(num_workers=0)
    with pytest.raises(ValueError, match="worker_ramp_interval"):
        RunConfig(worker_ramp_interval=-1)
    with pytest.raises(ValueError, match="max_persistent_errors"):
        RunConfig(max_persistent_errors=0)


def test_ramp_is_off_by_default(tmp_path: Path) -> None:
    run = _make_run(tmp_path / "run.db", config=RunConfig(num_workers=4))
    assert run.config.worker_ramp_interval == 0.0
    assert run.config.num_workers == 4


class _HangingTransport(SpyTransport):
    """Resolves nothing: every request parks until the run is torn down."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    @override
    async def resolve(
        self,
        handle: NoopHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


async def test_a_cancelled_run_finalizes_as_interrupted(
    tmp_path: Path,
) -> None:
    """A killed run must not read as COMPLETED: operators resume on that."""
    transport = _HangingTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open()
    try:
        await _seed_in_progress(run.db)
        await run.db.restore_queue()
        task = asyncio.create_task(run.run())
        await asyncio.wait_for(transport.entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with run.db.session_factory() as session:
            status = (
                await session.execute(
                    sa.text("SELECT status FROM run_metadata WHERE id = 1")
                )
            ).scalar_one()
        assert RunStatus.from_code(status) is RunStatus.INTERRUPTED
    finally:
        await run.aclose()
