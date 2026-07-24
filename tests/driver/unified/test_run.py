"""Tests for the unified driver's concrete ``Run`` (:class:`ScrapeRun`).

Two layers:

* ``TestScrapeRunConformance`` binds the real ``ScrapeRun`` to the shared
  ``RunConformance`` suite over a temp-file DB, a trivial scraper, and a spy
  transport that is never hit while the queue is empty.
* ``Test*`` targeted cases pin the lifecycle wiring the conformance suite
  leaves out: open brings the transport up and aclose tears it down; the
  compactor startup check trains immediately at/over the threshold and seeds a
  ``Compactor`` below it; ``spawn_worker`` registers and its on-done callback
  deregisters; ``status`` walks unstarted -> done across a trivial empty run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

from jkent.common.decorators import entry, step
from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    RequestTimeoutException,
)
from jkent.data_types import (
    BaseScraper,
    DriverRequirement,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.database_engine.compression import (
    compress,
    get_compression_dict,
)
from jkent.driver.unified_driver.orchestration import Compactor
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport import NoopHandle, Transport
from tests.driver.unified.test_run_conformance import RunConformance

if TYPE_CHECKING:
    from pathlib import Path


class TrivialScraper(BaseScraper[dict]):
    """Minimal scraper: one @step, empty rate_limits.

    The entry yields no requests so a freshly opened run starts with an empty
    queue — the conformance invariant the spy transport relies on (a hit
    ``resolve`` is an assertion failure).
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        yield ParsedData({"ok": True})


def test_strictly_serial_scraper_caps_workers_to_one(tmp_path: Path) -> None:
    """A STRICTLY_SERIAL scraper forces num_workers to 1.

    Concurrent workers would interleave a stateful session and defeat the
    per-step priority ordering, so the contract is enforced at construction
    regardless of the requested worker count.
    """

    class _SerialScraper(TrivialScraper):
        driver_requirements = [DriverRequirement.STRICTLY_SERIAL]

    run = ScrapeRun(_SerialScraper(), tmp_path / "s.db", num_workers=4)
    assert run.num_workers == 1

    # A non-serial scraper keeps its requested count.
    plain = ScrapeRun(TrivialScraper(), tmp_path / "p.db", num_workers=4)
    assert plain.num_workers == 4


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

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        raise AssertionError("resolve must not be hit with an empty queue")

    async def resolve_archive(self, handle, queued, decision=None):
        raise AssertionError("resolve_archive must not be hit")

    async def finish_archiving(self, stream) -> None:
        return None


def _make_run(
    db_path: Path,
    transport: SpyTransport | None = None,
    **run_kwargs: Any,
) -> ScrapeRun:
    """A ScrapeRun over a fresh temp DB + trivial scraper + spy transport.

    Signals are off so tests don't fight the handlers; ``resume=False`` keeps
    the empty-queue invariant (no entry requests are auto-seeded) so the spy
    transport is never hit.
    """
    return ScrapeRun(
        TrivialScraper(),
        db_path,
        transport=transport if transport is not None else SpyTransport(),
        rate_limited=False,
        resume=False,
        **run_kwargs,
    )


# --- Conformance ---------------------------------------------------------


class _NoSignalScrapeRun(ScrapeRun):
    """``ScrapeRun`` whose bare ``open()`` suppresses signal handlers.

    The conformance suite calls the bare ``open()`` protocol method; tests
    must not let the run install process-wide signal handlers.
    """

    async def open(self, *, setup_signal_handlers: bool = False) -> None:
        await super().open(setup_signal_handlers=False)


class TestScrapeRunConformance(RunConformance):
    """Runs the shared conformance suite against the real ``ScrapeRun``."""

    @pytest.fixture
    def subject(self, tmp_path: Path) -> ScrapeRun:
        return _NoSignalScrapeRun(
            TrivialScraper(),
            tmp_path / "run.db",
            transport=SpyTransport(),
            rate_limited=False,
            resume=False,
        )


# --- Targeted lifecycle cases -------------------------------------------


async def test_open_brings_transport_up_then_aclose_down(
    tmp_path: Path,
) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)

    await run.open(setup_signal_handlers=False)
    assert transport.opened is True
    assert transport.closed is False
    assert run.transport is transport

    await run.aclose()
    assert transport.closed is True


async def test_public_accessors_expose_db_and_engine(tmp_path: Path) -> None:
    # The public surface hosts reach for
    # instead of run._db / run._db.engine.sync_engine.
    run = _make_run(tmp_path / "run.db")
    await run.open(setup_signal_handlers=False)
    try:
        assert run.db is run._db
        assert run.session_factory is run.db.session_factory
        assert run._engine is not None
        assert run.sync_engine is run._engine.sync_engine
    finally:
        await run.aclose()


async def test_make_engine_seam_builds_the_run_engine(tmp_path: Path) -> None:
    # Subclasses (jent's ReplayRun) override _make_engine to swap the pool.
    class _SeamRun(ScrapeRun):
        make_engine_calls = 0

        async def _make_engine(self):
            type(self).make_engine_calls += 1
            return await super()._make_engine()

    run = _SeamRun(
        TrivialScraper(),
        tmp_path / "run.db",
        transport=SpyTransport(),
        rate_limited=False,
        resume=False,
    )
    await run.open(setup_signal_handlers=False)
    try:
        assert _SeamRun.make_engine_calls == 1
        assert run._engine is not None
    finally:
        await run.aclose()


async def test_default_transport_is_an_httpx_transport(
    tmp_path: Path,
) -> None:
    # With no transport injected, open() builds and brings up an
    # HttpxTransport, and aclose() tears it down.
    run = ScrapeRun(
        TrivialScraper(),
        tmp_path / "run.db",
        rate_limited=False,
        resume=False,
    )
    await run.open(setup_signal_handlers=False)
    assert run.transport is not None  # built an HttpxTransport
    await run.aclose()


async def _insert_resolved(run: ScrapeRun, step_name: str, count: int) -> None:
    """Insert ``count`` resolved (response-bearing) rows for ``step_name``."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        for i in range(count):
            content = (
                f"<html><body>Opinion {step_name} {i} "
                f"lorem ipsum dolor sit</body></html>"
            ).encode()
            compressed = compress(content)
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location, response_status_code,
                        response_url, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES ('completed', 9, :qc, 'GET', :url, :cont, '', 200,
                        :url, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "qc": i + 1,
                    "url": f"https://example.com/{step_name}/{i}",
                    "cont": step_name,
                    "compressed": compressed,
                    "osize": len(content),
                    "csize": len(compressed),
                },
            )
        await session.commit()


async def test_compactor_startup_seeds_below_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 10)
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    # Init the DB without seeding compactors yet, then plant rows.
    await run._init_db()
    await _insert_resolved(run, "parse", 4)

    await run._seed_compactors()

    compactor = run.compactor_for("parse")
    assert compactor is not None
    assert compactor.count == 4  # seeded with current resolved count
    assert compactor.done is False
    # Below threshold => no dictionary trained at startup.
    sf = run._db._session_factory  # type: ignore[union-attr]
    assert await get_compression_dict(sf, "parse") is None

    await transport.open()
    await run.aclose()


async def test_compactor_startup_trains_at_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 8)
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run._init_db()
    await _insert_resolved(run, "parse", 8)

    await run._seed_compactors()

    # At/over threshold with no dict => trained now, no live compactor seeded.
    assert run.compactor_for("parse") is None
    dict_result = await get_compression_dict(
        run._db._session_factory,  # type: ignore[union-attr]
        "parse",
    )
    assert dict_result is not None

    await transport.open()
    await run.aclose()


async def test_compactor_startup_skips_when_dict_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 4)
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run._init_db()
    await _insert_resolved(run, "parse", 3)
    # Plant a pre-existing dictionary for the step.
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                "INSERT INTO compression_dicts "
                "(continuation, version, dictionary_data, sample_count) "
                "VALUES ('parse', 1, :d, 1)"
            ),
            {"d": compress(b"x")},
        )
        await session.commit()

    await run._seed_compactors()

    # A step that already has a dictionary gets no compactor.
    assert run.compactor_for("parse") is None

    await transport.open()
    await run.aclose()


async def test_spawn_worker_registers_and_deregisters(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open(setup_signal_handlers=False)
    try:
        run.stop()  # so the spawned worker exits on its first idle check
        before = run.active_worker_count
        worker_id = run.spawn_worker()
        assert isinstance(worker_id, int)
        assert run.active_worker_count == before + 1

        # The on-done callback deregisters the worker once it exits.
        task = run._worker_tasks[worker_id]
        await task
        await asyncio.sleep(0)  # let the done-callback fire
        assert run.active_worker_count == before
    finally:
        await run.aclose()


async def test_status_transitions_unstarted_to_done(tmp_path: Path) -> None:
    transport = SpyTransport()
    run = _make_run(tmp_path / "run.db", transport=transport)
    await run.open(setup_signal_handlers=False)
    try:
        assert await run.status() == "unstarted"
        run.stop()
        await run.run()
        assert await run.status() == "done"
    finally:
        await run.aclose()


# --- Graceful resume (T1.4) ---------------------------------------------


class _ServingTransport(SpyTransport):
    """A SpyTransport that resolves every request to a trivial 200 response."""

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        return Response(
            status_code=200,
            headers={},
            content=b"<html></html>",
            text="<html></html>",
            url=queued.request.request.url,
            request=queued.request,
        )


async def _seed_in_progress(run: ScrapeRun) -> None:
    """Insert one ``in_progress`` row addressed to the ``parse`` step."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        await session.execute(
            sa.text(
                """
                INSERT INTO requests (
                    status, priority, queue_counter, request_type, method,
                    url, continuation, current_location)
                VALUES ('in_progress', 5, 1, 'navigating', 'GET',
                    'http://127.0.0.1/page1', 'parse', '')
                """
            )
        )
        await session.commit()


async def _status_counts(run: ScrapeRun) -> dict[str, int]:
    """Group the requests table by status into a {status: count} dict."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        result = await session.execute(
            sa.text("SELECT status, COUNT(*) FROM requests GROUP BY status")
        )
        return {row[0]: row[1] for row in result.all()}


async def test_resume_resets_in_progress_to_pending(tmp_path: Path) -> None:
    # Seed an interrupted (in_progress) row on a temp-file DB.
    db_path = tmp_path / "run.db"
    seed_run = _make_run(db_path)
    await seed_run._init_db()
    await _seed_in_progress(seed_run)
    assert await _status_counts(seed_run) == {"in_progress": 1}
    await seed_run.aclose()

    # Reopen the SAME db with resume=True: restore_queue resets it to pending.
    resumed = ScrapeRun(
        TrivialScraper(),
        db_path,
        transport=_ServingTransport(),
        rate_limited=False,
        resume=True,
    )
    await resumed.open(setup_signal_handlers=False)
    try:
        assert await _status_counts(resumed) == {"pending": 1}

        # Bonus: a subsequent run() drains the restored row to completion.
        await resumed.run()
        assert await _status_counts(resumed) == {"completed": 1}
    finally:
        await resumed.aclose()


# --- Error budget (max_persistent_errors) --------------------------------


def _persistent_exc() -> PersistentHTTPResponseException:
    return PersistentHTTPResponseException(404, "http://127.0.0.1/gone")


async def test_error_budget_stops_the_run_gracefully(tmp_path: Path) -> None:
    # Non-transient errors are charged via the _store_error funnel; hitting
    # the budget sets the stop event (graceful, resumable) exactly once.
    run = _make_run(tmp_path / "run.db", max_persistent_errors=2)
    await run.open(setup_signal_handlers=False)
    try:
        await run._store_error(_persistent_exc(), request_url="u1")
        assert not run.stop_event.is_set()
        # Unclassified exceptions never retry either, so they count too.
        await run._store_error(ValueError("parse bug"), request_url="u2")
        assert run.stop_event.is_set()
        assert run.persistent_error_count == 2
        # Stragglers past the threshold still count, without re-stopping.
        await run._store_error(_persistent_exc(), request_url="u3")
        assert run.persistent_error_count == 3
    finally:
        await run.aclose()


async def test_transient_errors_do_not_charge_the_budget(
    tmp_path: Path,
) -> None:
    # A transient that exhausted its retries stores an error row but is the
    # circuit breaker's signal, not scraper breakage: never budgeted.
    run = _make_run(tmp_path / "run.db", max_persistent_errors=1)
    await run.open(setup_signal_handlers=False)
    try:
        for _ in range(3):
            await run._store_error(
                RequestTimeoutException("http://127.0.0.1/slow", 30.0)
            )
        assert run.persistent_error_count == 0
        assert not run.stop_event.is_set()
    finally:
        await run.aclose()


async def test_unbudgeted_run_counts_but_never_stops(tmp_path: Path) -> None:
    # Default None = unlimited: the meter still runs for host visibility.
    run = _make_run(tmp_path / "run.db")
    await run.open(setup_signal_handlers=False)
    try:
        for _ in range(5):
            await run._store_error(_persistent_exc())
        assert run.persistent_error_count == 5
        assert not run.stop_event.is_set()
    finally:
        await run.aclose()
