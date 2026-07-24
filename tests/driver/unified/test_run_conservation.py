"""Generative conservation rigs for ``Run``: cross-worker exactly-once.

The worker conformance suite is single-worker by construction, so it
cannot see cross-worker races; these rigs drive the real ``ScrapeRun``
stack (real queue, storage, continuation — DB-backed) with a counting
transport and assert the global laws:

- exactly-once: with N seeded requests and W concurrent workers, every
  request is resolved exactly once and every row ends ``completed``
  (a double-dequeue across workers would show up as a duplicate id);
- stop/resume round trip: ``stop()`` fired mid-run loses nothing — the
  first run completes what it resolved, leaves the rest pending, and a
  fresh resumed run over the same DB finishes exactly the remainder, so
  across both runs each request is resolved exactly once.

Sync + asyncio.run per example, factory-built subjects (no fixtures in
``@given`` bodies) — the same pattern as the other generative rigs.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
from collections import Counter
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from hypothesis import given
from hypothesis import strategies as st

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver.unified_driver.transport import NoopHandle, Transport
from tests.driver.unified.test_run import _NoSignalScrapeRun

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from jkent.driver.unified_driver.run import ScrapeRun

pytestmark = pytest.mark.generative


class DrainScraper(BaseScraper[dict]):
    """One no-op step: every seeded request completes without yielding more."""

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        return
        yield  # pragma: no cover - makes this a generator

    @step
    def parse(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        return
        yield  # pragma: no cover - makes this a generator


class CountingTransport(Transport[NoopHandle]):
    """Resolves everything with 200 + HTML, logging each request id.

    ``stop_after`` + ``on_stop`` script a graceful shutdown: as the k-th
    resolve starts, ``on_stop`` (wired to ``run.stop()``) fires — the
    in-flight requests finish, new dequeues stop.
    """

    def __init__(self, stop_after: int | None = None) -> None:
        self.resolved: list[int] = []
        self.stop_after = stop_after
        self.on_stop: Callable[[], None] | None = None
        self._handles: dict[int, NoopHandle] = {}
        self.closed = False

    async def open(self) -> None:
        return None

    async def aclose(self) -> None:
        self.closed = True

    async def acquire(self, worker_id: int) -> NoopHandle:
        return self._handles.setdefault(worker_id, NoopHandle())

    async def release(self, worker_id: int) -> None:
        self._handles.pop(worker_id, None)

    async def resolve(
        self, handle: Any, queued: Any, await_conditions: Any = ()
    ) -> Response:
        self.resolved.append(queued.request_id)
        if (
            self.stop_after is not None
            and len(self.resolved) == self.stop_after
            and self.on_stop is not None
        ):
            self.on_stop()
        return Response(
            status_code=200,
            headers={},
            content=b"<html></html>",
            text="<html></html>",
            url=queued.request.request.url,
            request=queued.request,
        )

    async def resolve_archive(
        self, handle: Any, queued: Any, decision: Any = None
    ) -> Any:
        raise AssertionError("archive is not part of the conservation rigs")

    async def finish_archiving(self, stream: Any) -> None:
        return None


async def _seed_pending(run: ScrapeRun, count: int) -> list[int]:
    """Insert ``count`` pending request rows; return their ids."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        for i in range(count):
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location)
                    VALUES ('pending', 9, :qc, 'GET', :url, 'parse', '')
                    """
                ),
                {"qc": i + 1, "url": f"https://conserve.test/{i}"},
            )
        await session.commit()
        rows = await session.execute(
            sa.text("SELECT id FROM requests ORDER BY id")
        )
        return [row[0] for row in rows]


async def _status_counts(run: ScrapeRun) -> dict[str, int]:
    """Request-row counts by status."""
    sf = run._db._session_factory  # type: ignore[union-attr]
    async with sf() as session:
        rows = await session.execute(
            sa.text("SELECT status, COUNT(*) FROM requests GROUP BY status")
        )
        return {row[0]: row[1] for row in rows.all()}


class _RunFactory:
    """Builds ScrapeRuns over per-call DB paths under one temp dir."""

    def __init__(self, workdir: Path, schema_template: Path) -> None:
        self.workdir = workdir
        self.schema_template = schema_template
        self._counter = itertools.count()

    def fresh_db(self) -> Path:
        db_path = self.workdir / f"run-{next(self._counter)}.db"
        shutil.copy(self.schema_template, db_path)
        return db_path

    def make_run(
        self,
        db_path: Path,
        transport: CountingTransport,
        *,
        num_workers: int,
        resume: bool,
    ) -> ScrapeRun:
        return _NoSignalScrapeRun(
            DrainScraper(),
            db_path,
            transport=transport,
            rate_limited=False,
            resume=resume,
            num_workers=num_workers,
        )


@pytest.fixture(scope="module")
def run_factory(
    schema_template: Path, tmp_path_factory: pytest.TempPathFactory
) -> _RunFactory:
    return _RunFactory(
        tmp_path_factory.mktemp("conservation"), schema_template
    )


@given(
    n=st.integers(min_value=0, max_value=10),
    workers=st.integers(min_value=1, max_value=4),
)
def test_workers_complete_each_request_exactly_once(
    run_factory: _RunFactory, n: int, workers: int
) -> None:
    """N requests × W workers: every request resolved once, all completed."""

    async def drive() -> tuple[list[int], list[int], dict[str, int], str]:
        transport = CountingTransport()
        run = run_factory.make_run(
            run_factory.fresh_db(),
            transport,
            num_workers=workers,
            resume=False,
        )
        await run.open()
        try:
            ids = await _seed_pending(run, n)
            await run.run()
            counts = await _status_counts(run)
            status = await run.status()
        finally:
            await run.aclose()
        return ids, transport.resolved, counts, status

    ids, resolved, counts, status = asyncio.run(drive())

    assert sorted(resolved) == ids  # exactly once each — no double-dequeue
    assert counts.get("completed", 0) == len(ids)
    assert counts.get("pending", 0) == 0
    assert counts.get("in_progress", 0) == 0
    assert status == "done"


@st.composite
def _stop_resume_scenarios(draw: st.DrawFn) -> tuple[int, int, int]:
    """(seeded count, stop after k-th resolve, workers)."""
    n = draw(st.integers(min_value=1, max_value=10))
    stop_after = draw(st.integers(min_value=1, max_value=n))
    workers = draw(st.integers(min_value=1, max_value=4))
    return n, stop_after, workers


@given(scenario=_stop_resume_scenarios())
def test_stop_then_resume_completes_exactly_once_overall(
    run_factory: _RunFactory, scenario: tuple[int, int, int]
) -> None:
    """stop() mid-run + a resumed run = every request done exactly once."""
    n, stop_after, workers = scenario

    async def drive() -> tuple[
        list[int], list[int], list[int], dict[str, int]
    ]:
        db_path = run_factory.fresh_db()

        first_transport = CountingTransport(stop_after=stop_after)
        first = run_factory.make_run(
            db_path, first_transport, num_workers=workers, resume=False
        )
        first_transport.on_stop = first.stop
        await first.open()
        try:
            ids = await _seed_pending(first, n)
            await first.run()
            mid_counts = await _status_counts(first)
        finally:
            await first.aclose()

        # Interrupted run accounting: everything it resolved completed,
        # everything else is still pending — nothing lost, nothing doubled.
        assert mid_counts.get("completed", 0) == len(first_transport.resolved)
        assert mid_counts.get("in_progress", 0) == 0
        assert mid_counts.get("pending", 0) == n - len(
            first_transport.resolved
        )

        second_transport = CountingTransport()
        second = run_factory.make_run(
            db_path, second_transport, num_workers=workers, resume=True
        )
        await second.open()
        try:
            await second.run()
            final_counts = await _status_counts(second)
        finally:
            await second.aclose()

        return (
            ids,
            first_transport.resolved,
            second_transport.resolved,
            final_counts,
        )

    ids, first_resolved, second_resolved, final_counts = asyncio.run(drive())

    # The two runs partition the work: exactly once each, overall.
    assert sorted(first_resolved + second_resolved) == ids
    assert not (Counter(first_resolved) & Counter(second_resolved))
    assert len(first_resolved) >= 1  # the stop trigger fired
    assert final_counts.get("completed", 0) == len(ids)
    assert final_counts.get("pending", 0) == 0
