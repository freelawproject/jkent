"""End-to-end integration for the unified driver (A6 — Track A capstone).

Assembles a real ``ScrapeRun`` with the live ``HttpxTransport``, queue, storage,
workers, compactors, and rate limiter, and drives a small scrape
against the in-thread aiohttp test server (the ``server_url`` fixture). This is
the proof the units compose: unit + conformance passing never exercises the
real worker → transport → step → storage path, nor the run's wiring of
the per-step compactors and the rate-limit gate.

Asserted outcomes of a full run:
  - every request resolved and was marked completed, with no errors;
  - each ``ParsedData`` reached ``on_data`` and a ``results`` row;
  - the rate-limit gate was consulted once per request;
  - the per-step compactors recorded completions.

Graceful, resumable stop: a stop signalled before any work leaves the seeded
entry request pending (nothing completed), so the run can be resumed later.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import Any

from typing_extensions import override

from jkent.common.decorators import entry, step
from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.unified_driver import RateLimiter, ScrapeRun
from jkent.driver.unified_driver.rate_limiter import RateLimiters
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks

# --- A small decorated scraper -------------------------------------------


class _E2EScraper(BaseScraper[dict[str, Any]]):
    """entry → /test, then fan out N child pages, each yielding one datum."""

    base = "http://127.0.0.1"
    page_count = 5

    @entry(dict)
    def start(self) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/test"
            ),
            step="parse_entry",
        )

    @step
    def parse_entry(
        self, response: Response
    ) -> Generator[Request, None, None]:
        for i in range(1, self.page_count + 1):
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url=f"{self.base}/page{i}"
                ),
                step="parse_page",
                accumulated_data={"page": i},
            )

    @step
    def parse_page(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(
            data={"page": response.request.accumulated_data["page"]}
        )


def _make_scraper(server_url: str, *, page_count: int = 5) -> _E2EScraper:
    scraper = _E2EScraper()
    scraper.base = server_url
    scraper.page_count = page_count
    return scraper


class _GateSpy(RateLimiter):
    """A ``RateLimiter`` that proxies an inner limiter and counts ``gate``."""

    def __init__(self, inner: RateLimiter) -> None:
        self._inner = inner
        self.gate_calls = 0

    async def gate(self, request: Request) -> None:
        self.gate_calls += 1
        return await self._inner.gate(request)

    @override
    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
    ) -> None:
        self._inner.record_response(
            status_code, retry_after=retry_after, url=url
        )


def _counts(db_path: Path) -> dict[str, int]:
    """Read row counts straight from the closed run DB file."""
    conn = sqlite3.connect(str(db_path))
    try:
        q = conn.execute
        return {
            "completed": q(
                "SELECT COUNT(*) FROM requests WHERE status = ?",
                (RequestStatus.COMPLETED.code,),
            ).fetchone()[0],
            "with_response": q(
                "SELECT COUNT(*) FROM requests "
                "WHERE response_status_code IS NOT NULL"
            ).fetchone()[0],
            "pending": q(
                "SELECT COUNT(*) FROM requests WHERE status = ?",
                (RequestStatus.PENDING.code,),
            ).fetchone()[0],
            "results": q("SELECT COUNT(*) FROM results").fetchone()[0],
            "errors": q("SELECT COUNT(*) FROM errors").fetchone()[0],
        }
    finally:
        conn.close()


# --- The capstone --------------------------------------------------------


async def test_end_to_end_scrape_composes(
    server_url: str, tmp_path: Path
) -> None:
    """A real scrape resolves, parses, persists, and wires every collaborator."""
    results: list[dict[str, Any]] = []

    async def on_data(data: Any) -> None:
        results.append(data)

    db_path = tmp_path / "run.db"
    scraper = _make_scraper(server_url, page_count=5)
    run = ScrapeRun(
        scraper,
        db_path,
        config=RunConfig(num_workers=2, rate_limited=False),
        hooks=RunHooks(on_data=on_data),
    )
    await run.open()

    # Spy the gate after open() — workers are built from run.collaborators
    # at spawn time, so swapping the limiter there is what they see.
    lanes = run.collaborators.rate_limiters
    gate = _GateSpy(lanes[DEFAULT_RATE_LIMIT])
    run._collaborators = replace(
        run.collaborators,
        rate_limiters=RateLimiters({**lanes, DEFAULT_RATE_LIMIT: gate}),
    )

    try:
        await run.run()

        # Observable: every page's datum flowed to on_data.
        assert sorted(r["page"] for r in results) == [1, 2, 3, 4, 5]

        # The rate-limit gate was consulted once per request (entry + 5 pages).
        assert gate.gate_calls == 6

        # Compactors recorded completions for each step.
        entry_compactor = run.compactors.for_step("parse_entry")
        page_compactor = run.compactors.for_step("parse_page")
        assert entry_compactor is not None and entry_compactor.count == 1
        assert page_compactor is not None and page_compactor.count == 5

        # A drained, unstopped run reports done.
        assert await run.status() == "done"
    finally:
        await run.aclose()

    # Persisted: every request resolved + completed, 5 results, no errors.
    counts = _counts(db_path)
    assert counts["completed"] == 6
    assert counts["with_response"] == 6
    assert counts["results"] == 5
    assert counts["errors"] == 0
    assert counts["pending"] == 0


async def test_graceful_stop_leaves_work_resumable(
    server_url: str, tmp_path: Path
) -> None:
    """A stop signalled before any work completes nothing and keeps it pending."""
    db_path = tmp_path / "run.db"
    scraper = _make_scraper(server_url, page_count=5)
    run = ScrapeRun(
        scraper,
        db_path,
        config=RunConfig(num_workers=2, rate_limited=False),
    )
    await run.open()
    run.stop()  # graceful shutdown requested before run()
    try:
        await run.run()
    finally:
        await run.aclose()

    # Nothing processed; the seeded entry request is still pending (resumable).
    counts = _counts(db_path)
    assert counts["completed"] == 0
    assert counts["results"] == 0
    assert counts["pending"] >= 1
