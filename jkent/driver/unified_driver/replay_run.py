"""Replay run: a :class:`ScrapeRun` configured to replay from source DB(s).

Handles the *orchestration* half of replay (the parts that aren't execution):
it swaps in a :class:`ReplayTransport` (index lookups instead of network), a
:class:`ReplayWorker` (miss-policy failure routing), and a
:class:`ReplayStorage` (stub/delete/finalize terminal states), then adds the
two Run-level hooks the replay workflow needs — seeding mode-3 reseedable anchors
into the queue on ``open`` and finalizing stubs on ``aclose``.

The replay does no network I/O, so the run is unrate-limited by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jkent.driver.unified_driver.persistence import ReplayStorage
from jkent.driver.unified_driver.replay_worker import ReplayWorker
from jkent.driver.unified_driver.run import ScrapeRun
from jkent.driver.unified_driver.transport.replay_transport import (
    MatchMode,
    ReplayTransport,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jkent.data_types import BaseScraper

# Miss policies: what to do when a yielded request has no source-DB match.
MissPolicy = str  # Literal["raise", "skip", "stub"]


class ReplayRun(ScrapeRun):
    """A :class:`ScrapeRun` that replays a scraper from previous-run DB(s).

    Args:
        source_db_paths: Source DB paths in priority order.
        miss_policy: ``raise`` / ``skip`` / ``stub`` on a source-index miss.
        mode: Replay mode (see :data:`MatchMode`).
        trust_subtree_after_retry: When False (default), children of a
            retry-eligible parent are force-missed; when True they are looked
            up normally.
    """

    def __init__(
        self,
        scraper: BaseScraper[Any],
        db_path: Path,
        *,
        source_db_paths: list[Path],
        miss_policy: str = "stub",
        mode: MatchMode = "curr-error-free",
        trust_subtree_after_retry: bool = False,
        **kwargs: Any,
    ) -> None:
        self._miss_policy = miss_policy
        self._trust_subtree_after_retry = trust_subtree_after_retry
        transport = ReplayTransport(
            source_db_paths, mode=mode, scraper=scraper
        )
        super().__init__(
            scraper,
            db_path,
            transport=transport,
            rate_limited=kwargs.pop("rate_limited", False),
            **kwargs,
        )

    def _make_storage(self) -> ReplayStorage:
        assert self._db is not None
        return ReplayStorage(self._db, max_backoff_time=self.max_backoff_time)

    def _make_worker(self, worker_id: int) -> ReplayWorker:
        assert self._queue is not None
        assert self._transport is not None
        assert self._rate_limiter is not None
        assert self._continuation is not None
        assert self._storage is not None
        assert self._monitor is not None
        return ReplayWorker(
            worker_id,
            queue=self._queue,
            transport=self._transport,
            rate_limiter=self._rate_limiter,
            continuation=self._continuation,
            storage=self._storage,
            stop_event=self.stop_event,
            scraper=self.scraper,
            archive_handler=self._archive_handler,
            on_request_duration=self._monitor.record_request_duration,
            compactor_for=self.compactor_for,
            store_error=self._store_error,
            track_speculation=(
                self._speculation.track_outcome
                if self._speculation is not None
                else None
            ),
            miss_policy=self._miss_policy,
            trust_subtree_after_retry=self._trust_subtree_after_retry,
        )

    async def open(self, *, setup_signal_handlers: bool = True) -> None:
        """Bring the run up, then seed any mode-3 reseedable anchors."""
        await super().open(setup_signal_handlers=setup_signal_handlers)
        await self._seed_anchors()

    async def _seed_anchors(self) -> None:
        """Insert mode-3 anchor rows as pending entry requests (no-op otherwise).

        Their dedup_keys were excluded from the index, so they miss and get
        stubbed for re-fetch by a downstream run.
        """
        assert self._db is not None
        assert isinstance(self._transport, ReplayTransport)
        for row in self._transport.seed_anchor_rows():  # type: ignore
            await self._db.insert_request(  # type: ignore
                priority=row["priority"],
                parent_id=None,
                method=row["method"],
                url=row["url"],
                headers_json=row["headers_json"],
                cookies_json=row["cookies_json"],
                body=row["body"],
                continuation=row["continuation"],
                current_location=row["current_location"],
                accumulated_data_json=row["accumulated_data_json"],
                permanent_json=row["permanent_json"],
                dedup_key=row["deduplication_key"],
                verify=row["verify"],
                bypass_rate_limit=bool(row["bypass_rate_limit"]),
                request_type=row["request_type"],
                expected_type=row["expected_type"],
                via_json=row["via_json"],
                timeout_json=row["timeout_json"],
                json_data=row["json_data"],
                files_json=row["files_json"],
                auth_json=row["auth_json"],
                allow_redirects=bool(row["allow_redirects"]),
                proxies_json=row["proxies_json"],
                stream=bool(row["stream"]),
                cert_json=row["cert_json"],
            )

    async def aclose(self) -> None:
        """Finalize stubs (drop descendants, stub → pending), then tear down."""
        if self._db is not None and isinstance(self._storage, ReplayStorage):
            await self._storage.finalize_stubs()
        await super().aclose()
