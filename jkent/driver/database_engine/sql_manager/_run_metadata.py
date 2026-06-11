"""Run metadata operations for SQLManager."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func, select, update

from jkent.driver.database_engine.models import (
    Request,
    RequestStatus,
    RunMetadata,
    RunStatus,
)
from jkent.driver.database_engine.timestamps import now

if TYPE_CHECKING:
    import asyncio

    from sqlalchemy.ext.asyncio import async_sessionmaker

logger = logging.getLogger(__name__)


class RunMetadataMixin:
    """RunMetadata table database operations."""

    _lock: asyncio.Lock  # type: ignore[misc]
    _session_factory: async_sessionmaker  # type: ignore[misc]

    async def init_run_metadata(
        self,
        scraper_name: str,
        scraper_version: str | None,
        num_workers: int,
        max_backoff_time: float,
        jitter: float = 0.0,
        speculation_config: dict[str, dict[str, int]] | None = None,
        browser_config: dict[str, Any] | None = None,
        seed_params: list[dict[str, dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize run metadata in database.

        Only creates a new entry if one doesn't exist.

        Args:
            scraper_name: Name of the scraper class.
            scraper_version: Version string if available.
            num_workers: Number of concurrent workers.
            max_backoff_time: Maximum total backoff time before failure.
            jitter: Fraction of each retry backoff drawn as jitter, recorded
                so a corpus says how spread out its retries were.
            speculation_config: Optional dict mapping continuation name to
                {"threshold": int, "speculation": int} for speculative handling.
            browser_config: Optional dict with browser configuration for Playwright
                driver (browser_type, headless, viewport, user_agent, etc.).
            seed_params: Optional list of {entry_name: kwargs} dicts for
                initial_seed() invocation. Stored for run resumability.
        """
        async with self._lock, self._session_factory() as session:
            result = await session.execute(
                select(RunMetadata.id).where(RunMetadata.id == 1)
            )
            if result.scalar() is not None:
                return

            run = RunMetadata(
                id=1,
                scraper_name=scraper_name,
                scraper_version=scraper_version,
                jitter=jitter,
                num_workers=num_workers,
                max_backoff_time=max_backoff_time,
                speculation_config_json=(
                    json.dumps(speculation_config)
                    if speculation_config
                    else None
                ),
                browser_config_json=(
                    json.dumps(browser_config) if browser_config else None
                ),
                seed_params_json=(
                    json.dumps(seed_params)
                    if seed_params is not None
                    else None
                ),
            )
            session.add(run)
            await session.commit()

    async def get_seed_params(
        self,
    ) -> list[dict[str, dict[str, Any]]] | None:
        """Get the seed parameters for initial_seed() from run metadata.

        Returns:
            List of {entry_name: kwargs} dicts, or None if not stored.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunMetadata.seed_params_json).where(RunMetadata.id == 1)
            )
            val = result.scalar()
            if val:
                return json.loads(val)
            return None

    async def update_seed_params(
        self,
        seed_params: list[dict[str, dict[str, Any]]],
    ) -> None:
        """Overwrite the stored ``seed_params_json`` in run metadata.

        Used by ``--add-params`` to keep the stored intent in sync so that
        speculation filtering during ``run()`` doesn't drop templates added
        to an existing run.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == 1)
                .values(seed_params_json=json.dumps(seed_params))
            )
            await session.commit()

    async def restore_queue(self) -> int:
        """Restore pending requests from database on startup.

        Resets any in_progress requests to pending (they were interrupted).

        Returns:
            Number of pending requests after restoration.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(Request)
                .where(Request.status == RequestStatus.IN_PROGRESS)
                .values(status=RequestStatus.PENDING)
            )
            result = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status == RequestStatus.PENDING)
            )
            count = result.scalar() or 0
            await session.commit()
            return count

    async def close_run(self) -> None:
        """Clean up database state on driver close.

        Resets in_progress requests to pending and updates run status.

        Note: this is best-effort cleanup on the way down. Any failure is
        logged and swallowed rather than raised, so a close that cannot
        commit may leave rows in 'in_progress' and the run status as
        'running'. This is acceptable because the next startup's
        restore_queue() resets in_progress -> pending, and a resumed run
        overwrites the stored status when it starts; a stale 'running' is
        only ever observed on a database no run has reopened.
        """
        async with self._lock:
            try:
                async with self._session_factory() as session:
                    await session.execute(
                        update(Request)
                        .where(Request.status == RequestStatus.IN_PROGRESS)
                        .values(status=RequestStatus.PENDING)
                    )
                    await session.execute(
                        update(RunMetadata)
                        .where(RunMetadata.id == 1)
                        .values(
                            status=sa.case(
                                (
                                    RunMetadata.status == RunStatus.RUNNING,
                                    RunStatus.INTERRUPTED,
                                ),
                                else_=RunMetadata.status,
                            ),
                            ended_at=now(),
                        )
                    )
                    await session.commit()
            except Exception as e:
                logger.warning(f"Failed to update state on close: {e}")

    async def update_run_status_running(self) -> None:
        """Mark run as running."""
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == 1)
                .values(
                    status=RunStatus.RUNNING,
                    started_at=now(),
                )
            )
            await session.commit()

    async def finalize_run(self, status: RunStatus, error: str | None) -> None:
        """Finalize run with a final status and optional error.

        Args:
            status: Final status — any :class:`RunStatus` other than
                ``RUNNING``, which goes through
                :meth:`update_run_status_running` so ``started_at`` is set.
            error: Error message if status is error.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == 1)
                .values(
                    status=status,
                    ended_at=now(),
                    error_message=error,
                )
            )
            await session.commit()

    async def update_run_status(self, status: RunStatus) -> None:
        """Update run status.

        Args:
            status: New :class:`RunStatus`.
        """
        if status == RunStatus.RUNNING:
            await self.update_run_status_running()
        else:
            await self.finalize_run(status, None)

    async def save_browser_cookies(self, cookies_json: str) -> None:
        """Save browser cookies to run metadata for resume.

        Args:
            cookies_json: JSON-encoded browser cookies.
        """
        async with self._lock, self._session_factory() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == 1)
                .values(browser_cookies_json=cookies_json)
            )
            await session.commit()

    async def get_browser_cookies(self) -> str | None:
        """Get saved browser cookies from run metadata.

        Returns:
            JSON-encoded browser cookies, or None if not saved.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunMetadata.browser_cookies_json).where(
                    RunMetadata.id == 1
                )
            )
            return result.scalar()

    async def has_any_requests(self) -> bool:
        """Check if there are any requests in the database.

        Returns:
            True if there are any requests, False otherwise.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count()).select_from(Request)
            )
            return (result.scalar() or 0) > 0

    async def get_run_metadata(self) -> dict[str, Any] | None:
        """Get run metadata from database.

        Returns:
            Dict with run metadata or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(RunMetadata).where(RunMetadata.id == 1)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None

            return {
                "scraper_name": row.scraper_name,
                "scraper_version": row.scraper_version,
                "status": row.status,
                "created_at": row.created_at,
                "started_at": row.started_at,
                "ended_at": row.ended_at,
                "error_message": row.error_message,
                "jitter": row.jitter,
                "num_workers": row.num_workers,
                "max_backoff_time": row.max_backoff_time,
                "speculation_config": (
                    json.loads(row.speculation_config_json)
                    if row.speculation_config_json
                    else None
                ),
                "browser_config": (
                    json.loads(row.browser_config_json)
                    if row.browser_config_json
                    else None
                ),
            }
