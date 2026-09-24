"""Run metadata operations for SQLManager."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Final

from pydantic import AliasChoices, Field
from sqlalchemy import func, select, update

from jkent.common.serialization import (
    JsonColumn,
    dump_json_or_none,
    parse_json_column,
)
from jkent.driver.database_engine.models import (
    Request,
    RequestStatus,
    RunMetadata,
    RunStatus,
)
from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._types import RowModel
from jkent.driver.database_engine.timestamps import now

logger = logging.getLogger(__name__)

#: The id of the single ``run_metadata`` row. The table is a singleton — one
#: run per database — and this names that convention wherever it is queried.
RUN_METADATA_ID: Final = 1

#: The seed parameter sets a run was started with: ``{entry_name: kwargs}``
#: dicts, one per ``initial_seed()`` invocation.
SeedParams = list[dict[str, dict[str, Any]]]


class RunMetadataRecord(RowModel):
    """The run's metadata row, as :meth:`get_run_metadata` returns it.

    ``seed_params_json`` comes back decoded as ``seed_params``; ``None``
    where nothing was stored.
    """

    scraper_name: str
    scraper_version: str | None
    status: RunStatus
    created_at: datetime | None
    started_at: datetime | None
    ended_at: datetime | None
    error_message: str | None
    jitter: float
    num_workers: int
    max_backoff_time: float
    seed_params: Annotated[SeedParams | None, JsonColumn] = Field(
        default=None,
        validation_alias=AliasChoices("seed_params", "seed_params_json"),
    )


class RunMetadataMixin(SQLManagerBase):
    """RunMetadata table database operations."""

    async def init_run_metadata(
        self,
        scraper_name: str,
        scraper_version: str | None,
        num_workers: int,
        max_backoff_time: float,
        jitter: float = 0.0,
        seed_params: SeedParams | None = None,
    ) -> None:
        """Initialize run metadata in database, or refresh it on resume.

        Creates the singleton row on a fresh database. On an existing one
        — ``run.py`` calls this on every open, resume included — the row
        is kept but the columns that describe *this* session's
        configuration are re-stamped: ``scraper_version``,
        ``num_workers``, ``max_backoff_time`` and ``jitter``. Without
        that they stay frozen at whatever the first invocation used, so a
        run resumed with ``--workers 8`` after crashing at 4 still reports
        4, and a corpus resumed after a scraper fix records the version of
        the code that produced only its first half — which is the one
        attribute a corpus most needs to be right about, since an audit
        uses it to decide whether a missing field is site drift or a
        scraper bug.

        ``seed_params_json`` is write-once: a run database is pinned to
        the seed set it was created with, and a resume passes none.

        What does not describe the session is checked, not re-stamped: a
        database belongs to the scraper that created it, and seed params
        passed again (a fresh run that failed before seeding) must be the
        recorded ones.

        Args:
            scraper_name: Name of the scraper class.
            scraper_version: Version string if available.
            num_workers: Number of concurrent workers.
            max_backoff_time: Maximum total backoff time before failure.
            jitter: Fraction of each retry backoff drawn as jitter, recorded
                so a corpus says how spread out its retries were.
            seed_params: Optional list of {entry_name: kwargs} dicts for
                initial_seed() invocation. Stored for run resumability.

        Raises:
            ValueError: The database was created by another scraper, or
                with other seed params. Nothing is written.
        """
        seed_params_json = dump_json_or_none(seed_params)
        async with self._write_session() as session:
            existing = (
                await session.execute(
                    select(
                        RunMetadata.scraper_name, RunMetadata.seed_params_json
                    ).where(RunMetadata.id == RUN_METADATA_ID)
                )
            ).first()
            if existing is not None:
                if existing.scraper_name != scraper_name:
                    raise ValueError(
                        f"run database belongs to scraper "
                        f"{existing.scraper_name!r}, not {scraper_name!r}; "
                        "use a new run database for a different scraper."
                    )
                if seed_params is not None and parse_json_column(
                    existing.seed_params_json
                ) != parse_json_column(seed_params_json):
                    raise ValueError(
                        f"run database was created with seed_params "
                        f"{existing.seed_params_json}, not {seed_params_json}; "
                        "a run is pinned to its seed set, so recreate the "
                        "run database to run different params."
                    )
                await session.execute(
                    update(RunMetadata)
                    .where(RunMetadata.id == RUN_METADATA_ID)
                    .values(
                        scraper_version=scraper_version,
                        num_workers=num_workers,
                        max_backoff_time=max_backoff_time,
                        jitter=jitter,
                    )
                )
                await session.commit()
                return

            run = RunMetadata(
                id=RUN_METADATA_ID,
                scraper_name=scraper_name,
                scraper_version=scraper_version,
                jitter=jitter,
                num_workers=num_workers,
                max_backoff_time=max_backoff_time,
                seed_params_json=seed_params_json,
            )
            session.add(run)
            await session.commit()

    async def get_seed_params(self) -> SeedParams | None:
        """The seed parameters for initial_seed(), or None if not stored."""
        metadata = await self.get_run_metadata()
        return metadata.seed_params if metadata is not None else None

    @staticmethod
    async def _release_in_progress(session: Any) -> None:
        """Reset in_progress requests to pending (they were interrupted)."""
        await session.execute(
            update(Request)
            .where(Request.status == RequestStatus.IN_PROGRESS)
            .values(status=RequestStatus.PENDING)
        )

    async def restore_queue(self) -> int:
        """Restore pending requests from database on startup.

        Resets any in_progress requests to pending (they were interrupted).

        Returns:
            Number of pending requests after restoration.
        """
        async with self._write_session() as session:
            await self._release_in_progress(session)
            pending = await session.execute(
                select(func.count())
                .select_from(Request)
                .where(Request.status == RequestStatus.PENDING)
            )
            count = pending.scalar() or 0
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

        The metadata update only touches a row still in RUNNING — this close
        is then the thing that ended the run, so it stamps ``ended_at``. A
        run already finalized (COMPLETED / ERROR / INTERRUPTED) keeps the
        ``ended_at`` that finalization wrote: restamping it here would make
        the recorded duration grow with every later open/close of the file
        (post-run processing, inspection sessions).
        """
        try:
            async with self._write_session() as session:
                await self._release_in_progress(session)
                await session.execute(
                    update(RunMetadata)
                    .where(
                        RunMetadata.id == RUN_METADATA_ID,
                        RunMetadata.status == RunStatus.RUNNING,
                    )
                    .values(
                        status=RunStatus.INTERRUPTED,
                        ended_at=now(),
                    )
                )
                await session.commit()
        except Exception as e:
            logger.warning(
                "Failed to update state on close: %s", e, exc_info=True
            )

    async def update_run_status_running(self) -> None:
        """Mark run as running.

        ``started_at`` is stamped only the first time, so a resumed run's
        duration spans every session; a previous session's ``ended_at`` is
        cleared.
        """
        async with self._write_session() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == RUN_METADATA_ID)
                .values(
                    status=RunStatus.RUNNING,
                    started_at=func.coalesce(RunMetadata.started_at, now()),
                    ended_at=None,
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
        async with self._write_session() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == RUN_METADATA_ID)
                .values(
                    status=status,
                    ended_at=now(),
                    error_message=error,
                )
            )
            await session.commit()

    async def save_browser_cookies(self, cookies_json: str) -> None:
        """Save browser cookies to run metadata for resume.

        Args:
            cookies_json: JSON-encoded browser cookies.
        """
        async with self._write_session() as session:
            await session.execute(
                update(RunMetadata)
                .where(RunMetadata.id == RUN_METADATA_ID)
                .values(browser_cookies_json=cookies_json)
            )
            await session.commit()

    async def get_browser_cookies(self) -> str | None:
        """Get saved browser cookies from run metadata.

        Returns:
            JSON-encoded browser cookies, or None if not saved.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(RunMetadata.browser_cookies_json).where(
                    RunMetadata.id == RUN_METADATA_ID
                )
            )
            return result.scalar()

    async def get_run_metadata(self) -> RunMetadataRecord | None:
        """Get run metadata from database.

        Returns:
            :class:`RunMetadataRecord` or None if not found.
        """
        async with self.session_factory() as session:
            row = await session.get(RunMetadata, RUN_METADATA_ID)
            return RunMetadataRecord.model_validate(row) if row else None
