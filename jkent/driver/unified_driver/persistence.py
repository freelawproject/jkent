"""Concrete persistence components for the unified driver.

``RequestQueue`` owns the unified driver's DB-backed queue: it subclasses
``database_engine.queue.RequestQueueDB`` (the shared (de)serialization /
dequeue / staged-enqueue methods) and adds the unified-specific glue — the
progress-emitting ``enqueue_request``. :data:`ProgressCallback` is the event
hook every driver component forwards to (:func:`no_progress` when unset).
``ResponseStorage`` is ``database_engine.storage.ResponseStorageDB`` (the
lifecycle / response / result storage methods) under its driver-facing name.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from jkent.common.exceptions import TransientException
from jkent.data_types import Request, Response
from jkent.driver.database_engine.errors import error_message
from jkent.driver.database_engine.queue import RequestQueueDB
from jkent.driver.database_engine.storage import ResponseStorageDB

if TYPE_CHECKING:
    from jkent.common.rate_limits import RateLimitTable
    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)

#: A progress-event hook: ``(event_type, data)``.
ProgressCallback: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[None]]


async def no_progress(event_type: str, data: dict[str, Any]) -> None:
    """The :data:`ProgressCallback` for a run nobody is watching."""


class RequestQueue(RequestQueueDB):
    """DB-backed request queue (enqueue/dequeue/(de)serialize) for the unified driver."""

    def __init__(
        self,
        db: SQLManager,
        *,
        on_progress: ProgressCallback | None = None,
        rate_limits: RateLimitTable | None = None,
    ) -> None:
        super().__init__(db, rate_limits=rate_limits)
        self._on_progress = on_progress or no_progress

    async def enqueue_request(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None = None,
    ) -> None:
        """Enqueue a new request to the database.

        Persists the request to SQLite.

        Args:
            new_request: The new request to enqueue.
            context: Response or originating request for URL resolution.
            parent_request_id: Optional parent request ID for tracking request relationships.
        """
        request_data, progress_event = await self._prepare_enqueue(
            new_request, context, parent_request_id
        )
        # A deduplicated insert gets no progress event, matching the staged
        # path.
        inserted = await self.db.insert_request(request_data)
        if inserted.inserted:
            await self._on_progress("request_enqueued", progress_event)


#: Response/result storage and retry/backoff handling for the unified driver.
ResponseStorage = ResponseStorageDB


class ErrorSink(Protocol):
    """Where a worker reports that something went wrong.

    One seam for the whole failure surface, so a caller cannot report half
    of a failure. The two methods are different reports:

    * :meth:`request_failed` is *terminal* — the request is done and it
      lost. It writes both halves of the record (the request row's status
      and ``last_error``, and the ``errors`` row).
    * :meth:`record` only files the error. For a failure the caller is
      going to handle some other way — a replay host stubs the row and
      walks to a reseedable anchor — the diagnosis is still worth keeping
      even though "failed" is the wrong verdict for the row.

    A run also charges non-transient reports against its error budget, so
    routing every failure through here is what keeps that count honest.
    """

    async def request_failed(
        self,
        request_id: int,
        exc: Exception,
        *,
        request_url: str | None = None,
    ) -> None:
        """Mark ``request_id`` failed and file ``exc`` against it."""
        ...

    async def record(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        """File ``exc`` without changing any request's status.

        ``request_id`` is None for errors raised outside a request.
        """
        ...


class RowOnlyErrorSink:
    """The :class:`ErrorSink` for a worker with no run behind it.

    Marks the request row failed so the queue's accounting stays honest, and
    files nothing — there is no ``errors`` table owner to file it with. What
    :meth:`record` is handed is logged instead, so the diagnosis is not lost
    without a trace. Direct worker construction (tests, minimal hosts) gets
    this by default.
    """

    def __init__(self, storage: ResponseStorage) -> None:
        self._storage = storage

    async def request_failed(
        self,
        request_id: int,
        exc: Exception,
        *,
        request_url: str | None = None,
    ) -> None:
        await self._storage.mark_request_failed(request_id, error_message(exc))

    async def record(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        # request_failed needs no log of its own: the worker logs a terminal
        # failure before it reports one.
        logger.warning(
            "Unfiled error on request %s (%s): %s",
            request_id,
            request_url,
            exc,
            exc_info=exc,
        )


class ErrorBudget:
    """The run's :class:`ErrorSink`: files every error and meters the budget.

    Every worker failure path funnels through here, so this is the single
    counting site. Transient-exhausted errors are filed but not charged —
    server distress is the circuit breaker's signal; the budget guards
    against *scraper* breakage: persistent HTTP, assumption violations, and
    unclassified exceptions, none of which retrying will fix. Exhausting the
    budget calls ``stop`` once (a graceful, resumable shutdown), never
    mid-request. The count runs whether or not a budget is set, so hosts can
    watch scraper health on unbudgeted runs too.
    """

    def __init__(
        self,
        db: SQLManager,
        *,
        max_persistent_errors: int | None,
        stop: Callable[[], None],
    ) -> None:
        self._db = db
        self.max_persistent_errors = max_persistent_errors
        self._stop = stop
        self._persistent_error_count = 0
        self._exhausted = False

    @property
    def persistent_error_count(self) -> int:
        """Never-retried failures filed so far (the error-budget meter)."""
        return self._persistent_error_count

    @property
    def exhausted(self) -> bool:
        """Whether the budget has been hit (and ``stop`` called)."""
        return self._exhausted

    def exhaustion_message(self) -> str:
        """The run's ``error_message`` for a budget stop: count and limit."""
        return (
            f"Error budget exhausted: {self._persistent_error_count} "
            f"persistent error(s) reached "
            f"max_persistent_errors={self.max_persistent_errors}"
        )

    async def request_failed(
        self,
        request_id: int,
        exc: Exception,
        *,
        request_url: str | None = None,
    ) -> None:
        """Mark the request failed and file ``exc`` against it.

        Both halves of the record — the request row's status and
        ``last_error``, and the ``errors`` row — are written in one
        transaction, so a FAILED row always has the error that explains it.
        """
        await self._db.fail_request(request_id, exc, request_url=request_url)
        self._charge(exc)

    async def record(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        """Persist an error row; charge a non-transient one to the budget."""
        await self._db.store_error(
            exc, request_id=request_id, request_url=request_url
        )
        self._charge(exc)

    def _charge(self, exc: Exception) -> None:
        """Count a non-transient error; stop the run once, at the budget."""
        if isinstance(exc, TransientException):
            return
        self._persistent_error_count += 1
        if (
            self.max_persistent_errors is not None
            and self._persistent_error_count >= self.max_persistent_errors
            and not self._exhausted
        ):
            self._exhausted = True
            logger.error(
                "Stopping run: %d persistent errors reached the "
                "max_persistent_errors budget (%d)",
                self._persistent_error_count,
                self.max_persistent_errors,
            )
            self._stop()
