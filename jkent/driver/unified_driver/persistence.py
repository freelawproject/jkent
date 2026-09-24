"""Concrete persistence components for the unified driver.

``RequestQueue`` is ``database_engine.queue.RequestQueue`` (the
(de)serialization / dequeue / staged-enqueue methods) and ``ResponseStorage``
is ``database_engine.storage.ResponseStorage`` (the lifecycle / response /
result storage methods), both re-exported here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from jkent.common.exceptions import TransientException
from jkent.driver.database_engine.errors import error_message
from jkent.driver.database_engine.queue import RequestQueue as RequestQueue
from jkent.driver.database_engine.storage import (
    ResponseStorage as ResponseStorage,
)

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)


class ErrorSink(Protocol):
    """Where a worker reports that something went wrong.

    One seam for the whole failure surface, so a caller cannot report half
    of a failure. The two methods are different reports:

    * :meth:`request_failed` is *terminal* — the request is done and it
      lost. It writes both halves of the record (the request row's status
      and ``last_error``, and the ``errors`` row).
    * :meth:`record` only files the error. It is replay's: a replay host
      stubs the row and walks to a reseedable anchor, and the diagnosis is
      still worth keeping even though "failed" is the wrong verdict for
      the row. Replay resolves those failures itself, so :meth:`record` is
      never charged against the error budget.

    A failure raised after a request completed is neither: that work is
    best-effort, must leave the row valid when it fails, and is logged
    where it runs.
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

    def __init__(self, db: SQLManager) -> None:
        """Bind the sink to the run database that owns the request rows.

        Args:
            db: The run database whose request rows are marked failed.
        """
        self._db = db

    async def request_failed(
        self,
        request_id: int,
        exc: Exception,
        *,
        request_url: str | None = None,
    ) -> None:
        """Mark the request failed; file nothing.

        Args:
            request_id: The failed request's row id.
            exc: The exception that ended it.
            request_url: The request's URL, for symmetry with
                :class:`ErrorBudget`; unused here.
        """
        await self._db.mark_request_failed(request_id, error_message(exc))

    async def record(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        """Log the error instead of filing it.

        Args:
            exc: The exception to report.
            request_id: The request it belongs to, if any.
            request_url: That request's URL, if known.
        """
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

    Every terminal worker failure funnels through :meth:`request_failed`,
    so this is the single counting site. Transient-exhausted errors are
    filed but not charged — server distress is the circuit breaker's signal;
    the budget guards against *scraper* breakage: persistent HTTP, assumption
    violations, and unclassified exceptions, none of which retrying will
    fix. Exhausting the budget calls ``stop`` once (a graceful, resumable
    shutdown), never mid-request. The count runs whether or not a budget is
    set, so hosts can watch scraper health on unbudgeted runs too.

    The count is per session: it starts at 0, so resuming a stopped run
    grants it a fresh budget.
    """

    def __init__(
        self,
        db: SQLManager,
        *,
        max_persistent_errors: int | None,
        stop: Callable[[], None],
    ) -> None:
        """Build the run's error sink and budget meter.

        Args:
            db: The run database the errors are filed in.
            max_persistent_errors: The budget, or ``None`` for unlimited
                (the count still runs).
            stop: Called once when the budget is exhausted, for a
                graceful, resumable shutdown.

        Raises:
            ValueError: ``max_persistent_errors`` is below 1; use ``None``
                for no budget.
        """
        if max_persistent_errors is not None and max_persistent_errors < 1:
            raise ValueError(
                f"max_persistent_errors must be >= 1 or None, "
                f"got {max_persistent_errors}"
            )
        self._db = db
        self.max_persistent_errors = max_persistent_errors
        self._stop = stop
        self._persistent_error_count = 0
        self._exhausted = False

    @property
    def persistent_error_count(self) -> int:
        """Never-retried terminal failures this session (the budget meter)."""
        return self._persistent_error_count

    @property
    def exhausted(self) -> bool:
        """Whether the budget has been hit (and ``stop`` called)."""
        return self._exhausted

    def exhaustion_message(self) -> str:
        """The run's ``error_message`` for a budget stop.

        Returns:
            A message naming the persistent-error count and the limit.
        """
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
        A row enters FAILED once, so a request is charged at most once; a
        second report raises (see
        :class:`~jkent.driver.database_engine.sql_manager.IllegalTransitionError`)
        and charges nothing.
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
        """Persist an error row without charging the budget."""
        await self._db.store_error(
            exc, request_id=request_id, request_url=request_url
        )

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
