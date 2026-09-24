"""ResponseStorageDB - request lifecycle and response/result storage.

Marks requests completed/failed, handles retry backoff, and stores responses /
archived files / results. Owned by the unified driver. All methods are pure DB
operations (no driver glue); the host supplies ``db: SQLManager`` and the
retry configuration.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from jkent.common.exceptions import ScraperConfigError, TransientException
from jkent.common.serialization import (
    dump_json,
    dump_json_or_none,
)
from jkent.data_types import (
    ArchiveResponse,
    Request,
    Response,
)
from jkent.driver.database_engine.compression import (
    compress_response,
    decompress_response,
)
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.database_engine.errors import error_message
from jkent.driver.database_engine.sql_manager import (
    ResultInsert,
    SQLManager,
    StoredResponse,
)

logger = logging.getLogger(__name__)

#: Floor on a retry wait. Retrying inside a second is not a backoff, it is a
#: second attempt — and at that scale the jitter below cannot separate two
#: workers by anything meaningful either.
MIN_RETRY_DELAY_S = 1.0

#: Default share of the computed backoff drawn as jitter, applied
#: symmetrically (a 0.3 fraction means the wait lands in +/-30% of nominal).
#: Sized to actually decorrelate a pool that failed together: at a few percent
#: the draws overlap, so a co-failed group stays a convoy across every retry
#: generation and re-arrives as the burst that tripped the server in the first
#: place. +/-30% separates the early generations — where the nominal waits are
#: only seconds apart — while still keeping the expected delay on the nominal
#: exponential curve.
DEFAULT_RETRY_JITTER = 0.3


def serialize_result(
    data: Any,
    validation_errors: list[dict[str, Any]] | None = None,
) -> ResultInsert:
    """Serialize a scraped record (and its validation errors) for storage.

    The record is valid exactly when no ``validation_errors`` are given.
    A valid record refuses raw bytes like any JSON column. An invalid one is
    diagnostic — the raw ``failed_doc`` and pydantic's error dicts carry
    arbitrary scraped values — so it is written with ``bytes_as_repr`` and
    the ``str()`` fallback, and recording it cannot raise over the
    ``DataFormatAssumptionException`` the caller is recording.
    """
    invalid = validation_errors is not None
    return ResultInsert(
        result_type=type(data).__name__,
        data_json=dump_json(data, bytes_as_repr=invalid),
        is_valid=not invalid,
        validation_errors_json=dump_json_or_none(
            validation_errors, bytes_as_repr=True
        ),
    )


class ResponseStorageDB:
    """Request lifecycle management and response/result/file storage.

    Provides methods for marking requests completed/failed, handling retries,
    and storing responses, archived files, and scraped results.
    """

    def __init__(
        self,
        db: SQLManager,
        *,
        max_backoff_time: float = 3600.0,
        retry_base_delay: float = 1.0,
        retry_jitter: float = DEFAULT_RETRY_JITTER,
    ) -> None:
        # A jitter of 1 or more can draw a zero or negative wait (the floor
        # then catches a share of every generation, which re-synchronises the
        # pool), and a non-positive base never grows past the floor.
        if max_backoff_time <= 0:
            raise ValueError("max_backoff_time must be positive")
        if retry_base_delay <= 0:
            raise ValueError("retry_base_delay must be positive")
        if not 0 <= retry_jitter < 1:
            raise ValueError("retry_jitter must be in [0, 1)")
        self.db = db
        self.max_backoff_time = max_backoff_time
        self.retry_base_delay = retry_base_delay
        self.retry_jitter = retry_jitter

    async def mark_request_completed(
        self,
        request_id: int,
        *,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> None:
        """Mark a request as completed in the database.

        Args:
            request_id: The database ID of the request.
            speculation_outcome: The outcome of a probe completed without a
                stored response (``terminated_early``); ``None`` leaves the
                column alone.
        """
        await self.db.mark_request_completed(
            request_id, speculation_outcome=speculation_outcome
        )

    async def load_preresolved_response(
        self, request_id: int, request: Request
    ) -> Response | None:
        """Rebuild the stored response for a pre-resolved request.

        Returns None when no response is stored (which for a request flagged
        ``preresolved`` is an invariant violation the caller treats as an
        error). The body is decompressed with the same dictionary it was
        stored under; headers round-trip from JSON.
        """
        stored = await self.db.get_stored_response(request_id)
        if stored is None:
            return None
        if stored.content_compressed:
            content = await decompress_response(
                self.db,
                stored.content_compressed,
                stored.compression_dict_id,
            )
        else:
            content = b""
        headers: dict[str, str] = (
            json.loads(stored.response_headers_json)
            if stored.response_headers_json
            else {}
        )
        return Response(
            status_code=stored.response_status_code,
            headers=headers,
            content=content,
            url=stored.response_url,
            request=request,
        )

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        """Mark a request as failed in the database.

        Args:
            request_id: The database ID of the request.
            error_message: Error message describing the failure.
        """
        await self.db.mark_request_failed(request_id, error_message)

    async def handle_retry(
        self, request_id: int, error: Exception
    ) -> float | None:
        """Handle retry logic for transient errors with exponential backoff.

        The wait is ``retry_base_delay * 2 ** retry_count``, capped at a
        quarter of ``max_backoff_time``, then spread by +/-``retry_jitter``
        and floored at :data:`MIN_RETRY_DELAY_S` — and, where the error
        carries a server-sent ``Retry-After``, at that too.

        The jitter is what keeps a pool from retrying in lockstep. Workers
        that trip the same transient error at the same moment — a site
        throttling everything at once, which is the common case — would
        otherwise compute an identical delay and come back as a synchronised
        burst, re-tripping the same limit. Drawing each wait independently
        spreads that burst out.

        The floor is applied *after* the jitter so the minimum is a real
        guarantee rather than a nominal one a downward draw could undercut.
        The drawn value is what accumulates into ``cumulative_backoff`` and
        what the caller is told, because it is what the request actually
        waits.

        Args:
            request_id: The database ID of the request.
            error: The transient exception that was raised.

        Returns:
            The scheduled retry delay in seconds, or None if the request
            should be marked as failed.
        """
        retry_state = await self.db.get_retry_state(request_id)
        if retry_state is None:
            return None

        retry_count, cumulative_backoff = retry_state

        # Exponential backoff.
        nominal_delay = self.retry_base_delay * (2**retry_count)

        # Cap the nominal delay at max_backoff_time / 4.
        max_individual_delay = self.max_backoff_time / 4
        nominal_delay = min(nominal_delay, max_individual_delay)

        # Spread the wait so a pool that failed together does not return
        # together. Symmetric, so the expected delay is still the nominal
        # backoff curve rather than a drifted-up version of it.
        spread = nominal_delay * self.retry_jitter
        next_retry_delay = max(
            MIN_RETRY_DELAY_S,
            nominal_delay + random.uniform(-spread, spread),
        )

        # A server-sent Retry-After floors the wait: never come back
        # earlier than the server asked. The value arrives parsed and
        # clamped by the transport (see parse_retry_after), so a hostile
        # header cannot stall the schedule — though what it adds still
        # counts against the cumulative budget below.
        if (
            isinstance(error, TransientException)
            and error.retry_after is not None
        ):
            next_retry_delay = max(next_retry_delay, error.retry_after)

        # Give up only once the total would *exceed* max_backoff_time: a wait
        # that exactly fills the budget is within it, and a Retry-After
        # clamped to the budget would otherwise fail with no retry at all.
        new_cumulative_backoff = cumulative_backoff + next_retry_delay
        if new_cumulative_backoff > self.max_backoff_time:
            logger.warning(
                f"Request {request_id} exceeded max backoff time "
                f"({new_cumulative_backoff:.1f}s > {self.max_backoff_time:.1f}s)"
            )
            return None

        # Schedule retry by resetting to pending with updated backoff tracking
        await self.db.schedule_retry(
            request_id,
            new_cumulative_backoff,
            next_retry_delay,
            error_message(error),
        )

        logger.info(
            f"Request {request_id} scheduled for retry #{retry_count + 1} "
            f"(delay: {next_retry_delay:.1f}s, cumulative: {new_cumulative_backoff:.1f}s)"
        )

        return next_retry_delay

    async def store_response(
        self,
        request_id: int,
        response: Response,
        step: str,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> int:
        """Store an HTTP response in the database.

        For regular responses, content is compressed and stored on the
        request row. For ArchiveResponse, content is NOT stored (it's already
        on disk); instead, file metadata is stored in the archived_files
        table.

        Args:
            request_id: The database ID of the associated request.
            response: The Response object to store.
            step: The step method that will process this response.
            speculation_outcome: :class:`SpeculationOutcome` for a
                speculative request; None for a non-speculative one.

        Returns:
            ``request_id`` — the response is stored on that request's row.

        Raises:
            ScraperConfigError: ``response`` is an :class:`ArchiveResponse`
                that names no file — the archive handler lost the download.
        """
        # Checked before anything is written: an archive with no file has no
        # body and no archived_files row, so storing it would complete the
        # request with the download silently gone.
        if isinstance(response, ArchiveResponse) and not response.file_url:
            raise ScraperConfigError(
                f"archive response for {response.url} has no file_url; the "
                "archive handler returned no path for the file"
            )
        stored = StoredResponse(
            response_status_code=response.status_code,
            response_headers_json=dump_json_or_none(response.headers),
            response_url=response.url,
            content_size_original=len(response.content or b""),
            content_size_compressed=0,
        )
        # Archived files live on disk, so no body is stored. An empty body
        # is stored as NULL (not b"") so readers distinguish "no content"
        # via an IS NULL check and never feed an empty buffer to zstd.
        if not isinstance(response, ArchiveResponse) and response.content:
            compressed, dict_id = await compress_response(
                self.db, response.content, step
            )
            stored.content_compressed = compressed
            stored.content_size_compressed = len(compressed)
            stored.compression_dict_id = dict_id

        await self.db.store_response(
            request_id, stored, speculation_outcome=speculation_outcome
        )

        # For ArchiveResponse, also store file metadata in archived_files
        if isinstance(response, ArchiveResponse):
            # Get expected_type from the request if it's an archive request
            expected_type: str | None = None
            if (
                isinstance(response.request, Request)
                and response.request.archive
            ):
                expected_type = response.request.expected_type

            await self.db.store_archived_file(
                request_id=request_id,
                file_path=response.file_url,
                original_url=response.url,
                expected_type=expected_type,
                file_size=response.file_size,
                content_hash=response.content_hash,
            )

        return request_id

    async def _store_result(
        self,
        request_id: int,
        data: Any,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> int:
        """Store a scraped result in the database (own transaction).

        Args:
            request_id: The database ID of the request that produced this result.
            data: The scraped data to store.
            validation_errors: Validation errors, when the data failed
                validation; None marks the result valid.

        Returns:
            The database ID of the stored result.
        """
        return await self.db.store_result(
            request_id, serialize_result(data, validation_errors)
        )
