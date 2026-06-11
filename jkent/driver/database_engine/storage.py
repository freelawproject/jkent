"""ResponseStorageDB - request lifecycle and response/result storage.

Marks requests completed/failed, handles retry backoff, and stores responses /
archived files / results. Owned by the unified driver. All methods are pure DB
operations (no driver glue); the host supplies ``db: SQLManager`` and
``max_backoff_time``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from jkent.data_types import (
    ArchiveResponse,
    Request,
    Response,
)
from jkent.driver.database_engine.compression import (
    compress_response,
)
from jkent.driver.database_engine.sql_manager import SQLManager

logger = logging.getLogger(__name__)


class ResponseStorageDB:
    """Request lifecycle management and response/result/file storage.

    Provides methods for marking requests completed/failed, handling retries,
    and storing responses, archived files, and scraped results.
    """

    db: SQLManager  # type: ignore
    max_backoff_time: float  # type: ignore

    async def _mark_request_completed(self, request_id: int) -> None:
        """Mark a request as completed in the database.

        Args:
            request_id: The database ID of the request.
        """
        await self.db.mark_request_completed(request_id)

    async def _mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        """Mark a request as failed in the database.

        Args:
            request_id: The database ID of the request.
            error_message: Error message describing the failure.
        """
        await self.db.mark_request_failed(request_id, error_message)

    async def _handle_retry(
        self, request_id: int, error: Exception
    ) -> float | None:
        """Handle retry logic for transient errors with exponential backoff.

        Calculates the next retry delay using exponential backoff formula:
            next_retry_delay = base_delay * 2^retry_count

        Adds the delay to cumulative_backoff. If cumulative_backoff exceeds
        max_backoff_time, returns None to indicate the request should be
        marked as failed instead of retried.

        Args:
            request_id: The database ID of the request.
            error: The transient exception that was raised.

        Returns:
            The scheduled retry delay in seconds, or None if the request
            should be marked as failed.
        """
        # Get current retry state
        retry_state = await self.db.get_retry_state(request_id)
        if retry_state is None:
            return None

        retry_count, cumulative_backoff = retry_state

        # Calculate next retry delay with exponential backoff
        # Use a reasonable base delay (e.g., 1 second)
        retry_base_delay = 1.0
        next_retry_delay = retry_base_delay * (2**retry_count)

        # Cap individual retry delay at max_backoff_time / 4 to ensure
        # we don't have a single very long delay
        max_individual_delay = self.max_backoff_time / 4
        next_retry_delay = min(next_retry_delay, max_individual_delay)

        # Check if we would exceed max_backoff_time
        new_cumulative_backoff = cumulative_backoff + next_retry_delay
        if new_cumulative_backoff >= self.max_backoff_time:
            logger.warning(
                f"Request {request_id} exceeded max backoff time "
                f"({new_cumulative_backoff:.1f}s >= {self.max_backoff_time:.1f}s)"
            )
            return None

        # Schedule retry by resetting to pending with updated backoff tracking
        await self.db.schedule_retry(
            request_id, new_cumulative_backoff, next_retry_delay, str(error)
        )

        logger.info(
            f"Request {request_id} scheduled for retry #{retry_count + 1} "
            f"(delay: {next_retry_delay:.1f}s, cumulative: {new_cumulative_backoff:.1f}s)"
        )

        return next_retry_delay

    async def _store_response(
        self,
        request_id: int,
        response: Response,
        continuation: str,
        speculation_outcome: str | None = None,
    ) -> int:
        """Store an HTTP response in the database.

        For regular responses, content is compressed and stored in the responses table.
        For ArchiveResponse, content is NOT stored (it's already on disk); instead,
        file metadata is stored in the archived_files table.

        Args:
            request_id: The database ID of the associated request.
            response: The Response object to store.
            continuation: The continuation method that will process this response.
            speculation_outcome: For speculative requests: 'success', 'stopped', or 'skipped'.
                None for non-speculative requests.

        Returns:
            The database ID of the stored response.
        """
        # Serialize headers
        headers_json = (
            json.dumps(response.headers) if response.headers else None
        )

        # Check if this is an ArchiveResponse - file is already on disk
        is_archive = isinstance(response, ArchiveResponse)

        if is_archive:
            # For archived files, don't store content in database (it's on disk)
            # Store NULL for content to save space
            compressed = None
            content_size_original = (
                len(response.content) if response.content else 0
            )
            content_size_compressed = 0
            dict_id = None
        else:
            # Regular response - compress and store content
            content = response.content or b""
            content_size_original = len(content)

            if content_size_original > 0:
                compressed, dict_id = await compress_response(
                    self.db._session_factory,
                    content,
                    continuation,
                    db_lock=self.db._lock,
                )
                content_size_compressed = len(compressed)
            else:
                # Empty body: store NULL (not b"") so readers distinguish
                # "no content" via an IS NULL check and never feed an empty
                # buffer to zstd decompression.
                compressed = None
                dict_id = None
                content_size_compressed = 0

        await self.db.store_response(
            request_id=request_id,
            status_code=response.status_code,
            headers_json=headers_json,
            url=response.url,
            compressed_content=compressed,
            content_size_original=content_size_original,
            content_size_compressed=content_size_compressed,
            dict_id=dict_id,
            continuation=continuation,
            speculation_outcome=speculation_outcome,
        )

        # For ArchiveResponse, also store file metadata in archived_files
        if isinstance(response, ArchiveResponse) and response.file_url:
            # Get expected_type from the request if it's an archive request
            expected_type: str | None = None
            if (
                isinstance(response.request, Request)
                and response.request.archive
            ):
                expected_type = response.request.expected_type

            await self._store_archived_file(
                request_id=request_id,
                file_path=response.file_url,
                original_url=response.url,
                expected_type=expected_type,
                content=response.content,
            )

        return request_id

    async def _store_archived_file(
        self,
        request_id: int,
        file_path: str,
        original_url: str,
        expected_type: str | None,
        content: bytes | None,
    ) -> int:
        """Store archived file metadata in the database.

        Args:
            request_id: The database ID of the associated request.
            file_path: Local file system path where the file is stored.
            original_url: The URL the file was downloaded from.
            expected_type: Expected file type (pdf, audio, etc.).
            content: File content for computing hash and size.

        Returns:
            The database ID of the archived file record.
        """
        # Compute file size and content hash
        file_size = len(content) if content else 0
        content_hash = hashlib.sha256(content).hexdigest() if content else None

        return await self.db.store_archived_file(
            request_id=request_id,
            file_path=file_path,
            original_url=original_url,
            expected_type=expected_type,
            file_size=file_size,
            content_hash=content_hash,
        )

    @staticmethod
    def _serialize_result_for_storage(
        data: Any,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> tuple[str, str, str | None]:
        """Serialize result data + errors to (result_type, data_json, errors_json).

        Pulled out so the staging path can serialize without writing.
        """
        result_type = type(data).__name__

        if hasattr(data, "model_dump"):
            data_json = json.dumps(data.model_dump(mode="json"))
        elif hasattr(data, "dict"):
            data_json = json.dumps(data.dict())
        else:
            data_json = json.dumps(data)

        validation_errors_json: str | None = None
        if validation_errors:

            def make_serializable(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                if isinstance(obj, tuple):
                    return [make_serializable(item) for item in obj]
                if isinstance(obj, Exception):
                    return str(obj)
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

            validation_errors_json = json.dumps(
                make_serializable(validation_errors)
            )

        return result_type, data_json, validation_errors_json

    async def _store_result(
        self,
        request_id: int,
        data: Any,
        is_valid: bool = True,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> int:
        """Store a scraped result in the database.

        Args:
            request_id: The database ID of the request that produced this result.
            data: The scraped data to store.
            is_valid: Whether the data passed validation.
            validation_errors: List of validation errors if invalid.

        Returns:
            The database ID of the stored result.
        """
        result_type, data_json, validation_errors_json = (
            self._serialize_result_for_storage(data, validation_errors)
        )
        return await self.db.store_result(
            request_id=request_id,
            result_type=result_type,
            data_json=data_json,
            is_valid=is_valid,
            validation_errors_json=validation_errors_json,
        )
