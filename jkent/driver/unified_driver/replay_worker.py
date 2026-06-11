"""Replay worker: a :class:`PoolWorker` with the replay failure taxonomy.

A live run retries transient errors and marks failures; a *replay* run has no
network, so retrying just re-serves the same source row forever and a ``failed``
row must never reach the output DB. Instead every miss or error is funnelled
through the configured miss policy (``raise`` / ``skip`` / ``stub``):

- a :class:`ReplayMiss` from ``resolve`` (no source row) → miss policy;
- a ``TransientException`` from the continuation → transient miss policy
  (``stub`` walks to the nearest reseedable anchor so a downstream run re-fetches
  a clean subtree);
- any other continuation error → treated as a miss (the response was fine; the
  code/data shape is the problem — re-fetching won't help, so stub the row).

The cross-worker *retry-eligible parent* set lives on the
:class:`ReplayTransport`; children of a retried parent are force-missed here
unless ``trust_subtree_after_retry`` is set.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from jkent.common.exceptions import RequestFailedHalt, TransientException
from jkent.data_types import ArchiveResponse
from jkent.driver.unified_driver.persistence import ReplayStorage
from jkent.driver.unified_driver.transport import QueuedRequest
from jkent.driver.unified_driver.transport.replay_transport import (
    ReplayMiss,
    ReplayTransport,
)
from jkent.driver.unified_driver.worker import PoolWorker

if TYPE_CHECKING:
    from jkent.data_types import Request, Response

logger = logging.getLogger(__name__)


class ReplayWorker(PoolWorker):
    """A :class:`PoolWorker` that routes misses/errors through the miss policy.

    Args:
        miss_policy: What to do on a miss (``raise`` / ``skip`` / ``stub``).
        trust_subtree_after_retry: When False (default), children of a
            retry-eligible parent are unconditionally force-missed; when True
            they are looked up normally.
    """

    def __init__(
        self,
        *args: Any,
        miss_policy: str = "stub",
        trust_subtree_after_retry: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._miss_policy = miss_policy
        self._trust_subtree_after_retry = trust_subtree_after_retry

    @property
    def _replay_transport(self) -> ReplayTransport:
        assert isinstance(self._transport, ReplayTransport)
        return self._transport

    @property
    def _replay_storage(self) -> ReplayStorage:
        assert isinstance(self._storage, ReplayStorage)
        return self._storage

    async def _handle_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
    ) -> None:
        """Resolve from the index and route every miss/error to the policy."""
        # Force-miss children of a retried parent (unless trusted).
        if (
            parent_request_id is not None
            and not self._trust_subtree_after_retry
            and self._replay_transport.is_retry_eligible_parent(
                parent_request_id
            )
        ):
            await self._apply_miss_policy(request_id, request)
            return

        try:
            handle = await self._transport.acquire(self.worker_id)
            queued = QueuedRequest(
                request=request,
                request_id=request_id,
                parent_request_id=parent_request_id,
            )
            continuation_name = self._continuation_name(request)

            started = time.monotonic()
            try:
                if getattr(request, "archive", False):
                    response = await self._replay_resolve_archive(
                        handle, queued
                    )
                else:
                    response = await self._transport.resolve(
                        handle,
                        queued,
                        await_conditions=self._await_conditions(
                            continuation_name
                        ),
                    )
            except ReplayMiss:
                await self._apply_miss_policy(request_id, request)
                return
            duration_s = time.monotonic() - started

            if (
                getattr(request, "is_speculative", False)
                and self._track_speculation is not None
            ):
                await self._track_speculation(request, response)

            try:
                await self._continuation.complete_request(
                    request_id,
                    response,
                    request,
                    continuation_name,
                    page=getattr(handle, "page", None),
                )
            except RequestFailedHalt:
                raise
            except TransientException as exc:
                await self._apply_transient_miss_policy(
                    request_id, request, exc
                )
                return
            except Exception as exc:
                logger.warning(
                    "Replay continuation error treated as miss: url=%s "
                    "error=%s: %s",
                    self._request_url(request),
                    type(exc).__name__,
                    exc,
                )
                await self._apply_miss_policy(request_id, request)
                return

            if self._on_request_duration is not None:
                self._on_request_duration(duration_s)
            await self._record_for_compactor(continuation_name)

        except RequestFailedHalt:
            raise
        except TransientException as exc:
            # A transient from acquire/resolve itself (not the continuation):
            # replay can't retry it, so route through the transient policy.
            await self._apply_transient_miss_policy(request_id, request, exc)
        except Exception:
            logger.exception(
                "Replay worker %d error on request %d treated as miss",
                self.worker_id,
                request_id,
            )
            await self._apply_miss_policy(request_id, request)

    async def _replay_resolve_archive(
        self, handle: Any, queued: QueuedRequest
    ) -> Response:
        """Build an ``ArchiveResponse`` pointing at the source file verbatim.

        Replay never copies an archive: it streams nothing and references the
        stored file path directly, so the continuation stores an
        ``archived_files`` row pointing at the source DB's file.
        """
        stream = await self._transport.resolve_archive(handle, queued)
        try:
            # ReplayTransport returns a file-backed stream carrying the source
            # path; reference it verbatim (no copy).
            file_path = getattr(stream, "file_path", None)
            assert file_path is not None, "replay archive stream has no path"
            return ArchiveResponse(
                status_code=stream.status_code,
                headers=dict(stream.headers),
                content=b"",
                text="",
                url=stream.url,
                request=queued.request,
                file_url=file_path,
            )
        finally:
            await self._transport.finish_archiving(stream)

    async def _apply_miss_policy(
        self, request_id: int, request: Request
    ) -> None:
        """Translate a miss (or non-transient error) into a terminal state."""
        if self._miss_policy == "raise":
            raise RequestFailedHalt(
                f"replay miss: {self._request_url(request)}"
            )
        if self._miss_policy == "skip":
            logger.info(
                "Replay miss (skip): url=%s", self._request_url(request)
            )
            await self._replay_storage.delete_request_row(request_id)
            return
        logger.info("Replay miss (stub): url=%s", self._request_url(request))
        await self._replay_storage.stub_request(request_id)

    async def _apply_transient_miss_policy(
        self,
        request_id: int,
        request: Request,
        exc: TransientException,
    ) -> None:
        """Miss policy for a transient error raised from inside a step.

        ``stub`` walks the output-DB parent chain to the nearest reseedable
        anchor (or the root) and stubs that, so a downstream run re-fetches a
        clean subtree; ``raise`` halts; ``skip`` deletes the current row.
        """
        if self._miss_policy == "raise":
            raise RequestFailedHalt(
                f"transient error during replay: {exc}"
            ) from exc
        if self._miss_policy == "skip":
            logger.info(
                "Replay transient (skip): url=%s error=%s",
                self._request_url(request),
                exc,
            )
            await self._replay_storage.delete_request_row(request_id)
            return
        logger.warning(
            "Replay transient; walking to reseedable anchor: url=%s error=%s: %s",
            self._request_url(request),
            type(exc).__name__,
            exc,
        )
        await self._replay_storage.stub_with_reseedable_walk(request_id)
