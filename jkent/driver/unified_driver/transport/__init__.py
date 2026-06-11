"""The ``Transport`` seam: the one thing the unified driver's backends differ on.

The driver core owns orchestration — the queue, the worker pool, storage,
rate limiting, retries. A ``Transport`` owns the other half: turning a
request into a response, the per-worker resource that work runs on, and
recovery when that resource dies. HTTP, Playwright, and replay become
three implementations of this one interface instead of a base class plus
overrides.

Interface only — no implementations live here yet.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from jkent.data_types import (
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle

if TYPE_CHECKING:
    from jkent.data_types import (
        ArchiveDecision,
        Request,
        Response,
    )

# A single wait directive from a step's ``await_list``. Transports that
# can't honor waits (HTTP, replay) ignore them; Playwright applies them
# before snapshotting the DOM.
AwaitCondition: TypeAlias = (
    WaitForSelector | WaitForLoadState | WaitForURL | WaitForTimeout
)


@dataclass(frozen=True)
class QueuedRequest:
    """A request paired with the persistence ids the queue assigned it.

    Formalizes the tuple the queue hands a worker: the request spec plus
    its row id and its parent's row id. Transports that need to reach the
    database for execution use these — Playwright stages the parent tab
    from ``parent_request_id`` and tags captured incidentals with
    ``request_id``. HTTP and replay ignore the ids.
    """

    request: Request
    request_id: int
    parent_request_id: int | None = None


@runtime_checkable
class WorkerHandle(Protocol):
    """A transport's per-worker resource (a browser page, or nothing).

    Acquired once per worker via :meth:`Transport.acquire`, reused across
    requests, and reset between them. HTTP and replay hand back a no-op
    handle since they hold no per-worker state.
    """

    async def reset_for_reuse(self) -> None:
        """Clear per-request state so the handle is ready for the next request."""
        ...

    async def close(self) -> None:
        """Release the underlying per-worker resource."""
        ...


@runtime_checkable
class ArchiveStream(Protocol):
    """A streamed archive body plus its response metadata.

    Returned by :meth:`Transport.resolve_archive`. The caller iterates the
    body in chunks and writes it to storage, then hands this object back to
    :meth:`Transport.finish_archiving` to release any transport-side
    backing — e.g. the temp file a Playwright download is staged to before
    it can be streamed.
    """

    status_code: int
    headers: dict[str, str]
    url: str

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Iterate the response body in chunks."""
        ...


HandleT = TypeVar("HandleT", bound=WorkerHandle)


class Transport(AsyncLifecycle, Protocol[HandleT]):
    """Executes requests for the driver, owning its own resource lifecycle.

    Composes :class:`AsyncLifecycle` (``open``/``aclose`` for the run-scoped
    resource, e.g. a browser engine). Crash recovery is **internal**, not a
    caller concern: a dead resource surfaces as a ``TransientException`` from
    ``resolve`` (which poisons the worker's handle), and the next ``acquire``
    rebuilds it — escalating to a single-flight restart of the shared
    resource when a crash poisons the whole handle cache. Callers drive none
    of this; they retry transients and re-``acquire``.
    """

    async def acquire(self, worker_id: int) -> HandleT:
        """Get or create the per-worker handle for ``worker_id``.

        A handle poisoned by a crash is rebuilt here; when the shared
        resource itself is dead this escalates to a single-flight restart of
        it. May raise ``TransientException`` if the rebuild can't complete
        (e.g. a context that can't be restarted), so callers treat ``acquire``
        like ``resolve`` for retry purposes.
        """
        ...

    async def release(self, worker_id: int) -> None:
        """Close and forget the handle for ``worker_id`` (worker exiting)."""
        ...

    async def resolve(
        self,
        handle: HandleT,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        """Fetch ``queued.request`` and return its response.

        ``await_conditions`` come from the target step's ``await_list``.
        Transports that can't wait (HTTP, replay) ignore them; Playwright
        applies them before snapshotting. Playwright also stages the parent
        tab from ``queued.parent_request_id`` and persists captured
        incidentals against ``queued.request_id`` via its own DB reference.

        A dead browser surfaces as a ``TransientException`` (the handle is
        poisoned so the next ``acquire`` rebuilds and, if needed, restarts).
        """
        ...

    async def resolve_archive(
        self,
        handle: HandleT,
        queued: QueuedRequest,
        decision: ArchiveDecision | None = None,
    ) -> ArchiveStream:
        """Begin an archive download and return a stream of its body.

        All archiving is streamed: the caller reads the body in chunks from
        the returned :class:`ArchiveStream` and writes it to storage. How
        the transport produces those chunks is its own concern — Playwright
        can only obtain a download as a local file, so it stages to a temp
        file and streams from there, released later by ``finish_archiving``.
        ``queued.request`` is an archive request; ``decision`` is a
        pre-computed archive-handler verdict, consulted by the transport
        itself when None.
        """
        ...

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Release any transport-side backing for ``stream``.

        Called once the caller has fully consumed the stream and persisted
        the body — the place to delete a staged temp file. A no-op for
        transports that stream directly (HTTP, replay).
        """
        ...
