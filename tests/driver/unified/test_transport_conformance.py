"""Reusable conformance suite for ``Transport`` (jkent.driver.unified_driver.transport).

A ``Transport`` is the seam the driver backends differ on: it turns a request
into a response, owns a per-worker resource (``WorkerHandle``), and streams
archive bodies (``ArchiveStream``). HTTP, Playwright, and replay become three
implementations of this one interface.

``TransportConformance`` collects the method contract every implementation must
honor; subclass it and override ``subject`` to test a concrete transport.

Contract under test (see ``transport_contract.md``):

- Structural conformance: the subject exposes the ``Transport`` surface
  (``open``/``aclose``/``acquire``/``release``/``resolve``/``resolve_archive``/
  ``finish_archiving``). ``Transport`` is a generic ``Protocol`` and is **not**
  ``runtime_checkable``, so conformance is checked structurally, not via
  ``isinstance``.
- Lifecycle: ``open`` then ``aclose`` complete (``Transport`` composes
  ``AsyncLifecycle``).
- ``acquire(worker_id)`` returns a usable ``WorkerHandle`` (``reset_for_reuse``
  and ``close`` are awaitable no-throw).
- ``acquire`` is stable per ``worker_id`` (get-or-create) until released.
- ``release(worker_id)`` then ``acquire`` yields a fresh handle.
- ``resolve`` returns a ``Response`` whose ``.request`` is the queued request,
  and tolerates ``await_conditions``.
- ``resolve_archive`` returns an ``ArchiveStream`` with valid metadata
  (``status_code``/``headers``/``url``) before iteration, that async-iterates
  to ``bytes`` chunks.
- ``finish_archiving(stream)`` is awaitable and does not throw.
- ``WorkerHandle`` and ``ArchiveStream`` ARE ``runtime_checkable``, so the
  reference fakes satisfy ``isinstance``.

Out of scope: the poison/restart recovery model (``TransientException`` re-map,
single-flight restart, ``Recoverable``). That is transport-internal and not part
of the public method contract — see "Recovery model" in the contract doc.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    WaitForLoadState,
)
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    QueuedRequest,
    Transport,
    WorkerHandle,
)

if TYPE_CHECKING:
    from jkent.data_types import ArchiveDecision
    from jkent.driver.unified_driver.transport import AwaitCondition


# --- Reference in-memory transport ---------------------------------------


class FakeWorkerHandle:
    """No-op per-worker handle (the HTTP/replay shape: no per-worker state)."""

    def __init__(self) -> None:
        self.reset_count = 0
        self.closed = False

    async def reset_for_reuse(self) -> None:
        """Record a reset; idempotent and safe to call repeatedly."""
        self.reset_count += 1

    async def close(self) -> None:
        """Mark the handle released."""
        self.closed = True


class FakeArchiveStream:
    """Carries response metadata and yields a couple of canned byte chunks."""

    def __init__(
        self, status_code: int, headers: dict[str, str], url: str
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.url = url
        self.finished = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the body in chunks (single-pass)."""
        yield b"chunk-one"
        yield b"chunk-two"


class FakeTransport:
    """Minimal in-memory ``Transport`` returning canned responses/streams."""

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self._handles: dict[int, FakeWorkerHandle] = {}

    async def open(self) -> None:
        self.opened = True

    async def aclose(self) -> None:
        self.closed = True

    async def acquire(self, worker_id: int) -> FakeWorkerHandle:
        handle = self._handles.get(worker_id)
        if handle is None:
            handle = FakeWorkerHandle()
            self._handles[worker_id] = handle
        return handle

    async def release(self, worker_id: int) -> None:
        handle = self._handles.pop(worker_id, None)
        if handle is not None:
            await handle.close()

    async def resolve(
        self,
        handle: FakeWorkerHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        return Response(
            status_code=200,
            headers={"content-type": "text/html"},
            content=b"<html></html>",
            text="<html></html>",
            url=queued.request.request.url,
            request=queued.request,
        )

    async def resolve_archive(
        self,
        handle: FakeWorkerHandle,
        queued: QueuedRequest,
        decision: ArchiveDecision | None = None,
    ) -> FakeArchiveStream:
        return FakeArchiveStream(
            status_code=200,
            headers={"content-type": "application/pdf"},
            url=queued.request.request.url,
        )

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        if isinstance(stream, FakeArchiveStream):
            stream.finished = True


# --- Conformance suite ----------------------------------------------------


class TransportConformance:
    """Method-contract tests every ``Transport`` implementation must pass.

    Subclass and override :meth:`subject`. ``make_queued`` builds the
    ``QueuedRequest`` fed to ``resolve``/``resolve_archive``; override it if an
    implementation needs a specific request shape.
    """

    @pytest.fixture
    def subject(self) -> Transport[WorkerHandle]:
        """The transport under test."""
        raise NotImplementedError

    def make_queued(self, *, request_id: int = 1) -> QueuedRequest:
        """Build a ``QueuedRequest`` for a simple GET."""
        return QueuedRequest(
            request=Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url="https://example.com"
                ),
                continuation="parse",
            ),
            request_id=request_id,
        )

    def test_structural_conformance(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """The subject exposes the full ``Transport`` surface (no isinstance)."""
        for method in (
            "open",
            "aclose",
            "acquire",
            "release",
            "resolve",
            "resolve_archive",
            "finish_archiving",
        ):
            assert callable(getattr(subject, method))

    async def test_open_then_aclose(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``open`` then ``aclose`` complete without error."""
        await subject.open()
        await subject.aclose()

    async def test_acquire_returns_usable_handle(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``acquire`` returns a ``WorkerHandle`` whose reset/close are no-throw."""
        await subject.open()
        handle = await subject.acquire(0)
        assert isinstance(handle, WorkerHandle)
        await handle.reset_for_reuse()
        await handle.close()
        await subject.aclose()

    async def test_acquire_is_stable_per_worker(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """Repeated ``acquire`` for one worker returns the same handle."""
        await subject.open()
        first = await subject.acquire(7)
        second = await subject.acquire(7)
        assert first is second
        await subject.aclose()

    async def test_release_yields_fresh_handle(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """After ``release``, the next ``acquire`` produces a new handle."""
        await subject.open()
        first = await subject.acquire(3)
        await subject.release(3)
        second = await subject.acquire(3)
        assert first is not second
        await subject.aclose()

    async def test_resolve_returns_response_for_queued(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``resolve`` returns a ``Response`` carrying the queued request."""
        await subject.open()
        handle = await subject.acquire(0)
        queued = self.make_queued()
        response = await subject.resolve(handle, queued)
        assert isinstance(response, Response)
        assert response.request is queued.request
        await subject.aclose()

    async def test_resolve_tolerates_await_conditions(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``resolve`` accepts ``await_conditions`` (ignored by non-Playwright)."""
        await subject.open()
        handle = await subject.acquire(0)
        queued = self.make_queued()
        response = await subject.resolve(
            handle, queued, await_conditions=[WaitForLoadState()]
        )
        assert isinstance(response, Response)
        await subject.aclose()

    async def test_resolve_archive_metadata_and_chunks(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``resolve_archive`` yields valid metadata then ``bytes`` chunks."""
        await subject.open()
        handle = await subject.acquire(0)
        queued = self.make_queued()
        stream = await subject.resolve_archive(handle, queued)
        assert isinstance(stream, ArchiveStream)

        # Metadata is valid before iteration begins.
        assert isinstance(stream.status_code, int)
        assert isinstance(stream.headers, dict)
        assert isinstance(stream.url, str)

        chunks = [chunk async for chunk in stream]
        assert chunks  # at least one chunk
        assert all(isinstance(chunk, bytes) for chunk in chunks)
        await subject.aclose()

    async def test_finish_archiving_is_no_throw(
        self, subject: Transport[WorkerHandle]
    ) -> None:
        """``finish_archiving`` is awaitable and releases backing without error."""
        await subject.open()
        handle = await subject.acquire(0)
        queued = self.make_queued()
        stream = await subject.resolve_archive(handle, queued)
        async for _ in stream:
            pass
        await subject.finish_archiving(stream)
        await subject.aclose()


# --- Reference transport binds the suite ----------------------------------


class TestReferenceTransport(TransportConformance):
    """Runs the conformance suite against the in-memory reference fake."""

    @pytest.fixture
    def subject(self) -> Transport[FakeWorkerHandle]:
        return FakeTransport()


# --- runtime_checkable helper protocols -----------------------------------


def test_fake_handle_is_a_worker_handle() -> None:
    """``WorkerHandle`` is runtime_checkable; the fake satisfies isinstance."""
    assert isinstance(FakeWorkerHandle(), WorkerHandle)


def test_fake_stream_is_an_archive_stream() -> None:
    """``ArchiveStream`` is runtime_checkable; the fake satisfies isinstance."""
    stream = FakeArchiveStream(200, {}, "https://example.com/file.pdf")
    assert isinstance(stream, ArchiveStream)
