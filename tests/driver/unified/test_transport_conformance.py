"""Reusable conformance suite for ``Transport`` (jkent.driver.unified_driver.transport).

A ``Transport`` is the seam the driver backends differ on: it turns a request
into a response, owns a per-worker resource (``WorkerHandle``), and streams
archive bodies (``ArchiveStream``). HTTP, Playwright, and replay become three
implementations of this one interface.

``TransportConformance`` collects the method contract every implementation must
honor; subclass it and override ``subject`` to test a concrete transport.

Contract under test (see ``transport_contract.md``):

- Surface conformance: the subject is a ``Transport`` (a generic ``abc.ABC``),
  exposing ``open``/``aclose``/``acquire``/``release``/``resolve``/
  ``resolve_archive``/``finish_archiving``. Implementations (and the reference
  fakes) subclass it, so conformance is nominal — ``isinstance`` holds.
- Lifecycle: ``open`` then ``aclose`` complete (``Transport`` composes
  ``AsyncLifecycle``).
- ``acquire(worker_id)`` returns a usable ``WorkerHandle`` (``reset_for_reuse``
  and ``close`` are awaitable no-throw).
- ``acquire`` is stable per ``worker_id`` (get-or-create) until released.
- ``release(worker_id)`` then ``acquire`` yields a fresh handle.
- ``resolve`` returns a ``Response`` whose ``.request`` is the queued request,
  and tolerates ``await_conditions``.
- ``resolve`` honors the scraper's status classifier (the shared
  ``Transport.classify_and_raise``): transient -> ``HTTPResponseAssumptionException``,
  persistent -> ``PersistentHTTPResponseException`` (``SpeculationHTTPFailure``
  for speculative requests), successful -> a ``Response``. Codes absent from
  the scraper's map default to persistent. This half of the contract lives in
  its own suite, ``ClassificationConformance``, because bindings need a
  scraper-parameterized subject and a way to surface an arbitrary status+body
  (a live status server for the real transports).
- ``resolve_archive`` returns an ``ArchiveStream`` with valid metadata
  (``status_code``/``headers``/``url``) before iteration, that async-iterates
  to ``bytes`` chunks.
- ``finish_archiving(stream)`` is awaitable and does not throw.
- ``WorkerHandle`` and ``ArchiveStream`` are ``abc.ABC``s the reference fakes
  subclass, so they satisfy ``isinstance``.

Out of scope: the poison/restart recovery model (``TransientException`` re-map,
single-flight restart, ``Recoverable``). That is transport-internal and not part
of the public method contract — see "Recovery model" in the contract doc.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import pytest

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    SpeculationHTTPFailure,
)
from jkent.data_types import (
    BaseScraper,
    HTTPCodeType,
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


class FakeWorkerHandle(WorkerHandle):
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


class FakeArchiveStream(ArchiveStream):
    """Carries response metadata and yields a couple of canned byte chunks."""

    def __init__(
        self, status_code: int, headers: dict[str, str], url: str
    ) -> None:
        super().__init__(status_code=status_code, headers=headers, url=url)
        self.finished = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield the body in chunks (single-pass)."""
        yield b"chunk-one"
        yield b"chunk-two"


class FakeTransport(Transport[FakeWorkerHandle]):
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


# --- ABC subclass helper checks -------------------------------------------


def test_fake_handle_is_a_worker_handle() -> None:
    """``WorkerHandle`` is an ABC the fake subclasses, so isinstance holds."""
    assert isinstance(FakeWorkerHandle(), WorkerHandle)


def test_fake_stream_is_an_archive_stream() -> None:
    """``ArchiveStream`` is an ABC the fake subclasses, so isinstance holds."""
    stream = FakeArchiveStream(200, {}, "https://example.com/file.pdf")
    assert isinstance(stream, ArchiveStream)


# --- Classification conformance --------------------------------------------

# The success body carries a marker asserted by substring, not equality:
# browser transports surface a DOM snapshot (normalized by the browser),
# not the raw wire bytes.
CLASSIFY_MARKER = b"classification-marker"
CLASSIFY_BODY = b"<html><body>classification-marker</body></html>"

ClassifyOutcome = Literal["response", "transient", "persistent"]


class ClassifyScraper(BaseScraper[Any]):
    """Default-classifier scraper, concrete so browser bindings can instantiate."""

    def get_entry(self):  # type: ignore[no-untyped-def]
        return iter(())


class OverrideClassifyScraper(ClassifyScraper):
    """Inverts the defaults: 503 passes through, 418 fails fast, 200 retries."""

    HTTP_CODE_TYPES = {
        200: HTTPCodeType.TRANSIENT,
        418: HTTPCodeType.PERSISTENT,
        503: HTTPCodeType.SUCCESSFUL,
    }


class BodyClassifyingScraper(ClassifyScraper):
    """Retries any response whose body contains ``b'RETRY'`` (a flaky server)."""

    @classmethod
    def is_transient_error(
        cls,
        status_code: int,
        headers: Mapping[str, str] | None = None,
        content: bytes | None = None,
    ) -> bool:
        if content is not None and b"RETRY" in content:
            return True
        return super().is_transient_error(status_code, headers, content)


def classify_expected(
    scraper: type[BaseScraper[Any]], code: int, body: bytes
) -> ClassifyOutcome:
    """The scraper's own verdict — the oracle ``classify_and_raise`` follows."""
    if scraper.is_transient_error(code, None, body):
        return "transient"
    if scraper.is_persistent_error(code, None, body):
        return "persistent"
    return "response"


class ClassificationConformance:
    """Status-classification contract every ``Transport.resolve`` must honor.

    ``resolve`` must run the observed status/headers/body through the
    scraper's classifier (the shared ``Transport.classify_and_raise``) and map
    each verdict to the right exception — transient codes raise
    ``HTTPResponseAssumptionException``, persistent codes raise
    ``PersistentHTTPResponseException`` (narrowed to ``SpeculationHTTPFailure``
    on speculative requests), successful codes come back as a
    ``Response``. The scraper's classifier itself is the oracle
    (:func:`classify_expected`), so these tests pin delegation, not any
    particular default table.

    Subclass and implement :meth:`classify`: resolve ONE request through the
    transport under test such that the transport observes ``code`` + ``body``,
    returning the ``Response`` and letting any exception propagate.
    ``scraper`` is a zero-arg-constructible class; bindings whose transport
    wants an instance instantiate it.

    A deeper generative sweep of the same contract (arbitrary override maps,
    the full code matrix) runs against the cheap HTTP transport in
    ``test_httpx_classification``.
    """

    async def classify(
        self,
        scraper: type[BaseScraper[Any]],
        code: int,
        body: bytes,
        *,
        speculative: bool = False,
    ) -> Response:
        """Resolve one request that surfaces ``code`` + ``body``."""
        raise NotImplementedError

    async def _assert_outcome(
        self, scraper: type[BaseScraper[Any]], code: int, body: bytes
    ) -> None:
        outcome = classify_expected(scraper, code, body)
        if outcome == "response":
            resp = await self.classify(scraper, code, body)
            assert resp.status_code == code
            assert CLASSIFY_MARKER in resp.content
        elif outcome == "transient":
            with pytest.raises(HTTPResponseAssumptionException) as excinfo:
                await self.classify(scraper, code, body)
            self._assert_error_payload(excinfo.value, code)
        else:
            with pytest.raises(PersistentHTTPResponseException) as excinfo:
                await self.classify(scraper, code, body)
            self._assert_error_payload(excinfo.value, code)

    def _assert_error_payload(self, exc: Any, code: int) -> None:
        """The raised HTTP exception carries what the transport observed.

        The worker persists the failed exchange from these attributes, so a
        transport that classifies without attaching them silently loses the
        error body/headers from the run db.
        """
        assert exc.status_code == code
        assert exc.headers, "classified headers must travel on the exception"
        assert exc.body is not None and CLASSIFY_MARKER in exc.body, (
            "classified body must travel on the exception"
        )

    @pytest.mark.parametrize("code", [200, 404, 503])
    async def test_default_classification(self, code: int) -> None:
        """Framework defaults: 200 passes through, 404 fails fast, 503 retries."""
        await self._assert_outcome(ClassifyScraper, code, CLASSIFY_BODY)

    @pytest.mark.parametrize("code", [200, 418, 503])
    async def test_override_classification(self, code: int) -> None:
        """A scraper's ``HTTP_CODE_TYPES`` overrides beat the defaults."""
        await self._assert_outcome(
            OverrideClassifyScraper, code, CLASSIFY_BODY
        )

    async def test_unlisted_code_is_persistent(self) -> None:
        """A status in no bucket fails fast rather than passing through.

        520 (Cloudflare's "origin returned an unknown error") is absent from
        the default map; the classifier's fallback makes it persistent, so a
        transport must not surface it as a successful ``Response``.
        """
        with pytest.raises(PersistentHTTPResponseException):
            await self.classify(ClassifyScraper, 520, CLASSIFY_BODY)

    async def test_content_override_makes_a_200_transient(self) -> None:
        """A dynamic body-based classifier can retry a well-statused response."""
        with pytest.raises(HTTPResponseAssumptionException):
            await self.classify(
                BodyClassifyingScraper,
                200,
                b"<html><body>please RETRY later</body></html>",
            )

    async def test_content_override_leaves_a_clean_200_a_response(
        self,
    ) -> None:
        resp = await self.classify(BodyClassifyingScraper, 200, CLASSIFY_BODY)
        assert resp.status_code == 200
        assert CLASSIFY_MARKER in resp.content

    async def test_persistent_code_on_speculative_is_a_speculation_failure(
        self,
    ) -> None:
        """Speculative probes surface persistent codes as speculation outcomes.

        This is the signal ``SpeculationManager`` counts failures by, so a
        transport that skips classification silently breaks speculation.
        """
        with pytest.raises(SpeculationHTTPFailure):
            await self.classify(
                ClassifyScraper, 404, CLASSIFY_BODY, speculative=True
            )


# --- Reference classification binding ---------------------------------------


class ClassifyingFakeTransport(FakeTransport):
    """``FakeTransport`` that classifies a canned (status, body) in resolve."""

    def __init__(
        self,
        scraper: type[BaseScraper[Any]],
        status_code: int,
        body: bytes,
    ) -> None:
        super().__init__()
        self._scraper = scraper
        self._status_code = status_code
        self._body = body

    async def resolve(
        self,
        handle: FakeWorkerHandle,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        url = queued.request.request.url
        self.classify_and_raise(
            self._scraper,
            queued.request,
            status_code=self._status_code,
            headers={"content-type": "text/html"},
            body=self._body,
            url=url,
        )
        return Response(
            status_code=self._status_code,
            headers={"content-type": "text/html"},
            content=self._body,
            text=self._body.decode(),
            url=url,
            request=queued.request,
        )


class TestReferenceTransportClassification(ClassificationConformance):
    """Runs the classification suite against the in-memory reference fake."""

    async def classify(
        self,
        scraper: type[BaseScraper[Any]],
        code: int,
        body: bytes,
        *,
        speculative: bool = False,
    ) -> Response:
        transport = ClassifyingFakeTransport(scraper, code, body)
        await transport.open()
        handle = await transport.acquire(0)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/r"
            ),
            continuation="parse",
            is_speculative=speculative,
        )
        try:
            return await transport.resolve(
                handle, QueuedRequest(request=request, request_id=1)
            )
        finally:
            await transport.aclose()
