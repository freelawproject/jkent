"""Tests for the unified driver's concrete worker (:class:`PoolWorker`).

Two layers:

* ``TestPoolWorkerConformance`` binds the real ``PoolWorker`` to the shared
  ``WorkerConformance`` suite via adapter fakes that present the *real*
  collaborator interfaces (the worker calls ``queue._get_next_request``,
  ``transport.acquire/release/resolve``, ``continuation.complete_request``,
  ``storage._handle_retry/_mark_request_failed``) while exposing the harness's
  observable surface (``.put``/``__len__``, ``.failures``, and the
  ``processed``/``retried``/``skipped`` lists).
* ``Test*`` targeted cases use spies for the finer-grained contract points:
  gate-before-timer ordering, archive ``should_download`` skip bypassing the
  gate and the network, success reporting to monitor + compactor, and the
  no-retry persistent/arbitrary-error paths.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest

from jkent.common.exceptions import (
    PersistentHTTPResponseException,
    TransientException,
)
from jkent.data_types import (
    ArchiveDecision,
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.unified_driver.transport.playwright_transport import (
    ResolveTimeout,
)
from jkent.driver.unified_driver.worker import PoolWorker
from tests.driver.unified.test_worker_conformance import (
    WorkerConformance,
    WorkerHarness,
)


def _make_request(
    url: str = "https://example.com/p", *, archive: bool = False
):
    """A minimal BaseRequest with the ``parse`` continuation."""
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        continuation="parse",
        current_location="https://example.com",
        archive=archive,
        expected_type="pdf" if archive else None,
    )


def _make_response(request: Any) -> Response:
    return Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url=request.request.url,
    )


# --- Adapter fakes presenting the real collaborator interfaces -----------


@dataclass
class AdapterQueue:
    """Real-interface queue keyed by request id, with the harness surface.

    ``put``/``__len__`` are the harness observable surface; the worker pulls
    via ``_get_next_request`` which maps each id to a simple BaseRequest.
    """

    _items: deque[int] = field(default_factory=deque)
    _requests: dict[int, Any] = field(default_factory=dict)

    def put(self, request_id: int) -> None:
        self._items.append(request_id)
        self._requests.setdefault(request_id, _make_request())

    async def _get_next_request(self):
        if not self._items:
            return None
        request_id = self._items.popleft()
        return (request_id, self._requests[request_id], None)

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class AdapterTransport:
    """Real-interface transport that raises scripted per-id failures.

    ``failures`` maps a request id to the exception its *first* resolve
    raises; transient failures clear after firing so the retry succeeds.
    """

    failures: dict[int, Exception] = field(default_factory=dict)
    acquired: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)

    async def acquire(self, worker_id: int) -> object:
        self.acquired.add(worker_id)
        return object()

    async def release(self, worker_id: int) -> None:
        self.released.add(worker_id)

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        exc = self.failures.get(queued.request_id)
        if exc is not None:
            if isinstance(exc, TransientException):
                del self.failures[queued.request_id]
            raise exc
        return _make_response(queued.request)

    async def resolve_archive(self, handle, queued, decision=None):
        raise AssertionError("archive not used in conformance")

    async def finish_archiving(self, stream) -> None:
        return None


@dataclass
class AdapterContinuation:
    """Continuation whose ``complete_request`` records the id as processed."""

    processed: list[int]

    async def complete_request(
        self, request_id, response, request, continuation_name, **_: Any
    ) -> None:
        self.processed.append(request_id)


@dataclass
class AdapterStorage:
    """Storage whose retry re-enqueues and whose mark-failed records a skip."""

    queue: AdapterQueue
    retried: list[int]
    skipped: list[int]

    async def _handle_retry(self, request_id, error):
        self.retried.append(request_id)
        self.queue.put(request_id)  # so the retried item completes next pass
        return 1.0

    async def _mark_request_failed(self, request_id, error_message) -> None:
        self.skipped.append(request_id)

    async def _mark_request_completed(self, request_id) -> None:
        return None


class NoopRateLimiter:
    """A rate limiter that never throttles."""

    async def gate(self, request) -> None:
        return None

    @property
    def max_rate_per_second(self) -> float | None:
        return None


class StubScraper:
    """Scraper exposing get_continuation; steps carry no await_list metadata."""

    driver_requirements: list = []

    def get_continuation(self, name: str):
        def parse(_response):
            yield None

        return parse


class TestPoolWorkerConformance(WorkerConformance):
    """Runs the shared conformance suite against the real ``PoolWorker``."""

    @pytest.fixture
    def subject(self) -> WorkerHarness:
        queue = AdapterQueue()
        transport = AdapterTransport()
        stop_event = asyncio.Event()
        processed: list[int] = []
        retried: list[int] = []
        skipped: list[int] = []
        continuation = AdapterContinuation(processed=processed)
        storage = AdapterStorage(queue=queue, retried=retried, skipped=skipped)
        worker = PoolWorker(
            worker_id=1,
            queue=queue,  # type: ignore[arg-type]
            transport=transport,  # type: ignore[arg-type]
            rate_limiter=NoopRateLimiter(),
            continuation=continuation,  # type: ignore[arg-type]
            storage=storage,  # type: ignore[arg-type]
            stop_event=stop_event,
            scraper=StubScraper(),
            archive_handler=None,
        )
        return WorkerHarness(
            worker=worker,
            queue=queue,  # type: ignore[arg-type]
            transport=transport,  # type: ignore[arg-type]
            stop_event=stop_event,
            processed=processed,
            retried=retried,
            skipped=skipped,
        )


# --- Targeted spies ------------------------------------------------------


@dataclass
class SpyOrderRateLimiter:
    """Records gate calls against a shared event log for ordering checks."""

    log: list[str]

    async def gate(self, request) -> None:
        self.log.append("gate")

    @property
    def max_rate_per_second(self) -> float | None:
        return None


@dataclass
class SpyOrderTransport:
    """Logs resolve into a shared event log (the start of the timed region)."""

    log: list[str]

    async def acquire(self, worker_id: int) -> object:
        return object()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        self.log.append("resolve")
        return _make_response(queued.request)


@dataclass
class RecordingContinuation:
    completed: list[int] = field(default_factory=list)

    async def complete_request(
        self, request_id, response, request, continuation_name, **_: Any
    ) -> None:
        self.completed.append(request_id)


@dataclass
class RecordingStorage:
    retried: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)

    async def _handle_retry(self, request_id, error):
        self.retried.append(request_id)
        return 1.0

    async def _mark_request_failed(self, request_id, error_message) -> None:
        self.failed.append((request_id, error_message))

    async def _mark_request_completed(self, request_id) -> None:
        self.completed.append(request_id)


def _single_request_queue(request) -> AdapterQueue:
    queue = AdapterQueue()
    queue._items.append(1)
    queue._requests[1] = request
    return queue


async def test_gate_is_awaited_before_the_timed_region() -> None:
    log: list[str] = []
    request = _make_request()
    queue = _single_request_queue(request)
    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=SpyOrderTransport(log=log),  # type: ignore[arg-type]
        rate_limiter=SpyOrderRateLimiter(log=log),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=RecordingStorage(),  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
    )

    await worker.run()

    assert log == ["gate", "resolve"]


@dataclass
class SkipArchiveHandler:
    """Archive handler that declines the download (file already present)."""

    should_download_calls: int = 0
    save_calls: int = 0

    async def should_download(self, **_: Any) -> ArchiveDecision:
        self.should_download_calls += 1
        return ArchiveDecision(download=False, file_url="/tmp/cached.pdf")

    async def save_stream(self, **_: Any) -> str:
        self.save_calls += 1
        return "/tmp/should-not-happen.pdf"


@dataclass
class TrackingRateLimiter:
    gate_calls: int = 0

    async def gate(self, request) -> None:
        self.gate_calls += 1

    @property
    def max_rate_per_second(self) -> float | None:
        return None


@dataclass
class NetworkAssertTransport:
    """Transport that fails the test if any network method is touched."""

    async def acquire(self, worker_id: int) -> object:
        return object()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        raise AssertionError("resolve must not be called on skipped archive")

    async def resolve_archive(self, handle, queued, decision=None):
        raise AssertionError(
            "resolve_archive must not be called on skipped archive"
        )

    async def finish_archiving(self, stream) -> None:
        return None


async def test_skipped_archive_bypasses_gate_and_network() -> None:
    handler = SkipArchiveHandler()
    limiter = TrackingRateLimiter()
    request = _make_request(archive=True)
    queue = _single_request_queue(request)
    continuation = RecordingContinuation()
    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=NetworkAssertTransport(),  # type: ignore[arg-type]
        rate_limiter=limiter,
        continuation=continuation,  # type: ignore[arg-type]
        storage=RecordingStorage(),  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=handler,
    )

    await worker.run()

    assert handler.should_download_calls == 1
    assert limiter.gate_calls == 0  # skip did not consume a token
    assert handler.save_calls == 0  # no download performed
    assert continuation.completed == [
        1
    ]  # still persisted (skip ArchiveResponse)


class _RecordingMonitor:
    def __init__(self) -> None:
        self.durations: list[float] = []

    def __call__(self, duration_s: float) -> None:
        self.durations.append(duration_s)


class _FakeCompactor:
    def __init__(self) -> None:
        self.records = 0

    async def record_request(self) -> bool:
        self.records += 1
        return False


async def test_success_reports_duration_and_counts_toward_compactor() -> None:
    request = _make_request()
    queue = _single_request_queue(request)
    monitor = _RecordingMonitor()
    compactor = _FakeCompactor()
    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=SpyOrderTransport(log=[]),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=RecordingStorage(),  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
        on_request_duration=monitor,
        compactor_for=lambda _step: compactor,  # type: ignore[arg-type, return-value]
    )

    await worker.run()

    assert len(monitor.durations) == 1
    assert monitor.durations[0] >= 0.0
    assert compactor.records == 1  # the step's compactor was bumped


async def test_archive_request_does_not_count_toward_compactor() -> None:
    """An archive request is timed but must NOT bump the compactor.

    Archive responses store file metadata, not a compressible body, so
    counting one would eventually trip the compactor into training a
    compression dict over zero responses (ValueError).
    """
    request = _make_request(archive=True)
    queue = _single_request_queue(request)
    monitor = _RecordingMonitor()
    compactor = _FakeCompactor()
    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=NetworkAssertTransport(),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=RecordingStorage(),  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=SkipArchiveHandler(),
        on_request_duration=monitor,
        compactor_for=lambda _step: compactor,  # type: ignore[arg-type, return-value]
    )

    await worker.run()

    assert len(monitor.durations) == 1  # still timed
    assert compactor.records == 0  # but NOT counted toward compaction


@dataclass
class FailingTransport:
    """Transport whose resolve raises a fixed exception."""

    exc: Exception

    async def acquire(self, worker_id: int) -> object:
        return object()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        raise self.exc


async def test_persistent_http_records_outcome_and_is_not_retried() -> None:
    storage = RecordingStorage()
    stored_errors: list[int] = []
    request = _make_request("https://example.com/gone")
    queue = _single_request_queue(request)
    exc = PersistentHTTPResponseException(
        status_code=404, url="https://example.com/gone"
    )

    async def store_error(e, *, request_id, request_url):
        stored_errors.append(request_id)

    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=FailingTransport(exc=exc),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
        store_error=store_error,
    )

    await worker.run()

    assert storage.retried == []  # not retried
    assert [rid for rid, _ in storage.failed] == [1]  # recorded as failed
    assert stored_errors == [1]  # error stored


async def test_arbitrary_exception_marks_failed_and_stores_error() -> None:
    storage = RecordingStorage()
    stored: list[tuple[int, str | None]] = []
    request = _make_request("https://example.com/boom")
    queue = _single_request_queue(request)

    async def store_error(e, *, request_id, request_url):
        stored.append((request_id, request_url))

    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=FailingTransport(  # type: ignore[arg-type]
            exc=ValueError("unexpected")
        ),
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
        store_error=store_error,
    )

    await worker.run()

    assert storage.retried == []
    assert [rid for rid, _ in storage.failed] == [1]
    assert stored == [(1, "https://example.com/boom")]


# --- Transient retry: max-backoff give-up + strictly-serial idle ---------


@dataclass
class _TransientTransport:
    """Resolve always raises a transient error (the retry path under test)."""

    async def acquire(self, worker_id: int) -> object:
        return object()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        raise TransientException("flaky")


@dataclass
class _RetryStorage:
    """``_handle_retry`` returns a fixed delay (or None) without re-enqueueing."""

    delay: float | None
    retried: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

    async def _handle_retry(self, request_id, error):
        self.retried.append(request_id)
        return self.delay

    async def _mark_request_failed(self, request_id, error_message) -> None:
        self.failed.append((request_id, error_message))

    async def _mark_request_completed(self, request_id) -> None:
        return None


class _SerialScraper(StubScraper):
    """A strictly-serial scraper (transient retries idle the worker)."""

    driver_requirements = [DriverRequirement.STRICTLY_SERIAL]


def _transient_worker(
    storage: _RetryStorage, *, scraper: StubScraper, stored: list
) -> PoolWorker:
    async def store_error(e, *, request_id, request_url):
        stored.append((request_id, request_url))

    queue = _single_request_queue(_make_request())
    return PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=_TransientTransport(),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=scraper,
        archive_handler=None,
        store_error=store_error,
    )


async def test_transient_max_backoff_marks_failed_and_stores_error() -> None:
    """A None retry delay (backoff exhausted) → mark failed + store the error."""
    storage = _RetryStorage(delay=None)
    stored: list = []
    worker = _transient_worker(storage, scraper=StubScraper(), stored=stored)

    await worker.run()

    assert storage.retried == [1]
    assert [rid for rid, _ in storage.failed] == [1]
    assert stored == [(1, "https://example.com/p")]


async def test_transient_with_debug_response_persists_snapshot() -> None:
    """A transient carrying a debug_response stores it before the retry.

    PlaywrightTransport's ResolveTimeout attaches the partial-DOM snapshot;
    the worker persists it (so the failed attempt is inspectable) and then
    proceeds with the normal transient handling.
    """
    request = _make_request()
    snapshot = _make_response(request)

    class _TimeoutTransport:
        async def acquire(self, worker_id: int) -> object:
            return object()

        async def release(self, worker_id: int) -> None:
            return None

        async def resolve(self, handle, queued, await_conditions=()):
            raise ResolveTimeout("timeout", debug_response=snapshot)

    class _SnapshotStorage:
        def __init__(self) -> None:
            self.stored: list = []
            self.retried: list = []
            self.failed: list = []

        async def _store_response(
            self, request_id, response, continuation, speculation_outcome=None
        ) -> int:
            self.stored.append((request_id, response, continuation))
            return 1

        async def _handle_retry(self, request_id, error):
            self.retried.append(request_id)
            return None  # backoff exhausted → worker marks failed, loop ends

        async def _mark_request_failed(
            self, request_id, error_message
        ) -> None:
            self.failed.append((request_id, error_message))

    storage = _SnapshotStorage()
    worker = PoolWorker(
        worker_id=1,
        queue=_single_request_queue(request),  # type: ignore[arg-type]
        transport=_TimeoutTransport(),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
        store_error=lambda *a, **k: _noop(),
    )

    await worker.run()

    # The partial DOM was stored (against the request's continuation) before
    # the retry was handled.
    assert storage.stored == [(1, snapshot, "parse")]
    assert storage.retried == [1]


async def _noop() -> None:
    return None


async def test_strictly_serial_idles_until_retry_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A serial scraper waits the retry delay on the stop event after a transient."""
    waits: list[float] = []

    async def fake_wait_for(awaitable: Any, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()  # the stop_event.wait() coroutine; don't leave it pending
        raise TimeoutError  # mimic "delay elapsed, no stop"

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    storage = _RetryStorage(delay=2.5)
    worker = _transient_worker(storage, scraper=_SerialScraper(), stored=[])

    await worker.run()

    assert storage.retried == [1]
    assert waits == [2.5]  # idled once, for the scheduled retry delay


async def test_non_serial_does_not_idle_after_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-serial scraper re-queues and moves on — no idle wait."""
    waits: list[float] = []

    async def fake_wait_for(awaitable: Any, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()
        raise TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    storage = _RetryStorage(delay=2.5)
    worker = _transient_worker(storage, scraper=StubScraper(), stored=[])

    await worker.run()

    assert storage.retried == [1]
    assert waits == []  # not strictly serial → never idles
