"""Tests for the unified driver's concrete worker (:class:`PoolWorker`).

Two layers:

* ``TestPoolWorkerConformance`` binds the real ``PoolWorker`` to the shared
  ``WorkerConformance`` suite via adapter fakes that present the *real*
  collaborator interfaces (the worker calls ``queue.get_next_request``,
  ``transport.acquire/release/resolve``, ``continuation.complete_request``,
  ``storage.handle_retry/mark_request_failed``) while exposing the harness's
  observable surface (``.put``/``__len__``, ``.failures``, and the
  ``processed``/``retried``/``skipped`` lists).
* ``Test*`` targeted cases use spies for the finer-grained contract points:
  gate ordering, archive ``should_download`` skip bypassing the gate and the
  network, success reporting to the compactor, and the no-retry
  persistent/arbitrary-error paths.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    SpeculationHTTPFailure,
    TransientException,
)
from jkent.data_types import (
    ArchiveDecision,
    BaseScraper,
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
    _SCRIPT,
    WorkerConformance,
    WorkerHarness,
    _halt_scenarios,
    _stop_scenarios,
)


def _make_request(
    url: str = "https://example.com/p", *, archive: bool = False
):
    """A minimal Request with the ``parse`` continuation."""
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

    ``put``/``pending_ids``/``__len__`` are the harness observable surface;
    the worker pulls via ``get_next_request`` which maps each id to a simple
    Request.
    """

    _items: deque[int] = field(default_factory=deque)
    _requests: dict[int, Any] = field(default_factory=dict)
    _in_flight: int = 0

    def put(self, request_id: int) -> None:
        self._items.append(request_id)
        self._requests.setdefault(request_id, _make_request())

    async def get_next_request(self):
        if not self._items:
            return None
        request_id = self._items.popleft()
        self._in_flight += 1
        return (request_id, self._requests[request_id], None, False)

    async def seconds_until_next_pending(self) -> float | None:
        # No backoff model: an empty harness queue is durably idle.
        return None

    @property
    def in_flight_count(self) -> int:
        return self._in_flight

    def request_done(self) -> None:
        self._in_flight -= 1

    async def restamp_request_start(self, request_id: int) -> None:
        return None

    def pending_ids(self) -> list[int]:
        """The ids still queued, in dequeue order (harness observability)."""
        return list(self._items)

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
    acquire_count: int = 0
    release_count: int = 0

    async def acquire(self, worker_id: int) -> object:
        self.acquired.add(worker_id)
        self.acquire_count += 1
        return object()

    async def release(self, worker_id: int) -> None:
        self.released.add(worker_id)
        self.release_count += 1

    async def resolve(self, handle, queued, await_conditions=()) -> Response:
        # Yield like a real I/O-bound transport so the conformance stopper can
        # interleave a genuine mid-run stop (see FakeTransport.resolve).
        await asyncio.sleep(0)
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

    async def handle_retry(self, request_id, error):
        self.retried.append(request_id)
        self.queue.put(request_id)  # so the retried item completes next pass
        return 1.0

    async def mark_request_failed(self, request_id, error_message) -> None:
        self.skipped.append(request_id)

    async def mark_request_completed(self, request_id) -> None:
        return None


class NoopRateLimiter:
    """A rate limiter that never throttles."""

    async def gate(self, request) -> None:
        return None

    @property
    def max_rate_per_second(self) -> float | None:
        return None


class StubScraper(BaseScraper[Any]):
    """Scraper exposing get_continuation; steps carry no await_list metadata."""

    def get_continuation(self, name: str):  # type: ignore[override]
        def parse(_response):
            yield None

        return parse


class TestPoolWorkerConformance(WorkerConformance):
    """Runs the shared conformance suite against the real ``PoolWorker``."""

    # Thin @given wrappers over the base's check_* bodies: each binding owns
    # its own function objects so hypothesis's ``differing_executors`` health
    # check doesn't see one test shared across subclasses.

    @pytest.mark.generative
    @given(scripts=st.lists(_SCRIPT, max_size=10))
    def test_failure_soup_partitions_the_queue(
        self, scripts: list[str]
    ) -> None:
        self.check_failure_soup_partitions_the_queue(scripts)

    @pytest.mark.generative
    @given(scenario=_halt_scenarios())
    def test_halt_accounts_for_every_request(
        self, scenario: tuple[list[str], int]
    ) -> None:
        self.check_halt_accounts_for_every_request(scenario)

    @pytest.mark.generative
    @given(scenario=_stop_scenarios())
    def test_stop_never_loses_or_duplicates_work(
        self, scenario: tuple[int, int]
    ) -> None:
        self.check_stop_never_loses_or_duplicates_work(scenario)

    def make_harness(self) -> WorkerHarness:
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
    """Logs resolve into a shared event log for ordering checks."""

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
    stored: list[tuple[int, Response, str]] = field(default_factory=list)

    async def handle_retry(self, request_id, error):
        self.retried.append(request_id)
        return 1.0

    async def mark_request_failed(self, request_id, error_message) -> None:
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id) -> None:
        self.completed.append(request_id)

    async def store_response(
        self, request_id, response, continuation, speculation_outcome=None
    ) -> int:
        self.stored.append((request_id, response, continuation))
        return request_id


def _single_request_queue(request) -> AdapterQueue:
    queue = AdapterQueue()
    queue._items.append(1)
    queue._requests[1] = request
    return queue


async def test_gate_is_awaited_before_resolve() -> None:
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


# --- Circuit-breaker wiring -----------------------------------------------


@dataclass
class SpyCircuitBreaker:
    """Records the worker's breaker calls into a shared event log."""

    log: list[str] = field(default_factory=list)

    async def gate(self) -> None:
        self.log.append("breaker.gate")

    def record_failure(self) -> None:
        self.log.append("failure")

    def record_success(self) -> None:
        self.log.append("success")


def _spy_breaker_worker(
    request, transport, breaker: SpyCircuitBreaker, **overrides: Any
) -> PoolWorker:
    kwargs: dict[str, Any] = {
        "queue": _single_request_queue(request),
        "transport": transport,
        "rate_limiter": NoopRateLimiter(),
        "continuation": RecordingContinuation(),
        "storage": RecordingStorage(),
        "stop_event": asyncio.Event(),
        "scraper": StubScraper(),
        "archive_handler": None,
        "circuit_breaker": breaker,
        **overrides,
    }
    return PoolWorker(worker_id=1, **kwargs)


async def test_breaker_gates_before_rate_limiter_and_records_success() -> None:
    log: list[str] = []
    breaker = SpyCircuitBreaker(log=log)
    worker = _spy_breaker_worker(
        _make_request(),
        SpyOrderTransport(log=log),  # type: ignore[arg-type]
        breaker,
        rate_limiter=SpyOrderRateLimiter(log=log),
    )

    await worker.run()

    assert log == ["breaker.gate", "gate", "resolve", "success"]


async def test_transient_failure_records_failure_on_breaker() -> None:
    # FailingTransport is defined below with the persistent-path tests;
    # module-level definitions all exist by test-call time.
    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(),
        FailingTransport(TransientException("server buckling")),  # type: ignore[arg-type]
        breaker,
    )

    await worker.run()

    assert breaker.log == ["breaker.gate", "failure"]


async def test_persistent_http_records_success_on_breaker() -> None:
    # A persistent status is the server *answering* — availability evidence,
    # not distress; the breaker must not count it toward a trip.
    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(),
        FailingTransport(  # type: ignore[arg-type]
            PersistentHTTPResponseException(404, "https://example.com/p")
        ),
        breaker,
    )

    await worker.run()

    assert breaker.log == ["breaker.gate", "success"]


async def test_skipped_archive_never_touches_the_breaker() -> None:
    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(archive=True),
        NetworkAssertTransport(),  # type: ignore[arg-type]
        breaker,
        archive_handler=SkipArchiveHandler(),
    )

    await worker.run()

    assert breaker.log == []  # no network I/O: no gate, no evidence


class _FakeCompactor:
    def __init__(self) -> None:
        self.records = 0

    async def record_request(self) -> bool:
        self.records += 1
        return False


async def test_success_counts_toward_compactor() -> None:
    request = _make_request()
    queue = _single_request_queue(request)
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
        compactor_for=lambda _step: compactor,  # type: ignore[arg-type, return-value]
    )

    await worker.run()

    assert compactor.records == 1  # the step's compactor was bumped
    assert queue.in_flight_count == 0  # the claim was released


async def test_archive_request_does_not_count_toward_compactor() -> None:
    """An archive request must NOT bump the compactor.

    Archive responses store file metadata, not a compressible body, so
    counting one would eventually trip the compactor into training a
    compression dict over zero responses (ValueError).
    """
    request = _make_request(archive=True)
    queue = _single_request_queue(request)
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
        compactor_for=lambda _step: compactor,  # type: ignore[arg-type, return-value]
    )

    await worker.run()

    assert compactor.records == 0  # NOT counted toward compaction


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
    assert (
        storage.stored == []
    )  # no payload on the exception → nothing to store


async def test_persistent_http_with_payload_stores_the_failed_response() -> (
    None
):
    """A persistent HTTP error's observed body/headers land in storage.

    ``classify_and_raise`` attaches what the transport observed to the
    exception; the worker must persist it (before marking failed) so a 403
    block page is inspectable from the run db, not just a status code in the
    errors table.
    """
    storage = RecordingStorage()
    request = _make_request("https://example.com/blocked")
    queue = _single_request_queue(request)
    exc = PersistentHTTPResponseException(
        status_code=403,
        url="https://example.com/blocked",
        headers={"server": "cloudflare"},
        body=b"<html>Access denied</html>",
    )

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
        store_error=lambda *a, **k: _noop(),
    )

    await worker.run()

    assert [rid for rid, _ in storage.failed] == [1]
    [(request_id, response, continuation)] = storage.stored
    assert request_id == 1
    assert continuation == "parse"
    assert response.status_code == 403
    assert response.headers == {"server": "cloudflare"}
    assert response.content == b"<html>Access denied</html>"
    assert response.url == "https://example.com/blocked"


async def test_transient_http_error_persists_body_snapshot() -> None:
    """A transient HTTP error carrying body/headers stores them before retry.

    Mirrors the debug_response path: the latest failed attempt stays
    inspectable in the run db (a later success or attempt overwrites it).
    """
    storage = RecordingStorage()
    request = _make_request("https://example.com/flaky")
    queue = _single_request_queue(request)
    exc = HTTPResponseAssumptionException(
        status_code=503,
        expected_codes=[200],
        url="https://example.com/flaky",
        headers={"retry-after": "60"},
        body=b"<html>Maintenance</html>",
    )

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
        store_error=lambda *a, **k: _noop(),
    )

    await worker.run()

    assert storage.retried == [1]  # still routed through retry handling
    [(request_id, response, continuation)] = storage.stored
    assert request_id == 1
    assert continuation == "parse"
    assert response.status_code == 503
    assert response.headers == {"retry-after": "60"}
    assert response.content == b"<html>Maintenance</html>"


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
    """``handle_retry`` returns a fixed delay (or None) without re-enqueueing."""

    delay: float | None
    retried: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

    async def handle_retry(self, request_id, error):
        self.retried.append(request_id)
        return self.delay

    async def mark_request_failed(self, request_id, error_message) -> None:
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id) -> None:
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

        async def store_response(
            self, request_id, response, continuation, speculation_outcome=None
        ) -> int:
            self.stored.append((request_id, response, continuation))
            return 1

        async def handle_retry(self, request_id, error):
            self.retried.append(request_id)
            return None  # backoff exhausted → worker marks failed, loop ends

        async def mark_request_failed(self, request_id, error_message) -> None:
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
        # Real asyncio.wait_for raises asyncio.TimeoutError on timeout. On 3.10
        # that's a distinct class from the builtin TimeoutError, and the worker
        # suppresses asyncio.TimeoutError specifically, so mimic it faithfully.
        raise asyncio.TimeoutError  # mimic "delay elapsed, no stop"

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    storage = _RetryStorage(delay=2.5)
    worker = _transient_worker(storage, scraper=_SerialScraper(), stored=[])

    await worker.run()

    assert storage.retried == [1]
    assert waits == [2.5]  # idled once, for the scheduled retry delay


@dataclass
class _ScheduledRetryQueue:
    """Empty queue that reports a future-scheduled retry, then drains.

    ``get_next_request`` is always empty; ``seconds_until_next_pending``
    returns each value in ``delays`` in turn (a positive delay == a retry in
    backoff, ``None`` == durably idle), so a worker exercises the wait-for-the-
    scheduled-retry path before retiring.
    """

    delays: list[float | None]
    _calls: int = 0

    async def get_next_request(self):
        return None

    async def seconds_until_next_pending(self) -> float | None:
        value = (
            self.delays[self._calls]
            if self._calls < len(self.delays)
            else None
        )
        self._calls += 1
        return value

    async def restamp_request_start(self, request_id: int) -> None:
        return None

    @property
    def in_flight_count(self) -> int:
        return 0  # nothing is ever dequeued from this queue


class _ReleaseOnlyTransport:
    """Transport whose only reachable method is ``release`` (no work pulled)."""

    async def release(self, worker_id: int) -> None:
        return None


async def test_idle_worker_waits_for_scheduled_retry_then_retires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty queue with a scheduled retry makes the worker wait, not retire.

    Regression: the worker used to return on the first empty dequeue,
    abandoning a backoff retry. It now sleeps until the retry is ready (or
    the stop event), re-checks, and only retires once nothing is pending now
    or later and nothing is in flight.
    """
    waits: list[float] = []

    async def fake_wait_for(awaitable: Any, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError  # delay elapsed, no stop

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    queue = _ScheduledRetryQueue(delays=[2.5, None])
    worker = PoolWorker(
        worker_id=1,
        queue=queue,  # type: ignore[arg-type]
        transport=_ReleaseOnlyTransport(),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=_RetryStorage(delay=None),  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
    )

    await worker.run()

    assert waits == [2.5]  # waited once for the scheduled retry, then retired


class _BlockingFanOutContinuation:
    """``complete_request(1)`` enqueues request 2, then blocks until request 2
    completes — so only an idle *sibling* can process the fan-out child."""

    def __init__(self, queue: AdapterQueue) -> None:
        self.queue = queue
        self.completed: list[int] = []
        self._child_done = asyncio.Event()

    async def complete_request(
        self, request_id, response, request, continuation_name, **_: Any
    ) -> None:
        if request_id == 1:
            self.queue.put(2)
            await self._child_done.wait()
        else:
            self._child_done.set()
        self.completed.append(request_id)


async def test_idle_worker_waits_on_sibling_in_flight_and_takes_fanout() -> (
    None
):
    """An idle worker must not retire while a sibling holds a request in flight.

    The pool is pinned — a retired worker is never replaced — so a momentary
    lull (empty queue while a sibling's continuation is still running) must
    not shrink the pool. Worker 1's continuation blocks until the child it
    enqueues is completed, so the run can only finish if worker 2 stayed
    alive through the lull and picked the child up.
    """
    queue = AdapterQueue()
    queue.put(1)
    continuation = _BlockingFanOutContinuation(queue)

    def make_worker(worker_id: int) -> PoolWorker:
        worker = PoolWorker(
            worker_id=worker_id,
            queue=queue,  # type: ignore[arg-type]
            transport=SpyOrderTransport(log=[]),  # type: ignore[arg-type]
            rate_limiter=NoopRateLimiter(),
            continuation=continuation,  # type: ignore[arg-type]
            storage=RecordingStorage(),  # type: ignore[arg-type]
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            archive_handler=None,
        )
        worker.IN_FLIGHT_POLL_INTERVAL_S = 0.01  # keep the lull polls fast
        return worker

    # Start worker 1 and let it claim request 1 before worker 2 ever sees
    # the queue, so worker 2's first read is the lull: empty queue, one
    # sibling in flight.
    task_1 = asyncio.create_task(make_worker(1).run())
    while queue.in_flight_count == 0:
        await asyncio.sleep(0)
    task_2 = asyncio.create_task(make_worker(2).run())

    # Deadlocks (worker 2 retiring at the lull) fail fast via the timeout.
    await asyncio.wait_for(asyncio.gather(task_1, task_2), timeout=5.0)

    assert continuation.completed == [2, 1]  # the child ran, on worker 2
    assert queue.in_flight_count == 0  # every claim was released


async def test_speculation_http_failure_on_non_speculative_is_failed() -> None:
    """SpeculationHTTPFailure on a non-speculative request fails, not completes.

    Regression: the handler used to fall through to mark_request_completed for
    any request, silently recording a persistent HTTP failure as a success.
    """
    request = _make_request()  # is_speculative defaults False

    class _SpecFailTransport:
        async def acquire(self, worker_id: int) -> object:
            return object()

        async def release(self, worker_id: int) -> None:
            return None

        async def resolve(self, handle, queued, await_conditions=()):  # type: ignore[no-untyped-def]
            raise SpeculationHTTPFailure(
                status_code=404, url="https://example.com/p"
            )

    @dataclass
    class _OutcomeStorage:
        failed: list[int] = field(default_factory=list)
        completed: list[int] = field(default_factory=list)

        async def mark_request_failed(self, request_id, error_message) -> None:
            self.failed.append(request_id)

        async def mark_request_completed(self, request_id) -> None:
            self.completed.append(request_id)

    storage = _OutcomeStorage()
    stored: list = []

    async def store_error(e, *, request_id, request_url):
        stored.append((request_id, request_url))

    worker = PoolWorker(
        worker_id=1,
        queue=_single_request_queue(request),  # type: ignore[arg-type]
        transport=_SpecFailTransport(),  # type: ignore[arg-type]
        rate_limiter=NoopRateLimiter(),
        continuation=RecordingContinuation(),  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        stop_event=asyncio.Event(),
        scraper=StubScraper(),
        archive_handler=None,
        track_speculation=None,
        store_error=store_error,
    )

    await worker.run()

    assert storage.failed == [1]
    assert storage.completed == []
    assert stored == [(1, "https://example.com/p")]


async def test_non_serial_does_not_idle_after_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-serial scraper re-queues and moves on — no idle wait."""
    waits: list[float] = []

    async def fake_wait_for(awaitable: Any, timeout: float) -> None:
        waits.append(timeout)
        awaitable.close()
        # Real asyncio.wait_for raises asyncio.TimeoutError; on 3.10 that is a
        # distinct class from the builtin TimeoutError and the worker suppresses
        # asyncio.TimeoutError specifically. Raise the same class so this stays
        # faithful if the non-serial path ever does reach wait_for.
        raise asyncio.TimeoutError  # delay elapsed, no stop

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    storage = _RetryStorage(delay=2.5)
    worker = _transient_worker(storage, scraper=StubScraper(), stored=[])

    await worker.run()

    assert storage.retried == [1]
    assert waits == []  # not strictly serial → never idles
