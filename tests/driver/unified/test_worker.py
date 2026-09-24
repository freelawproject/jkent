"""Tests for the unified driver's concrete worker (:class:`PoolWorker`).

Two layers:

* ``TestPoolWorkerConformance`` binds the real ``PoolWorker`` to the shared
  ``WorkerConformance`` suite via adapter fakes that implement the
  collaborator methods the worker calls (``queue.get_next_request``,
  ``transport.acquire/release/resolve``, ``step.complete_request``,
  ``storage.handle_retry/mark_request_failed``) while exposing the harness's
  observable surface (``.put``/``__len__``, ``.failures``, and the
  ``processed``/``retried`` lists).
* Targeted tests use spies for the finer-grained contract points: dequeue
  retries on a locked database, gate ordering, archive ``should_download``
  skip bypassing the gate and the network, circuit-breaker evidence, success
  reporting to the compactor, the no-retry persistent/arbitrary-error paths,
  transient max-backoff give-up, rate-limiter feedback, and idle/fan-out
  retirement.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from collections import deque
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from hypothesis import given
from sqlalchemy.exc import OperationalError
from typing_extensions import override

from jkent import observability as obs
from jkent.common.exceptions import (
    HTMLStructuralAssumptionException,
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    ResolveTimeout,
    ScraperConfigError,
    SpeculationHTTPFailure,
    TransientException,
    TransientKind,
)
from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    ArchiveDecision,
    ArchiveResponse,
    BaseScraper,
    DriverRequirement,
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from jkent.driver.archive_handler import LocalAsyncStreamingArchiveHandler
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.unified_driver.compaction import Compactors
from jkent.driver.unified_driver.persistence import RowOnlyErrorSink
from jkent.driver.unified_driver.rate_limiter import (
    RateLimiter,
    RateLimiters,
)
from jkent.driver.unified_driver.transport import (
    ArchiveStream,
    AwaitCondition,
    NoopHandle,
    QueuedRequest,
)
from jkent.driver.unified_driver.wiring import RunCollaborators
from jkent.driver.unified_driver.worker import PoolWorker
from tests.driver.unified.test_worker_conformance import (
    FakeRetryPolicy,
    Script,
    WorkerConformance,
    WorkerHarness,
    _alphabet_scenarios,
    _stop_scenarios,
)


def _collabs(**kw: Any) -> RunCollaborators:
    """Collaborators for a standalone worker.

    Every test names its queue/transport/executor/storage; the sink
    defaults to the row-only one so failure paths still mark rows failed
    (what the recording storages observe), and everything else takes the
    null-object default.
    """
    kw.setdefault("error_sink", RowOnlyErrorSink(kw["storage"]))
    # Tests name one limiter; it gates the default lane, as in a real run.
    if "rate_limiter" in kw:
        kw["rate_limiters"] = RateLimiters(
            {
                DEFAULT_RATE_LIMIT: kw.pop("rate_limiter"),
                NO_RATE_LIMIT: NoopRateLimiter(),
            }
        )
    return RunCollaborators(**kw)


def _make_request(
    url: str = "https://example.com/p", *, archive: bool = False
):
    """A minimal Request with the ``parse`` step."""
    return Request(
        request=HTTPRequestParams(method=HttpMethod.GET, url=url),
        step="parse",
        current_location="https://example.com",
        archive=archive,
        expected_type="pdf" if archive else None,
    )


def _observed(
    request: Any,
    *,
    status_code: int,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> Response:
    """The exchange a transport observed, as ``classify_and_raise`` builds it."""
    return Response(
        status_code=status_code,
        headers=headers,
        content=body,
        text=body.decode(),
        url=url,
        request=request,
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


# --- Adapter fakes for the collaborators the worker calls ---------------


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

    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
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

    ``failures`` maps a request id to the exceptions its successive resolves
    raise, oldest first, each firing once (see ``FakeTransport``): a
    transient scripted ``times=k`` fails k attempts then succeeds.
    """

    failures: dict[int, deque[Exception]] = field(default_factory=dict)
    acquired: set[int] = field(default_factory=set)
    released: set[int] = field(default_factory=set)
    acquire_count: int = 0
    release_count: int = 0

    def fail(self, request_id: int, exc: Exception, *, times: int = 1) -> None:
        """Script ``exc`` for the next ``times`` resolves of ``request_id``."""
        self.failures.setdefault(request_id, deque()).extend([exc] * times)

    async def acquire(self, worker_id: int) -> object:
        self.acquired.add(worker_id)
        self.acquire_count += 1
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        self.released.add(worker_id)
        self.release_count += 1

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        # Yield like a real I/O-bound transport so the conformance stopper can
        # interleave a genuine mid-run stop (see FakeTransport.resolve).
        await asyncio.sleep(0)
        pending = self.failures.get(queued.request_id)
        if pending:
            raise pending.popleft()
        return _make_response(queued.request)

    async def resolve_archive(
        self, handle: object, queued: QueuedRequest
    ) -> ArchiveStream:
        raise AssertionError("archive not used in conformance")

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        return None


@dataclass
class AdapterExecutor:
    """Step whose ``complete_request`` records the id as processed."""

    processed: list[int]

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        **_: Any,
    ) -> None:
        self.processed.append(request_id)


@dataclass
class AdapterStorage:
    """Storage whose ``handle_retry`` is the harness's retry policy.

    The policy re-enqueues a granted retry (as the real storage schedules
    it) and answers ``None`` once the id's budget is spent; the worker's
    ``_fail_request`` then lands here via the row-only sink, so ``failed``
    is the harness's view of "marked failed".
    """

    retry_policy: FakeRetryPolicy
    failed: list[int]

    async def handle_retry(
        self, request_id: int, error: Exception
    ) -> float | None:
        return self.retry_policy.handle_retry(request_id)

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        self.failed.append(request_id)

    async def mark_request_completed(self, request_id: int) -> None:
        return None


@dataclass
class RecordingSink:
    """An ``ErrorSink`` that records both kinds of report.

    ``failed`` is the terminal report (the worker's ``_fail_request``), which
    on a real run writes the request row *and* the errors row; ``recorded``
    is the file-it-only report.
    """

    failed: list[tuple[int, Exception, str | None]] = field(
        default_factory=list
    )
    recorded: list[tuple[int | None, str | None]] = field(default_factory=list)

    @property
    def failed_ids(self) -> list[int]:
        return [rid for rid, _exc, _url in self.failed]

    async def request_failed(
        self,
        request_id: int,
        exc: Exception,
        *,
        request_url: str | None = None,
    ) -> None:
        self.failed.append((request_id, exc, request_url))

    async def record(
        self,
        exc: Exception,
        *,
        request_id: int | None = None,
        request_url: str | None = None,
    ) -> None:
        self.recorded.append((request_id, request_url))


class NoopRateLimiter(RateLimiter):
    """A rate limiter that never throttles."""

    async def gate(self, request: Request) -> None:
        return None


class StubScraper(BaseScraper[Any]):
    """Scraper exposing get_step; steps carry no await_list metadata."""

    @override
    def get_step(self, name: str):
        def parse(_response):
            yield None

        return parse


class TestPoolWorkerConformance(WorkerConformance):
    """Runs the shared conformance suite against the real ``PoolWorker``."""

    # Thin @given wrappers over the base's check_* bodies: each binding owns
    # its own function objects so hypothesis's ``differing_executors`` health
    # check doesn't see one test shared across subclasses.

    @pytest.mark.generative
    @given(scenario=_alphabet_scenarios())
    def test_failure_alphabet_conserves_every_request(
        self, scenario: tuple[list[Script], int | None]
    ) -> None:
        self.check_failure_alphabet_conserves_every_request(scenario)

    @pytest.mark.generative
    @given(scenario=_stop_scenarios())
    def test_stop_never_loses_or_duplicates_work(
        self, scenario: tuple[int, int]
    ) -> None:
        self.check_stop_never_loses_or_duplicates_work(scenario)

    @override
    def make_harness(self) -> WorkerHarness:
        queue = AdapterQueue()
        transport = AdapterTransport()
        stop_event = asyncio.Event()
        processed: list[int] = []
        retried: list[int] = []
        failed: list[int] = []
        retry_policy = FakeRetryPolicy(queue=queue, retried=retried)
        executor = AdapterExecutor(processed=processed)
        storage = AdapterStorage(retry_policy=retry_policy, failed=failed)
        worker = PoolWorker(
            1,
            _collabs(
                queue=queue,
                transport=transport,
                rate_limiter=NoopRateLimiter(),
                executor=executor,
                storage=storage,
                stop_event=stop_event,
                scraper=StubScraper(),
            ),
        )
        return WorkerHarness(
            worker=worker,
            queue=queue,  # type: ignore[arg-type]
            transport=transport,  # type: ignore[arg-type]
            stop_event=stop_event,
            processed=processed,
            retried=retried,
            failed=failed,
            retry_policy=retry_policy,
        )


# --- Targeted spies ------------------------------------------------------


@dataclass
class SpyOrderRateLimiter(RateLimiter):
    """Records gate calls against a shared event log for ordering checks."""

    log: list[str]

    async def gate(self, request: Request) -> None:
        self.log.append("gate")


@dataclass
class SpyOrderTransport:
    """Logs resolve into a shared event log for ordering checks."""

    log: list[str]

    async def acquire(self, worker_id: int) -> object:
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        self.log.append("resolve")
        return _make_response(queued.request)


@dataclass
class RecordingExecutor:
    completed: list[int] = field(default_factory=list)

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        **_: Any,
    ) -> None:
        self.completed.append(request_id)


@dataclass
class RecordingStorage:
    retried: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    stored: list[tuple[int, Response, str]] = field(default_factory=list)

    async def handle_retry(
        self, request_id: int, error: Exception
    ) -> float | None:
        self.retried.append(request_id)
        return 1.0

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id: int) -> None:
        self.completed.append(request_id)

    async def store_response(
        self,
        request_id: int,
        response: Response,
        step: str,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> int:
        self.stored.append((request_id, response, step))
        return request_id


def _single_request_queue(request: Request) -> AdapterQueue:
    queue = AdapterQueue()
    queue._items.append(1)
    queue._requests[1] = request
    return queue


@dataclass
class LockingQueue(AdapterQueue):
    """Queue whose dequeue raises "database is locked" the first N times."""

    locked_attempts: int = 0
    attempts: int = 0

    @override
    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
        self.attempts += 1
        if self.attempts <= self.locked_attempts:
            raise OperationalError(
                "UPDATE requests SET status=?",
                {},
                sqlite3.OperationalError("database is locked"),
            )
        return await super().get_next_request()


async def test_dequeue_retries_a_locked_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pool is pinned: a worker that raises here is never replaced, so a
    # momentary lock would permanently cost the run a worker.
    monkeypatch.setattr(PoolWorker, "DEQUEUE_RETRY_DELAYS_S", (0.0, 0.0, 0.0))
    queue = LockingQueue(locked_attempts=2)
    queue.put(1)
    executor = RecordingExecutor()
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=SpyOrderTransport(log=[]),
            rate_limiter=NoopRateLimiter(),
            executor=executor,
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    await worker.run()

    # Two locked, the dequeue that lands, then the drain check that retires
    # the worker.
    assert queue.attempts == 4
    assert executor.completed == [1]


async def test_dequeue_gives_up_when_the_lock_outlives_the_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Retrying forever would hide a database that is genuinely stuck.
    monkeypatch.setattr(PoolWorker, "DEQUEUE_RETRY_DELAYS_S", (0.0, 0.0))
    queue = LockingQueue(locked_attempts=99)
    queue.put(1)
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=SpyOrderTransport(log=[]),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    with pytest.raises(OperationalError):
        await worker.run()

    assert queue.attempts == 3  # the ladder's two retries, then one more


async def test_gate_is_awaited_before_resolve() -> None:
    log: list[str] = []
    request = _make_request()
    queue = _single_request_queue(request)
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=SpyOrderTransport(log=log),
            rate_limiter=SpyOrderRateLimiter(log=log),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    await worker.run()

    assert log == ["gate", "resolve"]


@dataclass
class _Clock:
    """A settable stand-in for the worker module's ``time``."""

    now: float = 0.0

    def monotonic(self) -> float:
        return self.now


@dataclass
class _WaitingRateLimiter(RateLimiter):
    """A gate that advances the fake clock by ``wait_s``."""

    clock: _Clock
    wait_s: float

    async def gate(self, request: Request) -> None:
        self.clock.now += self.wait_s


@dataclass
class _RestampQueue(AdapterQueue):
    restamped: list[int] = field(default_factory=list)

    @override
    async def restamp_request_start(self, request_id: int) -> None:
        self.restamped.append(request_id)


@pytest.mark.parametrize(
    ("wait_s", "restamped"),
    [
        pytest.param(0.0, [], id="pass-through"),
        pytest.param(PoolWorker.RESTAMP_MIN_GATE_WAIT_S, [1], id="waited"),
    ],
)
async def test_gate_wait_restamps_request_start(
    monkeypatch: pytest.MonkeyPatch, wait_s: float, restamped: list[int]
) -> None:
    """The start stamp is rewritten only when the gates actually waited."""
    clock = _Clock()
    monkeypatch.setattr("jkent.driver.unified_driver.worker.time", clock)
    queue = _RestampQueue()
    queue.put(1)
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=SpyOrderTransport(log=[]),
            rate_limiter=_WaitingRateLimiter(clock=clock, wait_s=wait_s),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    await worker.run()

    assert queue.restamped == restamped


@dataclass
class _SlowAcquireTransport(SpyOrderTransport):
    """A lease that advances the fake clock, like a browser restart."""

    clock: _Clock = field(default_factory=_Clock)
    acquire_s: float = 0.0

    @override
    async def acquire(self, worker_id: int) -> object:
        self.clock.now += self.acquire_s
        return await super().acquire(worker_id)


async def test_slow_acquire_restamps_request_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow lease is pre-execute time, like a gate wait.

    ``acquire`` is where a poisoned browser handle is rebuilt; with the gates
    passing straight through, that restart would otherwise land in the
    request's DB-derived duration.
    """
    clock = _Clock()
    monkeypatch.setattr("jkent.driver.unified_driver.worker.time", clock)
    queue = _RestampQueue()
    queue.put(1)
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=_SlowAcquireTransport(
                log=[], clock=clock, acquire_s=10.0
            ),
            rate_limiter=_WaitingRateLimiter(clock=clock, wait_s=0.0),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    await worker.run()

    assert queue.restamped == [1]


@dataclass
class _PreresolvedQueue(AdapterQueue):
    """Queue whose every request is flagged pre-resolved."""

    @override
    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
        result = await super().get_next_request()
        return None if result is None else (*result[:3], True)


@dataclass
class _NoStoredResponseStorage(RecordingStorage):
    async def load_preresolved_response(
        self, request_id: int, request: Request
    ) -> Response | None:
        return None


async def test_preresolved_without_stored_response_fails_the_request() -> None:
    """A pre-resolved request with no stored response is failed, not run."""
    queue = _PreresolvedQueue()
    queue.put(1)
    storage = _NoStoredResponseStorage()
    executor = RecordingExecutor()
    sink = RecordingSink()
    transport = AdapterTransport()
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=transport,
            rate_limiter=NoopRateLimiter(),
            executor=executor,
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert executor.completed == []
    assert storage.retried == []
    assert sink.failed_ids == [1]
    exc = sink.failed[0][1]
    assert isinstance(exc, RuntimeError)
    assert "has no stored response" in str(exc)
    assert transport.acquire_count == 0


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
class TrackingRateLimiter(RateLimiter):
    gate_calls: int = 0

    async def gate(self, request: Request) -> None:
        self.gate_calls += 1


@dataclass
class NetworkAssertTransport:
    """Transport that fails the test if any network method is touched."""

    async def acquire(self, worker_id: int) -> object:
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        raise AssertionError("resolve must not be called on skipped archive")

    async def resolve_archive(
        self, handle: object, queued: QueuedRequest
    ) -> ArchiveStream:
        raise AssertionError(
            "resolve_archive must not be called on skipped archive"
        )

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        return None


async def test_skipped_archive_bypasses_gate_and_network() -> None:
    handler = SkipArchiveHandler()
    limiter = TrackingRateLimiter()
    request = _make_request(archive=True)
    queue = _single_request_queue(request)
    executor = RecordingExecutor()
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=NetworkAssertTransport(),
            rate_limiter=limiter,
            executor=executor,
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            archive_handler=handler,
        ),
    )

    await worker.run()

    assert handler.should_download_calls == 1
    assert limiter.gate_calls == 0  # skip did not consume a token
    assert handler.save_calls == 0  # no download performed
    assert executor.completed == [1]  # still persisted (skip ArchiveResponse)


class _ChunkStream(ArchiveStream):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(status_code=200, headers={}, url="https://e/f.pdf")
        self._chunks = chunks

    async def _iterate(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterate()


@dataclass
class StreamingTransport(NetworkAssertTransport):
    """Transport whose archive body arrives as ``chunks``."""

    chunks: list[bytes] = field(default_factory=list)

    @override
    async def resolve_archive(
        self, handle: object, queued: QueuedRequest
    ) -> ArchiveStream:
        return _ChunkStream(self.chunks)


@dataclass
class ResponseExecutor:
    responses: list[Response] = field(default_factory=list)

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        **_: Any,
    ) -> None:
        self.responses.append(response)


@pytest.mark.parametrize(
    "chunks",
    [[b"%PDF-1.4 ", b"body \x00\x01", b"\xff"], []],
    ids=["file", "empty file"],
)
async def test_downloaded_archive_carries_its_size_and_digest(
    tmp_path: Path, chunks: list[bytes]
) -> None:
    """What was streamed to disk is what ``archived_files`` will record."""
    executor = ResponseExecutor()
    worker = PoolWorker(
        1,
        _collabs(
            queue=_single_request_queue(_make_request(archive=True)),
            transport=StreamingTransport(chunks=chunks),
            rate_limiter=NoopRateLimiter(),
            executor=executor,
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            archive_handler=LocalAsyncStreamingArchiveHandler(tmp_path),
        ),
    )

    await worker.run()

    (response,) = executor.responses
    assert isinstance(response, ArchiveResponse)
    body = b"".join(chunks)
    assert Path(response.file_url).read_bytes() == body
    assert response.file_size == len(body)
    assert response.content_hash == hashlib.sha256(body).hexdigest()


@dataclass
class FailingTransport:
    """Transport whose resolve raises a fixed exception."""

    exc: Exception

    async def acquire(self, worker_id: int) -> object:
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        raise self.exc


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
    request: Request,
    transport: object,
    breaker: SpyCircuitBreaker,
    **overrides: Any,
) -> PoolWorker:
    kwargs: dict[str, Any] = {
        "queue": _single_request_queue(request),
        "transport": transport,
        "rate_limiter": NoopRateLimiter(),
        "executor": RecordingExecutor(),
        "storage": RecordingStorage(),
        "stop_event": asyncio.Event(),
        "scraper": StubScraper(),
        "circuit_breaker": breaker,
        **overrides,
    }
    return PoolWorker(1, _collabs(**kwargs))


class _OkTransport:
    """A transport whose resolve simply succeeds."""

    async def acquire(self, worker_id: int) -> object:
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        return _make_response(queued.request)


async def test_server_answer_credits_breaker_even_if_step_fails() -> None:
    """A 200 whose step then breaks still counts as availability.

    The breaker tracks whether the *site* is up. A response that arrived and
    then failed to parse is evidence the site is up and the scraper is wrong,
    so the consecutive-failure count must still reset — otherwise a broken
    parser looks like an outage and opens the circuit.
    """

    class _BadExecutor:
        async def complete_request(self, *_a: Any, **_k: Any) -> None:
            raise HTMLStructuralAssumptionException(
                selector="//tr",
                selector_type="xpath",
                description="rows",
                expected_min=1,
                expected_max=None,
                actual_count=0,
                request_url="https://example.com/p",
            )

    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(),
        _OkTransport(),  # resolve succeeds
        breaker,
        executor=_BadExecutor(),
        error_sink=RecordingSink(),
    )

    await worker.run()

    assert "success" in breaker.log
    assert "failure" not in breaker.log


async def test_unanswered_failure_is_no_breaker_evidence() -> None:
    """A crash before any response reaches neither side of the breaker.

    An unclassified exception out of ``resolve`` says nothing about the
    server — it is neither availability nor distress, and recording either
    would be a made-up measurement.
    """
    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(),
        FailingTransport(exc=ValueError("resolve blew up")),
        breaker,
        error_sink=RecordingSink(),
    )

    await worker.run()

    assert breaker.log == ["breaker.gate"]


async def test_breaker_gates_before_rate_limiter_and_records_success() -> None:
    log: list[str] = []
    breaker = SpyCircuitBreaker(log=log)
    worker = _spy_breaker_worker(
        _make_request(),
        SpyOrderTransport(log=log),
        breaker,
        rate_limiter=SpyOrderRateLimiter(log=log),
    )

    await worker.run()

    assert log == ["breaker.gate", "gate", "resolve", "success"]


async def test_transient_failure_records_failure_on_breaker() -> None:
    breaker = SpyCircuitBreaker()
    worker = _spy_breaker_worker(
        _make_request(),
        FailingTransport(
            TransientException("server buckling", kind=TransientKind.NETWORK)
        ),
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
        FailingTransport(
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
        NetworkAssertTransport(),
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
        1,
        _collabs(
            queue=queue,
            transport=SpyOrderTransport(log=[]),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            compactors=Compactors({"parse": compactor}),  # type: ignore[dict-item]
        ),
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
        1,
        _collabs(
            queue=queue,
            transport=NetworkAssertTransport(),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            archive_handler=SkipArchiveHandler(),
            compactors=Compactors({"parse": compactor}),  # type: ignore[dict-item]
        ),
    )

    await worker.run()

    assert compactor.records == 0  # NOT counted toward compaction


async def test_persistent_http_records_outcome_and_is_not_retried() -> None:
    storage = RecordingStorage()
    sink = RecordingSink()
    request = _make_request("https://example.com/gone")
    queue = _single_request_queue(request)
    exc = PersistentHTTPResponseException(
        status_code=404, url="https://example.com/gone"
    )

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=exc),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert storage.retried == []  # not retried
    # One terminal report carries the id, the exception and the url — the
    # errors row and the request's last_error are written from it together.
    assert sink.failed == [(1, exc, "https://example.com/gone")]
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
    sink = RecordingSink()
    request = _make_request("https://example.com/blocked")
    queue = _single_request_queue(request)
    exc = PersistentHTTPResponseException(
        status_code=403,
        url="https://example.com/blocked",
        debug_response=_observed(
            request,
            status_code=403,
            url="https://example.com/blocked",
            headers={"server": "cloudflare"},
            body=b"<html>Access denied</html>",
        ),
    )

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=exc),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert sink.failed_ids == [1]
    [(request_id, response, step)] = storage.stored
    assert request_id == 1
    assert step == "parse"
    assert response.status_code == 403
    assert response.headers == {"server": "cloudflare"}
    assert response.content == b"<html>Access denied</html>"
    assert response.url == "https://example.com/blocked"


async def test_non_http_persistent_is_failed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``PersistentException`` that isn't HTTP takes the persistent arm.

    Before this arm existed these fell through to the generic
    ``except Exception``. The observable contract: no retry, marked failed on
    the first occurrence, an error row stored — and the request span labelled
    ``persistent`` rather than ``error``, so an assumption break is
    distinguishable from an unclassified crash in the outcome metric.
    """
    storage = RecordingStorage()
    request = _make_request("https://example.com/moved")
    queue = _single_request_queue(request)
    exc = HTMLStructuralAssumptionException(
        selector="//table//tr",
        selector_type="xpath",
        description="result rows",
        expected_min=1,
        expected_max=None,
        actual_count=0,
        request_url="https://example.com/moved",
    )
    sink = RecordingSink()
    outcomes: list[Any] = []

    def set_attribute(key: str, value: Any) -> None:
        if key == "jkent.outcome":
            outcomes.append(value)

    @contextmanager
    def recording_span(**_: Any) -> Iterator[Any]:
        yield SimpleNamespace(set_attribute=set_attribute)

    monkeypatch.setattr(obs, "request_span", recording_span)

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=exc),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert outcomes == [obs.Outcome.PERSISTENT]
    assert storage.retried == []
    # The terminal report carries the exception itself, so the errors row is
    # written from the same object that set the request's last_error.
    assert sink.failed == [(1, exc, "https://example.com/moved")]


async def test_non_http_persistent_does_not_count_against_the_breaker() -> (
    None
):
    """An assumption break is scraper breakage, not server distress.

    The breaker's failure count exists to back off a struggling server; a
    selector that stopped matching says nothing about server health, so the
    persistent arm must leave the breaker alone (the transient arm is the
    single failure-count site).
    """
    breaker = SpyCircuitBreaker()
    request = _make_request("https://example.com/moved")
    queue = _single_request_queue(request)

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=ScraperConfigError("no such step")),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=RecordingStorage(),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=RecordingSink(),
            circuit_breaker=breaker,
        ),
    )

    await worker.run()

    # Only the pre-resolve gate call; no failure or success recorded.
    assert breaker.log == ["breaker.gate"]


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
        debug_response=_observed(
            request,
            status_code=503,
            url="https://example.com/flaky",
            headers={"retry-after": "60"},
            body=b"<html>Maintenance</html>",
        ),
    )

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=exc),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=RecordingSink(),
        ),
    )

    await worker.run()

    assert storage.retried == [1]  # still routed through retry handling
    [(request_id, response, step)] = storage.stored
    assert request_id == 1
    assert step == "parse"
    assert response.status_code == 503
    assert response.headers == {"retry-after": "60"}
    assert response.content == b"<html>Maintenance</html>"


async def test_arbitrary_exception_marks_failed_and_stores_error() -> None:
    storage = RecordingStorage()
    sink = RecordingSink()
    request = _make_request("https://example.com/boom")
    queue = _single_request_queue(request)

    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=FailingTransport(exc=ValueError("unexpected")),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert storage.retried == []
    assert sink.failed == [(1, sink.failed[0][1], "https://example.com/boom")]
    assert isinstance(sink.failed[0][1], ValueError)


# --- Transient retry: max-backoff give-up + strictly-serial idle ---------


@dataclass
class _TransientTransport:
    """Resolve always raises a transient error (the retry path under test)."""

    async def acquire(self, worker_id: int) -> object:
        return NoopHandle()

    async def release(self, worker_id: int) -> None:
        return None

    async def resolve(
        self,
        handle: object,
        queued: QueuedRequest,
        await_conditions: Sequence[AwaitCondition] = (),
    ) -> Response:
        raise TransientException("flaky", kind=TransientKind.NETWORK)


@dataclass
class _RetryStorage:
    """``handle_retry`` returns a fixed delay (or None) without re-enqueueing."""

    delay: float | None
    retried: list[int] = field(default_factory=list)
    failed: list[tuple[int, str]] = field(default_factory=list)

    async def handle_retry(
        self, request_id: int, error: Exception
    ) -> float | None:
        self.retried.append(request_id)
        return self.delay

    async def mark_request_failed(
        self, request_id: int, error_message: str
    ) -> None:
        self.failed.append((request_id, error_message))

    async def mark_request_completed(self, request_id: int) -> None:
        return None


class _SerialScraper(StubScraper):
    """A strictly-serial scraper (transient retries idle the worker)."""

    driver_requirements = [DriverRequirement.STRICTLY_SERIAL]


def _transient_worker(
    storage: _RetryStorage, *, scraper: StubScraper, sink: RecordingSink
) -> PoolWorker:
    queue = _single_request_queue(_make_request())
    return PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=_TransientTransport(),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=scraper,
            error_sink=sink,
        ),
    )


async def test_transient_max_backoff_marks_failed_and_stores_error() -> None:
    """A None retry delay (backoff exhausted) → mark failed + store the error."""
    storage = _RetryStorage(delay=None)
    sink = RecordingSink()
    worker = _transient_worker(storage, scraper=StubScraper(), sink=sink)

    await worker.run()

    assert storage.retried == [1]
    assert sink.failed_ids == [1]
    assert sink.failed[0][2] == "https://example.com/p"


# --- Rate limiter feedback ------------------------------------------------


@dataclass
class RecordingRateLimiter(RateLimiter):
    """Never throttles; records every record_response call."""

    records: list[tuple[int, float | None, str | None]] = field(
        default_factory=list
    )

    async def gate(self, request: Request) -> None:
        return None

    @override
    def record_response(
        self,
        status_code: int,
        *,
        retry_after: float | None = None,
        url: str | None = None,
    ) -> None:
        self.records.append((status_code, retry_after, url))


def _http_failure_worker(exc: Exception, limiter: RateLimiter) -> PoolWorker:
    """A worker whose single request's resolve raises ``exc``."""

    @dataclass
    class _RaisingTransport:
        async def acquire(self, worker_id: int) -> object:
            return NoopHandle()

        async def release(self, worker_id: int) -> None:
            return None

        async def resolve(
            self,
            handle: object,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            raise exc

    return PoolWorker(
        1,
        _collabs(
            queue=_single_request_queue(_make_request()),
            transport=_RaisingTransport(),
            rate_limiter=limiter,
            executor=RecordingExecutor(),
            storage=_RetryStorage(delay=None),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=RecordingSink(),
        ),
    )


async def test_classified_transient_http_failure_feeds_rate_limiter() -> None:
    """A classified 429 reaches record_response with its Retry-After."""
    limiter = RecordingRateLimiter()
    worker = _http_failure_worker(
        HTTPResponseAssumptionException(
            429, [200], "https://example.com/p", retry_after=9.0
        ),
        limiter,
    )

    await worker.run()

    assert limiter.records == [(429, 9.0, "https://example.com/p")]


async def test_persistent_http_failure_feeds_rate_limiter() -> None:
    """A persistently-classified 429 is still rate feedback (no retry)."""
    limiter = RecordingRateLimiter()
    worker = _http_failure_worker(
        PersistentHTTPResponseException(
            429, "https://example.com/p", retry_after=30.0
        ),
        limiter,
    )

    await worker.run()

    assert limiter.records == [(429, 30.0, "https://example.com/p")]


async def test_unclassified_failure_does_not_feed_rate_limiter() -> None:
    """A non-HTTP transient (no status) tells the limiter nothing."""
    limiter = RecordingRateLimiter()
    worker = _http_failure_worker(
        TransientException("flaky", kind=TransientKind.NETWORK), limiter
    )

    await worker.run()

    assert limiter.records == []


async def test_outcome_state_does_not_leak_between_requests() -> None:
    """Each request's breaker and lane report reflects only that request.

    A worker keeps its outcome evidence on the instance, reset per request.
    Without the reset, the 200 would go on crediting the breaker for the
    unanswered crash after it, and the 429 would be re-reported to the lane
    on every later request.
    """
    exchanges: list[Exception | None] = [
        HTTPResponseAssumptionException(
            429, [200], "https://example.com/1", retry_after=30.0
        ),
        None,
        ValueError("crashed before any answer"),
    ]

    class _ScriptedTransport(_OkTransport):
        @override
        async def resolve(
            self,
            handle: object,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            exc = exchanges.pop(0)
            if exc is not None:
                raise exc
            return _make_response(queued.request)

    queue = AdapterQueue()
    for request_id in (1, 2, 3):
        queue.put(request_id)
    breaker = SpyCircuitBreaker()
    limiter = RecordingRateLimiter()
    worker = PoolWorker(
        1,
        _collabs(
            queue=queue,
            transport=_ScriptedTransport(),
            rate_limiter=limiter,
            executor=RecordingExecutor(),
            storage=_RetryStorage(delay=None),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=RecordingSink(),
            circuit_breaker=breaker,
        ),
    )

    await worker.run()

    assert exchanges == []
    assert breaker.log == [
        *("breaker.gate", "failure"),
        *("breaker.gate", "success"),
        "breaker.gate",
    ]
    assert limiter.records == [(429, 30.0, "https://example.com/1")]


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
            return NoopHandle()

        async def release(self, worker_id: int) -> None:
            return None

        async def resolve(
            self,
            handle: object,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            raise ResolveTimeout(
                url=request.request.url,
                timeout_seconds=30.0,
                debug_response=snapshot,
            )

    class _SnapshotStorage:
        def __init__(self) -> None:
            self.stored: list[tuple[int, Response, str]] = []
            self.retried: list[int] = []
            self.failed: list[tuple[int, str]] = []

        async def store_response(
            self,
            request_id: int,
            response: Response,
            step: str,
            speculation_outcome: SpeculationOutcome | None = None,
        ) -> int:
            self.stored.append((request_id, response, step))
            return 1

        async def handle_retry(
            self, request_id: int, error: Exception
        ) -> float | None:
            self.retried.append(request_id)
            return None  # backoff exhausted → worker marks failed, loop ends

        async def mark_request_failed(
            self, request_id: int, error_message: str
        ) -> None:
            self.failed.append((request_id, error_message))

    storage = _SnapshotStorage()
    worker = PoolWorker(
        1,
        _collabs(
            queue=_single_request_queue(request),
            transport=_TimeoutTransport(),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=RecordingSink(),
        ),
    )

    await worker.run()

    # The partial DOM was stored (against the request's step) before
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
    worker = _transient_worker(
        storage, scraper=_SerialScraper(), sink=RecordingSink()
    )

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

    async def get_next_request(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
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
        1,
        _collabs(
            queue=queue,
            transport=_ReleaseOnlyTransport(),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=_RetryStorage(delay=None),
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
        ),
    )

    await worker.run()

    assert waits == [2.5]  # waited once for the scheduled retry, then retired


class _BlockingFanOutExecutor:
    """``complete_request(1)`` enqueues request 2, then blocks until request 2
    completes — so only an idle *sibling* can process the fan-out child."""

    def __init__(self, queue: AdapterQueue) -> None:
        self.queue = queue
        self.completed: list[int] = []
        self._child_done = asyncio.Event()

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        **_: Any,
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
    lull (empty queue while a sibling's step is still running) must
    not shrink the pool. Worker 1's step blocks until the child it
    enqueues is completed, so the run can only finish if worker 2 stayed
    alive through the lull and picked the child up.
    """
    queue = AdapterQueue()
    queue.put(1)
    executor = _BlockingFanOutExecutor(queue)

    def make_worker(worker_id: int) -> PoolWorker:
        worker = PoolWorker(
            worker_id,
            _collabs(
                queue=queue,
                transport=SpyOrderTransport(log=[]),
                rate_limiter=NoopRateLimiter(),
                executor=executor,
                storage=RecordingStorage(),
                stop_event=asyncio.Event(),
                scraper=StubScraper(),
            ),
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

    assert executor.completed == [2, 1]  # the child ran, on worker 2
    assert queue.in_flight_count == 0  # every claim was released


async def test_speculation_http_failure_on_non_speculative_is_failed() -> None:
    """SpeculationHTTPFailure on a non-speculative request fails, not completes.

    Regression: the handler used to fall through to mark_request_completed for
    any request, silently recording a persistent HTTP failure as a success.
    """
    request = _make_request()  # is_speculative defaults False

    class _SpecFailTransport:
        async def acquire(self, worker_id: int) -> object:
            return NoopHandle()

        async def release(self, worker_id: int) -> None:
            return None

        async def resolve(
            self,
            handle: object,
            queued: QueuedRequest,
            await_conditions: Sequence[AwaitCondition] = (),
        ) -> Response:
            raise SpeculationHTTPFailure(
                status_code=404, url="https://example.com/p"
            )

    @dataclass
    class _OutcomeStorage:
        failed: list[int] = field(default_factory=list)
        completed: list[int] = field(default_factory=list)

        async def mark_request_failed(
            self, request_id: int, error_message: str
        ) -> None:
            self.failed.append(request_id)

        async def mark_request_completed(self, request_id: int) -> None:
            self.completed.append(request_id)

    storage = _OutcomeStorage()
    sink = RecordingSink()

    worker = PoolWorker(
        1,
        _collabs(
            queue=_single_request_queue(request),
            transport=_SpecFailTransport(),
            rate_limiter=NoopRateLimiter(),
            executor=RecordingExecutor(),
            storage=storage,
            stop_event=asyncio.Event(),
            scraper=StubScraper(),
            error_sink=sink,
        ),
    )

    await worker.run()

    assert sink.failed_ids == [1]
    assert storage.completed == []
    assert sink.failed[0][2] == "https://example.com/p"


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
    worker = _transient_worker(
        storage, scraper=StubScraper(), sink=RecordingSink()
    )

    await worker.run()

    assert storage.retried == [1]
    assert waits == []  # not strictly serial → never idles
