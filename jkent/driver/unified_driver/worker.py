"""Concrete worker: the per-worker execution loop for a unified-driver run.

It takes its collaborators as one
:class:`~jkent.driver.unified_driver.wiring.RunCollaborators`: it leases a
handle from the ``Transport``, pulls rows from the ``RequestQueue``, gates on
the ``CircuitBreaker`` and ``RateLimiter``, resolves via the transport, and
hands the response to the ``StepExecutor``. Retries/skips/marks go
through ``ResponseStorage``; failures are reported to the ``ErrorSink``; step
completions to the ``Compactors`` registry.

Transport recovery is opaque here: a dead resource
arrives as a ``TransientException`` and is retried like any other; the rebuild
happens inside the next ``transport.acquire``.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

from sqlalchemy.exc import OperationalError

from jkent import observability as obs
from jkent.common.decorator_metadata import get_step_metadata
from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentException,
    PersistentHTTPResponseException,
    RequestFailedHalt,
    SpeculationHTTPFailure,
    TransientException,
)
from jkent.data_types import ArchiveResponse, DriverRequirement, Response
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.unified_driver.lifecycle import sleep_unless_stopped
from jkent.driver.unified_driver.transport import QueuedRequest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from jkent.data_types import ArchiveDecision, Request
    from jkent.driver.unified_driver.rate_limiter import RateLimiter
    from jkent.driver.unified_driver.transport import (
        ArchiveStream,
        AwaitCondition,
    )
    from jkent.driver.unified_driver.wiring import RunCollaborators

logger = logging.getLogger(__name__)

#: Outcomes of a probe that found nothing: stored, completed, no step.
_MISSES = frozenset({SpeculationOutcome.MISS, SpeculationOutcome.STOPPED})


class PoolWorker:
    """A single unit of execution within a run.

    Leases a per-worker handle from the transport, then loops: pull the next
    request from the queue, resolve it via the transport, persist the result,
    and route failures by the exception taxonomy (transient → retry,
    persistent → fail, halt/skip → propagate). Exits on the stop event or a
    drained queue — nothing pending and nothing in flight anywhere in the
    pool — releasing its handle on the way out. The pool is pinned at spawn
    time and never re-grown, so retirement is deliberately conservative.

    A worker observes transport failure at point of use — it is the one
    holding the handle when the resource dies — and recovers by asking the
    transport to restart, then renewing its lease (see
    ``PlaywrightTransport.restart``).

    **Extension points.** Subclasses replacing the request lifecycle (jent's
    replay worker overrides :meth:`_handle_one` to route every miss/error
    through a replay miss policy) build on these members. They carry no
    stability promise here; a subclass pins the ones it uses on its side
    (jent: ``tests/replay/test_poolworker_seam.py``):

    * hooks: ``_handle_one``, ``_execute_preresolved``
    * helpers: ``_step_name``, ``_await_conditions``,
      ``_record_for_compactor``, ``_store_error_for``, ``_fail_request``,
      ``_report_outcome``, ``_request_url``
    * collaborators: ``_transport``, ``_storage``, ``_executor`` (and
      ``StepExecutor.complete_request``), ``_track_speculation``,
      ``_circuit_breaker``, plus ``worker_id`` — each an alias of the
      matching :class:`RunCollaborators` field, also reachable as
      ``self.collaborators``
    """

    # How long an idle worker sleeps before re-checking the queue while a
    # sibling still holds a request in flight (whose step may enqueue
    # children). Local SQLite reads make this poll effectively free.
    IN_FLIGHT_POLL_INTERVAL_S = 0.5

    # Backoff ladder for a dequeue that hits a SQLite lock error. SQLite's own
    # busy_timeout has already elapsed by the time one surfaces, so these
    # delays are about riding out a burst of contention, not waiting on a
    # single lock. Exhausting the ladder re-raises.
    DEQUEUE_RETRY_DELAYS_S = (0.1, 0.5, 2.0, 5.0)

    # Only re-stamp started_at when the lease and the breaker/rate gates
    # actually held the request at least this long. The restamp exists so
    # DB-derived durations exclude those waits; when they return immediately
    # (a live handle, tokens available, the none lane, replay — the common
    # case) the dequeue stamp is already honest and the restamp would be a
    # pure extra write transaction per request.
    RESTAMP_MIN_GATE_WAIT_S = 0.01

    # Outcomes that count as server distress. Everything else is either
    # availability evidence (see ``_server_answered``) or no evidence at all.
    #
    # The breaker measures *server availability*, not our success, and that is
    # the whole distinction: an assumption break or an unclassified crash is
    # our code or the page's shape being wrong, and says nothing about the
    # server's health. Only a transient counts against it.
    BREAKER_FAILURE_OUTCOMES: ClassVar[frozenset[obs.Outcome]] = frozenset(
        {obs.Outcome.TRANSIENT}
    )

    def __init__(
        self, worker_id: int, collaborators: RunCollaborators
    ) -> None:
        self.worker_id = worker_id
        self.collaborators = collaborators
        # Aliases for the extension surface (see the class docstring):
        # subclasses read these names directly.
        self._queue = collaborators.queue
        self._transport = collaborators.transport
        self._rate_limiters = collaborators.rate_limiters
        # One breaker shared pool-wide, so the failure count is a run-level
        # signal; the default NoopBreaker (direct construction in tests and
        # minimal hosts) never blocks and hears nothing.
        self._circuit_breaker = collaborators.circuit_breaker
        self._executor = collaborators.executor
        self._storage = collaborators.storage
        self._stop_event = collaborators.stop_event
        self._scraper = collaborators.scraper
        self._archive_handler = collaborators.archive_handler
        self._compactors = collaborators.compactors
        self._error_sink = collaborators.error_sink
        speculation = collaborators.speculation
        self._track_speculation: (
            Callable[[Request, Response], Awaitable[SpeculationOutcome | None]]
            | None
        ) = speculation.track_outcome if speculation is not None else None
        self._speculation_stopped: Callable[[Request], bool] | None = (
            speculation.has_stopped if speculation is not None else None
        )
        # Whether the request currently in flight reached the server. Set the
        # moment a resolve returns and read by _report_outcome at the end of
        # the request; instance state rather than a parameter because the
        # evidence appears mid-``_execute_one`` but is reported from
        # ``_handle_one``'s finally, and a worker runs one request at a time.
        self._server_answered = False
        # The exception the in-flight request failed with, if any — stashed
        # by _handle_one's arms (same instance-state rationale as
        # _server_answered) and read by _report_outcome, which extracts the
        # HTTP signal (status, Retry-After) for the rate limiter.
        self._outcome_exc: Exception | None = None
        # The limiter that gated the request in hand, so its HTTP feedback
        # goes back to the same lane. None until the gate is reached.
        self._outcome_limiter: RateLimiter | None = None

    @property
    def _strictly_serial(self) -> bool:
        """Whether the scraper requires strictly-serial processing."""
        return (
            DriverRequirement.STRICTLY_SERIAL
            in self._scraper.driver_requirements
        )

    async def run(self) -> None:
        """Process requests until shutdown or the queue is durably empty."""
        # Bind the scraper label for this worker task's whole lifetime so even
        # the dequeue path's DB-lock waits are attributed. The contextvar is
        # per-task (each worker is its own asyncio.Task), so this does not bleed
        # across workers.
        with obs.labeled(scraper=self._scraper.__class__.__name__):
            try:
                await self._run_loop()
            finally:
                await self._transport.release(self.worker_id)

    async def _dequeue_next(
        self,
    ) -> tuple[int, Request, int | None, bool] | None:
        """Dequeue the next request, riding out transient SQLite lock errors.

        The pool is pinned — a worker that raises is never replaced — so an
        ``OperationalError`` escaping here permanently costs the run a worker
        over what is usually a momentary contention. Retry a few times first;
        a lock that outlives the whole ladder is a real problem and still
        propagates.

        Returns ``None`` when nothing is dequeuable, exactly as
        ``get_next_request`` does — including when a retry was cut short by
        shutdown, which the caller re-checks.
        """
        attempts = len(self.DEQUEUE_RETRY_DELAYS_S) + 1
        for attempt, delay in enumerate(self.DEQUEUE_RETRY_DELAYS_S, start=1):
            try:
                return await self._queue.get_next_request()
            except OperationalError as exc:
                logger.warning(
                    "Worker %d could not dequeue (attempt %d/%d), retrying "
                    "in %.1fs: %s",
                    self.worker_id,
                    attempt,
                    attempts,
                    delay,
                    exc,
                )
            if await sleep_unless_stopped(self._stop_event, delay):
                return None
        return await self._queue.get_next_request()

    async def _run_loop(self) -> None:
        """The dequeue/handle loop, run inside the scraper label scope."""
        while not self._stop_event.is_set():
            result = await self._dequeue_next()
            if result is None:
                if self._stop_event.is_set():
                    return
                # No request is ready right now. That can mean the queue
                # is truly drained, OR that the only remaining work is
                # retries still in their backoff window (pending rows with
                # a future started_at, which the dequeue skips), OR that a
                # sibling worker holds a request in flight whose step
                # may yet enqueue children. The pool is pinned — a retired
                # worker is never replaced — so only retire when none of
                # those can produce work: nothing pending now or later AND
                # nothing in flight. Otherwise sleep and re-check.
                delay = await self._queue.seconds_until_next_pending()
                if delay is None:
                    if self._queue.in_flight_count == 0:
                        return  # drained: nothing pending, nothing in flight
                    delay = self.IN_FLIGHT_POLL_INTERVAL_S
                idle_start = time.monotonic()
                await sleep_unless_stopped(self._stop_event, delay)
                # The pinned pool's utilization signal: time this worker sat
                # with nothing dequeuable. Compare against
                # request.duration{phase=total} to size num_workers.
                obs.instruments().worker_idle.record(
                    time.monotonic() - idle_start, obs.current_labels()
                )
                continue
            request_id, request, parent_request_id, preresolved = result
            try:
                await self._handle_one(
                    request_id, request, parent_request_id, preresolved
                )
            finally:
                # Balance get_next_request's in-flight claim on every exit —
                # including halt/cancel — so idle siblings never wait on a
                # request nobody is handling.
                self._queue.request_done()

    async def _handle_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
        preresolved: bool = False,
    ) -> None:
        """Lease, gate, resolve, persist, and route failures for one request."""

        # Compute the target step up front so it labels the whole request span
        # (and every phase/metric under it), including failure paths.
        step_name = self._step_name(request)
        outcome = obs.Outcome.OK
        # A pre-resolved request and a skipped archive download do no network
        # I/O, so neither is evidence the site is up; ``_execute_one`` flips
        # this the moment a real resolve returns.
        self._server_answered = False
        self._outcome_exc = None
        self._outcome_limiter = None
        with (
            obs.labeled(step=step_name),
            obs.request_span(
                scraper=self._scraper.__class__.__name__,
                step=step_name,
            ) as span,
        ):
            try:
                if preresolved:
                    await self._execute_preresolved(
                        request_id, request, step_name
                    )
                else:
                    await self._execute_one(
                        request_id,
                        request,
                        parent_request_id,
                        step_name,
                    )
            except RequestFailedHalt:
                outcome = obs.Outcome.HALT
                raise  # propagate, stops the run
            except TransientException as e:
                outcome = obs.Outcome.TRANSIENT
                self._outcome_exc = e
                await self._handle_transient(request_id, request, e)
            except SpeculationHTTPFailure as e:
                outcome = obs.Outcome.SPECULATION_HTTP
                self._server_answered = True
                self._outcome_exc = e
                await self._handle_speculation_http(request_id, request, e)
            except PersistentHTTPResponseException as e:
                outcome = obs.Outcome.PERSISTENT_HTTP
                self._server_answered = True
                self._outcome_exc = e
                # Classifier said this status is persistent: no retry. Persist
                # the observed response (body/headers travel on the exception)
                # so the failure is inspectable from the run db.
                logger.warning(
                    "Worker %d persistent HTTP %s on request %d: %s",
                    self.worker_id,
                    e.status_code,
                    request_id,
                    e.url,
                )
                if e.debug_response is not None:
                    await self._storage.store_response(
                        request_id, e.debug_response, step_name
                    )
                await self._fail_request(request_id, e, e.url)
            except PersistentException as e:
                outcome = obs.Outcome.PERSISTENT
                self._outcome_exc = e
                # A scraper assumption or config violation: the site (or the
                # scraper) is wrong in a way retrying cannot fix, and it is
                # not server distress — so no retry and no breaker signal.
                # The traceback is the point here (unlike the HTTP arm above,
                # where the status says everything): it names the scraper
                # line whose assumption broke.
                logger.warning(
                    "Worker %d persistent failure on request %d: %s",
                    self.worker_id,
                    request_id,
                    e,
                    exc_info=True,
                )
                await self._fail_request(
                    request_id, e, self._request_url(request)
                )
            except Exception as e:
                outcome = obs.Outcome.ERROR
                self._outcome_exc = e
                logger.exception(
                    "Worker %d error processing request %d",
                    self.worker_id,
                    request_id,
                )
                await self._fail_request(
                    request_id, e, self._request_url(request)
                )
            finally:
                self._report_outcome(outcome)
                span.set_attribute("jkent.outcome", outcome)

    async def _execute_preresolved(
        self,
        request_id: int,
        request: Request,
        step_name: str,
    ) -> None:
        """Run a pre-resolved request's step without the transport.

        The response was stored at enqueue time (promoted from a captured
        incidental sub-request), so there is no lease, no rate-limit gate, and
        no network I/O: load the stored response and run the step.
        ``store_response=False`` because the row already holds it.
        """
        response = await self._storage.load_preresolved_response(
            request_id, request
        )
        if response is None:
            # preresolved implies a stored response; a missing one is a bug in
            # the enqueue path, not a retryable condition.
            raise RuntimeError(
                f"pre-resolved request {request_id} has no stored response"
            )
        with obs.phase(obs.Phase.STEP):
            await self._executor.complete_request(
                request_id,
                response,
                request,
                step_name,
                page=None,
                store_response=False,
            )

    async def _handle_transient(
        self, request_id: int, request: Request, e: TransientException
    ) -> None:
        """Route a transient failure: persist debug snapshot, retry or fail."""
        # The breaker is told by _report_outcome, not here — every arm reports
        # through the one table.
        # Whatever the transport observed before failing — a classified HTTP
        # error's body, or the partial DOM snapshotted before a timeout —
        # arrives in one shape on the exception. Persist it for debugging
        # before the retry; the next attempt (or a success) overwrites it.
        debug_response = e.debug_response
        if debug_response is not None:
            await self._storage.store_response(
                request_id,
                debug_response,
                self._step_name(request),
            )
        retry_delay = await self._storage.handle_retry(request_id, e)
        if retry_delay is None:
            # Max backoff exceeded (or no retry state): give up — mark
            # failed and store the error.
            await self._fail_request(request_id, e, self._request_url(request))
            return
        # A retry rate that climbs with worker count is server pushback —
        # the counter-signal against raising num_workers.
        obs.instruments().request_retries.add(1, obs.current_labels())
        if self._strictly_serial:
            # Strict serialization: idle until the just-scheduled retry is
            # ready rather than pulling other pending work. Stop-event-aware
            # so a shutdown during the wait stays prompt.
            await sleep_unless_stopped(self._stop_event, retry_delay)

    async def _handle_speculation_http(
        self, request_id: int, request: Request, e: SpeculationHTTPFailure
    ) -> None:
        """Route a persistent-HTTP result on a speculative probe."""
        track_speculation = self._track_speculation
        if not request.is_speculative or track_speculation is None:
            # SpeculationHTTPFailure only makes sense for a speculative
            # probe with a tracker wired up. If it ever reaches a
            # non-speculative request (or one with no tracker), do NOT
            # silently mark it completed — that would record a persistent
            # HTTP failure as a success and drop it. Treat it as a failure.
            logger.warning(
                "Worker %d got SpeculationHTTPFailure on non-speculative "
                "request %d (HTTP %s): %s",
                self.worker_id,
                request_id,
                e.status_code,
                e.url,
            )
            await self._fail_request(request_id, e, e.url)
        else:
            # Persistent HTTP on a speculative probe: a miss (not an error).
            # Store what came back with its outcome, then mark complete. No
            # retry, no step, no error row.
            logger.info(
                "Worker %d speculation probe HTTP %s on request %d: %s",
                self.worker_id,
                e.status_code,
                request_id,
                e.url,
            )
            # Status alone where the transport observed no body.
            response = e.debug_response or Response(
                status_code=e.status_code,
                headers={},
                content=b"",
                url=e.url,
                request=request,
            )
            await self._store_miss(
                request_id,
                request,
                response,
                await track_speculation(request, response),
            )

    async def _store_miss(
        self,
        request_id: int,
        request: Request,
        response: Response,
        outcome: SpeculationOutcome | None,
    ) -> None:
        """Store a probe that found nothing, and complete it without its step."""
        await self._storage.store_response(
            request_id, response, self._step_name(request), outcome
        )
        await self._storage.mark_request_completed(request_id)

    async def _execute_one(
        self,
        request_id: int,
        request: Request,
        parent_request_id: int | None,
        step_name: str,
    ) -> None:
        """The success path: lease, gate, resolve, persist, run step.

        Failures propagate to :meth:`_handle_one`, which routes them by the
        exception taxonomy.

        Sets :attr:`_server_answered` as soon as a resolve returns, so
        :meth:`_report_outcome` can credit the breaker even when a later step
        (the step) is what failed.
        """
        # Everything before the resolve is pre-execute wait: the lease (a
        # poisoned browser handle is rebuilt there, which can take seconds),
        # the archive pre-check and the gates. See the restamp below.
        waiting_since = time.monotonic()
        # A probe queued before its template stopped is not worth a fetch:
        # complete it unfetched, before it takes a lease or a gate slot.
        if (
            request.is_speculative
            and self._speculation_stopped is not None
            and self._speculation_stopped(request)
        ):
            await self._storage.mark_request_completed(
                request_id,
                speculation_outcome=SpeculationOutcome.TERMINATED_EARLY,
            )
            return
        # Lease at the top of each attempt; a poisoned handle is rebuilt
        # here. acquire can raise TransientException — same handling as
        # resolve below.
        handle = await self._transport.acquire(self.worker_id)

        queued = QueuedRequest(
            request=request,
            request_id=request_id,
            parent_request_id=parent_request_id,
        )
        is_archive = request.archive

        # Archive pre-check BEFORE gating: a skipped download does no
        # network I/O, so it must not consume a rate-limiter token.
        archive_decision: ArchiveDecision | None = None
        skip_download = False
        if is_archive:
            archive_decision = await self._archive_should_download(request)
            skip_download = not archive_decision.download

        # Pick the request's lane once; the same limiter hears the outcome
        # (``_report_outcome``). Gate outside the timed region (and skip it
        # for a skipped download). Circuit breaker first: an open circuit
        # must not claim a rate-limiter slot, and when it closes the limiter
        # re-spaces the released workers. The ``none`` lane and replay
        # gate through a NoopRateLimiter.
        limiter = self._rate_limiters.for_request(request)
        self._outcome_limiter = limiter
        if not skip_download:
            with obs.phase(obs.Phase.CIRCUIT_BREAKER_GATE):
                await self._circuit_breaker.gate()
            with obs.phase(obs.Phase.RATE_LIMITER_GATE):
                await limiter.gate(request)

        # Re-stamp the persisted start after the gate so a DB-derived
        # duration reflects the execute region, not time spent leasing a
        # handle or waiting for a rate-limiter token (started_at was stamped
        # at dequeue) — but only when that actually took time; an immediate
        # pass-through leaves the dequeue stamp accurate and skips the write.
        if time.monotonic() - waiting_since >= self.RESTAMP_MIN_GATE_WAIT_S:
            await self._queue.restamp_request_start(request_id)

        with obs.phase(obs.Phase.TRANSPORT_RESOLVE):
            if is_archive:
                response = await self._resolve_archive(
                    handle,
                    queued,
                    archive_decision,
                    skip_download=skip_download,
                )
            else:
                response = await self._transport.resolve(
                    handle,
                    queued,
                    await_conditions=self._await_conditions(step_name),
                )

        # The site responded. Record it before the step runs: what
        # the step makes of the response is not the server's doing,
        # and a skipped archive download (no network I/O) never gets here.
        if not skip_download:
            self._server_answered = True

        # Track speculation outcome for @speculate requests before the
        # step runs (on the success path).
        speculation_outcome: SpeculationOutcome | None = None
        if request.is_speculative and self._track_speculation is not None:
            speculation_outcome = await self._track_speculation(
                request, response
            )

        if speculation_outcome in _MISSES:
            # A 2xx the scraper's actually_successful rejects: the page is
            # not the record, so keep it but run no step on it.
            await self._store_miss(
                request_id, request, response, speculation_outcome
            )
        else:
            # Persist + run step + mark complete. A Playwright
            # WorkerPage handle carries a live ``.page`` (for autowait); the
            # ``WorkerHandle`` default is None, which is what HTTP/replay
            # noop handles hand back.
            with obs.phase(obs.Phase.STEP):
                await self._executor.complete_request(
                    request_id,
                    response,
                    request,
                    step_name,
                    page=handle.page,
                    speculation_outcome=speculation_outcome,
                )

        # Count toward the step's compactor — only requests that store a
        # response body; archive requests persist file metadata only.
        if not is_archive:
            await self._record_for_compactor(step_name)

    async def _resolve_archive(
        self,
        handle: Any,
        queued: QueuedRequest,
        decision: ArchiveDecision | None,
        *,
        skip_download: bool,
    ) -> Response:
        """Resolve an archive request into an ``ArchiveResponse``.

        On a skip decision, returns a synthetic response pointing at the
        existing file with no network I/O. Otherwise streams the body via the
        transport, saves it through the archive handler — measuring and
        hashing the bytes on their way through — and releases the
        transport-side backing with ``finish_archiving``.
        """
        request = queued.request
        if skip_download:
            assert decision is not None
            return ArchiveResponse(
                status_code=200,
                headers={},
                content=b"",
                url=request.request.url,
                request=request,
                file_url=decision.file_url,
            )

        stream: ArchiveStream = await self._transport.resolve_archive(
            handle, queued
        )
        digest = hashlib.sha256()
        size = 0

        async def measured() -> AsyncIterator[bytes]:
            nonlocal size
            async for chunk in stream:
                digest.update(chunk)
                size += len(chunk)
                yield chunk

        try:
            file_url = await self._archive_handler.save_stream(
                url=request.request.url,
                deduplication_key=request.effective_deduplication_key,
                expected_type=request.expected_type,
                hash_header_value=None,
                chunks=measured(),
            )
        finally:
            await self._transport.finish_archiving(stream)

        return ArchiveResponse(
            status_code=stream.status_code,
            headers=dict(stream.headers),
            content=b"",
            url=request.request.url,
            request=request,
            file_url=file_url,
            file_size=size,
            content_hash=digest.hexdigest(),
        )

    async def _archive_should_download(
        self, request: Request
    ) -> ArchiveDecision:
        """Consult the archive handler's ``should_download`` for ``request``."""
        return await self._archive_handler.should_download(
            url=request.request.url,
            deduplication_key=request.effective_deduplication_key,
            expected_type=request.expected_type,
            hash_header_value=None,
        )

    def _step_name(self, request: Request) -> str:
        """Resolve the request's step to its method name."""
        step = request.step
        if isinstance(step, str):
            return step
        return step.__name__

    def _await_conditions(self, step_name: str) -> Sequence[AwaitCondition]:
        """Derive resolve await-conditions from the target step's await_list."""
        if not step_name:
            return ()
        step = self._scraper.get_step(step_name)
        metadata = get_step_metadata(step)
        if metadata is None:
            return ()
        return tuple(metadata.await_list)

    async def _record_for_compactor(self, step_name: str) -> None:
        """Count one completed request toward its step's compactor, if any."""
        if step_name:
            await self._compactors.record(step_name)

    def _report_outcome(self, outcome: obs.Outcome) -> None:
        """Tell the circuit breaker and rate limiter how one request went.

        The single notification site, reached from ``_handle_one``'s
        ``finally`` so every arm — including ones added later — reports
        exactly once. Two independent facts decide the signal:

        * :attr:`_server_answered` — the site responded at all. That is
          availability evidence whatever happened next, so a 200 whose
          step then raised an assumption error still resets the
          breaker's consecutive-failure count. The server is not the thing
          that broke.
        * :attr:`BREAKER_FAILURE_OUTCOMES` — the outcome is server distress.

        Neither, and the breaker hears nothing: a skipped archive download
        and a pre-resolved request never touched the network, and an
        unclassified crash before a resolve says nothing about the site.

        The rate limiter's feedback (:meth:`RateLimiter.record_response`)
        is reported here too, from the stashed ``_outcome_exc`` to the
        stashed ``_outcome_limiter`` — the lane that gated the request — so
        every classified HTTP failure — transient, persistent, or a failed
        speculation probe — reaches its own lane exactly once.

        Speculation tracking and compactor accounting are deliberately NOT
        here: both are success-path-only and both are order-sensitive
        (speculation must be recorded *before* the step runs, since
        the step can enqueue further probes; the compactor counts
        *after*). Reporting them post-hoc alongside the breaker would
        silently reorder them.
        """
        if self._server_answered:
            self._circuit_breaker.record_success()
        elif outcome in self.BREAKER_FAILURE_OUTCOMES:
            self._circuit_breaker.record_failure()
        # The request's lane hears every classified HTTP failure — status
        # and any Retry-After — so an adaptive limiter can slow that lane
        # down. Static limiters inherit the no-op default. The persistent
        # arm matters too: a scraper that reclassifies 429 as persistent is
        # still being throttled. No limiter means the request never reached
        # the gate, so it cannot have an HTTP outcome to report.
        exc = self._outcome_exc
        limiter = self._outcome_limiter
        if limiter is not None and isinstance(
            exc,
            HTTPResponseAssumptionException
            | PersistentHTTPResponseException
            | SpeculationHTTPFailure,
        ):
            limiter.record_response(
                exc.status_code, retry_after=exc.retry_after, url=exc.url
            )

    async def _fail_request(
        self, request_id: int, exc: Exception, request_url: str | None
    ) -> None:
        """Report ``request_id`` as terminally failed because of ``exc``.

        Forwards to the run's
        :meth:`~jkent.driver.unified_driver.persistence.ErrorSink.request_failed`.
        The default sink
        (:class:`~jkent.driver.unified_driver.persistence.RowOnlyErrorSink`)
        marks the row and files nothing.
        """
        await self._error_sink.request_failed(
            request_id, exc, request_url=request_url
        )

    async def _store_error_for(
        self, exc: Exception, request_id: int, request_url: str | None
    ) -> None:
        """File an error without changing the request's status.

        For a failure the caller routes some other way — a replay host
        stubs the row and walks to a reseedable anchor — where the diagnosis is
        still worth keeping. The terminal report is :meth:`_fail_request`.
        """
        await self._error_sink.record(
            exc, request_id=request_id, request_url=request_url
        )

    def _request_url(self, request: Request) -> str | None:
        """Best-effort URL for error reporting."""
        try:
            return request.request.url
        except AttributeError:
            return None
