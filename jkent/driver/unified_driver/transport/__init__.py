"""The ``Transport`` seam: the one thing the unified driver's backends differ on.

The driver core owns orchestration — the queue, the worker pool, storage,
rate limiting, retries. A ``Transport`` owns the other half: turning a
request into a response, the per-worker resource that work runs on, and
recovery when that resource dies. jkent ships two implementations — HTTP
(httpx) and browser (Playwright/Camoufox) — and hosts add their own: a
replay transport serving responses from previous-run DBs plugs into this
same interface, with no jkent changes.

Mostly interface (the ``Transport`` ABC and its ``WorkerHandle`` /
``ArchiveStream`` collaborators); the handful of concrete bases every
transport would otherwise duplicate live here too — :class:`NoopHandle` and
:class:`StatelessTransport` (HTTP, and any transport with no per-worker
resource, e.g. replay) and :class:`FileArchiveStream` (file-backed archive
bodies).
"""

from __future__ import annotations

import abc
import asyncio
import math
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Generic,
    TypeAlias,
    TypeVar,
)

from typing_extensions import override

from jkent.common.exceptions import (
    HTTPResponseAssumptionException,
    PersistentHTTPResponseException,
    SpeculationHTTPFailure,
)
from jkent.common.request import DEFAULT_TIMEOUT_S
from jkent.contracts import ensure
from jkent.data_types import (
    HTTPCodeType,
    Response,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)
from jkent.driver.handles import NoopHandle, WorkerHandle
from jkent.driver.unified_driver.lifecycle import AsyncLifecycle

if TYPE_CHECKING:
    from collections.abc import Mapping

    from jkent.data_types import (
        BaseScraper,
        Request,
    )
    from jkent.driver.database_engine.sql_manager import SQLManager

#: Upper clamp for a server-sent ``Retry-After``, in seconds. Parsed and
#: clamped once, in :func:`parse_retry_after`, so both consumers — the retry
#: scheduler's per-request floor and the rate limiter's global pause — can
#: trust the value: a buggy or hostile header (``Retry-After: 86400``) must
#: slow the run down, never stall it.
MAX_RETRY_AFTER_S: Final[float] = 300.0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """``headers[name]``, looked up case-insensitively (``name`` lowercase)."""
    return next((v for k, v in headers.items() if k.lower() == name), None)


def _http_date(value: str) -> datetime | None:
    """An aware datetime from an HTTP-date, or ``None`` if unparseable."""
    try:
        when = parsedate_to_datetime(value.strip())
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when


@ensure(
    lambda result: (  # pyrefly: ignore[implicit-any-lambda]
        result is None
        or (math.isfinite(result) and 0.0 <= result <= MAX_RETRY_AFTER_S)
    ),
    "a parsed Retry-After is a finite number of seconds within the clamp",
)
def parse_retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Parse a ``Retry-After`` header into clamped seconds, if present.

    Accepts both RFC 9110 forms — delta-seconds and HTTP-date — with a
    case-insensitive lookup, since transports differ in header casing. An
    HTTP-date is measured against the response's ``Date`` header when it
    parses, so server clock skew cancels out, else against our clock.
    Returns seconds clamped to ``[0, MAX_RETRY_AFTER_S]``, or ``None`` when
    the header is absent or unparseable (a malformed value is a server bug;
    "no signal" beats guessing). ``nan`` counts as unparseable: ``float``
    accepts it, but it defeats both clamps, and a NaN pause would park the
    rate limiter's lane for the rest of the run.
    """
    if not headers:
        return None
    value = _header(headers, "retry-after")
    if value is None:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        when = _http_date(value)
        if when is None:
            return None
        # Measure against the response's own Date, so a skewed server clock
        # skews both ends alike; our clock only when Date is missing.
        date = _header(headers, "date")
        now = _http_date(date) if date is not None else None
        if now is None:
            now = datetime.now(timezone.utc)
        seconds = (when - now).total_seconds()
    if math.isnan(seconds):
        return None
    return min(max(seconds, 0.0), MAX_RETRY_AFTER_S)


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


class ArchiveStream(abc.ABC):
    """A streamed archive body plus its response metadata.

    Returned by :meth:`Transport.resolve_archive`. The caller iterates the
    body in chunks and writes it to storage, then hands this object back to
    :meth:`Transport.finish_archiving` — exactly once, whether or not the
    body was read to the end — to release it: :meth:`aclose` and any
    transport-side backing, e.g. the temp file a Playwright download is
    staged to before it can be streamed.

    The metadata trio (``status_code``/``headers``/``url``) is stored by
    this base ``__init__``; subclasses add their own body source and
    implement :meth:`__aiter__`, and :meth:`aclose` when that source holds
    anything open.
    """

    status_code: int
    headers: dict[str, str]
    url: str

    def __init__(
        self, *, status_code: int, headers: dict[str, str], url: str
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.url = url

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[bytes]:
        """Iterate the response body in chunks."""

    async def aclose(self) -> None:
        """Release the body source, read to the end or not. Idempotent."""
        return None


class FileArchiveStream(ArchiveStream):
    """An :class:`ArchiveStream` that reads a local file in chunks.

    Shared by the transports that can only surface an archive as a file on
    disk: Playwright stages a download to a temp file (deleted by its
    ``finish_archiving``), and replay points at a stored archive file it
    does not own (its ``finish_archiving`` is the inherited no-op). The
    deletion policy lives in each transport, not here.
    """

    def __init__(
        self,
        *,
        status_code: int,
        headers: dict[str, str],
        url: str,
        file_path: str,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__(status_code=status_code, headers=headers, url=url)
        self.file_path = file_path
        self._chunk_size = chunk_size
        self._iterators: list[AsyncGenerator[bytes, None]] = []

    def __aiter__(self) -> AsyncIterator[bytes]:
        iterator = self._chunks()
        self._iterators.append(iterator)
        return iterator

    @override
    async def aclose(self) -> None:
        """Close every iterator handed out, and with it the file."""
        while self._iterators:
            await self._iterators.pop().aclose()

    async def _chunks(self) -> AsyncGenerator[bytes, None]:
        with await asyncio.to_thread(open, self.file_path, "rb") as handle:
            while True:
                chunk = await asyncio.to_thread(handle.read, self._chunk_size)
                if not chunk:
                    break
                yield chunk


HandleT = TypeVar("HandleT", bound=WorkerHandle)


class Transport(AsyncLifecycle, Generic[HandleT]):
    """Executes requests for the driver, owning its own resource lifecycle.

    Composes :class:`AsyncLifecycle` (``open``/``aclose`` for the run-scoped
    resource, e.g. a browser engine). Crash recovery is **internal**, not a
    caller concern: a dead resource surfaces as a ``TransientException`` from
    ``resolve`` (which poisons the worker's handle), and the next ``acquire``
    rebuilds it — escalating to a single-flight restart of the shared
    resource when a crash poisons the whole handle cache. Callers drive none
    of this; they retry transients and re-``acquire``.
    """

    #: Whether ``resolve`` records the sub-requests a navigation fires into
    #: the ``incidental_requests`` table. Only a transport driving a real
    #: page can (Playwright); HTTP and replay see one exchange and no
    #: sub-resources. The step executor reads this to reject an
    #: ``incidental=`` step up front with an actionable message, rather than
    #: letting it fail later as a cardinality violation against zero
    #: captures — which reads as "the site changed", not "wrong transport".
    captures_incidentals: ClassVar[bool] = False

    @property
    def timeout(self) -> float:
        """Seconds a request without its own ``timeout`` waits.

        The one default shared by every transport. Implementations that
        accept a ``timeout`` kwarg override this with the configured value;
        transports with no network wait of their own (replay) inherit it so
        the attribute is part of the seam, not a per-class convention.
        """
        return DEFAULT_TIMEOUT_S

    @abc.abstractmethod
    async def acquire(self, worker_id: int) -> HandleT:
        """Get or create the per-worker handle for ``worker_id``.

        A handle poisoned by a crash is rebuilt here; when the shared
        resource itself is dead this escalates to a single-flight restart of
        it. May raise ``TransientException`` if the rebuild can't complete
        (e.g. a context that can't be restarted), so callers treat ``acquire``
        like ``resolve`` for retry purposes.
        """

    @abc.abstractmethod
    async def release(self, worker_id: int) -> None:
        """Close and forget the handle for ``worker_id`` (worker exiting)."""

    @abc.abstractmethod
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

    @abc.abstractmethod
    async def resolve_archive(
        self,
        handle: HandleT,
        queued: QueuedRequest,
    ) -> ArchiveStream:
        """Begin an archive download and return a stream of its body.

        All archiving is streamed: the caller reads the body in chunks from
        the returned :class:`ArchiveStream` and writes it to storage. How
        the transport produces those chunks is its own concern — Playwright
        can only obtain a download as a local file, so it stages to a temp
        file and streams from there, released later by ``finish_archiving``.
        ``queued.request`` is an archive request the worker has already
        decided to download.
        """

    async def finish_archiving(self, stream: ArchiveStream) -> None:
        """Release ``stream`` and any transport-side backing for it.

        Called exactly once per :meth:`resolve_archive`, whether the caller
        read the body to the end, stopped early, or failed. This default
        closes the stream, which is all a transport owning nothing else
        needs (replay reads a file it does not own); HTTP closes its
        streaming connection and Playwright also deletes its staged temp
        file by overriding this.
        """
        await stream.aclose()

    def bind_run_db(self, db: SQLManager) -> None:
        """Receive the run's database handle, before ``open`` is called.

        Most transports never touch the run database — they turn a request
        into a response and the driver persists it. Two do: a browser
        transport writes the sub-requests a navigation captured into
        ``incidental_requests``, and a replay transport copies those same
        rows forward from the corpus it is replaying, so that
        ``incidental=`` steps promote under replay exactly as they did live.

        This exists because a transport handed to ``ScrapeRun(transport=...)``
        is constructed *before* the run's database, so it cannot be given one
        at construction the way the bootstrapper gives
        :class:`PlaywrightTransport` its own. Called once, after the database
        is up and before :meth:`open`. Defaults to ignoring it.
        """
        return None

    async def export_cookies(self) -> str | None:
        """Serialize the transport's cookie jar, or None if there is none.

        The run persists the result on close and feeds it back to
        :meth:`import_cookies` on the next open, so a browser session
        survives a restart. None means nothing to persist (no jar, or it is
        gone) and leaves the saved jar alone; an empty jar is not None but
        its serialization (``"[]"``), so a logout overwrites the saved
        session. Defaults to None for transports with no jar of
        their own to carry across runs — HTTP (httpx holds cookies on the
        client, for the run's lifetime) and replay.
        """
        return None

    async def import_cookies(self, cookies_json: str) -> None:
        """Load a jar previously produced by :meth:`export_cookies`.

        Called once at run open with whatever the previous run persisted.
        Defaults to a no-op, paired with the default ``export_cookies``.
        """
        return None

    def classify_and_raise(
        self,
        scraper: type[BaseScraper[Any]] | BaseScraper[Any],
        request: Request,
        *,
        status_code: int,
        headers: Mapping[str, str] | None,
        body: bytes | None,
        url: str,
    ) -> None:
        """Consult the scraper's classifier and raise if the status is an error.

        The status-classification half of the ``resolve`` contract, shared by
        every transport: ``resolve`` must run the observed status/headers/body
        through the scraper's :meth:`~jkent.data_types.BaseScraper.classify`
        and map each verdict to the right exception —

        - transient  -> :class:`HTTPResponseAssumptionException` (retryable),
        - persistent -> :class:`PersistentHTTPResponseException`, narrowed to
          :class:`SpeculationHTTPFailure` for speculative requests so the
          worker records a speculation outcome instead of an error row,
        - successful -> return silently; the caller then returns its
          :class:`Response`. Note the default classifier treats codes
          absent from the scraper's map as persistent, so only codes the
          scraper claims as successful pass through.

        ``headers`` and ``body`` are best-effort: ``None`` where a transport
        hasn't observed them (streaming), or a reconstruction (Playwright's
        DOM snapshot and synthesized headers) rather than the raw wire bytes.
        Scraper classifiers must tolerate that.

        An exchange whose body was observed is reassembled into a
        ``Response`` and travels on the raised exception as
        ``debug_response``, so the worker persists the failed exchange to the
        run db the same way it persists a successful one — one shape for
        every failure that has a body. A
        ``Retry-After`` header is parsed and clamped here too (see
        :func:`parse_retry_after`) and travels as ``retry_after``, feeding
        both the retry scheduler's floor and the rate limiter's pause.
        """
        verdict = scraper.classify(status_code, headers, body)
        if verdict is HTTPCodeType.SUCCESSFUL:
            return
        retry_after = parse_retry_after(headers)
        debug_response = self._debug_response(
            request,
            status_code=status_code,
            headers=headers,
            body=body,
            url=url,
        )
        if verdict is HTTPCodeType.TRANSIENT:
            raise HTTPResponseAssumptionException(
                status_code=status_code,
                # The scraper's own map is what "expected" means here; the
                # message is only honest if it names those codes.
                expected_codes=sorted(
                    code
                    for code, code_type in (
                        scraper.active_http_code_types().items()
                    )
                    if code_type is HTTPCodeType.SUCCESSFUL
                ),
                url=url,
                debug_response=debug_response,
                retry_after=retry_after,
            )
        if getattr(request, "is_speculative", False):
            raise SpeculationHTTPFailure(
                status_code,
                url,
                debug_response=debug_response,
                retry_after=retry_after,
            )
        raise PersistentHTTPResponseException(
            status_code,
            url,
            debug_response=debug_response,
            retry_after=retry_after,
        )

    @staticmethod
    def _debug_response(
        request: Request,
        *,
        status_code: int,
        headers: Mapping[str, str] | None,
        body: bytes | None,
        url: str,
    ) -> Response | None:
        """Reassemble the observed exchange, or None if no body was observed.

        ``body is None`` means the transport never read one (streaming);
        storing that as an empty response would write over a previous
        attempt's real body, so it yields None. An empty body the server
        did send (``b""``) is an observation and is kept.
        """
        if body is None:
            return None
        return Response(
            status_code=status_code,
            headers=dict(headers) if headers else {},
            content=body,
            url=url,
            request=request,
        )


class StatelessTransport(Transport[NoopHandle]):
    """A transport with no per-worker resource: every handle is a no-op.

    Owns the handle cache such transports would otherwise each repeat —
    HTTP (httpx pools internally) and replay (reads from a source DB). A
    worker's handle is stable from ``acquire`` until ``release``.
    """

    def __init__(self) -> None:
        self._handles: dict[int, NoopHandle] = {}

    async def acquire(self, worker_id: int) -> NoopHandle:
        """Get-or-create the worker's handle, stable until release."""
        return self._handles.setdefault(worker_id, NoopHandle())

    async def release(self, worker_id: int) -> None:
        """Drop the worker's handle; the next acquire makes a fresh one."""
        handle = self._handles.pop(worker_id, None)
        if handle is not None:
            await handle.close()
