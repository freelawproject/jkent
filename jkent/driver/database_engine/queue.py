"""RequestQueueDB - DB-backed request queue operations for the unified driver."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import ValidationError

from jkent.common.exceptions import ScraperConfigError
from jkent.common.rate_limits import RateLimitTable
from jkent.common.request import (
    SkipDeduplicationCheck,
    serialize_url_and_body,
)
from jkent.common.serialization import dump_json, dump_json_or_none
from jkent.contracts import require
from jkent.data_types import (
    HTTPRequestParams,
    Request,
    RequestData,
    Response,
    via_from_json,
)
from jkent.driver.database_engine.enums import RequestType
from jkent.driver.database_engine.sql_manager import (
    DequeuedRow,
    InsertResult,
    RequestInsert,
    SQLManager,
)

__all__ = [
    "Dequeued",
    "EnqueuedEvent",
    "RequestQueueDB",
    "serialize_url_and_body",
]

if TYPE_CHECKING:
    from collections.abc import Callable

    from jkent.driver.database_engine.sql_manager import StoredResponse
    from jkent.driver.database_engine.staging import StagedWrites


class Dequeued(NamedTuple):
    """What :meth:`RequestQueueDB.get_next_request` hands the worker."""

    request_id: int
    request: Request
    parent_request_id: int | None
    #: True when the row already carries its response (promoted from a
    #: captured incidental), so the worker skips the transport.
    preresolved: bool


class EnqueuedEvent(NamedTuple):
    """The ``request_enqueued`` progress event for one enqueued request.

    A named shape rather than a bare dict so the three hops it makes — built
    in :meth:`RequestQueueDB._prepare_enqueue`, buffered by
    :class:`~jkent.driver.database_engine.staging.StagedWrites`, returned by
    its ``flush`` — are checked instead of stringly-keyed.
    :meth:`payload` is the untyped form the
    :data:`~jkent.driver.unified_driver.persistence.ProgressCallback`
    boundary takes, built in one place.

    ``url`` and ``step`` are both ``str``, so build one by keyword: a
    positional call that transposes the two still type-checks.
    """

    url: str
    step: str
    priority: int

    def payload(self) -> dict[str, Any]:
        """The event as a progress callback's ``data`` mapping."""
        return self._asdict()


def _step_name(step: str | Callable[..., Any]) -> str:
    """*step* as the row stores it: a name the worker can ``getattr``.

    The row stores the step by name and the worker resolves it with
    getattr(scraper, name). A callable with no ``__name__`` (a
    functools.partial, a callable object) has no name to resolve, so refuse
    it here, on the step that yielded it, not at dequeue.

    Raises:
        ScraperConfigError: when *step* is a callable with no ``__name__``.
    """
    if isinstance(step, str):
        return step
    name = getattr(step, "__name__", None)
    if not isinstance(name, str):
        raise ScraperConfigError(
            f"step {step!r} has no __name__; pass the scraper "
            "method or its name"
        )
    return name


def _check_dequeuable(request_data: RequestInsert) -> None:
    """Refuse a row the dequeue could not decode, at enqueue.

    An insert carries the JSON columns as text, so anything ``json.dumps``
    can spell inserts cleanly; the types are only enforced on the way back
    out, against :class:`DequeuedRow` (``headers``/``cookies`` as
    ``dict[str, str]``, ``timeout`` as a float or a connect/read pair).
    ``HTTPRequestParams`` is an unvalidated dataclass and ``permanent``
    headers merge into it unchecked, so ``headers={"X-Attempt": 3}`` reaches
    the column intact — and pydantic will not coerce that int back to a str.

    By then it is far too late: ``dequeue_next_request`` validates *after*
    its commit, so the row is already ``in_progress`` and no second dequeue
    will ever offer it again, and the ``ValidationError`` escapes the
    worker's dequeue above its ``request_done`` guard, retiring a worker
    from a pinned pool. Round-tripping the row through the dequeue types
    here puts the failure on the step that yielded it, the way the unknown
    rate-limit lane and the nameless callable step already do.

    Raises:
        ScraperConfigError: when the row would not decode at dequeue.
    """
    try:
        DequeuedRow.model_validate(
            {**request_data.model_dump(), "id": 0, "preresolved": False}
        )
    except ValidationError as exc:
        raise ScraperConfigError(
            f"request {request_data.url} cannot be stored as queued: {exc}"
        ) from exc


class RequestQueueDB:
    """DB-backed queue: enqueue (staged), dequeue, (de)serialization.

    Provides methods for persisting requests to SQLite with deduplication
    and reconstructing request objects from database rows.

    Also tracks how many dequeued requests are still being handled
    (``in_flight_count``): a worker pool is pinned — retired workers are
    never replaced — so an idle worker must distinguish a drained queue from
    a momentary lull while a sibling's in-flight request may still enqueue
    children. The count is in-memory (all workers share one queue instance),
    so stale ``in_progress`` rows from a crashed prior run can never wedge
    idle workers the way a DB-status check could.
    """

    def __init__(
        self, db: SQLManager, *, rate_limits: RateLimitTable | None = None
    ) -> None:
        """
        Args:
            db: The run database.
            rate_limits: The scraper's rate-limit lanes, which encode a
                request's ``rate_limit`` name to the integer the row stores
                and back. Defaults to the two framework lanes only, so a
                queue built without a scraper still round-trips ``None``
                and ``"none"`` but rejects a scraper-declared lane.
        """
        self.db = db
        self._in_flight = 0
        self._rate_limits = rate_limits or RateLimitTable.bare()

    @property
    def in_flight_count(self) -> int:
        """Dequeued requests not yet marked done via :meth:`request_done`."""
        return self._in_flight

    @require(
        lambda self: self.in_flight_count > 0,  # pyrefly: ignore[implicit-any-lambda]
        "request_done pairs with a dequeue that returned a request",
    )
    def request_done(self) -> None:
        """Mark one dequeued request as fully handled (children enqueued).

        One call per dequeue that returned a request, and no more: the
        count is the pinned pool's only termination signal, and a negative
        one never returns to zero, so an unpaired call leaves every idle
        worker polling a drained queue forever. The contract catches that
        under test; the floor keeps a release build failing the run rather
        than hanging it.
        """
        self._in_flight = max(0, self._in_flight - 1)

    async def _prepare_enqueue(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None,
    ) -> tuple[RequestInsert, EnqueuedEvent]:
        """Resolve, serialize, and dedup/parent-resolve a request for enqueue.

        The shared core of both enqueue paths — the immediate
        ``RequestQueue.enqueue_request`` and the staged
        ``_stage_enqueue_request`` — so the two cannot drift on serialization,
        the dedup-key / priority / parent-id rules, or the progress payload.

        Returns ``(request_data, progress_event)`` where ``request_data``
        already carries the effective ``priority``, ``deduplication_key``,
        and ``parent_request_id``, ready for ``insert_request`` /
        ``stage_request``.
        """
        # Await any async field resolvers first: serialization needs concrete
        # values, and the regenerated dedup key must hash them.
        new_request = await new_request.resolve_deferred_fields()
        resolved_request: Request = new_request.resolve_from(context)

        request_data = self._insert_payload(resolved_request)
        request_data.parent_request_id = parent_request_id

        progress_event = EnqueuedEvent(
            url=request_data.url,
            step=request_data.step,
            priority=resolved_request.effective_priority,
        )
        return request_data, progress_event

    async def insert_root_request(
        self, request: Request, *, deduplicate: bool = True
    ) -> InsertResult:
        """Insert a request with no parent to resolve against (own transaction).

        The enqueue path for entry requests and speculative probes, which
        have no response or request context. Probes pass
        ``deduplicate=False``: a deduplicated probe would never run, and the
        speculation tracker would never see its outcome.
        """
        # A hand-built request can still carry async field resolvers.
        request = await request.resolve_deferred_fields()
        request_data = self._insert_payload(request)
        if not deduplicate:
            request_data.deduplication_key = None
        return await self.db.insert_request(request_data)

    def _insert_payload(self, request: Request) -> RequestInsert:
        """The serialized row with the effective priority and dedup key."""
        request_data = self.serialize_request(request)
        request_data.priority = request.effective_priority
        request_data.deduplication_key = request.effective_deduplication_key
        return request_data

    async def _stage_enqueue_request(
        self,
        new_request: Request,
        context: Response | Request,
        parent_request_id: int | None,
        staged: StagedWrites,
        preresolved_response: StoredResponse | None = None,
    ) -> None:
        """Stage an enqueue for the parent step's flush.

        Mirrors ``enqueue_request`` but defers the DB insert and progress
        event until ``staged.flush()`` is called. When ``preresolved_response``
        is set the row is inserted pre-resolved (response stored, worker skips
        the transport).
        """
        request_data, progress_event = await self._prepare_enqueue(
            new_request, context, parent_request_id
        )
        staged.stage_request(
            request_data=request_data,
            progress_event=progress_event,
            preresolved_response=preresolved_response,
        )

    def serialize_request(
        self,
        request: Request,
    ) -> RequestInsert:
        """Serialize a Request to a :class:`RequestInsert` for DB storage.

        ``priority``, ``deduplication_key``, and ``parent_request_id`` are
        left at their defaults — the enqueue paths fill them in.

        Args:
            request: The request to serialize.

        Returns:
            RequestInsert with the serialized request data.
        """
        http_request = request.request

        step = _step_name(request.step)

        # Determine request type and expected_type
        request_type: RequestType
        if request.archive:
            request_type = RequestType.ARCHIVE
            expected_type = request.expected_type
        elif request.nonnavigating:
            request_type = RequestType.NON_NAVIGATING
            expected_type = None
        else:
            request_type = RequestType.NAVIGATING
            expected_type = None

        # Fold query params into the URL and encode the body. Shared with
        # replay's key derivation so the two encodings can't drift.
        url, body, body_is_form = serialize_url_and_body(http_request)

        request_data = RequestInsert(
            request_type=request_type,
            method=http_request.method,
            url=url,
            headers_json=dump_json_or_none(http_request.headers),
            cookies_json=dump_json_or_none(http_request.cookies),
            body=body,
            body_is_form=body_is_form,
            step=step,
            current_location=request.current_location,
            accumulated_data_json=dump_json_or_none(request.accumulated_data),
            permanent_json=dump_json_or_none(request.permanent),
            expected_type=expected_type,
            is_speculative=request.is_speculative,
            speculation_tracking_id=request.speculation_tracking_id,
            speculative_index=request.speculative_index,
            verify_json=dump_json(http_request.verify),
            via_json=request.via.to_json()
            if request.via is not None
            else None,
            # By lane code; an unknown name raises here, at enqueue.
            rate_limit=self._rate_limits.encode(request.rate_limit),
            # a timeout=(connect, read) tuple stores as a JSON list;
            # DequeuedRow re-tuples it.
            timeout_json=dump_json_or_none(http_request.timeout),
            json_data=dump_json_or_none(http_request.json),
            reseedable=request.reseedable,
        )
        # Refuse a row the dequeue could not decode, here on the step that
        # yielded it.
        _check_dequeuable(request_data)
        return request_data

    async def get_next_request(self) -> Dequeued | None:
        """Get the next pending request from the database.

        Returns:
            :class:`Dequeued`, or None if queue is empty.

        Notes:
            - Skips requests in retry backoff (started_at > current time)
        """
        # Atomically dequeue the next pending request (UPDATE ... RETURNING,
        # so two workers can never claim the same row).
        #
        # The in-flight count is raised BEFORE the dequeue so a claim is never
        # invisible: a sibling that reads an empty queue while this dequeue is
        # mid-commit still sees in_flight_count > 0 and keeps waiting instead
        # of retiring. The caller owns the matching request_done().
        self._in_flight += 1
        try:
            row = await self.db.dequeue_next_request()
        except BaseException:
            self._in_flight -= 1
            raise

        if row is None:
            self._in_flight -= 1
            return None

        return Dequeued(
            row.id,
            self._deserialize_request(row),
            row.parent_request_id,
            row.preresolved,
        )

    async def seconds_until_next_pending(self) -> float | None:
        """Delay until the soonest pending request is dequeuable.

        0.0 if one is ready now, the positive gap until the soonest retry
        still in backoff, or None if nothing is pending. See
        :meth:`SQLManager.seconds_until_next_pending`.
        """
        return await self.db.seconds_until_next_pending()

    async def restamp_request_start(self, request_id: int) -> None:
        """Re-stamp a request's start to now, after the rate-limit gate."""
        await self.db.restamp_request_start(request_id)

    def _deserialize_request(self, row: DequeuedRow) -> Request:
        """Deserialize a dequeued database row to a Request.

        Args:
            row: :class:`DequeuedRow` from ``dequeue_next_request``.

        Returns:
            Reconstructed Request with appropriate flags set based on
            request_type (see :class:`RequestType`).

        Note:
            ``HTTPRequestParams.params`` is *not* restored — it is folded
            into the stored URL on serialize (see ``serialize_url_and_body``),
            which is the single source of truth for the request target. The
            httpx transport sends the URL as-is and never re-sends ``params``,
            so the reconstructed request carries ``params=None`` with the
            query already in the URL. This is intentional: restoring ``params``
            would either drop the query (transport ignores ``params``) or
            double-encode it on re-serialization.
        """
        decoded_body = self._decode_body(row)

        http_params = HTTPRequestParams(
            method=row.method,
            url=row.url,
            headers=row.headers_json,
            cookies=row.cookies_json,
            data=decoded_body,
            json=row.json_data,
            timeout=row.timeout_json,
            verify=row.verify_json,
        )

        via = None
        if row.via_json:
            try:
                via = via_from_json(row.via_json)
            except ValueError as exc:
                # pydantic names the field path, not the row it came from.
                raise ValueError(
                    f"request {row.id}: via_json does not match a via shape"
                ) from exc

        # One constructor call with every kwarg named, so the checker
        # verifies each one against ``Request``'s fields; the request_type
        # only varies the three flag/extra fields. (A ``**kwargs`` dict here
        # would type-check as ``Any`` and let a renamed or retyped field
        # through silently.)
        archive = row.request_type == RequestType.ARCHIVE
        return Request(
            request=http_params,
            step=row.step,
            current_location=row.current_location,
            accumulated_data=row.accumulated_data_json or {},
            permanent=row.permanent_json or {},
            priority=row.priority,
            # A NULL key is the stored form of the opt-out (explicit
            # SkipDeduplicationCheck, or ``deduplicate=False``): every other
            # enqueue stores the effective key, so NULL never means "unset".
            deduplication_key=(
                row.deduplication_key
                if row.deduplication_key is not None
                else SkipDeduplicationCheck.SKIP
            ),
            via=via,
            rate_limit=self._rate_limits.decode(row.rate_limit),
            reseedable=row.reseedable,
            # Stored for every request type, so restored for every one; a
            # probe dequeued without them is never tracked as a probe.
            is_speculative=row.is_speculative,
            speculation_tracking_id=row.speculation_tracking_id,
            speculative_index=row.speculative_index,
            archive=archive,
            nonnavigating=row.request_type == RequestType.NON_NAVIGATING,
            # Only an archive row carries one; serialize_request stores NULL
            # for every other type.
            expected_type=row.expected_type if archive else None,
        )

    @staticmethod
    def _decode_body(row: DequeuedRow) -> RequestData:
        """The row's stored body as ``HTTPRequestParams.data`` took it.

        The inverse of ``encode_body``: a JSON body is a dict or a pair list
        (re-tupled), anything else is raw bytes, verbatim.

        Raises:
            ValueError: when a ``body_is_form`` body decodes to neither a
                mapping nor a pair list — ``encode_body`` only sets the flag
                for those two, so anything else is a row no ``RequestData``
                can hold.
        """
        if row.body is None or not row.body_is_form:
            return row.body
        decoded: Any = json.loads(row.body)
        if isinstance(decoded, list):
            pairs: list[tuple[str, Any]] = [
                (name, value) for name, value in decoded
            ]
            return pairs
        if isinstance(decoded, dict):
            fields: dict[str, Any] = decoded
            return fields
        raise ValueError(
            f"request {row.id}: body_is_form body decoded to "
            f"{type(decoded).__name__}, not a mapping or pair list"
        )
