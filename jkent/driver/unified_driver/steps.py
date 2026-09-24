"""StepExecutor - run a scraper step and persist its yields.

Collaborators are injected explicitly:

* ``db``      - SQLManager, used only for the atomic ``StagedWrites.flush``.
* ``scraper`` - resolves step names.
* ``queue``   - RequestQueue, stages enqueues (reusing its (de)serialization).
* ``storage`` - ResponseStorage, stores the response, serializes results,
  and marks the request completed when there is no step.

Speculation outcome tracking lives in the worker (its ``track_speculation``
callback), not here: the executor stores whatever ``speculation_outcome`` the
worker hands it alongside the response.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json as _json
import logging
import time
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from jkent.common.decorator_metadata import get_step_metadata
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    IncidentalRequestAssumptionException,
    ScraperConfigError,
)
from jkent.common.selectors import Selector
from jkent.data_types import (
    Multiple,
    ParsedData,
    Request,
    Response,
    Singular,
)
from jkent.driver.database_engine.enums import SpeculationOutcome
from jkent.driver.database_engine.queue import _step_name
from jkent.driver.database_engine.sql_manager import (
    CompressedPayload,
    StoredResponse,
)
from jkent.driver.database_engine.staging import StagedWrites
from jkent.driver.database_engine.storage import serialize_result

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from jkent.common.selector_observer import SelectorObserver, SelectorQuery
    from jkent.data_types import ScraperYield, SkipDeduplicationCheck
    from jkent.driver.database_engine.sql_manager import (
        IncidentalRequestRecord,
        IncidentalStorageRecord,
        SQLManager,
    )
    from jkent.driver.unified_driver.persistence import (
        RequestQueue,
        ResponseStorage,
    )

logger = logging.getLogger(__name__)

# Seconds between re-snapshots when autowait cannot target the missing
# element and polls the page for a change instead.
AUTOWAIT_POLL_INTERVAL = 0.25

# Seconds before re-waiting on a selector the page already matched when the
# step failed again on the snapshot: that wait returns at once, so without a
# pause a DOM that keeps changing (a clock, a banner) re-runs the step
# back to back for the whole budget. About one rendered frame.
AUTOWAIT_RECHECK_INTERVAL = 0.016

# Headers that describe the transport's body, not a re-encoded DOM snapshot.
_SNAPSHOT_STALE_HEADERS = frozenset(
    {"content-type", "content-length", "content-encoding", "transfer-encoding"}
)


class AutowaitPage(Protocol):
    """Minimal page surface the autowait loop drives.

    A Playwright page satisfies it as is, so a wait timeout is Playwright's
    own ``TimeoutError``.
    """

    url: str

    async def wait_for_selector(
        self, selector: str, *, timeout: int
    ) -> object:
        """Wait up to ``timeout`` ms for ``selector`` to match.

        Args:
            selector: The selector to wait for.
            timeout: Milliseconds to wait before giving up.

        Returns:
            The matched element handle, if the page provides one.

        Raises:
            PlaywrightTimeoutError: If ``selector`` does not match within
                ``timeout``.
            PlaywrightError: If the page can't evaluate it right now (it is
                navigating, or closed).
        """
        ...

    async def content(self) -> str:
        """Snapshot the page's current DOM.

        Returns:
            The page's serialized HTML.

        Raises:
            PlaywrightError: If the page can't be read right now (it is
                navigating, or closed).
        """
        ...


def _has_ambiguous_parent(query: SelectorQuery) -> bool:
    """Whether some ancestor of ``query`` matched more than one element.

    The observer records which selectors a scoped query ran under, not which
    of their matches, so such a query composes to a page-wide selector that
    can't single out the element the step failed on. Distinct elements are
    counted, not ``match_count``, which a parent queried twice on the same
    element doubles.

    Args:
        query: The failing observer query.

    Returns:
        ``True`` when the composed selector is ambiguous.
    """
    parent = query.parent
    while parent is not None:
        if parent.distinct_match_count > 1:
            return True
        parent = parent.parent
    return False


def _snapshot_headers(headers: dict[str, str]) -> dict[str, str]:
    """The headers to store with an autowait DOM snapshot.

    The snapshot's body is ``page.content()`` encoded as UTF-8, not the bytes
    the transport received, so the original type, length and encoding
    headers would misdescribe it.

    Args:
        headers: The original response's headers.

    Returns:
        Those headers without the body-describing ones, plus a
        ``Content-Type`` naming the snapshot's own charset.
    """
    kept = {
        name: value
        for name, value in headers.items()
        if name.lower() not in _SNAPSHOT_STALE_HEADERS
    }
    kept["Content-Type"] = "text/html; charset=utf-8"
    return kept


def _describe_unanswered(unanswered: list[IncidentalRequestRecord]) -> str:
    """Name the captures that got no response, with why where recorded.

    Args:
        unanswered: The captures to name.

    Returns:
        Each URL, followed by its failure reason in parentheses when the
        capture recorded one.
    """
    return ", ".join(
        f"{rec.url} ({rec.failure_reason})" if rec.failure_reason else rec.url
        for rec in unanswered
    )


def _sent_request(
    rec: IncidentalRequestRecord, bodies: dict[int, bytes]
) -> str:
    """What a capture sent: its method, captured URL and request-body hash.

    Args:
        rec: The capture.
        bodies: Request bodies by capture id.

    Returns:
        The three, JSON-encoded.
    """
    body = bodies.get(rec.id)
    return _json.dumps(
        [
            rec.method.upper(),
            rec.url,
            hashlib.sha256(body).hexdigest() if body is not None else None,
        ]
    )


def _capture_identities(
    matched: list[IncidentalRequestRecord], bodies: dict[int, bytes]
) -> list[str]:
    """Hash what each capture sent: method, captured URL and request body.

    Captures that sent the same request (a page polling one endpoint) are
    told apart by their ordinal among those identical requests, so each
    keeps its own identity while unrelated captures don't shift it.

    Args:
        matched: The captures, in capture order.
        bodies: Request bodies by capture id.

    Returns:
        Each capture's identity, in ``matched`` order.
    """
    seen: Counter[str] = Counter()
    identities: list[str] = []
    for rec in matched:
        sent = _sent_request(rec, bodies)
        identity = _json.dumps([sent, seen[sent]])
        seen[sent] += 1
        identities.append(hashlib.sha256(identity.encode()).hexdigest())
    return identities


class _ParentCaptures:
    """One parent navigation's incidental captures, read at most once.

    Every incidental-carrying yield of a step, and every autowait retry of
    it, matches against the same captures, so the rows (with their headers
    parsed) and their request bodies are each loaded in one query on first
    use and reused. Response storage rows hold the bodies, and a step
    promotes only a few of a page's captures, so only the rows a match asked
    for (:meth:`want_storage`) are read, together, on the first
    :meth:`storage` call.
    """

    def __init__(self, db: SQLManager, parent_request_id: int) -> None:
        self._db = db
        self._parent_request_id = parent_request_id
        self._records: (
            list[tuple[IncidentalRequestRecord, dict[str, str]]] | None
        ) = None
        self._bodies: dict[int, bytes] | None = None
        self._wanted_storage_ids: set[int] = set()
        self._storages: dict[int, IncidentalStorageRecord] = {}

    async def records(
        self,
    ) -> list[tuple[IncidentalRequestRecord, dict[str, str]]]:
        """Each capture with its parsed request headers.

        Returns:
            ``(capture, headers)`` pairs, in capture order.
        """
        if self._records is None:
            self._records = [
                (
                    rec,
                    _json.loads(rec.headers_json) if rec.headers_json else {},
                )
                for rec in await self._db.get_incidental_requests(
                    self._parent_request_id
                )
            ]
        return self._records

    async def bodies(self) -> dict[int, bytes]:
        """The request body the browser sent on each capture.

        Returns:
            Each body by capture id; a capture that carried none is absent.
        """
        if self._bodies is None:
            self._bodies = await self._db.get_incidental_request_bodies(
                self._parent_request_id
            )
        return self._bodies

    def want_storage(self, storage_id: int) -> None:
        """Note a storage row a promoted child will read.

        Args:
            storage_id: The matched capture's storage row.
        """
        self._wanted_storage_ids.add(storage_id)

    async def storage(self, storage_id: int) -> IncidentalStorageRecord | None:
        """A matched capture's response storage row.

        The first call for a row not yet read reads every wanted row with
        it, in one query: the step has finished matching by the time its
        children are staged.

        Args:
            storage_id: The storage row to return.

        Returns:
            The row, or ``None`` if it doesn't exist.
        """
        if storage_id not in self._storages:
            self._storages |= await self._db.get_incidental_request_storages(
                (self._wanted_storage_ids | {storage_id})
                - self._storages.keys()
            )
        return self._storages.get(storage_id)


class StepExecutor:
    """Store a response, run its step, and persist the yields atomically."""

    def __init__(
        self,
        db: SQLManager,
        scraper: Any,  # BaseScraper-like: needs get_step
        queue: RequestQueue,
        storage: ResponseStorage,
        *,
        handle_data: Callable[[Any], Awaitable[None]] | None = None,
        on_invalid_data: Callable[[DeferredValidation[Any]], Awaitable[None]]
        | None = None,
        captures_incidentals: bool,
    ) -> None:
        """Bind the executor to one run's collaborators.

        Args:
            db: The run database.
            scraper: The scraper whose steps are executed.
            queue: The queue yielded requests are enqueued on.
            storage: The run's response storage.
            handle_data: Called with each valid result, if set.
            on_invalid_data: Called with each failed validation, if set.
            captures_incidentals: Whether the transport records incidental
                sub-requests; off disables incidental promotion.
        """
        self.db = db
        self.scraper = scraper
        self.queue = queue
        self.storage = storage
        self._handle_data = handle_data
        self.on_invalid_data = on_invalid_data
        # Read off the run's transport (``Transport.captures_incidentals``)
        # so an ``incidental=`` step meets a transport-shaped error instead
        # of a cardinality violation against zero captures.
        self._captures_incidentals = captures_incidentals

    async def complete_request(
        self,
        request_id: int,
        response: Response,
        request: Request,
        step_name: str,
        *,
        page: AutowaitPage | None = None,
        store_response: bool = True,
        speculation_outcome: SpeculationOutcome | None = None,
    ) -> None:
        """Store response, run step, flush its yields, mark completed.

        ``speculation_outcome`` is what the worker's speculation tracker
        recorded for a probe (``hit``), stored with the response and again
        by the write that completes the request, which is the only one when
        ``store_response`` is off.
        """
        if store_response:
            await self.storage.store_response(
                request_id, response, step_name, speculation_outcome
            )

        if not step_name:
            await self.storage.mark_request_completed(
                request_id, speculation_outcome=speculation_outcome
            )
            return

        step = self.scraper.get_step(step_name)
        staged = StagedWrites(
            request_id=request_id, speculation_outcome=speculation_outcome
        )
        captures = _ParentCaptures(self.db, request_id)

        # Autowait dispatch: only a live page carrying an auto_await_timeout
        # takes the retry loop; everything else runs the normal generator path.
        auto_await_timeout: int | None = None
        if page is not None:
            metadata = get_step_metadata(step)
            auto_await_timeout = (
                metadata.auto_await_timeout if metadata else None
            )

        if page is not None and auto_await_timeout:
            await self._process_generator_with_autowait(
                step,
                response,
                request,
                request_id,
                step_name,
                auto_await_timeout,
                page=page,
                staged=staged,
                captures=captures,
                speculation_outcome=speculation_outcome,
            )
        else:
            gen = step(response)
            await self._process_generator_with_storage(
                gen,
                response,
                request,
                request_id,
                staged,
                captures,
            )

        await staged.flush(self.db)

    async def _process_generator_with_storage(
        self,
        gen: Generator[ScraperYield[Any], bool | None, None],
        response: Response,
        parent_request: Request,
        request_id: int,
        staged: StagedWrites,
        captures: _ParentCaptures,
    ) -> None:
        """Process generator yields, buffering all DB writes in ``staged``.

        A scraper assumption violation (e.g. HTMLStructuralAssumptionException)
        propagates to the caller; the autowait loop relies on that to drive its
        wait-and-retry, and otherwise it is the driver's terminal error.

        Yielded requests are staged only once the step has run to the end.
        Staging awaits their deferred field resolvers (a captcha solver, say),
        so a run that fails partway, which autowait then repeats, must not
        pay for them.

        Raises:
            ScraperConfigError: If the step yields anything other than
                ``ParsedData``, a ``Request`` or ``None``.
        """
        enqueues: list[Callable[[], Awaitable[None]]] = []
        for item in gen:
            match item:
                case ParsedData():
                    raw_data = item.unwrap()
                    if isinstance(raw_data, DeferredValidation):
                        try:
                            validated_data = raw_data.confirm()
                            self._stage_valid_result(staged, validated_data)
                        except DataFormatAssumptionException as e:
                            staged.stage_result(
                                serialize_result(e.failed_doc, e.errors)
                            )
                            if self.on_invalid_data:
                                staged.stage_callback(
                                    functools.partial(
                                        self.on_invalid_data, raw_data
                                    )
                                )
                    else:
                        self._stage_valid_result(staged, raw_data)

                case Request():
                    if item.incidental is not None:
                        await self._stage_preresolved_incidentals(
                            item,
                            response,
                            parent_request,
                            request_id,
                            staged,
                            captures,
                            enqueues,
                        )
                    else:
                        enqueues.append(
                            functools.partial(
                                self.queue._stage_enqueue_request,
                                item,
                                self._enqueue_ctx(
                                    item, response, parent_request
                                ),
                                request_id,
                                staged,
                            )
                        )

                case None:
                    pass

                case _:
                    raise ScraperConfigError(
                        f"Step yielded a {type(item).__name__}; a step may "
                        "yield only ParsedData, a Request, or None. Wrap "
                        "parsed results in ParsedData(...)."
                    )

        for enqueue in enqueues:
            await enqueue()

    def _stage_valid_result(self, staged: StagedWrites, data: Any) -> None:
        """Serialize a valid result, stage it, and stage its on-data callback."""
        staged.stage_result(serialize_result(data))
        if self._handle_data is not None:
            staged.stage_callback(functools.partial(self._handle_data, data))

    @staticmethod
    def _enqueue_ctx(
        req: Request, response: Response, parent_request: Request
    ) -> Response | Request:
        """Pick the URL-resolution context for an enqueued request.

        A navigating request resolves against the response it came from;
        a nonnavigating/archive request resolves against its parent.

        Args:
            req: The request being enqueued.
            response: The response its step ran on.
            parent_request: The request that produced that response.

        Returns:
            The response or request URLs resolve against.
        """
        if not req.nonnavigating and not req.archive:
            return response
        return parent_request

    async def _stage_preresolved_incidentals(
        self,
        item: Request,
        response: Response,
        parent_request: Request,
        parent_request_id: int,
        staged: StagedWrites,
        captures: _ParentCaptures,
        enqueues: list[Callable[[], Awaitable[None]]],
    ) -> None:
        """Promote captured incidentals into pre-resolved child requests.

        Matches ``item.incidental`` against the sub-requests the *parent*
        navigation captured (``parent_request_id``), enforces the spec's
        cardinality, and queues one pre-resolved request per match for
        staging — its promoted response stored in the same flush
        transaction. A matched capture with no response is skipped. The
        cardinality error surfaces here, attributed to the parent navigation,
        so an over/under-broad spec fails fast at enqueue rather than as a
        phantom request later.

        Args:
            item: The yielded request carrying the incidental spec.
            response: The parent navigation's response.
            parent_request: The parent navigation's request.
            parent_request_id: Its row id, which the captures hang off.
            staged: The flush transaction's staged writes.
            captures: That navigation's captures.
            enqueues: Where each child's staging is appended, to run once
                the step has finished.

        Raises:
            ScraperConfigError: If the run's transport records no
                incidental sub-requests.
            IncidentalRequestAssumptionException: If the match count
                violates the spec's cardinality.
        """  # noqa: DOC502 -- raised by _enforce_incidental_cardinality
        spec = item.incidental
        assert spec is not None
        if not self._captures_incidentals:
            # No captures will ever exist on this transport, so the
            # cardinality check below would fail with "found 0" — a site
            # diagnosis for what is really a wiring mistake. Say so instead.
            raise ScraperConfigError(
                f"Step yields incidental={spec!r}, but the run's transport "
                "does not capture incidental sub-requests. Incidentals are "
                "recorded by a browser transport only — declare a browser "
                "driver requirement (e.g. DriverRequirement.JS_EVAL) on the "
                "scraper."
            )
        bodies: dict[int, bytes] = (
            await captures.bodies()
            if spec.body_contains is not None
            or self._keys_from_captures(item, spec)
            else {}
        )
        matched: list[IncidentalRequestRecord] = []
        unanswered: list[IncidentalRequestRecord] = []
        for rec, headers in await captures.records():
            if not spec.matches(
                url=rec.url,
                method=rec.method,
                headers=headers,
                body=bodies.get(rec.id),
                resource_type=rec.resource_type,
            ):
                continue
            # A capture with no response has nothing to promote, so it is
            # no match.
            if rec.status_code is None:
                unanswered.append(rec)
            else:
                matched.append(rec)

        # A capture gets no response when the page or browser cancelled it,
        # the network failed it, or it was still in flight when the
        # navigation's captures were saved. None leaves anything to promote,
        # so all are skipped. A step that needs such a response should make
        # its parent wait (``await_list``) on a page element that appears
        # once that request completes, so the capture is answered before it
        # is saved.
        self._enforce_incidental_cardinality(
            spec, matched, unanswered, response.url
        )
        if unanswered:
            logger.info(
                f"Incidental {spec!r} skipped {len(unanswered)} capture(s) "
                f"that got no response: {_describe_unanswered(unanswered)}"
            )

        keys = self._incidental_dedup_keys(
            item, spec, matched, bodies, parent_request_id
        )
        for rec, key in zip(matched, keys, strict=True):
            child = replace(
                item,
                incidental=None,
                # The browser minted the response once; it can't be
                # re-fetched standalone.
                reseedable=False,
                deduplication_key=key,
            )
            if rec.storage_id is not None:
                captures.want_storage(rec.storage_id)
            enqueues.append(
                functools.partial(
                    self._stage_promoted_child,
                    child,
                    self._enqueue_ctx(child, response, parent_request),
                    parent_request_id,
                    staged,
                    rec,
                    captures,
                )
            )

    async def _stage_promoted_child(
        self,
        child: Request,
        context: Response | Request,
        parent_request_id: int,
        staged: StagedWrites,
        rec: IncidentalRequestRecord,
        captures: _ParentCaptures,
    ) -> None:
        """Stage one promoted child with its capture's response.

        Args:
            child: The child request.
            context: Its URL-resolution context.
            parent_request_id: The parent navigation's row id.
            staged: The flush transaction's staged writes.
            rec: The capture it is promoted from.
            captures: That navigation's captures, holding the storage rows.
        """
        storage = (
            await captures.storage(rec.storage_id)
            if rec.storage_id is not None
            else None
        )
        await self.queue._stage_enqueue_request(
            child,
            context,
            parent_request_id,
            staged,
            preresolved_response=self._build_preresolved_response(
                rec, storage
            ),
        )

    @staticmethod
    def _enforce_incidental_cardinality(
        spec: Singular | Multiple,
        matched: list[IncidentalRequestRecord],
        unanswered: list[IncidentalRequestRecord],
        request_url: str,
    ) -> None:
        """Check a promoted incidental's match count against its spec.

        Args:
            spec: The cardinality the yielded request declared.
            matched: The captures that matched it and got a response.
            unanswered: The captures that matched it but got no response,
                named in the error since they are often why it was raised.
            request_url: The parent navigation's URL, for the error.

        Raises:
            IncidentalRequestAssumptionException: If the count violates the
                spec's cardinality.
        """
        if isinstance(spec, Singular):
            if len(matched) == 1:
                return
            expected = "exactly 1 match"
        else:
            if matched:
                return
            expected = "at least 1 match"
        message = (
            f"{type(spec).__name__} incidental expected {expected}, "
            f"found {len(matched)}"
        )
        if unanswered:
            message += (
                f"; {len(unanswered)} more matched but got no response: "
                f"{_describe_unanswered(unanswered)}"
            )
        raise IncidentalRequestAssumptionException(
            message,
            request_url,
            match_spec=repr(spec),
            candidate_urls=[r.url for r in matched],
        )

    @staticmethod
    def _build_preresolved_response(
        rec: IncidentalRequestRecord, storage: IncidentalStorageRecord | None
    ) -> StoredResponse:
        """Build the response to store on a promoted incidental's request.

        Copies the already-compressed response body verbatim (no
        re-compression). Status and url come from the capture row, which
        records what *this* fetch saw; the response headers come from the
        shared storage row, so they are the first capture's (see
        ``replace_incidental_requests``). A capture with a status but no body
        (an excluded resource type) is stored with no body.

        Args:
            rec: The capture row to promote; it got a response.
            storage: Its storage row, if it has one.

        Returns:
            The response to store on the promoted request.
        """
        # Captures without a response never match (see the caller).
        assert rec.status_code is not None
        if storage is None:
            return StoredResponse(
                response_status_code=rec.status_code,
                response_url=rec.url,
            )
        return StoredResponse(
            response_status_code=rec.status_code,
            response_headers_json=storage.response_headers_json,
            response_url=rec.url,
            **storage.model_dump(include=set(CompressedPayload.model_fields)),
        )

    @staticmethod
    def _keys_from_captures(item: Request, spec: Singular | Multiple) -> bool:
        """Whether the children's dedup keys are built from the captures.

        Args:
            item: The yielded request carrying the incidental spec.
            spec: The cardinality that spec declared.

        Returns:
            ``True`` with no explicit key, or a string key fanned out
            over a ``Multiple``; the request bodies are then read.
        """
        key = item.deduplication_key
        return key is None or (
            isinstance(spec, Multiple) and isinstance(key, str)
        )

    @classmethod
    def _incidental_dedup_keys(
        cls,
        item: Request,
        spec: Singular | Multiple,
        matched: list[IncidentalRequestRecord],
        bodies: dict[int, bytes],
        parent_request_id: int,
    ) -> list[str | SkipDeduplicationCheck | None]:
        """The dedup key for each promoted incidental's child request.

        The yielded request is a placeholder (often the parent page's URL),
        so with no explicit key the child is keyed by the parent navigation,
        the step it continues to, and what the browser sent from it (see
        :func:`_capture_identities`). Hashing the placeholder would collapse
        captures from different parents that share a page URL (a single-page
        search, a form-driven listing); so would the capture alone, when the
        parents' state rides in a cookie or a header and their sub-requests
        are byte-identical. The step keeps two yields that hand one capture
        to different steps apart; the same capture handed to the same step
        twice (overlapping specs) is one child.

        An explicit key is the scraper's chosen identity and kept; for
        ``Multiple`` each match's capture identity is appended, so the
        matches stay distinct while one capture keys the same whatever
        order the browser captured it in.

        Args:
            item: The yielded request carrying the incidental spec.
            spec: The cardinality that spec declared.
            matched: The captures being promoted, in capture order.
            bodies: Request bodies by capture id; read only when
                :meth:`_keys_from_captures` holds.
            parent_request_id: The parent navigation's row id.

        Returns:
            Each child's ``deduplication_key``, in ``matched`` order.
        """
        key = item.deduplication_key
        if not cls._keys_from_captures(item, spec):
            return [key] * len(matched)
        identities = _capture_identities(matched, bodies)
        if key is not None:
            return [f"{key}:{identity}" for identity in identities]
        step = _step_name(item.step)
        return [
            hashlib.sha256(
                _json.dumps([parent_request_id, step, identity]).encode()
            ).hexdigest()
            for identity in identities
        ]

    async def _process_generator_with_autowait(
        self,
        step: Callable[
            [Response],
            Generator[ScraperYield[Any], bool | None, None],
        ],
        response: Response,
        parent_request: Request,
        request_id: int,
        step_name: str,
        auto_await_timeout: int,
        *,
        page: AutowaitPage,
        staged: StagedWrites,
        captures: _ParentCaptures,
        speculation_outcome: SpeculationOutcome | None,
    ) -> None:
        """Run the step, waiting on the live page for missing selectors.

        On an HTMLStructuralAssumptionException it waits for the page to
        change, re-snapshots the DOM, and retries until success or the
        ``auto_await_timeout`` (ms) elapses.

        When the failing selector can be targeted on the page it waits for
        that selector. ``wait_for_selector`` returns on the first match, so
        waiting can only cure too few matches: the wait targets the
        ``expected_min``-th match, too many matches raise at once, and so
        does a wait that leaves the DOM unchanged (the step would fail the
        same way on it again). A wait after a retry first pauses
        :data:`AUTOWAIT_RECHECK_INTERVAL`: the page already matched that
        selector, so the wait itself would return at once.

        A query scoped under a parent that matched several elements can't
        be targeted: the composed selector searches the whole page, where
        the other parents' matches already satisfy it. Nor can one whose
        chain doesn't compose (mixed grammars, an XPath union), nor one
        Playwright can't wait on (a ``text()`` or ``@href`` leaf), nor a
        failure the observer didn't record (no ``page`` injection, or a
        hand-raised exception), whose selector may have run scoped. Those
        cases poll
        the page every :data:`AUTOWAIT_POLL_INTERVAL` seconds for a changed
        DOM instead. The first such poll looks at once: the step has only
        seen the transport's bytes, and the browser's DOM almost always
        differs from them, so waiting a tick first would only add latency.

        A Playwright error from the page (it is navigating, or closed) ends
        the wait as a timeout would: the step's own error is raised.

        Only the last snapshot is stored, once the loop ends either way:
        the response row keeps what the step last ran on. A failure to store
        it while the step's own error is propagating is logged, so the
        step's error is the one raised; a cancelled loop stores nothing.

        Args:
            step: The step generator to drive.
            response: The response the step runs on.
            parent_request: The request that produced it.
            request_id: That request's row id.
            step_name: The step's name, stored with each re-snapshot.
            auto_await_timeout: Milliseconds to keep retrying for.
            page: The live page to wait on.
            staged: The flush transaction's staged writes.
            captures: The navigation's incidental captures.
            speculation_outcome: The probe outcome stored with the
                response, kept on each re-snapshot.

        Raises:
            HTMLStructuralAssumptionException: If the selector never
                resolves before the timeout, or the DOM stops changing.
        """
        start_time = time.monotonic()
        timeout_seconds = auto_await_timeout / 1000.0
        deadline = start_time + timeout_seconds
        original_response = response

        # Keep only the snapshot the step last ran on, stored once whether
        # that run succeeded or the loop gave up. Pass the probe's outcome
        # again: the store overwrites the whole response, outcome included.
        async def store_last_snapshot() -> None:
            if response is not original_response:
                await self.storage.store_response(
                    request_id, response, step_name, speculation_outcome
                )

        try:
            while True:
                try:
                    # Reset the staged buffer so a failed prior attempt's
                    # partial yields (and its deferred on_data/on_invalid_data
                    # callbacks) are discarded before the retry.
                    staged.reset()
                    gen = step(response)
                    await self._process_generator_with_storage(
                        gen,
                        response,
                        parent_request,
                        request_id,
                        staged,
                        captures,
                    )
                    break  # success

                except HTMLStructuralAssumptionException as e:
                    elapsed = time.monotonic() - start_time
                    if elapsed >= timeout_seconds:
                        logger.warning(
                            f"Autowait timeout exhausted "
                            f"({auto_await_timeout}ms) for request "
                            f"{request_id}"
                        )
                        raise

                    if (
                        e.expected_max is not None
                        and e.actual_count > e.expected_max
                    ):
                        logger.debug(
                            f"Selector over-matches ({e.actual_count} > "
                            f"{e.expected_max}), skipping autowait: "
                            f"{e.selector}"
                        )
                        raise

                    # The step wrapper records the observer on the
                    # per-execution Response, never on the shared
                    # StepMetadata.
                    wait_selector = self._page_wide_selector(
                        response.observer, e
                    )
                    if wait_selector is None:
                        html_content = await self._poll_for_changed_dom(
                            page,
                            response.content,
                            deadline,
                            look_now=response is original_response,
                        )
                        if html_content is None:
                            logger.warning(
                                f"Autowait: page unchanged within timeout "
                                f"for {e.selector}"
                            )
                            raise e from None
                    else:
                        if response is not original_response:
                            await asyncio.sleep(
                                min(
                                    AUTOWAIT_RECHECK_INTERVAL,
                                    deadline - time.monotonic(),
                                )
                            )
                        html_content = await self._wait_for_failing_selector(
                            e,
                            wait_selector,
                            page,
                            deadline - time.monotonic(),
                        )
                        if html_content is None:
                            raise e from None
                        if html_content.encode("utf-8") == response.content:
                            logger.warning(
                                f"Autowait: selector {e.selector} matched "
                                f"but the DOM is unchanged"
                            )
                            raise e from None

                    response = Response(
                        status_code=response.status_code,
                        url=page.url,
                        content=html_content.encode("utf-8"),
                        text=html_content,
                        headers=_snapshot_headers(original_response.headers),
                        request=response.request,
                    )
                    logger.info(
                        "Autowait: retrying step function with fresh DOM "
                        "snapshot"
                    )
        except Exception:
            # CancelledError is not an Exception: a cancelled loop skips the
            # store instead of writing after it was told to stop.
            try:
                await store_last_snapshot()
            except Exception:
                logger.exception(
                    f"Autowait: failed to store the last snapshot for "
                    f"request {request_id}"
                )
            raise
        await store_last_snapshot()

    def _page_wide_selector(
        self,
        observer: SelectorObserver | None,
        exc: HTMLStructuralAssumptionException,
    ) -> Selector | None:
        """The page-wide selector that finds the element ``exc`` failed on.

        Args:
            observer: The step's selector observer, if any.
            exc: The structural exception the step raised.

        Returns:
            The selector to wait on, or ``None`` when there is none to wait
            on: the exception carries no recorded query, the failing query
            ran under a parent that matched several elements, its chain
            doesn't compose (mixed grammars, or an XPath step with no single
            path), or the composed selector isn't one Playwright can wait on
            (``text()``, ``@href``, a cssselect-only pseudo-class). The
            exception's own selector may have run scoped, so it is never
            waited on as is.
        """
        # PageElement attaches the query whose count raised; a hand-raised
        # exception carries none.
        query = exc.query
        if observer is None or query is None or _has_ambiguous_parent(query):
            return None
        composed = observer.compose_absolute_selector(query)
        if composed is None:
            return None
        # A composed chain is all one grammar, the leaf's. Waitability is
        # checked on what is actually waited on: a parent in the chain can
        # be the part Playwright cannot evaluate.
        target = Selector.of(composed, query.selector_type)
        if not target.can_playwright_wait():
            logger.debug(
                f"Selector not Playwright-compatible, polling instead: "
                f"{composed}"
            )
            return None
        return target

    @staticmethod
    async def _wait_for_failing_selector(
        exc: HTMLStructuralAssumptionException,
        target: Selector,
        page: AutowaitPage,
        remaining_seconds: float,
    ) -> str | None:
        """Wait on the page for the selector the step failed on.

        Args:
            exc: The structural exception the step raised.
            target: Its page-wide selector, one Playwright can wait on.
            page: The live page to wait on.
            remaining_seconds: What is left of the autowait budget.

        Returns:
            The page's DOM once the selector matches, or ``None`` when it
            didn't match in time.
        """
        # Some matches exist already, so a plain wait would return at once;
        # wait for the last one still missing.
        if exc.actual_count > 0:
            target = target.nth(exc.expected_min)
        wait_selector = target.for_playwright()

        logger.info(f"Autowait: waiting for selector {wait_selector}")

        # Floor at 1ms: Playwright treats timeout=0 as "wait forever", so a
        # sub-millisecond remaining budget must not round down to 0.
        remaining_timeout = max(1, int(remaining_seconds * 1000))
        try:
            await page.wait_for_selector(
                wait_selector, timeout=remaining_timeout
            )
        except PlaywrightTimeoutError:
            logger.warning(
                f"Autowait failed: selector {wait_selector} "
                f"not found within timeout"
            )
            return None
        except PlaywrightError as e:
            logger.warning(
                f"Autowait failed: the page could not wait for "
                f"{wait_selector}: {e}"
            )
            return None
        return await StepExecutor._live_dom(page)

    @staticmethod
    async def _live_dom(page: AutowaitPage) -> str | None:
        """Snapshot the page's DOM, if the page can be read.

        Args:
            page: The live page to snapshot.

        Returns:
            The serialized DOM, or ``None`` when Playwright can't read it
            (the page is navigating, or closed).
        """
        try:
            return await page.content()
        except PlaywrightError as e:
            logger.warning(f"Autowait: could not snapshot the page: {e}")
            return None

    @staticmethod
    async def _poll_for_changed_dom(
        page: AutowaitPage,
        content: bytes,
        deadline: float,
        *,
        look_now: bool = False,
    ) -> str | None:
        """Re-snapshot the page until its DOM differs from ``content``.

        A snapshot that fails (the page is mid-navigation) is skipped and
        the next tick tries again.

        Args:
            page: The live page to snapshot.
            content: The DOM the step last failed on.
            deadline: The ``time.monotonic()`` the autowait budget ends at.
            look_now: Take the first snapshot without waiting a tick.

        Returns:
            The changed DOM, or ``None`` if it hadn't changed by
            ``deadline``.
        """
        wait = not look_now
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            if wait:
                await asyncio.sleep(min(AUTOWAIT_POLL_INTERVAL, remaining))
            wait = True
            html_content = await StepExecutor._live_dom(page)
            if (
                html_content is not None
                and html_content.encode("utf-8") != content
            ):
                return html_content
