"""Tests for incidental pre-resolution (``Request.incidental``).

A step yields a ``Request(incidental=Singular|Multiple(...))``; instead of a
network fetch, the ``StepExecutor`` matches the spec against the parent
navigation's captured incidental sub-requests and enqueues pre-resolved
child requests whose response is stored in the same flush transaction. The
worker later runs the child's step without touching the transport.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

import pytest

from jkent.common.exceptions import (
    IncidentalRequestAssumptionException,
    ScraperConfigError,
)
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Multiple,
    Request,
    Response,
    ScraperYield,
    Singular,
)
from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.sql_manager import (
    IncidentalCapture,
    SQLManager,
)
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from tests.db_queries import fetch_requests

# Reuse the step test's fakes/helpers.
from tests.driver.unified.test_steps import (
    FakeScraper,
    _make_executor,
    _seed_request,
)

if TYPE_CHECKING:
    pass


def _incidental(
    *,
    url: str,
    method: str = "GET",
    request_headers: dict[str, str] | None = None,
    request_body: bytes | None = None,
    response_body: bytes | None = b'{"ok": true}',
    status_code: int | None = 200,
    failure_reason: str | None = None,
) -> IncidentalCapture:
    """One captured incidental sub-request."""
    capture = IncidentalCapture(
        resource_type="xhr",
        method=method,
        url=url,
        headers_json=json.dumps(request_headers or {}),
        body=request_body,
        status_code=status_code,
        response_headers_json=json.dumps({"content-type": "application/json"}),
        failure_reason=failure_reason,
    )
    if response_body is not None:
        capture.set_body(response_body, compress(response_body))
    return capture


async def _seed(
    sql_manager: SQLManager, parent_id: int, *captures: IncidentalCapture
) -> None:
    """Store ``captures`` as the parent navigation's one batch."""
    await sql_manager.replace_incidental_requests(parent_id, list(captures))


def _response_for(request: Request) -> Response:
    return Response(
        request=request,
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="https://example.com/page",
    )


def _promote(
    spec: Singular | Multiple, dedup: str | None = "promote"
) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/detail"
        ),
        step="parse",
        nonnavigating=True,
        incidental=spec,
        deduplication_key=dedup,
    )


async def test_singular_preresolves_to_stored_response(
    sql_manager: SQLManager,
) -> None:
    """A Singular match enqueues one pre-resolved request carrying the body."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/c/courts/getcasedetaildata/1",
            response_body=b'{"parties": ["A", "B"]}',
        ),
        # A couple of noise incidentals that must not match.
        _incidental(url="https://example.com/app.js"),
        _incidental(url="https://example.com/style.css"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    queue = RequestQueue(sql_manager)
    dequeued = await queue.get_next_request()
    assert dequeued is not None
    child_id, child, returned_parent, preresolved = dequeued
    assert preresolved is True
    assert returned_parent == parent_id

    storage = ResponseStorage(sql_manager)
    response = await storage.load_preresolved_response(child_id, child)
    assert response is not None
    assert response.status_code == 200
    assert json.loads(response.text) == {"parties": ["A", "B"]}
    # The promoted response's URL is the captured incidental's URL, not the
    # request's representative URL.
    assert response.url.endswith("/getcasedetaildata/1")


async def test_singular_after_a_retried_navigation_finds_one_match(
    sql_manager: SQLManager,
) -> None:
    """A retried navigation's captures replace the first attempt's.

    Both attempts saw the detail call; a ``Singular`` must see it once.
    """
    parent_id, parent = await _seed_request(sql_manager)
    for attempt in (b'{"attempt": 1}', b'{"attempt": 2}'):
        await _seed(
            sql_manager,
            parent_id,
            _incidental(
                url="https://example.com/c/getcasedetaildata/1",
                response_body=attempt,
            ),
        )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    dequeued = await RequestQueue(sql_manager).get_next_request()
    assert dequeued is not None
    child_id, child, _, _ = dequeued
    response = await ResponseStorage(sql_manager).load_preresolved_response(
        child_id, child
    )
    assert response is not None
    assert json.loads(response.text) == {"attempt": 2}


async def test_singular_zero_matches_raises(sql_manager: SQLManager) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/app.js"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(IncidentalRequestAssumptionException):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_incidental_on_a_non_capturing_transport_is_a_config_error(
    sql_manager: SQLManager,
) -> None:
    """A transport that records no incidentals gets a wiring error, not a
    cardinality one.

    ``Transport.captures_incidentals`` is False for HTTP and replay, so the
    match set is empty by construction and the cardinality check would blame
    the site ("expected exactly 1 match, found 0") for what is a missing
    browser requirement. The executor must say so instead.
    """
    parent_id, parent = await _seed_request(sql_manager)

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(
        sql_manager, FakeScraper(yields), captures_incidentals=False
    )

    with pytest.raises(ScraperConfigError, match="does not capture"):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_singular_multiple_matches_raises(
    sql_manager: SQLManager,
) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/api/getcasedetaildata/1"),
        _incidental(url="https://example.com/api/getcasedetaildata/2"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(IncidentalRequestAssumptionException):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_multiple_fans_out_one_per_match(
    sql_manager: SQLManager,
) -> None:
    """Multiple enqueues one pre-resolved request per match, distinct keys."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetA"}',
            response_body=b'{"a": 1}',
        ),
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetB"}',
            response_body=b'{"b": 2}',
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/graphql", method="POST"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    # Two rows (the shared yield dedup key did not collapse the matches).
    pending = await fetch_requests(sql_manager, status="pending")
    assert len(pending) == 2
    # Both dequeue as pre-resolved, each promoting its own response body.
    queue = RequestQueue(sql_manager)
    storage = ResponseStorage(sql_manager)
    bodies = []
    while (dequeued := await queue.get_next_request()) is not None:
        child_id, child, _parent, preresolved = dequeued
        assert preresolved is True
        resp = await storage.load_preresolved_response(child_id, child)
        assert resp is not None
        bodies.append(json.loads(resp.text))
    assert {tuple(b.items()) for b in bodies} == {
        (("a", 1),),
        (("b", 2),),
    }


async def test_multiple_zero_matches_raises(sql_manager: SQLManager) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/app.js"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/graphql"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(IncidentalRequestAssumptionException):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_body_contains_disambiguates(sql_manager: SQLManager) -> None:
    """body_contains selects the right same-URL request by JSON deep-contains."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetA","variables":{"t":"tok1"}}',
            response_body=b'{"pick": "A"}',
        ),
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetB","variables":{"t":"tok2"}}',
            response_body=b'{"pick": "B"}',
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        # Pin operationName; ignore the volatile token in variables.
        yield _promote(
            Singular(url="*/graphql", body_contains={"operationName": "GetB"})
        )

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    queue = RequestQueue(sql_manager)
    dequeued = await queue.get_next_request()
    assert dequeued is not None
    child_id, child, _parent, preresolved = dequeued
    assert preresolved is True
    storage = ResponseStorage(sql_manager)
    response = await storage.load_preresolved_response(child_id, child)
    assert response is not None
    assert json.loads(response.text) == {"pick": "B"}


async def test_a_capture_that_got_no_response_is_not_promoted(
    sql_manager: SQLManager,
) -> None:
    """A browser-aborted fetch is not a 200 with an empty body."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/c/getcasedetaildata/1",
            response_body=None,
            status_code=None,
            failure_reason="net::ERR_ABORTED",
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(
        IncidentalRequestAssumptionException, match="net::ERR_ABORTED"
    ) as excinfo:
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )
    # Attributed to the navigation that yielded the spec, like any
    # cardinality error, not to the aborted sub-request.
    assert excinfo.value.request_url == "https://example.com/page"
    assert excinfo.value.match_spec == repr(
        Singular(url="*getcasedetaildata*")
    )


async def test_an_aborted_duplicate_does_not_count_toward_singular(
    sql_manager: SQLManager,
) -> None:
    """A superseded fetch the browser aborted leaves one real match."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/api/search?q=a",
            response_body=None,
            status_code=None,
            failure_reason="net::ERR_ABORTED",
        ),
        _incidental(url="https://example.com/api/search?q=ab"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*/api/search*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    dequeued = await RequestQueue(sql_manager).get_next_request()
    assert dequeued is not None
    response = await ResponseStorage(sql_manager).load_preresolved_response(
        dequeued.request_id, dequeued.request
    )
    assert response is not None
    assert response.url == "https://example.com/api/search?q=ab"


async def test_multiple_promotes_the_answered_matches_around_an_aborted_one(
    sql_manager: SQLManager,
) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/api/item/1"),
        _incidental(
            url="https://example.com/api/item/2",
            response_body=None,
            status_code=None,
            failure_reason="net::ERR_ABORTED",
        ),
        _incidental(url="https://example.com/api/item/3"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/api/item/*"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_a_status_of_zero_is_promoted_as_zero(
    sql_manager: SQLManager,
) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/c/getcasedetaildata/1", status_code=0
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    dequeued = await RequestQueue(sql_manager).get_next_request()
    assert dequeued is not None
    response = await ResponseStorage(sql_manager).load_preresolved_response(
        dequeued.request_id, dequeued.request
    )
    assert response is not None
    assert response.status_code == 0


async def _seed_parent(
    sql_manager: SQLManager, url: str
) -> tuple[int, Request]:
    """A dequeued parent navigation at ``url``."""
    queue = RequestQueue(sql_manager)
    await queue.insert_root_request(
        Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url=url),
            step="parse",
        )
    )
    dequeued = await queue.get_next_request()
    assert dequeued is not None
    request_id, restored, _, _ = dequeued
    assert isinstance(restored, Request)
    return request_id, restored


async def test_unkeyed_children_of_different_parents_stay_distinct(
    sql_manager: SQLManager,
) -> None:
    """Without an explicit key, a child is keyed by its capture.

    Both parents yield the same placeholder request, but each captured its
    own API call; hashing the placeholder would drop the second child.
    """

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*/api/search*"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    for page in (1, 2):
        parent_id, parent = await _seed_parent(
            sql_manager, f"https://example.com/search#{page}"
        )
        await _seed(
            sql_manager,
            parent_id,
            _incidental(url=f"https://example.com/api/search?page={page}"),
        )
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_unkeyed_children_differing_only_in_body_stay_distinct(
    sql_manager: SQLManager,
) -> None:
    """The capture key covers the request body (a POSTed search)."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetA"}',
        ),
        _incidental(
            url="https://example.com/graphql",
            method="POST",
            request_body=b'{"operationName":"GetB"}',
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/graphql"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_explicit_singular_key_dedups_across_parents(
    sql_manager: SQLManager,
) -> None:
    """A scraper-chosen key is its identity, so it still dedups."""

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*/api/case*"), dedup="case:A-1")

    executor = _make_executor(sql_manager, FakeScraper(yields))
    for page in (1, 2):
        parent_id, parent = await _seed_parent(
            sql_manager, f"https://example.com/docket/{page}"
        )
        await _seed(
            sql_manager,
            parent_id,
            _incidental(url=f"https://example.com/api/case?v={page}"),
        )
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )

    assert len(await fetch_requests(sql_manager, status="pending")) == 1


async def test_unkeyed_identical_captures_of_different_parents_stay_distinct(
    sql_manager: SQLManager,
) -> None:
    """Parents whose state rides in a cookie send byte-identical requests.

    Each page's ``GET /api/results`` answers a different page, so keying
    the child by the capture alone would drop the second one.
    """

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*/api/results"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    for page in (1, 2):
        parent_id, parent = await _seed_parent(
            sql_manager, f"https://example.com/listing#{page}"
        )
        await _seed(
            sql_manager,
            parent_id,
            _incidental(
                url="https://example.com/api/results",
                response_body=f'{{"page": {page}}}'.encode(),
            ),
        )
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_explicit_multiple_key_ignores_capture_order(
    sql_manager: SQLManager,
) -> None:
    """One capture under one scraper key dedups whatever order it came in."""

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/api/item/*"), dedup="case:A")

    executor = _make_executor(sql_manager, FakeScraper(yields))
    for page, items in ((1, ("x", "y")), (2, ("y", "x"))):
        parent_id, parent = await _seed_parent(
            sql_manager, f"https://example.com/docket/{page}"
        )
        await _seed(
            sql_manager,
            parent_id,
            *(
                _incidental(url=f"https://example.com/api/item/{item}")
                for item in items
            ),
        )
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_captures_are_read_once_per_step(
    sql_manager: SQLManager,
) -> None:
    """Several incidental yields share one read of the parent's captures."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/api/a"),
        _incidental(url="https://example.com/api/b"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Singular(url="*/api/a"), dedup=None)
        yield _promote(Singular(url="*/api/b"), dedup=None)

    reads: list[int] = []
    original = sql_manager.get_incidental_requests

    async def counting(parent_request_id: int) -> Any:
        reads.append(parent_request_id)
        return await original(parent_request_id)

    sql_manager.get_incidental_requests = counting  # type: ignore[method-assign]
    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert reads == [parent_id]
    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_unkeyed_identical_captures_each_get_a_child(
    sql_manager: SQLManager,
) -> None:
    """A page polling one endpoint yields one child per poll.

    The requests are byte-identical, so only their order tells them apart.
    """
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(
            url="https://example.com/api/status",
            response_body=b'{"state": "queued"}',
        ),
        _incidental(
            url="https://example.com/api/status",
            response_body=b'{"state": "done"}',
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/api/status"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert len(await fetch_requests(sql_manager, status="pending")) == 2


async def test_promotion_reads_bodies_and_storage_in_one_query_each(
    sql_manager: SQLManager,
) -> None:
    """Fanning out over many matches doesn't query once per match."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        *(
            _incidental(
                url="https://example.com/graphql",
                method="POST",
                request_body=f'{{"page": {n}}}'.encode(),
                response_body=f'{{"n": {n}}}'.encode(),
            )
            for n in range(5)
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(
            Multiple(url="*/graphql", body_contains="page"), dedup=None
        )

    calls: list[str] = []
    original_bodies = sql_manager.get_incidental_request_bodies
    original_storages = sql_manager.get_incidental_request_storages

    async def counting_bodies(parent_request_id: int) -> Any:
        calls.append("bodies")
        return await original_bodies(parent_request_id)

    async def counting_storages(storage_ids: Any) -> Any:
        calls.append("storages")
        return await original_storages(storage_ids)

    sql_manager.get_incidental_request_bodies = counting_bodies  # type: ignore[method-assign]
    sql_manager.get_incidental_request_storages = counting_storages  # type: ignore[method-assign]
    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert sorted(calls) == ["bodies", "storages"]
    assert len(await fetch_requests(sql_manager, status="pending")) == 5


@pytest.mark.parametrize(
    "failure_reason",
    [
        pytest.param("NS_ERROR_NET_RESET", id="network-failure"),
        pytest.param(None, id="still-in-flight"),
    ],
)
async def test_multiple_skips_a_matched_capture_with_no_response(
    sql_manager: SQLManager, failure_reason: str | None
) -> None:
    """Any capture without a response is skipped, not only a cancelled one.

    A step that needs it should make its parent wait for it instead.
    """
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        _incidental(url="https://example.com/api/results?page=1"),
        _incidental(
            url="https://example.com/api/results?page=2",
            response_body=None,
            status_code=None,
            failure_reason=failure_reason,
        ),
        _incidental(url="https://example.com/api/results?page=3"),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        yield _promote(Multiple(url="*/api/results*"), dedup=None)

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    pending = await fetch_requests(sql_manager, status="pending")
    assert len(pending) == 2


async def test_one_capture_handed_to_two_steps_gets_a_child_each(
    sql_manager: SQLManager,
) -> None:
    """Two yields over the same capture continuing to different steps.

    A key built from the parent and the capture alone gives both children
    one key, and the flush drops the second.
    """
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager, parent_id, _incidental(url="https://example.com/api/case")
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        for step in ("parse_case", "parse_parties"):
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET, url="https://example.com/detail"
                ),
                step=step,
                nonnavigating=True,
                incidental=Singular(url="*/api/case"),
            )

    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    pending = await fetch_requests(sql_manager, status="pending")
    assert sorted(r.step for r in pending) == [
        "parse_case",
        "parse_parties",
    ]


async def test_storage_rows_are_read_once_per_step(
    sql_manager: SQLManager,
) -> None:
    """One incidental yield per listing row shares one storage read.

    That read covers the matched captures only, not the page's scripts.
    """
    parent_id, parent = await _seed_request(sql_manager)
    await _seed(
        sql_manager,
        parent_id,
        *(
            _incidental(
                url=f"https://example.com/api/row/{n}",
                response_body=f'{{"n": {n}}}'.encode(),
            )
            for n in range(5)
        ),
        *(
            _incidental(
                url=f"https://example.com/static/bundle{n}.js",
                response_body=f"var bundle = {n};".encode(),
            )
            for n in range(3)
        ),
    )

    def yields(_r: Response) -> Generator[ScraperYield[Any], None, None]:
        for n in range(5):
            yield _promote(Singular(url=f"*/api/row/{n}"), dedup=None)

    calls: list[Any] = []
    original = sql_manager.get_incidental_request_storages

    async def counting(storage_ids: Any) -> Any:
        calls.append(storage_ids)
        return await original(storage_ids)

    sql_manager.get_incidental_request_storages = counting  # type: ignore[method-assign]
    executor = _make_executor(sql_manager, FakeScraper(yields))
    await executor.complete_request(
        parent_id, _response_for(parent), parent, "parse"
    )

    assert len(calls) == 1
    assert len(calls[0]) == 5
    assert len(await fetch_requests(sql_manager, status="pending")) == 5
