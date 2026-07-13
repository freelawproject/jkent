"""Tests for incidental pre-resolution (``Request.incidental``).

A step yields a ``Request(incidental=Singular|Multiple(...))``; instead of a
network fetch, the ``ContinuationExecutor`` matches the spec against the parent
navigation's captured incidental sub-requests and enqueues pre-resolved
child requests whose response is stored in the same flush transaction. The
worker later runs the child's continuation without touching the transport.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jkent.common.exceptions import IncidentalRequestAssumptionException
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
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import (
    RequestQueue,
    ResponseStorage,
)
from tests.db_queries import fetch_requests

# Reuse the continuation test's fakes/helpers.
from tests.driver.unified.test_continuation import (
    FakeScraper,
    _make_executor,
    _seed_request,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


async def _seed_incidental(
    sql_manager: SQLManager,
    parent_id: int,
    *,
    url: str,
    method: str = "GET",
    request_headers: dict[str, str] | None = None,
    request_body: bytes | None = None,
    response_body: bytes = b'{"ok": true}',
    status_code: int = 200,
) -> None:
    """Insert one captured incidental sub-request under ``parent_id``."""
    compressed = compress(response_body)
    await sql_manager.insert_incidental_request(
        parent_request_id=parent_id,
        resource_type="xhr",
        method=method,
        url=url,
        headers_json=json.dumps(request_headers or {}),
        body=request_body,
        status_code=status_code,
        response_headers_json=json.dumps({"content-type": "application/json"}),
        content_compressed=compressed,
        content_size_original=len(response_body),
        content_size_compressed=len(compressed),
    )


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
        continuation="parse",
        nonnavigating=True,
        incidental=spec,
        deduplication_key=dedup,
    )


async def test_singular_preresolves_to_stored_response(
    sql_manager: SQLManager,
) -> None:
    """A Singular match enqueues one pre-resolved request carrying the body."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/c/courts/getcasedetaildata/1",
        response_body=b'{"parties": ["A", "B"]}',
    )
    # A couple of noise incidentals that must not match.
    await _seed_incidental(
        sql_manager, parent_id, url="https://example.com/app.js"
    )
    await _seed_incidental(
        sql_manager, parent_id, url="https://example.com/style.css"
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
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


async def test_singular_zero_matches_raises(sql_manager: SQLManager) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed_incidental(
        sql_manager, parent_id, url="https://example.com/app.js"
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
        yield _promote(Singular(url="*getcasedetaildata*"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(IncidentalRequestAssumptionException):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_singular_multiple_matches_raises(
    sql_manager: SQLManager,
) -> None:
    parent_id, parent = await _seed_request(sql_manager)
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/api/getcasedetaildata/1",
    )
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/api/getcasedetaildata/2",
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
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
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/graphql",
        method="POST",
        request_body=b'{"operationName":"GetA"}',
        response_body=b'{"a": 1}',
    )
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/graphql",
        method="POST",
        request_body=b'{"operationName":"GetB"}',
        response_body=b'{"b": 2}',
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
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
    await _seed_incidental(
        sql_manager, parent_id, url="https://example.com/app.js"
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
        yield _promote(Multiple(url="*/graphql"))

    executor = _make_executor(sql_manager, FakeScraper(yields))
    with pytest.raises(IncidentalRequestAssumptionException):
        await executor.complete_request(
            parent_id, _response_for(parent), parent, "parse"
        )


async def test_body_contains_disambiguates(sql_manager: SQLManager) -> None:
    """body_contains selects the right same-URL request by JSON deep-contains."""
    parent_id, parent = await _seed_request(sql_manager)
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/graphql",
        method="POST",
        request_body=b'{"operationName":"GetA","variables":{"t":"tok1"}}',
        response_body=b'{"pick": "A"}',
    )
    await _seed_incidental(
        sql_manager,
        parent_id,
        url="https://example.com/graphql",
        method="POST",
        request_body=b'{"operationName":"GetB","variables":{"t":"tok2"}}',
        response_body=b'{"pick": "B"}',
    )

    def yields(_r: Response) -> Generator[ScraperYield, None, None]:
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
