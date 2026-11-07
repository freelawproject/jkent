"""What the queue round-trip rig cannot reach: the SQLite hop.

``test_queue_roundtrip.py`` is the Hypothesis rig over
``serialize_request``/``_deserialize_request`` and subsumes the field-level
laws. Its ``_trip`` is DB-less, though, so two things stay uncovered there and
are covered here: the bind processing of the coded-enum ``rate_limit`` column
(what integer actually lands on disk, and that it survives the CHECK
constraints), and ``RequestQueue.insert_root_request``, the entry-seeding path
that writes a row the rig never builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from pyrate_limiter import Duration, Rate

from jkent.data_types import (
    NO_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    RateLimitTable,
    Request,
)
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Request as RequestRow
from jkent.driver.database_engine.sql_manager import DequeuedRow, SQLManager
from jkent.driver.unified_driver.persistence import RequestQueue

if TYPE_CHECKING:
    pass

# The columns a ``DequeuedRow`` is built from — the same list the real
# dequeue path RETURNs.
#
# Selected off the model rather than as ``sa.text``: ``request_type`` and
# ``method`` are stored as integer codes, so a raw textual SELECT would hand
# ``_deserialize_request`` the codes instead of the members the real dequeue
# path gives it.
_SELECT = sa.select(
    *(getattr(RequestRow, name) for name in DequeuedRow.model_fields)
).where(RequestRow.id == 1)


async def _roundtrip(
    sql_manager: SQLManager,
    original: Request,
    *,
    rate_limits: RateLimitTable | None = None,
) -> tuple[dict[str, Any], Request]:
    """Serialize, persist, re-read, and deserialize ``original``.

    Returns the serialized column dict and the deserialized request.
    ``rate_limits`` is the scraper's lane table; the default is the bare
    two-lane one.
    """
    queue = RequestQueue(sql_manager, rate_limits=rate_limits)
    serialized = queue.serialize_request(original).model_dump()

    cols: dict[str, Any] = {
        **serialized,
        "priority": original.effective_priority,
        "status": RequestStatus.PENDING,
    }
    # Core insert against the model, not ``sa.text``, so the coded-enum
    # columns get their bind processing (a textual insert would try to store
    # the labels and trip the CHECK constraints).
    async with sql_manager.session_factory() as session:
        await session.execute(sa.insert(RequestRow).values(**cols))
        await session.commit()

    async with sql_manager.session_factory() as session:
        row = (await session.execute(_SELECT)).first()
    assert row is not None

    deserialized = queue._deserialize_request(DequeuedRow.model_validate(row))
    assert isinstance(deserialized, Request)
    return serialized, deserialized


async def test_rate_limit_none_lane_round_trips_as_code_1(
    sql_manager: SQLManager,
) -> None:
    """``rate_limit="none"`` stores as 1 — what ``bypass_rate_limit=True`` was."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/urgent",
        ),
        step="handle_urgent",
        current_location="",
        rate_limit=NO_RATE_LIMIT,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["rate_limit"] == 1
    assert deserialized.rate_limit == NO_RATE_LIMIT


async def test_rate_limit_default_lane_stores_0_and_reads_back_unset(
    sql_manager: SQLManager,
) -> None:
    """An unset ``rate_limit`` stores as 0 and comes back as None."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/normal",
        ),
        step="parse",
        current_location="",
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["rate_limit"] == 0
    assert deserialized.rate_limit is None


async def test_rate_limit_named_lane_round_trips_by_declaration_order(
    sql_manager: SQLManager,
) -> None:
    """A scraper's own lanes store as 2 + position and decode by name."""

    class _Laned(BaseScraper[dict[str, Any]]):
        named_rate_limits = {
            "downloads": [Rate(1, Duration.SECOND)],
            "search": [Rate(1, Duration.SECOND)],
        }

    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/q"
        ),
        step="parse",
        rate_limit="search",
    )
    serialized, deserialized = await _roundtrip(
        sql_manager, original, rate_limits=RateLimitTable.for_scraper(_Laned)
    )
    assert serialized["rate_limit"] == 3
    assert deserialized.rate_limit == "search"


async def test_entry_request_preserves_json_and_extended_fields(
    sql_manager: SQLManager,
) -> None:
    """Entry-point seeding must keep ``json`` (and the other extended fields).

    Regression: entry requests were seeded through a bespoke insert that
    accepted only a subset of columns — so a POST whose body lived in ``json``
    (e.g. the Arkansas ``caseinfo.arcourts.gov`` search) reached the DB with an
    empty ``json_data`` and was replayed/executed without its body. This drives
    the real ``RequestQueue.insert_root_request`` wiring through to the DB.
    """
    entry_request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://caseinfo.arcourts.gov/opad/api/cases/search",
            headers={"Content-Type": "application/json"},
            json={"courtName": "STATE OF ARKANSAS SUPREME COURT", "page": 1},
            timeout=360.0,
            verify=False,
        ),
        step="parse_search_results",
        deduplication_key="ar.supreme.page-1",
    )

    queue = RequestQueue(sql_manager)
    await queue.insert_root_request(entry_request)

    async with sql_manager.session_factory() as session:
        row = (await session.execute(_SELECT)).first()
    assert row is not None

    deserialized = queue._deserialize_request(DequeuedRow.model_validate(row))
    assert isinstance(deserialized, Request)
    assert deserialized.request.json == entry_request.request.json
    assert deserialized.request.timeout == entry_request.request.timeout
    assert deserialized.request.verify is False


async def test_bad_via_json_names_the_request_row(
    sql_manager: SQLManager,
) -> None:
    """A ``via_json`` that fails validation says which queue row it came from.

    pydantic's error names the field path but not the row, so a corrupt via
    in a large run db was untraceable. The original stays chained.
    """
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/v"
        ),
        step="parse",
    )
    await _roundtrip(sql_manager, original)
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.update(RequestRow)
            .where(RequestRow.id == 1)
            .values(via_json='{"type": "link", "description": "no selector"}')
        )
        await session.commit()
        row = (await session.execute(_SELECT)).first()
    assert row is not None

    queue = RequestQueue(sql_manager)
    with pytest.raises(ValueError, match=r"request 1\b.*via_json") as info:
        queue._deserialize_request(DequeuedRow.model_validate(row))
    assert isinstance(info.value.__cause__, ValidationError)
