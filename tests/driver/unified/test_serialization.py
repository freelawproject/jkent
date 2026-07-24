"""Request (de)serialization round-trips for the unified driver's queue.

A duplicate of the persistent driver's serialization battery, retargeted at the
unified driver's own ``RequestQueue`` (``database_engine.queue.RequestQueueDB``).
Each test serializes a ``Request`` with ``serialize_request``, writes it to the
``requests`` table, reads it back, and reconstructs it with
``_deserialize_request`` — isolating the DB serialization layer from the
in-memory ``resolve_from`` path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import sqlalchemy as sa

from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import (
    FilesType,
    HttpMethod,
    HTTPRequestParams,
    Request,
)
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver.persistence import RequestQueue
from jkent.driver.unified_driver.run import ScrapeRun

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

# The 29 columns ``_deserialize_request`` unpacks, in order.
_SELECT = sa.text(
    "SELECT id, request_type, method, url, headers_json, cookies_json, body, "
    "continuation, current_location, accumulated_data_json, permanent_json, "
    "expected_type, priority, is_speculative, speculation_id, verify, via_json, "
    "bypass_rate_limit, deduplication_key, timeout_json, json_data, files_json, "
    "auth_json, allow_redirects, proxies_json, stream, cert_json, "
    "archive_hash_header, reseedable "
    "FROM requests WHERE id = 1"
)


@pytest.fixture
async def sql_manager(tmp_path: Path) -> AsyncIterator[SQLManager]:
    """A fully-migrated SQLManager backed by a temp-file SQLite DB."""
    async with SQLManager.open(tmp_path / "test.db") as manager:
        yield manager


async def _roundtrip(
    sql_manager: SQLManager, original: Request
) -> tuple[dict[str, Any], Request]:
    """Serialize, persist, re-read, and deserialize ``original``.

    Returns the serialized column dict and the deserialized request.
    """
    queue = RequestQueue.__new__(RequestQueue)  # (de)serialize need no db
    serialized = queue.serialize_request(original)

    cols: dict[str, Any] = {
        **serialized,
        "priority": original.effective_priority,
        "status": "pending",
        "queue_counter": 1,
    }
    names = ", ".join(cols)
    placeholders = ", ".join(f":{k}" for k in cols)
    async with sql_manager._session_factory() as session:
        await session.execute(
            sa.text(f"INSERT INTO requests ({names}) VALUES ({placeholders})"),
            cols,
        )
        await session.commit()

    async with sql_manager._session_factory() as session:
        row = (await session.execute(_SELECT)).first()
    assert row is not None

    deserialized = queue._deserialize_request(row)  # type: ignore[arg-type]
    assert isinstance(deserialized, Request)
    return serialized, deserialized


async def test_navigating_request_round_trip(sql_manager: SQLManager) -> None:
    """A navigating Request is correctly serialized and deserialized."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/page",
            headers={"User-Agent": "Test", "Accept": "text/html"},
            cookies={"session": "abc123"},
        ),
        continuation="parse_page",
        current_location="https://example.com",
        accumulated_data={"key": "value", "count": 42},
        permanent={"headers": {"Authorization": "Bearer token"}},
        priority=5,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["request_type"] == "navigating"
    assert serialized["expected_type"] is None
    assert not deserialized.nonnavigating
    assert not deserialized.archive
    assert deserialized.request.method == original.request.method
    assert deserialized.request.url == original.request.url
    assert deserialized.request.headers == original.request.headers
    assert deserialized.request.cookies == original.request.cookies
    assert deserialized.continuation == original.continuation
    assert deserialized.current_location == original.current_location
    assert deserialized.accumulated_data == original.accumulated_data
    assert deserialized.permanent == original.permanent
    assert deserialized.priority == original.priority


async def test_non_navigating_request_round_trip(
    sql_manager: SQLManager,
) -> None:
    """A non-navigating Request with binary body round-trips correctly."""
    # Non-JSON bytes test raw binary preservation; JSON-like bytes get
    # decoded to dicts by design (for form data).
    original = Request(
        nonnavigating=True,
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://api.example.com/data",
            headers={"Content-Type": "application/octet-stream"},
            data=b"\x00\x01\x02\x03binary data\xff\xfe",
        ),
        continuation="process_api_response",
        current_location="https://example.com/main",
        accumulated_data={"items": [1, 2, 3]},
        permanent={"cookies": {"auth": "secret"}},
        priority=3,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["request_type"] == "non_navigating"
    assert serialized["expected_type"] is None
    assert deserialized.nonnavigating
    assert deserialized.request.method == original.request.method
    assert deserialized.request.url == original.request.url
    assert deserialized.request.headers == original.request.headers
    assert deserialized.request.data == original.request.data
    assert deserialized.continuation == original.continuation
    assert deserialized.current_location == original.current_location
    assert deserialized.accumulated_data == original.accumulated_data
    assert deserialized.permanent == original.permanent
    assert deserialized.priority == original.priority


async def test_archive_request_round_trip(sql_manager: SQLManager) -> None:
    """An archive Request is correctly serialized and deserialized."""
    original = Request(
        archive=True,
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/files/document.pdf",
            headers={"Accept": "application/pdf"},
        ),
        continuation="handle_download",
        current_location="https://example.com/documents",
        expected_type="pdf",
        accumulated_data={"document_id": "12345"},
        permanent={},
        priority=1,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["request_type"] == "archive"
    assert serialized["expected_type"] == "pdf"
    assert deserialized.archive
    assert deserialized.request.method == original.request.method
    assert deserialized.request.url == original.request.url
    assert deserialized.request.headers == original.request.headers
    assert deserialized.continuation == original.continuation
    assert deserialized.current_location == original.current_location
    assert deserialized.expected_type == original.expected_type
    assert deserialized.accumulated_data == original.accumulated_data
    assert deserialized.permanent == original.permanent
    assert deserialized.priority == original.priority


async def test_archive_request_without_expected_type(
    sql_manager: SQLManager,
) -> None:
    """Archive Request round-trip when expected_type is None."""
    original = Request(
        archive=True,
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/files/unknown",
        ),
        continuation="handle_download",
        current_location="https://example.com",
        expected_type=None,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["request_type"] == "archive"
    assert serialized["expected_type"] is None
    assert deserialized.archive
    assert deserialized.expected_type is None


async def test_request_with_binary_body(sql_manager: SQLManager) -> None:
    """Request round-trip with binary body data."""
    binary_body = b"\x00\x01\x02\xff\xfe\xfd"
    original = Request(
        nonnavigating=True,
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/upload",
            data=binary_body,
        ),
        continuation="handle_upload",
        current_location="",
    )

    _serialized, deserialized = await _roundtrip(sql_manager, original)

    assert deserialized.request.data == binary_body


async def test_empty_bytes_body_round_trip(sql_manager: SQLManager) -> None:
    """An explicitly empty bytes body ``b""`` survives as ``b""`` (not None).

    Distinct from a request with no body at all (which stores/returns None).
    """
    original = Request(
        nonnavigating=True,
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/empty",
            data=b"",
        ),
        continuation="handle",
        current_location="",
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["body"] == b""
    assert deserialized.request.data == b""


async def test_binary_files_round_trip(sql_manager: SQLManager) -> None:
    """File values carrying binary content round-trip via base64 -> bytes."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/upload",
            # Binary file content is intentionally out-of-type here: the test
            # exercises the serializer's bytes -> base64 path for file parts.
            files=cast(
                FilesType,
                {
                    "text_part": "plain string content",
                    "blob": b"\x00\x01\x02\xff\xfe",
                },
            ),
        ),
        continuation="handle_upload",
        current_location="",
    )

    _serialized, deserialized = await _roundtrip(sql_manager, original)

    assert deserialized.request.files is not None
    assert deserialized.request.files["text_part"] == "plain string content"
    assert deserialized.request.files["blob"] == b"\x00\x01\x02\xff\xfe"


async def test_verify_false_string_is_rejected(
    sql_manager: SQLManager,
) -> None:
    """A CA-bundle path literally equal to 'false' is rejected, not silently
    collapsed to verify=False (which would disable TLS verification)."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com",
            verify="false",
        ),
        continuation="parse",
        current_location="",
    )

    queue = RequestQueue.__new__(RequestQueue)
    with pytest.raises(ScraperConfigError):
        queue.serialize_request(original)


async def test_request_with_empty_optional_fields(
    sql_manager: SQLManager,
) -> None:
    """Request round-trip with minimal fields (empty optionals)."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com",
        ),
        continuation="parse",
        current_location="",
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["headers_json"] is None
    assert serialized["cookies_json"] is None
    assert serialized["body"] is None
    assert serialized["accumulated_data_json"] is None
    assert serialized["permanent_json"] is None
    assert deserialized.request.headers is None
    assert deserialized.request.cookies is None
    assert deserialized.request.data is None
    assert deserialized.accumulated_data == {}
    assert deserialized.permanent == {}


async def test_bypass_rate_limit_round_trip(sql_manager: SQLManager) -> None:
    """bypass_rate_limit=True round-trips through the DB correctly."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/urgent",
        ),
        continuation="handle_urgent",
        current_location="",
        bypass_rate_limit=True,
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["bypass_rate_limit"] is True
    assert deserialized.bypass_rate_limit is True


async def test_bypass_rate_limit_default_false(
    sql_manager: SQLManager,
) -> None:
    """bypass_rate_limit defaults to False when not set."""
    original = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url="https://example.com/normal",
        ),
        continuation="parse",
        current_location="",
    )

    serialized, deserialized = await _roundtrip(sql_manager, original)

    assert serialized["bypass_rate_limit"] is False
    assert deserialized.bypass_rate_limit is False


async def test_all_http_request_params_fields_round_trip(
    sql_manager: SQLManager,
) -> None:
    """Every non-default HTTPRequestParams field must survive a DB round-trip.

    Regression test: the Nevada Supreme Court scraper set ``timeout=360.0`` on
    archive HTTPRequestParams, but the queue silently dropped it (along with
    ``json``, ``files``, ``auth``, ``allow_redirects``, ``proxies``, ``stream``,
    ``cert``, ``archive_hash_header``) on serialize -> insert -> select ->
    deserialize. The DB layer is a separate place these fields can be lost.
    """
    original = Request(
        archive=True,
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://example.com/files/document.pdf",
            # `params` is encoded into the URL on serialize, so it is not
            # asserted as a separate field below.
            data={"form_field": "value"},
            json={"json_field": "value"},
            headers={"Accept": "application/pdf"},
            cookies={"session": "abc123"},
            files={"upload": "file.txt"},  # type: ignore[dict-item]
            auth=("user", "pass"),
            timeout=360.0,
            allow_redirects=False,
            proxies={"http": "http://proxy.example:3128"},
            verify=False,
            stream=True,
            cert="/path/to/cert.pem",
        ),
        continuation="handle_download",
        current_location="https://example.com/documents",
        expected_type="pdf",
        archive_hash_header="X-Content-SHA256",
        priority=1,
    )

    _serialized, deserialized = await _roundtrip(sql_manager, original)

    # URL/method are their own columns and known to round-trip; assert them
    # so a future schema change can't pass this test without exercising them.
    assert deserialized.request.url == original.request.url
    assert deserialized.request.method == original.request.method

    assert deserialized.request.data == original.request.data
    assert deserialized.request.json == original.request.json
    assert deserialized.request.headers == original.request.headers
    assert deserialized.request.cookies == original.request.cookies
    assert deserialized.request.files == original.request.files
    assert deserialized.request.auth == original.request.auth
    assert deserialized.request.timeout == original.request.timeout
    assert (
        deserialized.request.allow_redirects
        == original.request.allow_redirects
    )
    assert deserialized.request.proxies == original.request.proxies
    assert deserialized.request.verify == original.request.verify
    assert deserialized.request.stream == original.request.stream
    assert deserialized.request.cert == original.request.cert
    assert deserialized.archive_hash_header == original.archive_hash_header


async def test_entry_request_preserves_json_and_extended_fields(
    sql_manager: SQLManager,
) -> None:
    """Entry-point seeding must keep ``json`` (and the other extended fields).

    Regression: entry requests were seeded through a bespoke insert that
    accepted only a subset of columns — so a POST whose body lived in ``json``
    (e.g. the Arkansas ``caseinfo.arcourts.gov`` search) reached the DB with an
    empty ``json_data`` and was replayed/executed without its body. This drives
    the real ``ScrapeRun._enqueue_entry_request`` wiring through to the DB.
    """
    entry_request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.POST,
            url="https://caseinfo.arcourts.gov/opad/api/cases/search",
            headers={"Content-Type": "application/json"},
            json={"courtName": "STATE OF ARKANSAS SUPREME COURT", "page": 1},
            timeout=360.0,
            allow_redirects=False,
            stream=True,
        ),
        continuation="parse_search_results",
        deduplication_key="ar.supreme.page-1",
    )

    run = ScrapeRun.__new__(ScrapeRun)
    run._db = sql_manager
    queue = RequestQueue.__new__(RequestQueue)  # serialize needs no db
    await run._enqueue_entry_request(queue, entry_request)

    async with sql_manager._session_factory() as session:
        row = (await session.execute(_SELECT)).first()
    assert row is not None

    deserialized = RequestQueue._deserialize_request(queue, row)  # type: ignore[arg-type]
    assert isinstance(deserialized, Request)
    assert deserialized.request.json == entry_request.request.json
    assert deserialized.request.timeout == entry_request.request.timeout
    assert deserialized.request.allow_redirects is False
    assert deserialized.request.stream is True
