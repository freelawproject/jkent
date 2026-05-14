"""Tests for request type serialization and deserialization round-trips."""

from __future__ import annotations

import sqlalchemy as sa


class TestRequestTypeRoundTrip:
    """Tests for request type serialization and deserialization round-trips."""

    async def test_navigating_request_round_trip(self, initialized_db) -> None:
        """Test that a navigating Request is correctly serialized and deserialized."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        # Create a navigating Request with all fields populated
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

        # Serialize using the driver's method
        # We need a minimal driver instance just for serialization
        class MockScraper:
            pass

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        # Verify request_type is set correctly
        assert serialized["request_type"] == "navigating"
        assert serialized["expected_type"] is None

        # Insert into database
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        # Retrieve and deserialize
        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)

        # Verify it's the correct type
        assert isinstance(deserialized, Request)
        assert not deserialized.nonnavigating
        assert not deserialized.archive

        # Verify all fields match
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
        self, initialized_db
    ) -> None:
        """Test that a non-navigating Request is correctly serialized and deserialized."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        # Create a non-navigating Request with all fields populated
        # Note: Use non-JSON bytes to test raw binary preservation.
        # JSON-like bytes get decoded to dicts by design (for form data).
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

        # Serialize
        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        # Verify request_type is set correctly
        assert serialized["request_type"] == "non_navigating"
        assert serialized["expected_type"] is None

        # Insert into database
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        # Retrieve and deserialize
        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)

        # Verify it's the correct type
        assert isinstance(deserialized, Request)
        assert deserialized.nonnavigating

        # Verify all fields match
        assert deserialized.request.method == original.request.method
        assert deserialized.request.url == original.request.url
        assert deserialized.request.headers == original.request.headers
        assert deserialized.request.data == original.request.data
        assert deserialized.continuation == original.continuation
        assert deserialized.current_location == original.current_location
        assert deserialized.accumulated_data == original.accumulated_data
        assert deserialized.permanent == original.permanent
        assert deserialized.priority == original.priority

    async def test_archive_request_round_trip(self, initialized_db) -> None:
        """Test that an archive Request is correctly serialized and deserialized."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        # Create an archive Request with all fields populated
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
            priority=1,  # Default for archive Request
        )

        # Serialize
        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        # Verify request_type and expected_type are set correctly
        assert serialized["request_type"] == "archive"
        assert serialized["expected_type"] == "pdf"

        # Insert into database
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        # Retrieve and deserialize
        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)

        # Verify it's the correct type
        assert isinstance(deserialized, Request)
        assert deserialized.archive

        # Verify all fields match
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
        self, initialized_db
    ) -> None:
        """Test archive Request round-trip when expected_type is None."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        # Create an archive Request without expected_type
        original = Request(
            archive=True,
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com/files/unknown",
            ),
            continuation="handle_download",
            current_location="https://example.com",
            expected_type=None,  # No type hint
        )

        # Serialize
        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        assert serialized["request_type"] == "archive"
        assert serialized["expected_type"] is None

        # Insert and retrieve
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        deserialized = driver._deserialize_request(row)

        assert isinstance(deserialized, Request)
        assert deserialized.archive
        assert deserialized.expected_type is None

    async def test_request_with_binary_body(self, initialized_db) -> None:
        """Test request round-trip with binary body data."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

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

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        # Insert and retrieve
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        result = driver._deserialize_request(row)
        # Request returns BaseRequest directly
        deserialized = result if not isinstance(result, tuple) else result[0]

        assert deserialized.request.data == binary_body

    async def test_request_with_empty_optional_fields(
        self, initialized_db
    ) -> None:
        """Test request round-trip with minimal fields (empty optionals)."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        # Minimal request with empty optional fields
        original = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com",
            ),
            continuation="parse",
            current_location="",
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        # Verify optional fields are None/empty
        assert serialized["headers_json"] is None
        assert serialized["cookies_json"] is None
        assert serialized["body"] is None
        assert serialized["accumulated_data_json"] is None
        assert serialized["permanent_json"] is None

        # Insert and retrieve
        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        result = driver._deserialize_request(row)
        # Request returns BaseRequest directly
        deserialized = result if not isinstance(result, tuple) else result[0]

        # Verify deserialized correctly with empty defaults
        assert deserialized.request.headers is None
        assert deserialized.request.cookies is None
        assert deserialized.request.data is None
        assert deserialized.accumulated_data == {}
        assert deserialized.permanent == {}

    async def test_bypass_rate_limit_round_trip(self, initialized_db) -> None:
        """Test that bypass_rate_limit=True round-trips through DB correctly."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        original = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com/urgent",
            ),
            continuation="handle_urgent",
            current_location="",
            bypass_rate_limit=True,
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)
        assert serialized["bypass_rate_limit"] is True

        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type, bypass_rate_limit
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type, :bypass_rate_limit
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                    "bypass_rate_limit": serialized["bypass_rate_limit"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)
        assert deserialized.bypass_rate_limit is True

    async def test_all_http_request_params_fields_round_trip(
        self, initialized_db
    ) -> None:
        """Every non-default HTTPRequestParams field must survive DB round-trip.

        Regression test: the Nevada Supreme Court scraper set
        ``timeout=360.0`` on archive HTTPRequestParams, but the persistent
        queue silently dropped it (along with ``json``, ``files``, ``auth``,
        ``allow_redirects``, ``proxies``, ``stream``, ``cert``,
        ``archive_hash_header``) on serialize -> insert -> select ->
        deserialize, causing downloads to hang because httpx fell back to
        the client-level default timeout.

        This is a complement to
        tests/data_types/test_archive_request.py::
        test_resolve_from_preserves_all_http_request_params_fields, which
        only exercises the in-memory ``resolve_from`` path. The DB layer
        in ``_serialize_request`` / ``_deserialize_request`` is a separate
        place the same fields can be lost.
        """
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        original = Request(
            archive=True,
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url="https://example.com/files/document.pdf",
                # `params` is encoded into the URL on serialize, so we
                # don't assert it round-trips as a separate field below.
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

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type, verify,
                    timeout_json, json_data, files_json, auth_json,
                    allow_redirects, proxies_json, stream, cert_json,
                    archive_hash_header
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type, :verify,
                    :timeout_json, :json_data, :files_json, :auth_json,
                    :allow_redirects, :proxies_json, :stream, :cert_json,
                    :archive_hash_header
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                    "verify": serialized["verify"],
                    "timeout_json": serialized["timeout_json"],
                    "json_data": serialized["json_data"],
                    "files_json": serialized["files_json"],
                    "auth_json": serialized["auth_json"],
                    "allow_redirects": serialized["allow_redirects"],
                    "proxies_json": serialized["proxies_json"],
                    "stream": serialized["stream"],
                    "cert_json": serialized["cert_json"],
                    "archive_hash_header": serialized["archive_hash_header"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)
        assert isinstance(deserialized, Request)

        # URL and method are stored as their own columns and are known
        # to round-trip; check them so a future schema change can't make
        # this test pass without actually exercising them.
        assert deserialized.request.url == original.request.url
        assert deserialized.request.method == original.request.method

        # Every other HTTPRequestParams field that the scraper set
        # should survive the round-trip.
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

        # archive_hash_header lives on the Request, not HTTPRequestParams,
        # but it's another field the serializer silently drops, so guard
        # it here too.
        assert deserialized.archive_hash_header == original.archive_hash_header

    async def test_bypass_rate_limit_default_false(
        self, initialized_db
    ) -> None:
        """Test that bypass_rate_limit defaults to False when not set."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        original = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com/normal",
            ),
            continuation="parse",
            current_location="",
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)
        assert serialized["bypass_rate_limit"] is False

        engine, session_factory = initialized_db
        async with session_factory() as session:
            await session.execute(
                sa.text("""
                INSERT INTO requests (
                    status, priority, queue_counter, request_type,
                    method, url, headers_json, cookies_json, body,
                    continuation, current_location,
                    accumulated_data_json, permanent_json,
                    expected_type
                ) VALUES (
                    'pending', :priority, 1, :request_type,
                    :method, :url, :headers_json, :cookies_json, :body,
                    :continuation, :current_location,
                    :accumulated_data_json, :permanent_json,
                    :expected_type
                )
                """),
                {
                    "priority": original.priority,
                    "request_type": serialized["request_type"],
                    "method": serialized["method"],
                    "url": serialized["url"],
                    "headers_json": serialized["headers_json"],
                    "cookies_json": serialized["cookies_json"],
                    "body": serialized["body"],
                    "continuation": serialized["continuation"],
                    "current_location": serialized["current_location"],
                    "accumulated_data_json": serialized[
                        "accumulated_data_json"
                    ],
                    "permanent_json": serialized["permanent_json"],
                    "expected_type": serialized["expected_type"],
                },
            )
            await session.commit()

        async with session_factory() as session:
            result = await session.execute(
                sa.text("""
                SELECT id, request_type, method, url, headers_json, cookies_json, body,
                       continuation, current_location,
                       accumulated_data_json, permanent_json,
                       expected_type, priority,
                       is_speculative, speculation_id, verify, via_json,
                       bypass_rate_limit, deduplication_key,
                       timeout_json, json_data, files_json, auth_json,
                       allow_redirects, proxies_json, stream, cert_json,
                       archive_hash_header
                FROM requests WHERE id = 1
                """)
            )
            row = result.first()
        assert row is not None

        deserialized = driver._deserialize_request(row)
        assert deserialized.bypass_rate_limit is False
