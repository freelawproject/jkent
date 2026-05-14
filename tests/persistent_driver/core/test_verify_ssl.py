"""Tests that the verify parameter round-trips through serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

ALABAMA_URL = (
    "https://acis.alabama.gov/displaydocs2.cfm?no=7287&event=5CS0U0K6M"
)


class TestVerifyThroughPersistentDriverQueue:
    """verify=False round-trips through serialize → DB → deserialize → request."""

    async def test_verify_false_round_trip(self, initialized_db) -> None:
        """Insert a verify=False request, dequeue it, and confirm the flag survives."""
        from kent.driver.persistent_driver.sql_manager import SQLManager

        engine, session_factory = initialized_db
        sql = SQLManager(engine, session_factory)

        # Insert with verify="false" (the DB representation of verify=False)
        req_id = await sql.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url=ALABAMA_URL,
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse_page",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key=None,
            parent_id=None,
            verify="false",
        )
        assert req_id is not None

        # Dequeue and deserialize
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        row = await sql.dequeue_next_request()
        assert row is not None

        request = driver._deserialize_request(row[:28])
        assert request.request.verify is False

    async def test_verify_true_round_trip(self, initialized_db) -> None:
        """Default verify (None in DB) deserializes to True."""
        from kent.driver.persistent_driver.sql_manager import SQLManager

        engine, session_factory = initialized_db
        sql = SQLManager(engine, session_factory)

        await sql.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key=None,
            parent_id=None,
            # verify not passed — defaults to None
        )

        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        row = await sql.dequeue_next_request()
        assert row is not None

        request = driver._deserialize_request(row[:28])
        assert request.request.verify is True

    async def test_verify_ca_bundle_round_trip(self, initialized_db) -> None:
        """A CA bundle path string round-trips correctly."""
        from kent.driver.persistent_driver.sql_manager import SQLManager

        engine, session_factory = initialized_db
        sql = SQLManager(engine, session_factory)

        await sql.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key=None,
            parent_id=None,
            verify="/etc/ssl/certs/ca-bundle.crt",
        )

        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        row = await sql.dequeue_next_request()
        assert row is not None

        request = driver._deserialize_request(row[:28])
        assert request.request.verify == "/etc/ssl/certs/ca-bundle.crt"

    async def test_serialize_deserialize_verify_false(self) -> None:
        """_serialize_request produces verify='false' for verify=False."""
        from kent.data_types import HttpMethod, HTTPRequestParams, Request
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        req = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=ALABAMA_URL,
                verify=False,
            ),
            continuation="parse_page",
            current_location="",
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(req)
        assert serialized["verify"] == "false"

    async def test_serialize_deserialize_verify_true(self) -> None:
        """_serialize_request produces verify=None for verify=True (default)."""
        from kent.data_types import HttpMethod, HTTPRequestParams, Request
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )

        req = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://example.com",
            ),
            continuation="parse",
            current_location="",
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(req)
        assert serialized["verify"] is None

    async def test_verify_false_survives_resolve_from(self) -> None:
        """verify=False survives resolve_from (the enqueue_request path)."""
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Request,
            Response,
        )

        parent = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://judicial.alabama.gov/decision/supremecourtdecisions",
            ),
            continuation="parse_historical_decisions_list",
        )
        parent_response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="https://judicial.alabama.gov/decision/supremecourtdecisions",
            request=parent,
        )

        child = Request(
            archive=True,
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=ALABAMA_URL,
                verify=False,
            ),
            continuation="handle_historical_pdf_download",
        )

        resolved = child.resolve_from(parent_response)
        assert resolved.request.verify is False

    @pytest.mark.slow
    async def test_verify_false_end_to_end(self, db_path: Path) -> None:
        """Full end-to-end: insert verify=False request, dequeue, fetch via AsyncRequestManager."""
        from kent.common.request_manager import AsyncRequestManager
        from kent.data_types import HttpMethod, HTTPRequestParams, Request
        from kent.driver.persistent_driver.database import init_database
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.sql_manager import SQLManager

        engine, session_factory = await init_database(db_path)
        sql = SQLManager(engine, session_factory)

        # Create and serialize a verify=False request
        original = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=ALABAMA_URL,
                verify=False,
            ),
            continuation="parse_page",
            current_location="",
        )

        driver = PersistentDriver.__new__(PersistentDriver)
        serialized = driver._serialize_request(original)

        await sql.insert_request(
            priority=1,
            request_type=serialized["request_type"],
            method=serialized["method"],
            url=serialized["url"],
            headers_json=serialized["headers_json"],
            cookies_json=serialized["cookies_json"],
            body=serialized["body"],
            continuation=serialized["continuation"],
            current_location=serialized["current_location"],
            accumulated_data_json=serialized["accumulated_data_json"],
            permanent_json=serialized["permanent_json"],
            expected_type=serialized["expected_type"],
            dedup_key=None,
            parent_id=None,
            verify=serialized["verify"],
        )

        # Dequeue
        row = await sql.dequeue_next_request()
        assert row is not None
        request = driver._deserialize_request(row[:28])
        assert request.request.verify is False

        # Actually fetch through AsyncRequestManager
        manager = AsyncRequestManager()
        try:
            response = await manager.resolve_request(request)
            assert response.status_code == 200
            assert len(response.content) > 0
        finally:
            await manager.close()

        await engine.dispose()
