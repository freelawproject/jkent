"""Tests for result storage operations (_results.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import sqlalchemy as sa

from jkent.driver.database_engine.sql_manager import SQLManager

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    InsertRequest = Callable[..., Awaitable[int]]


class TestResultStorage:
    """Tests for result storage operations."""

    async def test_store_result_valid(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test storing a valid result."""
        request_id = await insert_request()

        result_id = await sql_manager.store_result(
            request_id=request_id,
            result_type="CaseData",
            data_json=json.dumps({"case_name": "Smith v. Jones", "id": 123}),
            is_valid=True,
            validation_errors_json=None,
        )

        assert result_id > 0

        # Verify result was stored
        async with sql_manager._session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT result_type, is_valid, data_json FROM results WHERE id = :id"
                ),
                {"id": result_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == "CaseData"
        assert row[1] == 1  # is_valid
        data = json.loads(row[2])
        assert data["case_name"] == "Smith v. Jones"

    async def test_store_result_invalid(
        self, sql_manager: SQLManager, insert_request: InsertRequest
    ) -> None:
        """Test storing an invalid result with validation errors."""
        request_id = await insert_request()

        validation_errors = [
            {"loc": ["docket_number"], "msg": "field required"}
        ]

        result_id = await sql_manager.store_result(
            request_id=request_id,
            result_type="CaseData",
            data_json=json.dumps({"case_name": "Incomplete"}),
            is_valid=False,
            validation_errors_json=json.dumps(validation_errors),
        )

        # Verify result was stored as invalid
        async with sql_manager._session_factory() as session:
            result = await session.execute(
                sa.text(
                    "SELECT is_valid, validation_errors_json FROM results WHERE id = :id"
                ),
                {"id": result_id},
            )
            row = result.first()
        assert row is not None
        assert row[0] == 0  # is_valid = False
        assert row[1] is not None
