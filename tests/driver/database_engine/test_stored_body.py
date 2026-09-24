"""Pin which request rows count as a stored body, and who counts them.

Dictionary training, recompression and the off-dictionary count all select
from the same predicate; these cases pin its semantics once, and pin each
consumer to it, so a consumer that drifts from the shared definition fails
here.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy import select

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.compression import (
    compress,
    count_off_dictionary,
)
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Request
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stored_body import (
    stored_body_clauses,
    training_sample_clauses,
)

_STEP = "parse"

#: A dictionary id no row is compressed against, so the off-dictionary
#: count reduces to the stored-body predicate alone.
_UNUSED_DICT_ID = 999_999


@pytest.mark.parametrize(
    ("step", "status", "status_code", "has_body", "stored", "trainable"),
    [
        pytest.param(
            _STEP,
            RequestStatus.COMPLETED,
            200,
            True,
            True,
            True,
            id="completed-with-body",
        ),
        pytest.param(
            _STEP,
            RequestStatus.FAILED,
            500,
            True,
            True,
            False,
            id="failed-error-page",
        ),
        pytest.param(
            _STEP,
            RequestStatus.PENDING,
            503,
            True,
            True,
            False,
            id="retried-debug-snapshot",
        ),
        pytest.param(
            _STEP,
            RequestStatus.COMPLETED,
            200,
            False,
            False,
            False,
            id="archive-status-no-body",
        ),
        pytest.param(
            _STEP,
            RequestStatus.COMPLETED,
            None,
            True,
            False,
            False,
            id="body-without-status-code",
        ),
        pytest.param(
            _STEP,
            RequestStatus.PENDING,
            None,
            False,
            False,
            False,
            id="unfetched",
        ),
        pytest.param(
            "other_step",
            RequestStatus.COMPLETED,
            200,
            True,
            False,
            False,
            id="other-step",
        ),
    ],
)
async def test_stored_body_predicate(
    sql_manager: SQLManager,
    step: str,
    status: RequestStatus,
    status_code: int | None,
    has_body: bool,
    stored: bool,
    trainable: bool,
) -> None:
    """One row is a stored body / a training sample exactly as pinned.

    ``resolved_response_count`` counts training samples and
    ``count_off_dictionary`` counts stored bodies not yet on the target
    dictionary; both must agree with the shared predicate.
    """
    compressed = compress(b"<html>body</html>") if has_body else None
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, method, url, step, "
                "current_location, response_status_code, content_compressed) "
                "VALUES (:status, 9, :method, 'https://stored.test/', :step, "
                "'', :status_code, :compressed)"
            ),
            {
                "status": status.code,
                "method": HttpMethod.GET.code,
                "step": step,
                "status_code": status_code,
                "compressed": compressed,
            },
        )
        await session.commit()

        stored_count = (
            await session.execute(
                select(sa.func.count())
                .select_from(Request)
                .where(*stored_body_clauses(_STEP))
            )
        ).scalar_one()
        trainable_count = (
            await session.execute(
                select(sa.func.count())
                .select_from(Request)
                .where(*training_sample_clauses(_STEP))
            )
        ).scalar_one()

    assert stored_count == int(stored)
    assert trainable_count == int(trainable)
    assert await sql_manager.resolved_response_count(_STEP) == int(trainable)
    assert await count_off_dictionary(
        sql_manager, _STEP, _UNUSED_DICT_ID
    ) == int(stored)
