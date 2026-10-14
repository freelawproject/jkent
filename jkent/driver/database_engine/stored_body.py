"""Which ``requests`` rows hold a stored response body.

One definition for every reader that selects stored bodies — compression's
training and recompression passes and the request mixin's counts — so they
cannot drift apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Request

if TYPE_CHECKING:
    import sqlalchemy as sa


def stored_body_clauses(step: str) -> tuple[sa.ColumnElement[bool], ...]:
    """``WHERE`` clauses selecting ``step``'s rows that hold a stored body.

    A stored body is a non-NULL ``content_compressed`` on a row with a
    ``response_status_code``, of any request status. Archive requests set a
    status code but store their file on disk, so they are excluded.
    Recompression and :func:`~jkent.driver.database_engine.compression.count_off_dictionary` select from this set.
    """
    return (
        Request.step == step,
        Request.response_status_code.isnot(None),
        Request.content_compressed.isnot(None),
    )


def training_sample_clauses(step: str) -> tuple[sa.ColumnElement[bool], ...]:
    """``WHERE`` clauses selecting ``step``'s dictionary-training samples.

    The :func:`stored_body_clauses` rows whose request is ``COMPLETED``:
    failed and retried rows can carry an error-page or debug-snapshot body,
    which does not belong in a step's training set.
    :func:`~jkent.driver.database_engine.compression.train_compression_dict` samples from this set and
    ``SQLManager.resolved_response_count`` counts it.
    """
    return (
        *stored_body_clauses(step),
        Request.status == RequestStatus.COMPLETED,
    )
