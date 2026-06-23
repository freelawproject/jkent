"""Pin ``PoolWorker``'s documented extension surface.

jent's ``ReplayWorker`` subclasses ``PoolWorker``, overrides ``_handle_one``,
and reimplements the request lifecycle against the protected members listed
in the class docstring. Those names are the blessed extension seam: renaming
one silently breaks jent, so this suite turns a rename into a jkent test
failure. It checks presence and signatures only — behavior is covered by the
worker/continuation suites.
"""

from __future__ import annotations

import inspect
from typing import Any

from jkent.driver.unified_driver.continuation import ContinuationExecutor
from jkent.driver.unified_driver.worker import PoolWorker


def _params(func) -> list[str]:
    return list(inspect.signature(func).parameters)


def test_lifecycle_hooks_exist_with_pinned_signatures() -> None:
    assert _params(PoolWorker._handle_one) == [
        "self",
        "request_id",
        "request",
        "parent_request_id",
        "preresolved",
    ]
    assert _params(PoolWorker._execute_preresolved) == [
        "self",
        "request_id",
        "request",
        "continuation_name",
    ]


def test_helper_methods_exist_with_pinned_signatures() -> None:
    assert _params(PoolWorker._continuation_name) == ["self", "request"]
    assert _params(PoolWorker._await_conditions) == [
        "self",
        "continuation_name",
    ]
    assert _params(PoolWorker._record_for_compactor) == [
        "self",
        "continuation_name",
    ]
    assert _params(PoolWorker._store_error_for) == [
        "self",
        "exc",
        "request_id",
        "request_url",
    ]
    assert _params(PoolWorker._request_url) == ["self", "request"]


def test_collaborator_attributes_are_assigned_by_init() -> None:
    # The constructor kwargs land on these protected attributes; subclasses
    # read them directly (e.g. ReplayWorker's _transport/_storage properties).
    # The test only checks assignment, so any placeholder satisfies init.
    anything: Any = object()
    worker = PoolWorker(
        7,
        queue=anything,
        transport=anything,
        rate_limiter=anything,
        continuation=anything,
        storage=anything,
        stop_event=anything,
        scraper=anything,
        archive_handler=anything,
        compactor_for=None,
        store_error=None,
        track_speculation=None,
        circuit_breaker=None,
    )
    assert worker.worker_id == 7
    for attr in (
        "_queue",
        "_transport",
        "_storage",
        "_continuation",
        "_track_speculation",
        "_circuit_breaker",
    ):
        assert hasattr(worker, attr)


def test_continuation_complete_request_signature() -> None:
    assert _params(ContinuationExecutor.complete_request) == [
        "self",
        "request_id",
        "response",
        "request",
        "continuation_name",
        "page",
        "store_response",
    ]
