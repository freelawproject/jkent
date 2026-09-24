"""SQLManager - Database operations for the unified driver.

This package provides a standalone class for all SQLite database operations,
enabling independent testing and programmatic inspection of the database
without requiring a full driver instance.

The SQLManager handles:
- Request queue operations (enqueue, dequeue, status updates)
- Response storage with compression
- Result storage with validation tracking
- Error tracking
- Run metadata management
- Speculative progress tracking
"""

from jkent.driver.database_engine.sql_manager._base import SQLManagerBase
from jkent.driver.database_engine.sql_manager._errors import ErrorsMixin
from jkent.driver.database_engine.sql_manager._requests import (
    RequestQueueMixin,
    RetryState,
)
from jkent.driver.database_engine.sql_manager._responses import (
    ResponseStorageMixin,
)
from jkent.driver.database_engine.sql_manager._results import (
    ResultStorageMixin,
)
from jkent.driver.database_engine.sql_manager._types import (
    CompressedPayload,
    DequeuedRow,
    IncidentalCapture,
    IncidentalRequestRecord,
    InsertResult,
    RequestInsert,
    ResultInsert,
    RowModel,
    StoredResponse,
    compute_cache_key,
)


class SQLManager(
    ErrorsMixin,
    RequestQueueMixin,
    ResponseStorageMixin,
    ResultStorageMixin,
    SQLManagerBase,
):
    """Database manager for the unified driver's run database.

    Provides all database operations needed by the driver in a standalone
    class that can be used independently for testing, inspection, and
    programmatic access to the SQLite database.

    Example::

        # Standalone usage for inspection
        async with SQLManager.open(db_path) as manager:
            params = await manager.get_seed_params()

        # With existing engine/session factory (for driver integration)
        manager = SQLManager(engine, session_factory)
        await manager.insert_request(params)
    """


__all__ = [
    "CompressedPayload",
    "DequeuedRow",
    "IncidentalCapture",
    "IncidentalRequestRecord",
    "InsertResult",
    "RequestInsert",
    "ResultInsert",
    "RetryState",
    "RowModel",
    "SQLManager",
    "StoredResponse",
    "compute_cache_key",
]
