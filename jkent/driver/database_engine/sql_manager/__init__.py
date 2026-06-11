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
from jkent.driver.database_engine.sql_manager._incidental_requests import (
    IncidentalRequestStorageMixin,
)
from jkent.driver.database_engine.sql_manager._requests import (
    DequeuedRow,
    RequestQueueMixin,
)
from jkent.driver.database_engine.sql_manager._responses import (
    CompressedResponse,
    ParentTabResponse,
    ResponseStorageMixin,
    StoredResponse,
)
from jkent.driver.database_engine.sql_manager._results import (
    ResultStorageMixin,
)
from jkent.driver.database_engine.sql_manager._run_metadata import (
    RUN_METADATA_ID,
    RunMetadataDict,
    RunMetadataMixin,
)
from jkent.driver.database_engine.sql_manager._speculation import (
    SpeculationMixin,
)
from jkent.driver.database_engine.sql_manager._types import (
    IncidentalRequestDict,
    IncidentalRequestParams,
    IncidentalRequestRecord,
    RequestInsert,
    compute_cache_key,
)


class SQLManager(
    RunMetadataMixin,
    ErrorsMixin,
    RequestQueueMixin,
    ResponseStorageMixin,
    IncidentalRequestStorageMixin,
    ResultStorageMixin,
    SpeculationMixin,
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
    "RUN_METADATA_ID",
    "CompressedResponse",
    "DequeuedRow",
    "IncidentalRequestDict",
    "IncidentalRequestParams",
    "IncidentalRequestRecord",
    "ParentTabResponse",
    "RequestInsert",
    "RunMetadataDict",
    "SQLManager",
    "StoredResponse",
    "compute_cache_key",
]
