"""SQLModel table definitions for the Kent persistent_driver database.

This module defines all database table models using SQLModel, replacing
the raw DDL in schema.py. These models are used with SQLAlchemy's async
engine for all database operations.

Tables:
- requests: HTTP request queue with status tracking, retry logic, and
  inline response storage (compressed HTTP responses with dictionary refs)
- compression_dicts: Versioned zstd dictionaries per-continuation
- results: Validated scraped data
- archived_files: Downloaded file metadata
- run_metadata: Single-row configuration and state
- errors: Detailed error tracking with type-specific fields
- rate_items: Rate limiting items
- speculation_tracking: Speculative protocol tracking state
- incidental_request_storage: Deduplicated content for browser requests
- incidental_requests: Browser-initiated network requests (Playwright)
- schema_info: Schema version tracking
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy import Column, LargeBinary
from sqlmodel import Field, SQLModel


class SchemaInfo(SQLModel, table=True):  # type: ignore[call-arg]
    """Schema version tracking."""

    __tablename__ = "schema_info"

    id: int | None = Field(default=None, primary_key=True)
    version: int
    applied_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class Request(SQLModel, table=True):  # type: ignore[call-arg]
    """HTTP request queue with status tracking and retry logic."""

    __tablename__ = "requests"
    __table_args__ = (
        sa.UniqueConstraint(
            "deduplication_key",
            name="uq_requests_dedup_key",
            sqlite_on_conflict="IGNORE",
        ),
        sa.Index(
            "idx_requests_status_priority",
            "status",
            "priority",
            "queue_counter",
        ),
        sa.Index("idx_requests_continuation", "continuation"),
        sa.Index("idx_requests_deduplication", "deduplication_key"),
        sa.Index("idx_requests_cache_key", "cache_key"),
        sa.Index("idx_requests_parent", "parent_request_id"),
        sa.Index("idx_requests_response_status_code", "response_status_code"),
        sa.Index("idx_requests_compression_dict", "compression_dict_id"),
    )

    id: int | None = Field(default=None, primary_key=True)

    # Queue management
    status: str = Field(
        default="pending",
        sa_column_kwargs={"server_default": sa.text("'pending'")},
    )
    priority: int = Field(
        default=9,
        sa_column_kwargs={"server_default": sa.text("9")},
    )
    queue_counter: int
    request_type: str = Field(
        default="navigating",
        sa_column_kwargs={"server_default": sa.text("'navigating'")},
    )

    # HTTP Request
    method: str
    url: str
    headers_json: str | None = None
    cookies_json: str | None = None
    body: bytes | None = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )

    # Scraper context
    continuation: str
    current_location: str = Field(
        default="",
        sa_column_kwargs={"server_default": sa.text("''")},
    )
    accumulated_data_json: str | None = None
    permanent_json: str | None = None
    deduplication_key: str | None = Field(default=None)
    cache_key: str | None = None

    # Archive-specific
    expected_type: str | None = None

    # Rate limit bypass
    bypass_rate_limit: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": sa.text("0")},
    )

    # Timestamps (human-readable)
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )
    started_at: str | None = None
    completed_at: str | None = None

    # High-precision monotonic timestamps (nanoseconds)
    created_at_ns: int | None = None
    started_at_ns: int | None = None
    completed_at_ns: int | None = None

    # Retry tracking
    retry_count: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    cumulative_backoff: float = Field(
        default=0.0,
        sa_column_kwargs={"server_default": sa.text("0.0")},
    )
    next_retry_delay: float | None = None
    last_error: str | None = None

    # Parent tracking
    parent_request_id: int | None = Field(
        default=None, foreign_key="requests.id"
    )

    # Speculation tracking
    is_speculative: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    speculation_id: str | None = None

    # --- Response fields (populated when response is received) ---
    # NULL response_status_code means no response has been stored yet.
    response_status_code: int | None = None
    response_headers_json: str | None = None
    response_url: str | None = None

    # Content (compressed)
    content_compressed: bytes | None = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )
    content_size_original: int | None = None
    content_size_compressed: int | None = None

    # Compression metadata
    compression_dict_id: int | None = Field(
        default=None, foreign_key="compression_dicts.id"
    )

    # Response timestamps
    response_created_at: str | None = None

    # Speculative request outcome tracking
    speculation_outcome: str | None = None

    # Playwright via field (ViaFormSubmit / ViaLink JSON)
    via_json: str | None = None

    # TLS verification override
    verify: str | None = None

    # --- Remaining HTTPRequestParams fields (added in v20) ---
    # The persistent queue originally only round-tripped a subset of
    # HTTPRequestParams. Scrapers that set timeout / json / files / auth /
    # allow_redirects / proxies / stream / cert had those values silently
    # dropped on enqueue, which is how the Nevada Supreme Court archive
    # downloads ended up hanging on the httpx client-level default timeout.
    timeout_json: str | None = None
    json_data: str | None = None
    files_json: str | None = None
    auth_json: str | None = None
    allow_redirects: bool = Field(
        default=True,
        sa_column_kwargs={"server_default": sa.text("1")},
    )
    proxies_json: str | None = None
    stream: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    cert_json: str | None = None

    # Request-level (not HTTPRequestParams) but also dropped pre-v20.
    archive_hash_header: str | None = None


class CompressionDict(SQLModel, table=True):  # type: ignore[call-arg]
    """Versioned zstd compression dictionaries per-continuation."""

    __tablename__ = "compression_dicts"
    __table_args__ = (
        sa.UniqueConstraint("continuation", "version"),
        sa.Index("idx_compression_dicts_continuation", "continuation"),
    )

    id: int | None = Field(default=None, primary_key=True)
    continuation: str
    version: int
    dictionary_data: bytes = Field(
        sa_column=Column(LargeBinary, nullable=False)
    )
    sample_count: int
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class Result(SQLModel, table=True):  # type: ignore[call-arg]
    """Validated scraped data results."""

    __tablename__ = "results"
    __table_args__ = (
        sa.Index("idx_results_type", "result_type"),
        sa.Index("idx_results_request", "request_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    request_id: int | None = Field(default=None, foreign_key="requests.id")

    # Result data
    result_type: str
    data_json: str

    # Validation status
    is_valid: bool = Field(
        default=True,
        sa_column_kwargs={"server_default": sa.text("1")},
    )
    validation_errors_json: str | None = None

    # Timestamps
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class ArchivedFile(SQLModel, table=True):  # type: ignore[call-arg]
    """Downloaded file metadata."""

    __tablename__ = "archived_files"
    __table_args__ = (
        sa.Index("idx_archived_files_request", "request_id"),
        sa.Index("idx_archived_files_hash", "content_hash"),
    )

    id: int | None = Field(default=None, primary_key=True)
    request_id: int = Field(foreign_key="requests.id")

    # File info
    file_path: str
    original_url: str
    expected_type: str | None = None
    file_size: int | None = None
    content_hash: str | None = None

    # Timestamps
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class RunMetadata(SQLModel, table=True):  # type: ignore[call-arg]
    """Single-row run configuration and state."""

    __tablename__ = "run_metadata"
    __table_args__ = (
        sa.CheckConstraint("id = 1", name="run_metadata_single_row"),
    )

    id: int | None = Field(default=None, primary_key=True)

    # Scraper identity
    scraper_name: str
    scraper_version: str | None = None

    # Run state
    status: str = Field(
        default="created",
        sa_column_kwargs={"server_default": sa.text("'created'")},
    )
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )
    started_at: str | None = None
    ended_at: str | None = None
    error_message: str | None = None

    # Invocation parameters
    params_json: str | None = None
    seed_params_json: str | None = None
    base_delay: float
    jitter: float
    num_workers: int
    max_backoff_time: float

    # Speculation configuration
    speculation_config_json: str | None = None

    # Browser configuration (Playwright driver)
    browser_config_json: str | None = None

    # Browser cookie persistence (Playwright driver, for resume)
    browser_cookies_json: str | None = None


class Error(SQLModel, table=True):  # type: ignore[call-arg]
    """Detailed error tracking with type-specific fields."""

    __tablename__ = "errors"
    __table_args__ = (
        sa.Index("idx_errors_request", "request_id"),
        sa.Index("idx_errors_type", "error_type"),
        sa.Index(
            "idx_errors_unresolved",
            "is_resolved",
            sqlite_where=sa.text("is_resolved = 0"),
        ),
    )

    id: int | None = Field(default=None, primary_key=True)
    request_id: int | None = Field(default=None, foreign_key="requests.id")

    # Error classification
    error_type: str
    error_class: str
    message: str
    request_url: str

    # Structured error data
    context_json: str | None = None

    # Structural errors (HTMLStructuralAssumptionException)
    selector: str | None = None
    selector_type: str | None = None
    expected_min: int | None = None
    expected_max: int | None = None
    actual_count: int | None = None

    # Validation errors (DataFormatAssumptionException)
    model_name: str | None = None
    validation_errors_json: str | None = None
    failed_doc_json: str | None = None

    # Transient errors
    status_code: int | None = None
    timeout_seconds: float | None = None

    # Stack trace
    traceback: str | None = None

    # Resolution tracking
    is_resolved: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    resolved_at: str | None = None
    resolution_notes: str | None = None
    resolution_type: str | None = None

    # Timestamps
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class RateItem(SQLModel, table=True):  # type: ignore[call-arg]
    """Rate limiting items."""

    __tablename__ = "rate_items"
    __table_args__ = (sa.Index("idx_rate_items_timestamp", "timestamp"),)

    id: int | None = Field(default=None, primary_key=True)
    name: str
    timestamp: int
    weight: int = Field(
        default=1,
        sa_column_kwargs={"server_default": sa.text("1")},
    )


class SpeculationTracking(SQLModel, table=True):  # type: ignore[call-arg]
    """Tracks speculation state for Speculative protocol entries."""

    __tablename__ = "speculation_tracking"
    __table_args__ = (sa.Index("idx_speculation_tracking_func", "func_name"),)

    id: int | None = Field(default=None, primary_key=True)
    func_name: str = Field(unique=True)
    highest_successful_id: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    consecutive_failures: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    current_ceiling: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    stopped: bool = Field(
        default=False,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    param_index: int = Field(
        default=0,
        sa_column_kwargs={"server_default": sa.text("0")},
    )
    template_json: str | None = None
    updated_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class Estimate(SQLModel, table=True):  # type: ignore[call-arg]
    """Estimated downstream result counts from EstimateData yields."""

    __tablename__ = "estimates"
    __table_args__ = (sa.Index("idx_estimates_request", "request_id"),)

    id: int | None = Field(default=None, primary_key=True)
    request_id: int = Field(foreign_key="requests.id")

    # Estimate parameters
    expected_types_json: str  # JSON list of type name strings
    min_count: int
    max_count: int | None = None

    # Timestamps
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )


class IncidentalRequestStorage(SQLModel, table=True):  # type: ignore[call-arg]
    """Deduplicated content storage for incidental browser requests."""

    __tablename__ = "incidental_request_storage"
    __table_args__ = (sa.Index("idx_irs_content_md5", "content_md5"),)

    id: int | None = Field(default=None, primary_key=True)
    resource_type: str
    url: str
    method: str
    body: bytes | None = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )
    status_code: int | None = None
    response_headers_json: str | None = None
    content_compressed: bytes | None = Field(
        default=None, sa_column=Column(LargeBinary, nullable=True)
    )
    content_size_original: int | None = None
    content_size_compressed: int | None = None
    compression_dict_id: int | None = Field(
        default=None, foreign_key="compression_dicts.id"
    )
    failure_reason: str | None = None
    content_md5: str | None = None


class IncidentalRequest(SQLModel, table=True):  # type: ignore[call-arg]
    """Browser-initiated network requests (Playwright driver)."""

    __tablename__ = "incidental_requests"
    __table_args__ = (
        sa.Index("idx_incidental_requests_parent", "parent_request_id"),
        sa.Index("idx_incidental_requests_storage", "storage_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    parent_request_id: int = Field(foreign_key="requests.id")
    url: str
    headers_json: str | None = None
    started_at_ns: int | None = None
    completed_at_ns: int | None = None
    from_cache: bool | None = None
    created_at: str | None = Field(
        default=None,
        sa_column_kwargs={"server_default": sa.text("CURRENT_TIMESTAMP")},
    )
    storage_id: int | None = Field(
        default=None, foreign_key="incidental_request_storage.id"
    )
