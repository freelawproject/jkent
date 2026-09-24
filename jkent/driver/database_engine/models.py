"""SQLAlchemy ORM table definitions.

Tables:
- requests: HTTP request queue with status tracking, retry logic, and
  inline response storage (compressed HTTP responses with dictionary refs)
- compression_dicts: Versioned zstd dictionaries per-step
- results: Validated scraped data
- archived_files: Downloaded file metadata
- run_metadata: Single-row configuration and state
- errors: Detailed error tracking with type-specific fields
- speculation_tracking: Speculative protocol tracking state
- incidental_request_storage: Deduplicated content for browser requests
- incidental_requests: Browser-initiated network requests (Playwright)
- schema_info: Schema version tracking

Columns whose values come from a fixed vocabulary are
:class:`~jkent.driver.database_engine.enums.CodedEnumType`: an ``INTEGER``
column holding the member's ``.code``, mapped back to the member on load. Each
one is paired with a ``code_check`` ``CHECK`` constraint, because an integer
column otherwise records nothing at all about its vocabulary — not even how
many values it has. Reading these columns outside the ORM means decoding with
``<Enum>.from_code()``; a raw ``SELECT status FROM requests`` yields ``1``.

Columns typed as a bare ``str`` are open vocabularies.

Timestamp columns are ``Mapped[datetime | None]``, mapped to
``timestamps.UtcDateTime`` by ``Base.type_annotation_map`` and so read back as a
UTC-aware ``datetime`` rather than as text needing ``fromisoformat``. Writing
one from Python requires an aware value; a naive one is refused rather than
guessed at. The value normally comes from ``timestamps.now_sql()`` and is the
millisecond text format documented there. Durations are computed with
``timestamps.epoch_seconds`` — subtracting two ``DateTime`` columns directly
compiles to a SQL ``-`` between two TEXT values, which SQLite coerces to 0
rather than rejecting, so the ORM-native spelling silently returns 0.0.

:class:`RowModel`, the pydantic base for models read from these rows, lives
here too: ``errors`` and ``sql_manager`` both build on it, and neither can
import it from the other without a cycle.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from pydantic import BaseModel, ConfigDict
from sqlalchemy import ForeignKey, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import (
    CodedEnumType,
    ErrorType,
    RequestStatus,
    RequestType,
    RunStatus,
    SelectorType,
    SpeculationOutcome,
    TransientKind,
    code_check,
)
from jkent.driver.database_engine.timestamps import UtcDateTime, now_sql

__all__ = [
    "ArchivedFile",
    "Base",
    "CompressionDict",
    "Error",
    "ErrorType",
    "IncidentalRequest",
    "IncidentalRequestStorage",
    "Request",
    "RequestStatus",
    "RequestType",
    "Result",
    "RowModel",
    "RunMetadata",
    "RunStatus",
    "SchemaInfo",
    "SelectorType",
    "SpeculationOutcome",
    "SpeculationTracking",
]


def json_checks(table: str, *columns: str) -> tuple[sa.CheckConstraint, ...]:
    """``CHECK (json_valid(col))`` for each of *columns* on *table*.

    The ``_json`` suffix is a naming convention the database cannot enforce, so
    without these a malformed value is only discovered when something tries to
    parse it — a run or two later, in a corpus, far from the writer that put it
    there. SQLite's ``json_valid`` moves that to the INSERT.

    NULL is accepted without an explicit ``IS NULL`` arm: ``json_valid(NULL)``
    is NULL, and a CHECK only fails on a definitively false result. The empty
    string is *not* accepted — it is not valid JSON — which is the one case
    worth knowing about, since "no value" here must be NULL rather than ``''``.

    Args:
        table: Table name, used to build the constraint names.
        columns: Column names to constrain.

    Returns:
        Constraints to splat into the table's ``__table_args__``.
    """
    return tuple(
        sa.CheckConstraint(
            f"json_valid({column})", name=f"ck_{table}_{column}_valid"
        )
        for column in columns
    )


class RowModel(BaseModel):
    """Base for row-shaped models: built from rows/ORM objects, no extras."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")


class Base(DeclarativeBase):
    """Declarative base owning the metadata that ``create_all`` builds from.

    ``type_annotation_map`` is what lets every timestamp column below be
    declared as a bare ``Mapped[datetime | None]``: without it each one would
    have to name :class:`~jkent.driver.database_engine.timestamps.UtcDateTime`
    explicitly, and one that forgot would take naive local time without
    complaint.
    """

    type_annotation_map = {datetime: UtcDateTime()}


class CompressedPayloadColumns:
    """The four columns a stored, compressed body occupies.

    Shared by ``requests`` (a response) and ``incidental_request_storage`` (a
    captured sub-request); ``sql_manager.CompressedPayload`` is the row model
    with the same fields. Declarative emits mixin columns after the ones the
    class declares itself, so these land at the end of both tables.
    """

    content_compressed: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        doc=(
            "zstd-compressed body, using the dictionary named by "
            "``compression_dict_id`` when one is set. NULL when there is no "
            "body."
        ),
    )
    content_size_original: Mapped[int | None] = mapped_column(
        doc="Uncompressed body length in bytes."
    )
    content_size_compressed: Mapped[int | None] = mapped_column(
        doc=(
            "Stored body length in bytes. Kept alongside the original size "
            "so compression ratios are reportable without decompressing."
        )
    )
    compression_dict_id: Mapped[int | None] = mapped_column(
        ForeignKey("compression_dicts.id"),
        doc=(
            "Dictionary this body was compressed against. NULL means it was "
            "compressed without one — for a response, the case before enough "
            "samples had accumulated for its step to train on."
        ),
    )


class SchemaInfo(Base):
    """Schema version tracking.

    One row per schema version applied to this database. A fresh database is
    stamped at ``database.BASELINE_VERSION``, and ``create_engine_and_init``
    refuses to open one stamped at any other version — there is no migration
    runner, so the supported answer is to recreate the run database.
    """

    __tablename__ = "schema_info"

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    version: Mapped[int] = mapped_column(
        doc="Schema version number this row records as applied."
    )
    applied_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Stamp time (UTC, millisecond resolution).",
    )


class Request(CompressedPayloadColumns, Base):
    """HTTP request queue with status tracking and retry logic.

    The central table: one row per request the scraper asked for, carrying the
    request itself, its queue state, and — once fetched — the response inline.
    Responses live on this row rather than in a separate table because every
    read of a response is keyed by its request, and the 1:1 join bought
    nothing.
    """

    __tablename__ = "requests"
    __table_args__ = (
        sa.UniqueConstraint(
            "deduplication_key",
            name="uq_requests_dedup_key",
            sqlite_on_conflict="IGNORE",
        ),
        # Ties within a priority dequeue FIFO by id. Named explicitly: with
        # ANALYZE statistics SQLite will not order by the rowid an index
        # carries implicitly, and falls back to a sort.
        sa.Index("idx_requests_status_priority", "status", "priority", "id"),
        sa.Index("idx_requests_step", "step"),
        sa.Index("idx_requests_cache_key", "cache_key"),
        sa.Index("idx_requests_parent", "parent_request_id"),
        sa.Index("idx_requests_response_status_code", "response_status_code"),
        sa.Index("idx_requests_compression_dict", "compression_dict_id"),
        sa.Index("idx_requests_speculation", "speculation_tracking_id"),
        # The vocabulary of each coded column, as the schema's only record of
        # it now that the values are integers.
        code_check("status", RequestStatus, "ck_requests_status"),
        code_check("request_type", RequestType, "ck_requests_request_type"),
        code_check("method", HttpMethod, "ck_requests_method"),
        code_check(
            "speculation_outcome",
            SpeculationOutcome,
            "ck_requests_speculation_outcome",
        ),
        *json_checks(
            "requests",
            "headers_json",
            "cookies_json",
            "accumulated_data_json",
            "permanent_json",
            "response_headers_json",
            "via_json",
            "timeout_json",
            "verify_json",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Surrogate key. Referenced by every other request-scoped table.",
    )

    # Queue management
    status: Mapped[RequestStatus] = mapped_column(
        CodedEnumType(RequestStatus),
        default=RequestStatus.PENDING,
        server_default=sa.text(str(RequestStatus.PENDING.code)),
        doc="Lifecycle state; see :class:`RequestStatus`.",
    )
    priority: Mapped[int] = mapped_column(
        default=9,
        server_default=sa.text("9"),
        doc=(
            "Dispatch priority, lower first. Scraper-supplied; 9 is the "
            "framework default for an unannotated request."
        ),
    )
    request_type: Mapped[RequestType] = mapped_column(
        CodedEnumType(RequestType),
        default=RequestType.NAVIGATING,
        server_default=sa.text(str(RequestType.NAVIGATING.code)),
        doc="Which driver path this takes; see :class:`RequestType`.",
    )

    # HTTP Request
    method: Mapped[HttpMethod] = mapped_column(
        CodedEnumType(HttpMethod),
        doc="HTTP method; see :class:`jkent.common.request.HttpMethod`.",
    )
    url: Mapped[str] = mapped_column(
        doc=(
            "Absolute request URL with query params already folded in — "
            "``HTTPRequestParams.params`` is not stored separately, this is "
            "the single source of truth for the target "
            "(see ``jkent.common.request.serialize_url_and_body``)."
        )
    )
    headers_json: Mapped[str | None] = mapped_column(
        doc="JSON object of request headers. NULL when none were set."
    )
    cookies_json: Mapped[str | None] = mapped_column(
        doc="JSON object of request cookies. NULL when none were set."
    )
    body: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        doc=(
            "Request body: raw bytes when ``body_is_form`` is false, the JSON "
            "of a dict or pair list of form fields when it is true. NULL "
            "for bodyless requests."
        ),
    )
    body_is_form: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "``body`` holds form data (a dict or pair list) as JSON, "
            "rather than raw bytes. "
            "Without it a raw body that happens to parse as JSON cannot be "
            "told apart from form data."
        ),
    )

    # Scraper context
    step: Mapped[str] = mapped_column(
        doc=(
            "Name of the scraper method that will be handed the response. "
            "Also the key compression dictionaries are trained per."
        )
    )
    current_location: Mapped[str] = mapped_column(
        default="",
        server_default=sa.text("''"),
        doc=(
            "The scraper's browsing location when this request was enqueued."
        ),
    )
    accumulated_data_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON of the partial record threaded down this branch, merged "
            "with each child's contribution as the subtree fans out."
        )
    )
    permanent_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON of fields that propagate unchanged to every descendant "
            "(court id, term, and similar run-wide context)."
        )
    )
    deduplication_key: Mapped[str | None] = mapped_column(
        doc=(
            "Scraper-supplied identity for work that must happen at most "
            "once per run. Enforced by ``uq_requests_dedup_key``, which uses "
            "ON CONFLICT IGNORE so a duplicate insert is a no-op rather than "
            "an error. NULL opts out of deduplication."
        )
    )
    cache_key: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        doc=(
            "Raw 16-byte MD5 digest over (method, url, body, headers_json) — "
            "see ``sql_manager.compute_cache_key``. Identifies requests whose "
            "stored response can be reused. Distinct from "
            "``deduplication_key``: this one is derived from the wire "
            "request, that one is the scraper's declared intent. Stored as a "
            "BLOB digest rather than hex text: the column is indexed and "
            "written on every request, and hex doubles both the row and the "
            "index entry to record nothing extra. MD5 is chosen for width, "
            "not strength — it is a lookup key within one run's database, "
            "never a trust boundary."
        ),
    )

    # Archive-specific
    expected_type: Mapped[str | None] = mapped_column(
        doc=(
            "Open-vocabulary file-kind hint from ``Request(archive=True, "
            "expected_type=...)`` — e.g. ``pdf``, ``audio``. Free-form on "
            "purpose: the archive handler uses it to hint what we are expecting. "
            "NULL on non-archive requests "
            "and on archive requests that gave no hint."
        )
    )

    # Rate-limit lane
    rate_limit: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc=(
            "Which of the scraper's rate-limit lanes gates this request, as "
            "the lane's code: 0 is the scraper's default ``rate_limits``, 1 "
            "is unlimited (fetches that do not hit the scraped origin), and "
            "2+ index ``named_rate_limits`` in declaration order — see "
            "``jkent.common.rate_limits.RateLimitTable``. The vocabulary is "
            "per scraper, so unlike the CodedEnum columns there is no CHECK "
            "constraint."
        ),
    )

    # Timestamps. Wall clock, millisecond resolution, UTC — see
    # ``database_engine.timestamps``. These are both the human-readable record
    # and what durations are measured from.
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Enqueue time (UTC, millisecond resolution).",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        doc=(
            "Dispatch time, same format as ``created_at``. Re-stamped after "
            "the rate-limit gate so queue wait is not counted as fetch time. "
            "NULL until claimed."
        )
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        doc=(
            "Terminal time, same format as ``created_at``. Subtract "
            "``started_at`` — via ``timestamps.epoch_seconds`` — for the "
            "request duration. NULL until done."
        )
    )

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc="Transient retries already attempted for this request.",
    )
    cumulative_backoff: Mapped[float] = mapped_column(
        default=0.0,
        server_default=sa.text("0.0"),
        doc=(
            "Seconds of backoff this request has been delayed in total, "
            "checked against the run's ``max_backoff_time`` to decide when "
            "retrying stops."
        ),
    )
    last_error: Mapped[str | None] = mapped_column(
        doc=(
            "Message from the most recent failed attempt, for triage without "
            "a join. The full record is a row in ``errors``."
        )
    )

    # Parent tracking. ON DELETE CASCADE makes deleting a request drop its whole
    # subtree (self-referential) in one statement.
    parent_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc=(
            "Request whose step queued this one. NULL on the seed "
            "requests a run starts from."
        ),
    )

    # Speculation tracking
    is_speculative: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "This request was guessed at by the ``Speculative`` protocol "
            "rather than observed in a response, so a miss is expected and "
            "is not an error."
        ),
    )
    speculation_tracking_id: Mapped[int | None] = mapped_column(
        ForeignKey("speculation_tracking.id"),
        doc=(
            "Speculation template that produced this request. The tracking "
            "row is upserted before its probes are enqueued, so this FK is "
            "always resolvable. NULL on non-speculative requests."
        ),
    )
    speculative_index: Mapped[int | None] = mapped_column(
        doc=(
            "Where in its template's sequence this probe sits — the integer "
            "passed to ``Speculative.from_int()``. NULL on non-speculative "
            "requests."
        )
    )

    # --- Response fields (populated when response is received) ---
    # NULL response_status_code means no response has been stored yet.
    response_status_code: Mapped[int | None] = mapped_column(
        doc=(
            "HTTP status of the stored response. NULL means no response has "
            "been stored yet — this is the marker code checks for presence "
            "with."
        )
    )
    response_headers_json: Mapped[str | None] = mapped_column(
        doc="JSON object of response headers."
    )
    response_url: Mapped[str | None] = mapped_column(
        doc=(
            "Final URL the response came from, which differs from ``url`` "
            "when the transport followed redirects."
        )
    )

    # Response timestamps
    response_created_at: Mapped[datetime | None] = mapped_column(
        doc="Response store time, same format as ``created_at``."
    )

    # Speculative request outcome tracking
    speculation_outcome: Mapped[SpeculationOutcome | None] = mapped_column(
        CodedEnumType(SpeculationOutcome),
        doc=(
            "How a speculative request resolved; see "
            ":class:`SpeculationOutcome`. NULL on non-speculative requests."
        ),
    )

    # Playwright via field (ViaFormSubmit / ViaLink JSON)
    via_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON ``ViaFormSubmit`` / ``ViaLink`` describing the page "
            "interaction that produces this request, for the Playwright "
            "transport to re-drive. NULL when the request is issued directly."
        )
    )

    # TLS verification override
    verify_json: Mapped[str] = mapped_column(
        default="true",
        server_default=sa.text("'true'"),
        doc=(
            "JSON ``HTTPRequestParams.verify``: ``true`` / ``false``, or a "
            "CA bundle path as a JSON string."
        ),
    )

    # --- Remaining HTTPRequestParams fields ---
    # The HTTPRequestParams that round-trip through the queue beyond
    # url/body/headers/cookies/verify_json: timeout / json.
    # (Scrapers that set e.g. timeout on archive downloads rely on these being
    # persisted, rather than falling back to the httpx client-level default.)
    timeout_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON ``HTTPRequestParams.timeout`` — a number, or a "
            "``[connect, read]`` pair. NULL uses the client default."
        )
    )
    json_data: Mapped[str | None] = mapped_column(
        doc=(
            "JSON ``HTTPRequestParams.json``, i.e. the body to send as JSON. "
            "Distinct from ``body``, which is already-encoded bytes."
        )
    )
    reseedable: Mapped[bool | None] = mapped_column(
        doc=(
            "Scraper hint for whether this request can be re-fetched "
            "standalone: true = stateless, false = depends on "
            "server-mirrored client state, NULL = unspecified."
        )
    )

    # A pre-resolved request already carries its response, populated at enqueue
    # time by promoting a captured incidental sub-request (see the driver's
    # ``Request.incidental`` handling). The worker dequeues it like any pending
    # request but skips the transport entirely and runs the step
    # against the already-stored response. Distinct from "a response happens to
    # be present" — a retry may store a debug snapshot on a still-pending row —
    # so it is an explicit flag rather than inferred from response presence.
    # sort_order=1 puts this after CompressedPayloadColumns' own columns, so
    # it is the last column of ``requests`` (mixin columns otherwise sort
    # after every column declared here).
    preresolved: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        sort_order=1,
        doc=(
            "This row's response was populated at enqueue time by promoting "
            "a captured incidental sub-request, so the worker runs the "
            "step and skips the transport entirely. An explicit flag "
            "rather than inferred from response presence, because a retry "
            "may store a debug snapshot on a still-pending row."
        ),
    )


class CompressionDict(Base):
    """Versioned zstd compression dictionaries per-step.

    Pages fetched by the same step share boilerplate, so a dictionary
    trained on a sample of them compresses each far better than standalone
    zstd. Rows are versioned rather than replaced: a stored response names the
    dictionary it was compressed against, so retraining must not invalidate
    what is already on disk.
    """

    __tablename__ = "compression_dicts"
    # The UNIQUE constraint's automatic index also serves step lookups
    # (a leading prefix), so a separate index on step would be redundant.
    __table_args__ = (sa.UniqueConstraint("step", "version"),)

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Surrogate key, referenced by the compressed-content columns.",
    )
    step: Mapped[str] = mapped_column(
        doc="Step method name whose responses this was trained on."
    )
    version: Mapped[int] = mapped_column(
        doc=(
            "Generation counter within a step, incremented on each "
            "retrain. Unique with ``step``."
        )
    )
    dictionary_data: Mapped[bytes] = mapped_column(
        LargeBinary, doc="The trained zstd dictionary itself."
    )
    sample_count: Mapped[int] = mapped_column(
        doc="Number of responses trained on, for deciding when to retrain."
    )
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Train time (UTC, millisecond resolution).",
    )


class Result(Base):
    """Validated scraped data results.

    One row per record a step emitted. Rows that failed validation are
    stored too, flagged rather than dropped, so a schema drift shows up as
    inspectable data instead of as an absence.
    """

    __tablename__ = "results"
    __table_args__ = (
        sa.Index("idx_results_type", "result_type"),
        sa.Index("idx_results_request", "request_id"),
        *json_checks("results", "data_json", "validation_errors_json"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    request_id: Mapped[int] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc=(
            "Request whose step emitted this record. NOT NULL: every "
            "writer already supplies one — ``store_result``, "
            "``store_result_in_session`` and the staging path all take a "
            "non-optional ``request_id`` — and ON DELETE CASCADE means a "
            "result cannot outlive the request it came from. Matches "
            "``archived_files.request_id``, which is the same relationship."
        ),
    )

    # Result data
    result_type: Mapped[str] = mapped_column(
        doc=(
            "``type(data).__name__`` of the emitted model — the scraper's "
            "own Pydantic class name. Open vocabulary by construction: every "
            "scraper defines its own models, so the set is not knowable here."
        )
    )
    data_json: Mapped[str] = mapped_column(
        doc=(
            "The record as JSON. Written whether or not it validated, so an "
            "invalid row can be inspected rather than only counted."
        )
    )

    # Validation status
    is_valid: Mapped[bool] = mapped_column(
        default=True,
        server_default=sa.text("1"),
        doc=(
            "Whether the record passed its model's validation. Redundant "
            "with ``validation_errors_json IS NULL`` and kept anyway: it is "
            "what the stats queries group by, and an indexable boolean beats "
            "a nullability test over a text column."
        ),
    )
    validation_errors_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON list of Pydantic error dicts when validation failed; NULL "
            "when it passed."
        )
    )

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Store time (UTC, millisecond resolution).",
    )


class ArchivedFile(Base):
    """Downloaded file metadata.

    File bytes live on disk, not here; this table is the index that maps a
    downloaded file back to the request that fetched it.
    """

    __tablename__ = "archived_files"
    __table_args__ = (
        sa.Index("idx_archived_files_request", "request_id"),
        sa.Index("idx_archived_files_hash", "content_hash"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    request_id: Mapped[int] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc="The archive request that downloaded this file.",
    )

    # File info
    file_path: Mapped[str] = mapped_column(
        doc="Path the bytes were written to. The file itself is not in the DB."
    )
    original_url: Mapped[str] = mapped_column(
        doc="URL the file was fetched from."
    )
    expected_type: Mapped[str | None] = mapped_column(
        doc=(
            "The requesting ``Request``'s ``expected_type`` hint, copied here "
            "so the archive index stands on its own."
        )
    )
    file_size: Mapped[int | None] = mapped_column(
        doc="Size in bytes of the file as written."
    )
    content_hash: Mapped[str | None] = mapped_column(
        doc=(
            "Hex digest of the file contents, for detecting that two "
            "requests fetched identical bytes."
        )
    )

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Store time (UTC, millisecond resolution).",
    )


class RunMetadata(Base):
    """Single-row run configuration and state.

    Holds what the run was invoked with and how it ended. Constrained to
    ``id = 1``: one database is one run, and the check makes a second row a
    write error rather than a silently-ignored ambiguity.
    """

    __tablename__ = "run_metadata"
    __table_args__ = (
        sa.CheckConstraint("id = 1", name="run_metadata_single_row"),
        code_check("status", RunStatus, "ck_run_metadata_status"),
        *json_checks(
            "run_metadata",
            "seed_params_json",
            "browser_cookies_json",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Always 1, enforced by the ``run_metadata_single_row`` check.",
    )

    # Scraper identity
    scraper_name: Mapped[str] = mapped_column(
        doc="Class name of the scraper this run drove."
    )
    scraper_version: Mapped[str | None] = mapped_column(
        doc=(
            "The scraper's declared version, so a corpus records which "
            "revision produced it."
        )
    )

    # Run state
    status: Mapped[RunStatus] = mapped_column(
        CodedEnumType(RunStatus),
        default=RunStatus.CREATED,
        server_default=sa.text(str(RunStatus.CREATED.code)),
        doc="State of the run as a whole; see :class:`RunStatus`.",
    )
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Database creation time (UTC, millisecond resolution).",
    )
    started_at: Mapped[datetime | None] = mapped_column(
        doc="When the run first entered ``running``. NULL if never started."
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        doc=(
            "When the run reached a terminal status. Also written by the "
            "best-effort close path, so it can be set while ``status`` is "
            "still ``running``."
        )
    )
    error_message: Mapped[str | None] = mapped_column(
        doc="The failure text when ``status`` is ``error``; NULL otherwise."
    )

    # Invocation parameters
    seed_params_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON of the seed parameter sets the run was created with. "
            "Written once; a run database is pinned to its seed set."
        )
    )
    jitter: Mapped[float] = mapped_column(
        doc=(
            "Fraction of a retry's backoff drawn as jitter, applied "
            "symmetrically — 0.05 spreads each wait across +/-5% of nominal. "
            "Recorded so a corpus says how spread out its retries were; the "
            "drawn value itself is not stored anywhere, it is folded into "
            "``requests.started_at``. See ``ResponseStorageDB.handle_retry``."
        )
    )
    num_workers: Mapped[int] = mapped_column(
        doc="Size of the worker pool this run was configured with."
    )
    max_backoff_time: Mapped[float] = mapped_column(
        doc=(
            "Ceiling on a request's ``cumulative_backoff`` before retrying "
            "stops."
        )
    )

    # Browser cookie persistence (Playwright driver, for resume)
    browser_cookies_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON browser cookie jar, checkpointed so a resumed run "
            "restarts with the session it had rather than a fresh one."
        )
    )


class Error(Base):
    """Detailed error tracking with type-specific fields.

    A row per raised exception, with the exception's own structured fields
    lifted into columns so failures are queryable — "which selector broke, on
    how many pages" — rather than only greppable. Type-specific columns are
    NULL for error types that do not carry them.
    """

    __tablename__ = "errors"
    __table_args__ = (
        sa.Index("idx_errors_request", "request_id"),
        sa.Index("idx_errors_type", "error_type"),
        code_check("error_type", ErrorType, "ck_errors_error_type"),
        code_check("selector_type", SelectorType, "ck_errors_selector_type"),
        code_check("kind", TransientKind, "ck_errors_kind"),
        *json_checks(
            "errors",
            "context_json",
            "validation_errors_json",
            "failed_doc_json",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    request_id: Mapped[int | None] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc=(
            "Request being processed when this was raised. NULL for errors "
            "raised outside a request context."
        ),
    )

    # Error classification
    error_type: Mapped[ErrorType] = mapped_column(
        CodedEnumType(ErrorType),
        doc=(
            "Bucket from ``errors.classify_error``; see :class:`ErrorType`. "
            "Decides which of the type-specific columns below are populated."
        ),
    )
    error_class: Mapped[str] = mapped_column(
        doc="``type(exc).__name__`` — the concrete exception class."
    )
    message: Mapped[str] = mapped_column(doc="The exception's message.")
    request_url: Mapped[str] = mapped_column(
        doc=(
            "URL that triggered the error, or the literal ``unknown`` when "
            "neither the exception nor the caller supplied one."
        )
    )

    # Structured error data
    context_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON of a ``ScraperAssumptionException``'s context dict, dumped "
            "with ``default=str`` since it can hold arbitrary scraped values."
        )
    )

    # Structural errors (HTMLStructuralAssumptionException)
    selector: Mapped[str | None] = mapped_column(
        doc="Structural errors: the selector whose match count was wrong."
    )
    selector_type: Mapped[SelectorType | None] = mapped_column(
        CodedEnumType(SelectorType),
        doc=(
            "Structural errors: the grammar ``selector`` is written in; see "
            ":class:`SelectorType`. NULL on every other error type."
        ),
    )
    expected_min: Mapped[int | None] = mapped_column(
        doc="Structural errors: minimum matches the scraper asserted."
    )
    expected_max: Mapped[int | None] = mapped_column(
        doc="Structural errors: maximum matches the scraper asserted."
    )
    actual_count: Mapped[int | None] = mapped_column(
        doc="Structural errors: matches the selector actually found."
    )

    # Validation errors (DataFormatAssumptionException)
    model_name: Mapped[str | None] = mapped_column(
        doc="Validation errors: the Pydantic model that rejected the record."
    )
    validation_errors_json: Mapped[str | None] = mapped_column(
        doc="Validation errors: JSON list of Pydantic error dicts."
    )
    failed_doc_json: Mapped[str | None] = mapped_column(
        doc=(
            "Validation errors: JSON of the record that failed, so the input "
            "is inspectable next to the complaint about it."
        )
    )

    # HTTP and timeout errors (HTTPResponseAssumptionException,
    # PersistentHTTPResponseException, RequestTimeoutException)
    status_code: Mapped[int | None] = mapped_column(
        doc=(
            "HTTP errors: the response status that raised. Populated for "
            "both ``transient`` and ``persistent`` ``error_type`` — "
            "``PersistentHTTPResponseException`` carries a status as well."
        )
    )
    timeout_seconds: Mapped[float | None] = mapped_column(
        doc="Timeout errors: the deadline that was exceeded."
    )

    # Stack trace
    traceback: Mapped[str | None] = mapped_column(
        doc="Formatted traceback, when one was captured."
    )

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Store time (UTC, millisecond resolution).",
    )

    kind: Mapped[TransientKind | None] = mapped_column(
        CodedEnumType(TransientKind),
        doc=(
            "Transient errors: which subsystem failed, from "
            ":class:`TransientKind`. NULL for persistent error types and "
            "for any transient exception raised without a ``kind``."
        ),
    )


class SpeculationTracking(Base):
    """Tracks speculation state for Speculative protocol entries.

    One row per speculation template. Persisted so a resumed run picks up
    probing where it left off instead of re-walking a sequence it has already
    established the end of.
    """

    __tablename__ = "speculation_tracking"
    # func_name is UNIQUE (below); its automatic index serves the lookups.
    __table_args__ = (
        *json_checks(
            "speculation_tracking", "template_json", "seed_value_json"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    func_name: Mapped[str] = mapped_column(
        unique=True,
        doc=(
            "State key of the ``Speculative`` entry point this state belongs "
            "to, ``{entry_name}:{param_index}``. Unique — one row per "
            "template. Requests point back here via "
            "``requests.speculation_tracking_id``."
        ),
    )
    highest_successful_id: Mapped[int | None] = mapped_column(
        doc=(
            "Largest speculated index that came back with a real resource. "
            "NULL until one has."
        ),
    )
    consecutive_failures: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc=(
            "Misses since the last success. Reaching the configured "
            "threshold sets ``stopped``."
        ),
    )
    current_ceiling: Mapped[int | None] = mapped_column(
        doc=(
            "Highest index probing is currently allowed to reach. NULL "
            "until the template has been seeded."
        ),
    )
    stopped: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "Probing has given up on this template; probes from it still "
            "queued are completed unfetched, with ``speculation_outcome = "
            "'terminated_early'``."
        ),
    )
    param_index: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc=(
            "Which seed parameter set this row tracks, for a template "
            "speculated independently per seed."
        ),
    )
    template_json: Mapped[str | None] = mapped_column(
        doc="JSON of the request template speculated indices are filled into."
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        onupdate=now_sql(),
        doc=(
            "Last mutation time (UTC, millisecond resolution). ``onupdate`` "
            "re-stamps this on any ordinary UPDATE, but SQLAlchemy does not "
            "apply it to an ON CONFLICT DO UPDATE SET clause — the upsert in "
            "``_speculation.py`` therefore sets it explicitly."
        ),
    )
    # The raw (pre-validation) seed value this template came from, as JSON —
    # e.g. the "[cursor_key]" reference string a host seeded with, so the host
    # can map state rows back to its cursor store without re-deriving jkent's
    # index assignment.
    seed_value_json: Mapped[str | None] = mapped_column(
        doc=(
            "Raw (pre-validation) seed value this template came from, as JSON."
        )
    )


class IncidentalRequestStorage(CompressedPayloadColumns, Base):
    """Content-addressed payload storage for incidental browser requests.

    One row per distinct payload in a crawl, identified solely by
    ``content_md5``. Everything about a *fetch* — url, method, status,
    resource type, request body, failure text — lives on
    ``incidental_requests``, because a payload served to twenty page loads
    is one row here and twenty rows there.
    """

    __tablename__ = "incidental_request_storage"
    __table_args__ = (
        sa.UniqueConstraint("content_md5", name="uq_irs_content_md5"),
        *json_checks("incidental_request_storage", "response_headers_json"),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Surrogate key, referenced by ``incidental_requests.storage_id``.",
    )
    response_headers_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON object of response headers from the capture that stored "
            "the payload first. Not part of this row's identity: later "
            "captures of the same bytes may have carried different headers."
        )
    )
    content_md5: Mapped[bytes] = mapped_column(
        LargeBinary,
        doc=(
            "Raw 16-byte MD5 digest of the body as delivered, before "
            "compression — so one payload is one row whatever dictionary "
            "``content_compressed`` was encoded against. This row's whole "
            "identity, and unique. NOT NULL: a capture with no body gets no "
            "row here at all (``incidental_requests.storage_id`` stays NULL)."
        ),
    )


class IncidentalRequest(Base):
    """Browser-initiated network requests (Playwright driver).

    One row per sub-request a page fired while a scraper request was being
    handled — the traffic the scraper did not ask for but which reveals the
    APIs behind the page. Everything particular to the fetch is here; the
    response *body* lives in ``incidental_request_storage``, keyed by its
    own hash, so a payload repeated across a crawl costs a row here and
    nothing there.
    """

    __tablename__ = "incidental_requests"
    __table_args__ = (
        sa.Index("idx_incidental_requests_parent", "parent_request_id"),
        sa.Index("idx_incidental_requests_storage", "storage_id"),
        *json_checks("incidental_requests", "headers_json"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    parent_request_id: Mapped[int] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc="The scraper request whose page load fired this sub-request.",
    )
    resource_type: Mapped[str] = mapped_column(
        doc=(
            "Playwright's ``request.resource_type`` — ``document``, ``xhr``, "
            "``fetch``, ``script``, and so on. Left open rather than made an "
            "enum: the vocabulary is the browser's, not ours."
        )
    )
    method: Mapped[str] = mapped_column(
        doc=(
            "HTTP method the browser used. Deliberately not the "
            "``http_method`` enum that ``requests.method`` uses: page "
            "JavaScript can issue an arbitrary method, and a ``CHECK`` here "
            "would turn odd traffic into a failed scrape."
        )
    )
    url: Mapped[str] = mapped_column(doc="URL the browser requested.")
    headers_json: Mapped[str | None] = mapped_column(
        doc="JSON object of request headers."
    )
    body: Mapped[bytes | None] = mapped_column(
        LargeBinary, doc="Request body the browser sent, if any."
    )
    status_code: Mapped[int | None] = mapped_column(
        doc="Response status. NULL when the request never got a response."
    )
    failure_reason: Mapped[str | None] = mapped_column(
        doc=(
            "Playwright's failure text when the browser request never "
            "completed. NULL on success."
        )
    )
    # Unlike the dropped ``requests.*_at_ns`` columns these are
    # ``time.time_ns()`` — wall clock on the Unix epoch, so they are
    # comparable across processes. Kept at nanosecond resolution because a
    # page's sub-requests routinely complete inside one millisecond.
    started_at_ns: Mapped[int | None] = mapped_column(
        doc="``time.time_ns()`` when the browser issued the request."
    )
    completed_at_ns: Mapped[int | None] = mapped_column(
        doc="``time.time_ns()`` when the response finished."
    )
    from_cache: Mapped[bool | None] = mapped_column(
        doc=(
            "Whether the response was served without hitting the network. "
            "Sourced from Playwright's ``response.from_service_worker``, the "
            "only cache signal it exposes. NULL when unknown, e.g. on a "
            "request that failed before responding."
        )
    )
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Store time (UTC, millisecond resolution).",
    )
    storage_id: Mapped[int | None] = mapped_column(
        ForeignKey("incidental_request_storage.id"),
        doc=(
            "The content-addressed payload for this occurrence, shared with "
            "every other capture of the same bytes. NULL when there was no "
            "body to store — a failed request, or a response with an empty "
            "one."
        ),
    )
