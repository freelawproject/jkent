"""SQLAlchemy ORM table definitions.

Tables:
- requests: HTTP request queue with status tracking, retry logic, and
  inline response storage (compressed HTTP responses with dictionary refs)
- compression_dicts: Versioned zstd dictionaries per-continuation
- results: Validated scraped data
- archived_files: Downloaded file metadata
- run_metadata: Single-row configuration and state
- errors: Detailed error tracking with type-specific fields
- speculation_tracking: Speculative protocol tracking state
- incidental_request_storage: Deduplicated content for browser requests
- incidental_requests: Browser-initiated network requests (Playwright)
- schema_info: Schema version tracking

Every column is written as an explicit ``mapped_column()`` call rather than a
bare ``Mapped[...]`` annotation: pyre reports a bare annotation as an
uninitialized attribute.

Every column also carries ``doc=``, which is what the reader of a row needs
and is the only place that documentation can live: SQLite has no column
comments, so ``comment=`` would be dropped on the floor, whereas ``doc=``
becomes the mapped attribute's docstring and shows up in editors and in
``help()``.

Columns whose values come from a fixed vocabulary are
:class:`~jkent.driver.database_engine.enums.CodedEnumType`: an ``INTEGER``
column holding the member's ``.code``, mapped back to the member on load. Each
one is paired with a ``code_check`` ``CHECK`` constraint, because an integer
column otherwise records nothing at all about its vocabulary — not even how
many values it has. Reading these columns outside the ORM means decoding with
``<Enum>.from_code()``; a raw ``SELECT status FROM requests`` yields ``1``.

Columns typed as a bare ``str`` are open vocabularies by design — a Python
class name, a Playwright resource type, a caller-supplied hint — and their
``doc=`` says where the value comes from.

Every ``*_json`` column carries a ``json_valid`` ``CHECK`` (see
:func:`json_checks`). They hold serialized text rather than SQLAlchemy's
``JSON`` type deliberately: the query layer owns (de)serialization today, and
``JSON`` would silently double-encode the already-serialized strings that
jent's replay seeding and the queue's own round-trip pass back in. The
constraint is what that choice was missing — it costs nothing and catches a
malformed value at the INSERT instead of a corpus later.

Nullable columns that carry a ``server_default`` deliberately do *not* set
``default=None``. A Python-side default of ``None`` would make the INSERT send
an explicit NULL and suppress the server default, so ``created_at`` and
friends would stop being populated.

Timestamp columns are ``Mapped[datetime | None]``, mapped to SQLAlchemy's
``DateTime`` by ``Base.type_annotation_map`` and so read back as ``datetime``
rather than as text needing ``fromisoformat``. The stored bytes are unchanged:
the value still comes from ``timestamps.now_sql()`` and is still the
millisecond text format documented there. Durations are still computed with
``timestamps.epoch_seconds`` — subtracting two ``DateTime`` columns directly
compiles to a SQL ``-`` between two TEXT values, which SQLite coerces to 0
rather than rejecting, so the ORM-native spelling silently returns 0.0.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import ForeignKey, LargeBinary
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import (
    CodedEnumType,
    RequestStatus,
    RequestType,
    RunStatus,
    SpeculationOutcome,
    code_check,
)
from jkent.driver.database_engine.timestamps import now_sql

__all__ = [
    "ArchivedFile",
    "Base",
    "CompressionDict",
    "Error",
    "IncidentalRequest",
    "IncidentalRequestStorage",
    "Request",
    "RequestStatus",
    "RequestType",
    "Result",
    "RunMetadata",
    "RunStatus",
    "SchemaInfo",
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


class Base(DeclarativeBase):
    """Declarative base owning the metadata that ``create_all`` builds from.

    ``type_annotation_map`` is what lets every timestamp column below be
    declared as a bare ``Mapped[datetime | None]``: without it each one would
    have to name ``sa.DateTime()`` explicitly.
    """

    type_annotation_map = {datetime: sa.DateTime()}


class SchemaInfo(Base):
    """Schema version tracking.

    One row per schema version applied to this database. A fresh database is
    stamped at ``database.BASELINE_VERSION``; there is no migration runner, so
    in practice this table records provenance rather than driving anything.
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


class Request(Base):
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
        sa.Index(
            "idx_requests_status_priority",
            "status",
            "priority",
            "queue_counter",
        ),
        sa.Index("idx_requests_continuation", "continuation"),
        sa.Index("idx_requests_cache_key", "cache_key"),
        sa.Index("idx_requests_parent", "parent_request_id"),
        sa.Index("idx_requests_response_status_code", "response_status_code"),
        sa.Index("idx_requests_compression_dict", "compression_dict_id"),
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
            "files_json",
            "auth_json",
            "proxies_json",
            "cert_json",
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
    queue_counter: Mapped[int] = mapped_column(
        doc=(
            "Monotonic enqueue sequence. Breaks ties within a priority so "
            "dispatch is FIFO rather than dependent on rowid ordering."
        )
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
        doc="HTTP method; see :class:`jkent.data_types.HttpMethod`.",
    )
    url: Mapped[str] = mapped_column(
        doc=(
            "Absolute request URL with query params already folded in — "
            "``HTTPRequestParams.params`` is not stored separately, this is "
            "the single source of truth for the target "
            "(see ``queue.serialize_url_and_body``)."
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
            "Encoded request body, as the transport will send it. NULL for "
            "bodyless requests."
        ),
    )

    # Scraper context
    continuation: Mapped[str] = mapped_column(
        doc=(
            "Name of the scraper method that will be handed the response. "
            "Also the key compression dictionaries are trained per."
        )
    )
    current_location: Mapped[str] = mapped_column(
        default="",
        server_default=sa.text("''"),
        doc=(
            "The scraper's browsing location when this request was enqueued, "
            "for restoring browser state on a resumed or replayed run. Empty "
            "string when unset."
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
    cache_key: Mapped[str | None] = mapped_column(
        doc=(
            "Hex SHA256 over (method, url, body, headers_json) — see "
            "``sql_manager.compute_cache_key``. Identifies requests whose "
            "stored response can be reused. Distinct from "
            "``deduplication_key``: this one is derived from the wire "
            "request, that one is the scraper's declared intent."
        )
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

    # Rate limit bypass
    bypass_rate_limit: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "Skip the rate-limit gate for this request. Set for fetches that "
            "do not hit the scraped origin."
        ),
    )

    # Timestamps. Wall clock, millisecond resolution, UTC — see
    # ``database_engine.timestamps``. These are both the human-readable record
    # and what durations are measured from; the parallel ``*_at_ns`` columns
    # that used to carry monotonic nanoseconds are gone, because their epoch
    # made them incomparable outside the process that wrote them.
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
    # subtree (self-referential) in one statement — relied on by jent's replay
    # stub/skip pruning (ReplayStorage).
    parent_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("requests.id", ondelete="CASCADE"),
        doc=(
            "Request whose continuation queued this one. NULL on the seed "
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
    speculation_id: Mapped[str | None] = mapped_column(
        doc=(
            "JSON array ``[func_name, param_index, spec_id]`` identifying "
            "which speculation template produced this request and where in "
            "its sequence it sits. Joins to ``speculation_tracking`` by "
            "``func_name``. NULL on non-speculative requests."
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

    # Content (compressed)
    content_compressed: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        doc=(
            "zstd-compressed response body, using the dictionary named by "
            "``compression_dict_id`` when one is set."
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

    # Compression metadata
    compression_dict_id: Mapped[int | None] = mapped_column(
        ForeignKey("compression_dicts.id"),
        doc=(
            "Dictionary this body was compressed against. NULL means it was "
            "compressed without one — the case before enough samples had "
            "accumulated for this continuation to train on."
        ),
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
    verify: Mapped[str | None] = mapped_column(
        doc=(
            'Serialized ``HTTPRequestParams.verify``: ``"true"`` / '
            '``"false"`` for the bool forms, or a CA bundle path. NULL '
            "leaves the client default in place."
        )
    )

    # --- Remaining HTTPRequestParams fields ---
    # The full set of HTTPRequestParams that round-trip through the queue:
    # timeout / json / files / auth / allow_redirects / proxies / stream / cert.
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
    files_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON ``HTTPRequestParams.files`` for multipart uploads. Binary "
            "and file-like content is base64-encoded (see "
            "``queue._serialize_files``) and reconstructs as bytes."
        )
    )
    auth_json: Mapped[str | None] = mapped_column(
        doc="JSON ``[username, password]`` for basic auth."
    )
    allow_redirects: Mapped[bool] = mapped_column(
        default=True,
        server_default=sa.text("1"),
        doc="Whether the transport follows 3xx responses for this request.",
    )
    proxies_json: Mapped[str | None] = mapped_column(
        doc="JSON scheme-to-proxy-URL mapping."
    )
    stream: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "Read the body incrementally instead of into memory, for large "
            "archive downloads."
        ),
    )
    cert_json: Mapped[str | None] = mapped_column(
        doc=("JSON client-certificate path, or a ``[cert, key]`` pair.")
    )

    # Request-level field (not part of HTTPRequestParams).
    archive_hash_header: Mapped[str | None] = mapped_column(
        doc=(
            "Name of a response header carrying the file's checksum, so an "
            "archive download can be verified against what the server "
            "claims. NULL when the site offers no such header."
        )
    )

    # reseedable marker: scraper-supplied hint for whether this
    # request can be re-fetched standalone. True = stateless; False = depends
    # on server-mirrored client state; NULL = unspecified. Used by
    # `pdd replay error-stubs` to pick the seed level when re-running errored
    # subtrees.
    reseedable: Mapped[bool | None] = mapped_column(
        doc=(
            "Scraper hint for whether this request can be re-fetched "
            "standalone: true = stateless, false = depends on "
            "server-mirrored client state, NULL = unspecified. Read by "
            "``pdd replay error-stubs`` to pick a seed level when re-running "
            "errored subtrees."
        )
    )

    # A pre-resolved request already carries its response, populated at enqueue
    # time by promoting a captured incidental sub-request (see the driver's
    # ``Request.incidental`` handling). The worker dequeues it like any pending
    # request but skips the transport entirely and runs the continuation
    # against the already-stored response. Distinct from "a response happens to
    # be present" — a retry may store a debug snapshot on a still-pending row —
    # so it is an explicit flag rather than inferred from response presence.
    # Kept LAST so migrations can add it with a plain ALTER ... ADD COLUMN
    # (appended column) without diverging from create_all's column order.
    preresolved: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "This row's response was populated at enqueue time by promoting "
            "a captured incidental sub-request, so the worker runs the "
            "continuation and skips the transport entirely. An explicit flag "
            "rather than inferred from response presence, because a retry "
            "may store a debug snapshot on a still-pending row."
        ),
    )


class CompressionDict(Base):
    """Versioned zstd compression dictionaries per-continuation.

    Pages fetched by the same continuation share boilerplate, so a dictionary
    trained on a sample of them compresses each far better than standalone
    zstd. Rows are versioned rather than replaced: a stored response names the
    dictionary it was compressed against, so retraining must not invalidate
    what is already on disk.
    """

    __tablename__ = "compression_dicts"
    __table_args__ = (
        sa.UniqueConstraint("continuation", "version"),
        sa.Index("idx_compression_dicts_continuation", "continuation"),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Surrogate key, referenced by the compressed-content columns.",
    )
    continuation: Mapped[str] = mapped_column(
        doc="Continuation method name whose responses this was trained on."
    )
    version: Mapped[int] = mapped_column(
        doc=(
            "Generation counter within a continuation, incremented on each "
            "retrain. Unique with ``continuation``."
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

    One row per record a continuation emitted. Rows that failed validation are
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
            "Request whose continuation emitted this record. NOT NULL: every "
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
            "params_json",
            "seed_params_json",
            "speculation_config_json",
            "browser_config_json",
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
    params_json: Mapped[str | None] = mapped_column(
        doc="JSON of the invocation parameters the run was started with."
    )
    seed_params_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON of the seed parameter sets, rewritten by ``--add-params`` "
            "so speculation filtering on resume sees templates added to an "
            "existing run."
        )
    )
    # ``base_delay`` is vestigial: always 0.0, never read for behaviour.
    # Request spacing is the rate limiter's job
    # (``unified_driver.rate_limiter``, driven by the scraper's
    # ``rate_limits``), and retry backoff reads its base from
    # ``ResponseStorage.retry_base_delay``, a constructor argument that is not
    # persisted. The column is NOT NULL so it cannot be dropped without a
    # table rebuild; documented as dead rather than left looking meaningful.
    base_delay: Mapped[float] = mapped_column(
        doc=(
            "Unused. Always 0.0 — request spacing comes from the rate "
            "limiter, not from this column."
        )
    )
    jitter: Mapped[float] = mapped_column(
        doc=(
            "Fraction of a retry's backoff drawn as jitter, applied "
            "symmetrically — 0.05 spreads each wait across +/-5% of nominal. "
            "Recorded so a corpus says how spread out its retries were; the "
            "drawn value itself is not stored anywhere, it is folded into "
            "``requests.started_at``. See ``ResponseStorage.handle_retry``."
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

    # Speculation configuration
    speculation_config_json: Mapped[str | None] = mapped_column(
        doc="JSON speculation tuning (failure thresholds, ceilings)."
    )

    # Browser configuration (Playwright driver)
    browser_config_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON browser configuration for the Playwright driver. NULL for "
            "an httpx-only run."
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

    This is a scraper-development record, not a substitute for application
    error reporting: it is scoped to one run's database, is read by ``jent``
    alongside the responses that produced it, and outlives the process only
    as part of that corpus.
    """

    __tablename__ = "errors"
    __table_args__ = (
        sa.Index("idx_errors_request", "request_id"),
        sa.Index("idx_errors_type", "error_type"),
        sa.Index("idx_errors_unresolved", "is_resolved"),
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
    error_type: Mapped[str] = mapped_column(
        doc=(
            "Bucket from ``errors.classify_error``: ``structural``, "
            "``validation``, ``transient``, ``persistent``, or ``unknown``. "
            "Decides which of the type-specific columns below are populated."
        )
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
    selector_type: Mapped[str | None] = mapped_column(
        doc="Structural errors: the selector dialect (CSS, XPath, ...)."
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

    # Resolution tracking
    is_resolved: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "Triage marker for whether this error has been dealt with. "
            "Indexed by the partial index ``idx_errors_unresolved``."
        ),
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        doc="When ``is_resolved`` was set."
    )
    resolution_notes: Mapped[str | None] = mapped_column(
        doc="Free-text note about how the error was resolved."
    )
    resolution_type: Mapped[str | None] = mapped_column(
        doc=(
            "Free-text label for the kind of resolution. No writer sets this "
            "yet; it exists for triage tooling to fill in."
        )
    )

    # Timestamps
    created_at: Mapped[datetime | None] = mapped_column(
        server_default=now_sql(),
        doc="Store time (UTC, millisecond resolution).",
    )


class SpeculationTracking(Base):
    """Tracks speculation state for Speculative protocol entries.

    One row per speculation template. Persisted so a resumed run picks up
    probing where it left off instead of re-walking a sequence it has already
    established the end of.
    """

    __tablename__ = "speculation_tracking"
    __table_args__ = (
        sa.Index("idx_speculation_tracking_func", "func_name"),
        *json_checks(
            "speculation_tracking", "template_json", "seed_value_json"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, doc="Surrogate key.")
    func_name: Mapped[str] = mapped_column(
        unique=True,
        doc=(
            "Name of the ``Speculative`` entry point this state belongs to. "
            "Unique — one row per template. Matches the first element of a "
            "request's ``speculation_id``."
        ),
    )
    highest_successful_id: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc="Largest speculated index that came back with a real resource.",
    )
    consecutive_failures: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc=(
            "Misses since the last success. Reaching the configured "
            "threshold sets ``stopped``."
        ),
    )
    current_ceiling: Mapped[int] = mapped_column(
        default=0,
        server_default=sa.text("0"),
        doc="Highest index probing is currently allowed to reach.",
    )
    stopped: Mapped[bool] = mapped_column(
        default=False,
        server_default=sa.text("0"),
        doc=(
            "Probing has given up on this template; further requests from it "
            "are recorded with ``speculation_outcome = 'skipped'``."
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
            "Last mutation time (UTC, millisecond resolution). Maintained by "
            "``onupdate``, so any UPDATE to this row re-stamps it without the "
            "writer having to say so."
        ),
    )
    # The raw (pre-validation) seed value this template came from, as JSON —
    # e.g. the "[cursor_key]" reference string a host seeded with, so the host
    # can map state rows back to its cursor store without re-deriving jkent's
    # index assignment. Declared last so migration_reset_v010.py's ALTER
    # append matches create_all's column order.
    seed_value_json: Mapped[str | None] = mapped_column(
        doc=(
            "Raw (pre-validation) seed value this template came from, as JSON."
        )
    )


class IncidentalRequestStorage(Base):
    """Deduplicated content storage for incidental browser requests."""

    __tablename__ = "incidental_request_storage"
    __table_args__ = (
        sa.Index("idx_irs_content_md5", "content_md5"),
        *json_checks("incidental_request_storage", "response_headers_json"),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        doc="Surrogate key, referenced by ``incidental_requests.storage_id``.",
    )
    resource_type: Mapped[str] = mapped_column(
        doc=(
            "Playwright's ``request.resource_type`` — ``document``, ``xhr``, "
            "``fetch``, ``script``, and so on. Left open rather than made an "
            "enum: the vocabulary is the browser's, not ours."
        )
    )
    url: Mapped[str] = mapped_column(doc="URL the browser requested.")
    method: Mapped[str] = mapped_column(
        doc=(
            "HTTP method the browser used. Deliberately not the "
            "``http_method`` enum that ``requests.method`` uses: page "
            "JavaScript can issue an arbitrary method, and a ``CHECK`` here "
            "would turn odd traffic into a failed scrape."
        )
    )
    body: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        doc="Request body, part of this row's identity for deduplication.",
    )
    status_code: Mapped[int | None] = mapped_column(
        doc="Response status. NULL when the request never got a response."
    )
    response_headers_json: Mapped[str | None] = mapped_column(
        doc="JSON object of response headers."
    )
    content_compressed: Mapped[bytes | None] = mapped_column(
        LargeBinary, doc="zstd-compressed response body."
    )
    content_size_original: Mapped[int | None] = mapped_column(
        doc="Uncompressed body length in bytes."
    )
    content_size_compressed: Mapped[int | None] = mapped_column(
        doc="Stored body length in bytes."
    )
    compression_dict_id: Mapped[int | None] = mapped_column(
        ForeignKey("compression_dicts.id"),
        doc="Dictionary this body was compressed against, if any.",
    )
    failure_reason: Mapped[str | None] = mapped_column(
        doc=(
            "Playwright's failure text when the browser request never "
            "completed. NULL on success."
        )
    )
    content_md5: Mapped[str | None] = mapped_column(
        doc=(
            "MD5 of the uncompressed body. Indexed, and used to find an "
            "existing row to reuse instead of storing the payload again."
        )
    )


class IncidentalRequest(Base):
    """Browser-initiated network requests (Playwright driver).

    One row per sub-request a page fired while a scraper request was being
    handled — the traffic the scraper did not ask for but which reveals the
    APIs behind the page. Content lives in ``incidental_request_storage`` so
    repeats across a crawl cost a row, not a payload.
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
    url: Mapped[str] = mapped_column(
        doc=(
            "URL the browser requested. Duplicated from the storage row so "
            "the capture log reads without a join."
        )
    )
    headers_json: Mapped[str | None] = mapped_column(
        doc=(
            "JSON object of request headers. Per-occurrence, unlike the "
            "shared payload, so it stays on this row."
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
            "The deduplicated payload for this occurrence. NULL when the "
            "content was not stored."
        ),
    )
