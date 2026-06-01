"""v15 → v16 data backfill: migrate incidental_requests content to storage table.

Reads from old incidental_requests columns, creates deduplicated rows in
incidental_request_storage (using content MD5 as dedup key), links them
back via storage_id FK, then recreates the table without the old columns.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import sqlalchemy as sa

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

BATCH_SIZE = 1000

KEPT_COLUMNS = [
    "id",
    "parent_request_id",
    "url",
    "headers_json",
    "started_at_ns",
    "completed_at_ns",
    "from_cache",
    "created_at",
    "storage_id",
]


async def migrate(engine: AsyncEngine) -> bool:
    """Backfill incidental_request_storage from old incidental_requests columns.

    Returns True on success, True (no-op) if old columns already removed.
    """
    # Check if old columns still exist (they won't on a fresh v16 DB)
    async with engine.begin() as conn:
        cols = await conn.run_sync(
            lambda c: [
                row[1]
                for row in c.execute(
                    sa.text("PRAGMA table_info(incidental_requests)")
                ).fetchall()
            ]
        )
        if "content_compressed" not in cols:
            logger.info(
                "Old columns already removed — no data migration needed."
            )
            return True

    # Backfill storage rows
    md5_to_storage_id: dict[str, int] = {}
    total_rows = 0
    deduped = 0
    created = 0

    async with engine.begin() as conn:
        count = await conn.run_sync(
            lambda c: c.execute(
                sa.text(
                    "SELECT COUNT(*) FROM incidental_requests WHERE storage_id IS NULL"
                )
            ).scalar()
        )
        logger.info(f"Rows to migrate: {count}")

        offset = 0
        while True:
            rows = await conn.run_sync(
                lambda c, off=offset: c.execute(  # type: ignore[misc]
                    sa.text(
                        "SELECT id, resource_type, method, url, body, status_code, "
                        "response_headers_json, content_compressed, content_size_original, "
                        "content_size_compressed, compression_dict_id, failure_reason "
                        "FROM incidental_requests WHERE storage_id IS NULL "
                        "ORDER BY id LIMIT :limit OFFSET :offset"
                    ),
                    {"limit": BATCH_SIZE, "offset": off},
                ).fetchall()
            )
            if not rows:
                break

            for row in rows:
                total_rows += 1
                (
                    ir_id,
                    resource_type,
                    method,
                    url,
                    body,
                    status_code,
                    resp_headers,
                    content_compressed,
                    size_orig,
                    size_comp,
                    dict_id,
                    failure,
                ) = row

                content_md5 = None
                if content_compressed is not None:
                    content_md5 = hashlib.md5(content_compressed).hexdigest()

                storage_id = (
                    md5_to_storage_id.get(content_md5) if content_md5 else None
                )

                if storage_id is None:
                    result = await conn.run_sync(
                        lambda c, **kw: (
                            c.execute(
                                sa.text(
                                    "INSERT INTO incidental_request_storage "
                                    "(resource_type, url, method, body, status_code, "
                                    "response_headers_json, content_compressed, "
                                    "content_size_original, content_size_compressed, "
                                    "compression_dict_id, failure_reason, content_md5) "
                                    "VALUES (:resource_type, :url, :method, :body, "
                                    ":status_code, :response_headers_json, "
                                    ":content_compressed, :content_size_original, "
                                    ":content_size_compressed, :compression_dict_id, "
                                    ":failure_reason, :content_md5)"
                                ),
                                kw,
                            ).lastrowid
                        ),
                        resource_type=resource_type or "",
                        url=url or "",
                        method=method or "GET",
                        body=body,
                        status_code=status_code,
                        response_headers_json=resp_headers,
                        content_compressed=content_compressed,
                        content_size_original=size_orig,
                        content_size_compressed=size_comp,
                        compression_dict_id=dict_id,
                        failure_reason=failure,
                        content_md5=content_md5,
                    )
                    storage_id = result
                    if content_md5:
                        md5_to_storage_id[content_md5] = storage_id
                    created += 1
                else:
                    deduped += 1

                await conn.run_sync(
                    lambda c, sid=storage_id, iid=ir_id: c.execute(  # type: ignore[misc]
                        sa.text(
                            "UPDATE incidental_requests SET storage_id = :sid WHERE id = :id"
                        ),
                        {"sid": sid, "id": iid},
                    )
                )

            offset += BATCH_SIZE
            logger.info(f"  Processed {total_rows}/{count} rows...")

    logger.info(
        f"Data migration complete: {total_rows} rows processed, "
        f"{created} storage rows created, {deduped} deduplicated."
    )

    # Drop old columns by recreating the table
    logger.info("Dropping old columns from incidental_requests...")
    kept_cols = ", ".join(KEPT_COLUMNS)
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "CREATE TABLE incidental_requests_new ("
                "  id INTEGER PRIMARY KEY,"
                "  parent_request_id INTEGER NOT NULL REFERENCES requests(id),"
                "  url TEXT NOT NULL,"
                "  headers_json TEXT,"
                "  started_at_ns INTEGER,"
                "  completed_at_ns INTEGER,"
                "  from_cache BOOLEAN,"
                "  created_at TEXT DEFAULT CURRENT_TIMESTAMP,"
                "  storage_id INTEGER REFERENCES incidental_request_storage(id)"
                ")"
            )
        )
        await conn.execute(
            sa.text(
                f"INSERT INTO incidental_requests_new ({kept_cols}) "
                f"SELECT {kept_cols} FROM incidental_requests"
            )
        )
        await conn.execute(sa.text("DROP TABLE incidental_requests"))
        await conn.execute(
            sa.text(
                "ALTER TABLE incidental_requests_new RENAME TO incidental_requests"
            )
        )
        await conn.execute(
            sa.text(
                "CREATE INDEX idx_incidental_requests_parent "
                "ON incidental_requests(parent_request_id)"
            )
        )
        await conn.execute(
            sa.text(
                "CREATE INDEX idx_incidental_requests_storage "
                "ON incidental_requests(storage_id)"
            )
        )

    # Reclaim space
    logger.info("Running VACUUM to reclaim space...")
    async with engine.begin() as conn:
        await conn.execute(sa.text("VACUUM"))

    return True
