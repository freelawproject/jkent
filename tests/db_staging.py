"""Shared ``requests``-table staging helpers for the driver tests.

Several rigs need to plant rows in the ``requests`` table by hand: a completed
parent whose cached body stages into a browser tab, or a pending/in-progress
row to act as a foreign-key target. The SQL is schema-coupled, so it lives here
once instead of being copy-pasted per test module (``test_playwright_transport``
and the form-conformance ``harness`` both build on these).

The inserted id comes back via ``RETURNING id`` rather than a follow-up
``SELECT ... WHERE queue_counter = :qc``: ``queue_counter`` is not unique, so a
rig sharing one DB across many rows (the form-conformance harness) could
otherwise match more than one row and raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from jkent.driver.database_engine.compression import compress

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker


async def insert_staged_parent(
    sf: async_sessionmaker, *, url: str, body: bytes, qc: int
) -> int:
    """Insert a completed parent row whose cached body stages into a tab."""
    compressed = compress(body)
    async with sf() as session:
        row_id = (
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location, response_status_code,
                        response_url, response_headers_json, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES ('completed', 9, :qc, 'GET', :url, 'parse', '', 200,
                        :url, NULL, :compressed, :osize, :csize, NULL)
                    RETURNING id
                    """
                ),
                {
                    "url": url,
                    "qc": qc,
                    "compressed": compressed,
                    "osize": len(body),
                    "csize": len(compressed),
                },
            )
        ).scalar_one()
        await session.commit()
        return row_id


async def insert_request_row(sf: async_sessionmaker, url: str, qc: int) -> int:
    """Insert a pending ``requests`` row (the FK target for incidentals)."""
    async with sf() as session:
        row_id = (
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location)
                    VALUES ('in_progress', 9, :qc, 'GET', :url, 'parse', '')
                    RETURNING id
                    """
                ),
                {"qc": qc, "url": url},
            )
        ).scalar_one()
        await session.commit()
        return row_id
