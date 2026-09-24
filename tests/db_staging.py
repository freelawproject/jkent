"""Shared ``requests``-table staging helpers for the driver tests.

Several rigs need to plant rows in the ``requests`` table by hand: a completed
parent whose cached body stages into a browser tab, or a pending/in-progress
row to act as a foreign-key target. The SQL is schema-coupled, so it lives here
once instead of being copy-pasted per test module (``test_playwright_transport``
and the form-conformance ``harness`` both build on these).

The inserted id comes back via ``RETURNING id``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.compression import compress
from jkent.driver.database_engine.enums import RequestStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def insert_staged_parent(
    sf: async_sessionmaker[AsyncSession], *, url: str, body: bytes
) -> int:
    """Insert a completed parent row whose cached body stages into a tab."""
    compressed = compress(body)
    async with sf() as session:
        row_id = (
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location, response_status_code,
                        response_url, response_headers_json, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES (:status, 9, :method, :url, 'parse', '', 200,
                        :url, NULL, :compressed, :osize, :csize, NULL)
                    RETURNING id
                    """
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": url,
                    "compressed": compressed,
                    "osize": len(body),
                    "csize": len(compressed),
                },
            )
        ).scalar_one()
        await session.commit()
        return row_id


async def insert_request_row(
    sf: async_sessionmaker[AsyncSession], url: str
) -> int:
    """Insert a pending ``requests`` row (the FK target for incidentals)."""
    async with sf() as session:
        row_id = (
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location)
                    VALUES (:status, 9, :method, :url, 'parse', '')
                    RETURNING id
                    """
                ),
                {
                    "status": RequestStatus.IN_PROGRESS.code,
                    "method": HttpMethod.GET.code,
                    "url": url,
                },
            )
        ).scalar_one()
        await session.commit()
        return row_id
