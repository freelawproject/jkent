"""Error paths + train/recompress flow for the zstd compression module.

Covers the unhappy paths — missing dictionaries, empty sample sets,
undecompressable samples, skipped-row accounting — plus the full
train -> recompress -> retrain version flow.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import sqlalchemy as sa

import jkent.driver.database_engine.compression as de_compression
from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import SQLManager

_HTML = (
    b"<html><body><h1>Case {n}</h1>"
    b"<p>Lorem ipsum dolor sit amet, the parties are John Doe and "
    b"Jane Smith of 123 Main Street, docket BCC-2024-{n}.</p>"
    b"</body></html>"
)


@pytest.fixture
def comp() -> Any:
    """The compression module under test."""
    return de_compression


async def _insert_responses(
    sql_manager: SQLManager,
    comp: Any,
    continuation: str,
    count: int,
    *,
    garbage: bool = False,
) -> None:
    """Insert ``count`` completed responses (optionally undecompressable)."""
    async with sql_manager.session_factory() as session:
        for i in range(count):
            content = _HTML.replace(b"{n}", str(i).encode())
            compressed = (
                b"not-zstd-at-all" if garbage else comp.compress(content)
            )
            await session.execute(
                sa.text(
                    "INSERT INTO requests (status, priority, queue_counter, "
                    "method, url, continuation, current_location, "
                    "response_status_code, response_url, content_compressed, "
                    "content_size_original, content_size_compressed) "
                    "VALUES (:status, 9, :qc, :method, :url, :cont, '', "
                    "200, :url, :compressed, :osize, :csize)"
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "qc": i + 1,
                    "url": f"https://comp.test/{continuation}/{i}",
                    "cont": continuation,
                    "compressed": compressed,
                    "osize": len(content),
                    "csize": len(compressed),
                },
            )
        await session.commit()


async def test_decompress_response_missing_dict(
    sql_manager: SQLManager, comp
) -> None:
    compressed = comp.compress(b"payload")
    with pytest.raises(ValueError, match="Dictionary 12345 not found"):
        await comp.decompress_response(
            sql_manager.session_factory, compressed, 12345
        )


async def test_get_dict_lookups_miss(sql_manager: SQLManager, comp) -> None:
    sf = sql_manager.session_factory
    assert await comp.get_compression_dict(sf, "no_such_step") is None
    assert await comp.get_dict_by_id(sf, 4242) is None


async def test_train_with_no_responses(sql_manager: SQLManager, comp) -> None:
    with pytest.raises(ValueError, match="No responses found"):
        await comp.train_compression_dict(
            sql_manager.session_factory, "empty_step", db_lock=asyncio.Lock()
        )


async def test_train_with_undecompressable_samples(
    sql_manager: SQLManager, comp
) -> None:
    await _insert_responses(sql_manager, comp, "bad_step", 5, garbage=True)
    with pytest.raises(ValueError, match="Could not decompress any samples"):
        await comp.train_compression_dict(
            sql_manager.session_factory, "bad_step", db_lock=asyncio.Lock()
        )


async def test_recompress_without_dictionary(
    sql_manager: SQLManager, comp
) -> None:
    sf = sql_manager.session_factory
    with pytest.raises(ValueError, match="No dictionary found for"):
        await comp.recompress_responses(
            sf, "no_such_step", db_lock=asyncio.Lock()
        )
    with pytest.raises(ValueError, match="No dictionary found with id"):
        await comp.recompress_responses(
            sf, "no_such_step", dict_id=777, db_lock=asyncio.Lock()
        )


async def test_train_then_recompress_round_trip(
    sql_manager: SQLManager, comp
) -> None:
    sf = sql_manager.session_factory
    lock = asyncio.Lock()
    await _insert_responses(sql_manager, comp, "parse", 30)

    dict_id = await comp.train_compression_dict(sf, "parse", db_lock=lock)
    found = await comp.get_compression_dict(sf, "parse")
    assert found is not None and found[0] == dict_id

    # chunk_size=7 with 30 rows forces several read-recompress-write pages,
    # including a short final one.
    stats = await comp.recompress_responses(
        sf, "parse", db_lock=lock, chunk_size=7
    )
    assert stats.recompressed_count == 30
    assert stats.skipped_count == 0
    assert stats.total_original_bytes > 0
    assert stats.total_compressed_bytes > 0

    # Every row now references the dictionary, and its content still
    # decompresses to the original bytes through the dict-aware path.
    async with sf() as session:
        rows = (
            await session.execute(
                sa.text(
                    "SELECT compression_dict_id, content_compressed, url "
                    "FROM requests"
                )
            )
        ).all()
    assert all(row[0] == dict_id for row in rows)
    sample = rows[0]
    content = await comp.decompress_response(sf, sample[1], sample[0])
    assert b"<html>" in content and b"docket BCC-2024-" in content

    # A second pass against the same dictionary is a no-op: every row is
    # already on the target dictionary, so nothing is rewritten.
    again = await comp.recompress_responses(
        sf, "parse", dict_id=dict_id, db_lock=lock
    )
    assert again == comp.RecompressStats(0, 0, 0, 0)

    # Retraining stores a new version; the latest wins the lookup.
    second_id = await comp.train_compression_dict(sf, "parse", db_lock=lock)
    assert second_id != dict_id
    latest = await comp.get_compression_dict(sf, "parse")
    assert latest is not None and latest[0] == second_id


async def test_recompress_reports_skipped_rows(
    sql_manager: SQLManager, comp
) -> None:
    """Rows whose stored bytes can't be processed are counted, not hidden."""
    sf = sql_manager.session_factory
    lock = asyncio.Lock()
    await _insert_responses(sql_manager, comp, "mixed", 30)
    await _insert_responses(sql_manager, comp, "mixed", 4, garbage=True)

    # Training tolerates the garbage rows (they're skipped as samples).
    dict_id = await comp.train_compression_dict(sf, "mixed", db_lock=lock)

    stats = await comp.recompress_responses(
        sf, "mixed", dict_id=dict_id, db_lock=lock, chunk_size=10
    )
    assert stats.recompressed_count == 30
    assert stats.skipped_count == 4

    # The skipped rows stay off the dictionary: a rerun would retry exactly
    # those and nothing else.
    async with sf() as session:
        remaining = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM requests WHERE continuation = "
                    "'mixed' AND compression_dict_id IS NULL"
                )
            )
        ).scalar_one()
    assert remaining == 4


def test_decompress_samples_byte_cap(comp) -> None:
    """Sample collection stops once the cumulative byte budget is met."""
    contents = [b"x" * 100 for _ in range(10)]
    rows = [(comp.compress(c), None) for c in contents]
    samples = comp._decompress_samples(rows, {}, "step", 250)
    assert len(samples) == 3
    assert all(s == b"x" * 100 for s in samples)
