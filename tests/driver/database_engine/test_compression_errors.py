"""Error paths + train/recompress flow for the zstd compression module.

Covers the unhappy paths — missing dictionaries, empty sample sets,
undecompressable samples, skipped-row accounting — plus the full
train -> recompress -> retrain version flow and the paths that run once a
dictionary exists: moving rows onto a newer one, and the concurrent-change
guard.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

import jkent.driver.database_engine.compression as de_compression
from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import SQLManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

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
    step: str,
    count: int,
    *,
    garbage: bool = False,
    body: bytes | None = None,
) -> None:
    """Insert ``count`` completed responses (optionally undecompressable).

    Each is ``body`` when given, else a distinct page of ``_HTML``.
    """
    async with sql_manager.session_factory() as session:
        for i in range(count):
            content = (
                body
                if body is not None
                else _HTML.replace(b"{n}", str(i).encode())
            )
            compressed = (
                b"not-zstd-at-all" if garbage else comp.compress(content)
            )
            await session.execute(
                sa.text(
                    "INSERT INTO requests (status, priority, "
                    "method, url, step, current_location, "
                    "response_status_code, response_url, content_compressed, "
                    "content_size_original, content_size_compressed) "
                    "VALUES (:status, 9, :method, :url, :cont, '', "
                    "200, :url, :compressed, :osize, :csize)"
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": f"https://comp.test/{step}/{i}",
                    "cont": step,
                    "compressed": compressed,
                    "osize": len(content),
                    "csize": len(compressed),
                },
            )
        await session.commit()


async def test_decompress_response_missing_dict(
    sql_manager: SQLManager, comp: Any
) -> None:
    compressed = comp.compress(b"payload")
    with pytest.raises(ValueError, match="Dictionary 12345 not found"):
        await comp.decompress_response(sql_manager, compressed, 12345)


async def test_get_dict_lookups_miss(
    sql_manager: SQLManager, comp: Any
) -> None:
    assert await comp.get_compression_dict(sql_manager, "no_such_step") is None
    assert await comp.get_dict_by_id(sql_manager, 4242) is None


async def test_train_with_no_responses(
    sql_manager: SQLManager, comp: Any
) -> None:
    with pytest.raises(ValueError, match="No responses found"):
        await comp.train_compression_dict(sql_manager, "empty_step")


async def test_train_with_undecompressable_samples(
    sql_manager: SQLManager, comp: Any
) -> None:
    await _insert_responses(sql_manager, comp, "bad_step", 5, garbage=True)
    with pytest.raises(ValueError, match="Could not decompress any samples"):
        await comp.train_compression_dict(sql_manager, "bad_step")


async def test_train_refuses_a_corpus_too_small_to_train_on(
    sql_manager: SQLManager, comp: Any
) -> None:
    """Eight one-byte bodies segfault zstd's trainer; refuse before calling it."""
    await _insert_responses(sql_manager, comp, "tiny", 8, body=b"1")
    with pytest.raises(ValueError, match="too little sample data"):
        await comp.train_compression_dict(sql_manager, "tiny")


async def test_recompress_without_dictionary(
    sql_manager: SQLManager, comp: Any
) -> None:
    with pytest.raises(ValueError, match="No dictionary found for"):
        await comp.recompress_responses(sql_manager, "no_such_step")
    with pytest.raises(ValueError, match="No dictionary found with id"):
        await comp.recompress_responses(
            sql_manager, "no_such_step", dict_id=777
        )


async def test_train_then_recompress_round_trip(
    sql_manager: SQLManager, comp: Any
) -> None:
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "parse", 30)

    dict_id = await comp.train_compression_dict(sql_manager, "parse")
    found = await comp.get_compression_dict(sql_manager, "parse")
    assert found is not None and found[0] == dict_id

    # chunk_size=7 with 30 rows forces several read-recompress-write pages,
    # including a short final one.
    stats = await comp.recompress_responses(sql_manager, "parse", chunk_size=7)
    assert stats.recompressed_count == 30
    assert stats.skipped_count == 0
    assert stats.total_original_bytes > 0
    assert stats.total_compressed_bytes > 0

    # Every row now references the dictionary, its content still
    # decompresses to the original bytes through the dict-aware path, and
    # the size columns describe the rewritten payload.
    async with sf() as session:
        rows = (
            await session.execute(
                sa.text(
                    "SELECT compression_dict_id, content_compressed, url, "
                    "content_size_original, content_size_compressed "
                    "FROM requests ORDER BY id"
                )
            )
        ).all()
    assert len(rows) == 30
    for i, (row_dict_id, compressed, url, osize, csize) in enumerate(rows):
        assert row_dict_id == dict_id
        assert url == f"https://comp.test/parse/{i}"
        expected = _HTML.replace(b"{n}", str(i).encode())
        assert await comp.decompress_response(
            sql_manager, compressed, row_dict_id
        ) == (expected)
        assert (osize, csize) == (len(expected), len(compressed))

    # A second pass against the same dictionary is a no-op: every row is
    # already on the target dictionary, so nothing is rewritten.
    again = await comp.recompress_responses(
        sql_manager, "parse", dict_id=dict_id
    )
    assert again == comp.RecompressStats(0, 0, 0, 0)

    # Retraining stores a new version; the latest wins the lookup.
    second_id = await comp.train_compression_dict(sql_manager, "parse")
    assert second_id != dict_id
    latest = await comp.get_compression_dict(sql_manager, "parse")
    assert latest is not None and latest[0] == second_id


async def test_recompress_reports_skipped_rows(
    sql_manager: SQLManager, comp: Any
) -> None:
    """Rows whose stored bytes can't be processed are counted, not hidden."""
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "mixed", 30)
    await _insert_responses(sql_manager, comp, "mixed", 4, garbage=True)

    # Training tolerates the garbage rows (they're skipped as samples).
    dict_id = await comp.train_compression_dict(sql_manager, "mixed")

    stats = await comp.recompress_responses(
        sql_manager, "mixed", dict_id=dict_id, chunk_size=10
    )
    assert stats.recompressed_count == 30
    assert stats.skipped_count == 4

    # The skipped rows stay off the dictionary: a rerun would retry exactly
    # those and nothing else.
    async with sf() as session:
        remaining = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM requests WHERE step = "
                    "'mixed' AND compression_dict_id IS NULL"
                )
            )
        ).scalar_one()
    assert remaining == 4


def test_decompress_samples_byte_cap(comp: Any) -> None:
    """Sample collection stops once the cumulative byte budget is met."""
    contents = [b"x" * 100 for _ in range(10)]
    rows = [(comp.compress(c), None) for c in contents]
    samples = comp._decompress_samples(rows, {}, "step", 250)
    assert len(samples) == 3
    assert all(s == b"x" * 100 for s in samples)


async def _bodies_by_id(
    sql_manager: SQLManager, comp: Any, step: str
) -> dict[int, tuple[int | None, bytes]]:
    """Each ``step`` row's dictionary id and decompressed body, by row id."""
    sf = sql_manager.session_factory
    async with sf() as session:
        rows = (
            await session.execute(
                sa.text(
                    "SELECT id, compression_dict_id, content_compressed "
                    "FROM requests WHERE step = :step"
                ),
                {"step": step},
            )
        ).all()
    return {
        row[0]: (
            row[1],
            await comp.decompress_response(sql_manager, row[2], row[1]),
        )
        for row in rows
    }


async def test_recompress_moves_rows_off_an_older_dictionary(
    sql_manager: SQLManager, comp: Any
) -> None:
    """Rows on dictionary A land on a newer dictionary B with their bodies
    unchanged."""
    await _insert_responses(sql_manager, comp, "parse", 30)
    originals = {
        rid: body
        for rid, (_, body) in (
            await _bodies_by_id(sql_manager, comp, "parse")
        ).items()
    }
    dict_a = await comp.train_compression_dict(sql_manager, "parse")
    await comp.recompress_responses(sql_manager, "parse", dict_id=dict_a)
    dict_b = await comp.train_compression_dict(sql_manager, "parse")
    assert dict_b != dict_a

    stats = await comp.recompress_responses(
        sql_manager, "parse", dict_id=dict_b, chunk_size=7
    )

    assert stats.recompressed_count == 30
    assert stats.skipped_count == 0
    after = await _bodies_by_id(sql_manager, comp, "parse")
    assert {rid: d for rid, (d, _) in after.items()} == dict.fromkeys(
        originals, dict_b
    )
    assert {rid: body for rid, (_, body) in after.items()} == originals


async def test_recompress_skips_a_row_changed_after_the_read(
    sql_manager: SQLManager, comp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row rewritten between the chunk read and the guarded UPDATE is
    counted as skipped and keeps the concurrent writer's bytes."""
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "parse", 30)
    dict_id = await comp.train_compression_dict(sql_manager, "parse")
    async with sf() as session:
        victim = (
            await session.execute(sa.text("SELECT MIN(id) FROM requests"))
        ).scalar_one()
    concurrent = comp.compress(b"<html>newer content</html>")
    real_write_session = comp.write_session

    @asynccontextmanager
    async def racing_write_session(
        factory: Any, db_lock: asyncio.Lock
    ) -> AsyncIterator[Any]:
        """Land a concurrent write to ``victim`` just before the chunk's."""
        async with real_write_session(factory, db_lock) as session:
            await session.execute(
                sa.text(
                    "UPDATE requests SET content_compressed = :c "
                    "WHERE id = :id"
                ),
                {"c": concurrent, "id": victim},
            )
            await session.commit()
        async with real_write_session(factory, db_lock) as session:
            yield session

    monkeypatch.setattr(comp, "write_session", racing_write_session)

    stats = await comp.recompress_responses(
        sql_manager, "parse", dict_id=dict_id
    )

    assert stats.recompressed_count == 29
    assert stats.skipped_count == 1
    async with sf() as session:
        row = (
            await session.execute(
                sa.text(
                    "SELECT content_compressed, compression_dict_id "
                    "FROM requests WHERE id = :id"
                ),
                {"id": victim},
            )
        ).one()
    assert (row[0], row[1]) == (concurrent, None)
