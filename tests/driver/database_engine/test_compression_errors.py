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
import zstandard as zstd

import jkent.driver.database_engine.compression as de_compression
from jkent.data_types import HttpMethod
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.sql_manager import SQLManager, StoredResponse
from jkent.driver.database_engine.sql_manager import (
    _compression as de_compression_db,
)

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
    """The pure half of the compression module under test."""
    return de_compression


@pytest.fixture
def comp_db() -> Any:
    """The manager-side half: the ``CompressionMixin`` module."""
    return de_compression_db


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
        await sql_manager.decompress_response(compressed, 12345)


async def test_get_dict_lookups_miss(
    sql_manager: SQLManager, comp: Any
) -> None:
    assert await sql_manager.get_compression_dict("no_such_step") is None
    assert await sql_manager.get_dict_by_id(4242) is None


async def test_cached_dictionaries_are_precomputed_for_compression(
    sql_manager: SQLManager, comp: Any
) -> None:
    """A cached dictionary carries its digest, so compressors reuse it.

    Without the precompute every ``ZstdCompressor`` built on the dictionary
    re-digests the blob, which dominates the cost of compressing a short
    body. Precomputing pins the compression parameters, so the observable
    signature is that ``level`` stops reaching zstd — which is why the
    module compresses at one level and
    :data:`~jkent.driver.database_engine.compression.DEFAULT_COMPRESSION_LEVEL`
    is the level the cache precomputes for.
    """
    await _insert_responses(sql_manager, comp, "parse", 30)
    dict_id = await sql_manager.train_compression_dict("parse")
    body = b"".join(_HTML.replace(b"{n}", str(i).encode()) for i in range(30))

    entry = await sql_manager.get_compression_dict("parse")
    assert entry is not None
    # The same object on every lookup: the digest is paid once, at fill.
    again = await sql_manager.get_compression_dict("parse")
    assert again is not None
    assert again.dictionary is entry.dictionary

    # A plain dictionary lets the level through (so the assertion below is
    # not vacuous); the cached one does not, because it is precomputed.
    plain = zstd.ZstdCompressionDict(entry.dictionary.as_bytes())
    assert comp.compress(body, level=1, dictionary=plain) != comp.compress(
        body, level=19, dictionary=plain
    )
    assert comp.compress(
        body, level=1, dictionary=entry.dictionary
    ) == comp.compress(body, level=19, dictionary=entry.dictionary)

    # The by-id path serves recompression, which compresses a whole chunk
    # per dictionary, so it precomputes too.
    sql_manager.compression_dict_cache.by_id.clear()
    by_id = await sql_manager.get_dict_by_id(dict_id)
    assert by_id is not None
    assert comp.compress(body, level=1, dictionary=by_id) == comp.compress(
        body, level=19, dictionary=by_id
    )

    # And a precomputed dictionary still round-trips against a plain one.
    assert (
        comp.decompress(
            comp.compress(body, dictionary=entry.dictionary), dictionary=plain
        )
        == body
    )


async def test_train_with_no_responses(
    sql_manager: SQLManager, comp: Any
) -> None:
    with pytest.raises(ValueError, match="No responses found"):
        await sql_manager.train_compression_dict("empty_step")


async def test_train_with_undecompressable_samples(
    sql_manager: SQLManager, comp: Any
) -> None:
    await _insert_responses(sql_manager, comp, "bad_step", 5, garbage=True)
    with pytest.raises(ValueError, match="Could not decompress any samples"):
        await sql_manager.train_compression_dict("bad_step")


async def test_train_refuses_a_corpus_too_small_to_train_on(
    sql_manager: SQLManager, comp: Any
) -> None:
    """Eight one-byte bodies segfault zstd's trainer; refuse before calling it."""
    await _insert_responses(sql_manager, comp, "tiny", 8, body=b"1")
    with pytest.raises(ValueError, match="too little sample data"):
        await sql_manager.train_compression_dict("tiny")


async def test_recompress_without_dictionary(
    sql_manager: SQLManager, comp: Any
) -> None:
    with pytest.raises(ValueError, match="No dictionary found for"):
        await sql_manager.recompress_responses("no_such_step")
    with pytest.raises(ValueError, match="No dictionary found with id"):
        await sql_manager.recompress_responses("no_such_step", dict_id=777)


async def test_train_then_recompress_round_trip(
    sql_manager: SQLManager, comp: Any
) -> None:
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "parse", 30)

    dict_id = await sql_manager.train_compression_dict("parse")
    found = await sql_manager.get_compression_dict("parse")
    assert found is not None and found[0] == dict_id

    # chunk_size=7 with 30 rows forces several read-recompress-write pages,
    # including a short final one.
    stats = await sql_manager.recompress_responses("parse", chunk_size=7)
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
        assert await sql_manager.decompress_response(
            compressed, row_dict_id
        ) == (expected)
        assert (osize, csize) == (len(expected), len(compressed))

    # A second pass against the same dictionary is a no-op: every row is
    # already on the target dictionary, so nothing is rewritten.
    again = await sql_manager.recompress_responses("parse", dict_id=dict_id)
    assert again == comp.RecompressStats(0, 0, 0, 0)

    # Retraining stores a new version; the latest wins the lookup.
    second_id = await sql_manager.train_compression_dict("parse")
    assert second_id != dict_id
    latest = await sql_manager.get_compression_dict("parse")
    assert latest is not None and latest[0] == second_id


async def test_recompress_rejects_another_steps_dictionary(
    sql_manager: SQLManager, comp: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dict_id trained for another step is refused, from the cache alone.

    The cached entry carries its step, so the guard needs no query of its
    own once the dictionary has been seen.
    """
    await _insert_responses(sql_manager, comp, "parse", 30)
    await _insert_responses(sql_manager, comp, "other", 30)
    await sql_manager.train_compression_dict("parse")
    other_id = await sql_manager.train_compression_dict("other")

    def no_session() -> Any:
        raise AssertionError("the step check should be a cache hit")

    monkeypatch.setattr(sql_manager, "session_factory", no_session)
    with pytest.raises(ValueError, match="trained for step 'other'"):
        await sql_manager.recompress_responses("parse", dict_id=other_id)

    # A cold cache reads the step along with the dictionary.
    monkeypatch.undo()
    sql_manager.compression_dict_cache.by_id.clear()
    with pytest.raises(ValueError, match="trained for step 'other'"):
        await sql_manager.recompress_responses("parse", dict_id=other_id)
    assert sql_manager.compression_dict_cache.by_id[other_id].step == "other"


async def test_recompress_reports_skipped_rows(
    sql_manager: SQLManager, comp: Any
) -> None:
    """Rows whose stored bytes can't be processed are counted, not hidden."""
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "mixed", 30)
    await _insert_responses(sql_manager, comp, "mixed", 4, garbage=True)

    # Training tolerates the garbage rows (they're skipped as samples).
    dict_id = await sql_manager.train_compression_dict("mixed")

    stats = await sql_manager.recompress_responses(
        "mixed", dict_id=dict_id, chunk_size=10
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


async def test_recompress_stops_when_a_page_cannot_be_rewritten(
    sql_manager: SQLManager, comp: Any
) -> None:
    """A page that produces no write ends the pass instead of spinning.

    Pages are "the next rows still off the dictionary", so the writes are
    what advance the walk. Rows that can never be rewritten stay in the
    predicate and would be handed back unchanged forever.
    """
    await _insert_responses(sql_manager, comp, "stuck", 30)
    dict_id = await sql_manager.train_compression_dict("stuck")
    assert (await sql_manager.recompress_responses("stuck")).skipped_count == 0

    # Now every row left off the dictionary is one no pass can rewrite.
    await _insert_responses(sql_manager, comp, "stuck", 3, garbage=True)
    assert await sql_manager.count_off_dictionary("stuck", dict_id) == 3

    stats = await asyncio.wait_for(
        sql_manager.recompress_responses("stuck", dict_id=dict_id),
        timeout=10,
    )

    assert stats.recompressed_count == 0
    assert stats.skipped_count == 3
    # Still off the dictionary, and still reported: this is the shape
    # ``Compactor.for_step`` logs as a step that cannot converge.
    assert await sql_manager.count_off_dictionary("stuck", dict_id) == 3


def test_decompress_samples_byte_cap(comp: Any) -> None:
    """Sample collection stops once the cumulative byte budget is met."""
    contents = [b"x" * 100 for _ in range(10)]
    rows = [(comp.compress(c), None) for c in contents]
    samples = comp._decompress_samples(rows, {}, "step", 250)
    assert len(samples) == 3
    assert all(s == b"x" * 100 for s in samples)


async def test_collect_training_samples_stops_at_the_byte_cap(
    sql_manager: SQLManager,
    comp: Any,
    comp_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection stops mid-walk, so the ids past the cap are never read.

    Batched so that the cap lands inside the second batch: the point of
    the walk is that the bodies behind the remaining ids are never
    fetched, not merely never decompressed.
    """
    await _insert_responses(sql_manager, comp, "capped", 40)
    monkeypatch.setattr(comp_db, "TRAIN_SAMPLE_BATCH_SIZE", 10)

    async with sql_manager.session_factory() as session:
        sample_ids = list(
            (
                await session.execute(
                    sa.text(
                        "SELECT id FROM requests WHERE step = 'capped' "
                        "ORDER BY id"
                    )
                )
            ).scalars()
        )
    assert len(sample_ids) == 40

    reads: list[int] = []
    original = comp_db._decompress_samples

    def counting(rows: Any, *args: Any) -> Any:
        reads.append(len(rows))
        return original(rows, *args)

    monkeypatch.setattr(comp_db, "_decompress_samples", counting)

    body_size = len(_HTML.replace(b"{n}", b"0"))
    samples = await sql_manager._collect_training_samples(
        "capped", sample_ids, body_size * 15
    )

    # Two batches read, not four: the walk stopped once the cap was met.
    assert reads == [10, 10]
    assert sum(len(sample) for sample in samples) >= body_size * 15
    assert len(samples) < len(sample_ids)


async def test_training_seeds_the_dictionary_cache(
    sql_manager: SQLManager, comp: Any
) -> None:
    """The dictionary a run just trained is cached, not merely invalidated.

    The compactor recompresses against it immediately, so making the next
    lookup a cache hit saves that pass a read and a precompute.
    """
    await _insert_responses(sql_manager, comp, "parse", 30)
    dict_id = await sql_manager.train_compression_dict("parse")

    cache = sql_manager.compression_dict_cache
    assert cache.latest["parse"] is not None
    assert cache.latest["parse"].dict_id == dict_id
    assert cache.by_id[dict_id] is cache.latest["parse"]
    assert cache.by_id[dict_id].step == "parse"

    # Seeded, not just stored: the cached entry is precomputed like any
    # other, so the level no longer reaches zstd.
    body = b"".join(_HTML.replace(b"{n}", str(i).encode()) for i in range(30))
    cached = cache.by_id[dict_id].dictionary
    assert comp.compress(body, level=1, dictionary=cached) == comp.compress(
        body, level=19, dictionary=cached
    )

    # And retraining replaces it rather than leaving the old version.
    second_id = await sql_manager.train_compression_dict("parse")
    assert second_id != dict_id
    assert cache.latest["parse"] is not None
    assert cache.latest["parse"].dict_id == second_id


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
            await sql_manager.decompress_response(row[2], row[1]),
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
    dict_a = await sql_manager.train_compression_dict("parse")
    await sql_manager.recompress_responses("parse", dict_id=dict_a)
    dict_b = await sql_manager.train_compression_dict("parse")
    assert dict_b != dict_a

    stats = await sql_manager.recompress_responses(
        "parse", dict_id=dict_b, chunk_size=7
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
    counted as skipped and is never clobbered with the stale bytes.

    The walk re-reads whatever is still off the dictionary, so the row comes
    back on a later page and is then recompressed from the concurrent
    writer's content. ``skipped_count`` therefore counts the skipped
    *attempt*, not a row left behind.
    """
    sf = sql_manager.session_factory
    await _insert_responses(sql_manager, comp, "parse", 30)
    dict_id = await sql_manager.train_compression_dict("parse")
    async with sf() as session:
        victim = (
            await session.execute(sa.text("SELECT MIN(id) FROM requests"))
        ).scalar_one()
    concurrent = comp.compress(b"<html>newer content</html>")
    real_write_session = sql_manager._write_session

    raced = False

    @asynccontextmanager
    async def racing_write_session() -> AsyncIterator[Any]:
        """Land a concurrent store to ``victim`` just before the first
        chunk's, and only that one: a writer that kept rewriting the row
        every page would just be the same race again.

        Through ``store_response``, the one writer of stored bodies, so the
        row is restamped the way a real re-store restamps it."""
        nonlocal raced
        if not raced:
            raced = True
            await sql_manager.store_response(
                victim,
                StoredResponse(
                    response_status_code=200,
                    response_url="https://comp.test/parse/0",
                    content_compressed=concurrent,
                    content_size_original=len(b"<html>newer content</html>"),
                    content_size_compressed=len(concurrent),
                ),
            )
        async with real_write_session() as session:
            yield session

    monkeypatch.setattr(sql_manager, "_write_session", racing_write_session)

    stats = await sql_manager.recompress_responses("parse", dict_id=dict_id)

    assert stats.recompressed_count == 30
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
    # Not clobbered: the row carries the concurrent writer's content, picked
    # up on a later page and recompressed against the target dictionary.
    assert row[0] != concurrent
    assert row[1] == dict_id
    assert (
        comp.decompress(
            row[0], dictionary=await sql_manager.get_dict_by_id(dict_id)
        )
        == b"<html>newer content</html>"
    )
