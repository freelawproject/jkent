"""Contract tests for ``Compactor`` (jkent.driver.unified_driver.compaction).

``Compactor`` tracks one scraper step's response count *in memory* (so it
never queries the DB to decide when to act) and, on the call that reaches the
threshold, owns a one-shot job: train a zstd dictionary for the step from its
stored responses and recompress them.

Contract under test:

- In-memory counting: each ``record_request()`` before the threshold bumps the
  count and returns ``False`` without touching the database.
- One-shot at threshold: the call that brings the count to ``threshold``
  trains a dictionary and recompresses the step's responses, then returns
  ``True``.
- Owns the work: after that call a compression dictionary exists for the step
  and every stored response for the step is recompressed against it.
- Inert afterwards: later calls return ``False`` and train no second
  dictionary.
- Below threshold: no dictionary is trained.
- Seeding: a Compactor seeded with ``count`` reaches the threshold after the
  remaining calls (resumed runs continue rather than restart).

The in-memory counting is exercised with hypothesis; the train+recompress
behavior against a real in-memory SQLite database with mock responses.

The generative DB-backed rigs at the bottom build their own in-memory
database per example (a function-scoped fixture cannot be reused across
Hypothesis examples, and the dictionary cache is keyed per engine, so a
fresh engine per example also keeps cached "no dictionary yet" results from
leaking between examples) and drive it with ``asyncio.run``:

- content preservation: after the compactor fires, every stored body
  decompresses (with the row's recorded dictionary) to the bytes inserted,
  and an empty body stays the NULL row the storage rule promises;
  a corpus zstd cannot train on leaves the compactor inert with nothing
  written, and never raises into the worker;
- convergence/idempotence: a first ``recompress_responses`` pass rewrites
  every selected row (``recompressed + skipped == selected``, chunked to
  force the multi-chunk path), a second pass rewrites nothing, and a
  resumed compactor mints no second dictionary;
- concurrency: any burst of ``record_request`` calls straddling the
  threshold trains exactly once and returns ``True`` exactly once.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from hypothesis import assume, given
from hypothesis import strategies as st
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.compression import (
    RecompressStats,
    compress,
    decompress,
    get_compression_dict,
    get_dict_by_id,
    recompress_responses,
    train_compression_dict,
)
from jkent.driver.database_engine.database import get_session_factory
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Base
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.unified_driver import Compactor, compaction

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Similar-but-varied HTML, so zstd has something to train a dictionary on.
_HTML_TEMPLATE = b"""
<html>
  <head><title>Opinion {n}</title></head>
  <body>
    <div class="case-header"><h1>Case Number: {n}</h1></div>
    <div class="opinion">
      <p>The court finds that the defendant in matter {n} is liable for
      damages. The plaintiff's motion for summary judgment is granted.
      The parties are John Doe and Jane Smith, 123 Main Street.</p>
    </div>
  </body>
</html>
"""


async def _insert_responses(
    session_factory: "async_sessionmaker[AsyncSession]", step: str, count: int
) -> None:
    """Insert ``count`` mock completed responses for ``step``."""
    async with session_factory() as session:
        for i in range(count):
            content = _HTML_TEMPLATE.replace(b"{n}", str(i).encode())
            compressed = compress(content)
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location, response_status_code,
                        response_url, content_compressed, content_size_original,
                        content_size_compressed, compression_dict_id)
                    VALUES (:status, 9, :method, :url, :cont, '', 200,
                        :url, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": f"https://example.com/{step}/{i}",
                    "cont": step,
                    "compressed": compressed,
                    "osize": len(content),
                    "csize": len(compressed),
                },
            )
        await session.commit()


# --- In-memory counting (no DB) ------------------------------------------


@pytest.mark.generative
@given(
    threshold=st.integers(min_value=2, max_value=200),
    seed=st.integers(min_value=0, max_value=150),
    data=st.data(),
)
def test_counts_in_memory_below_threshold(
    threshold: int, seed: int, data: st.DataObject
) -> None:
    assume(seed < threshold)
    calls = data.draw(st.integers(min_value=0, max_value=threshold - seed - 1))

    async def drive() -> tuple[Compactor, list[bool]]:
        # The database is never touched below the threshold.
        c = Compactor(
            "parse",
            cast("SQLManager", None),
            threshold=threshold,
            count=seed,
        )
        results = [await c.record_request() for _ in range(calls)]
        return c, results

    c, results = asyncio.run(drive())

    assert results == [False] * calls
    assert c.count == seed + calls
    assert c.done is False


# --- Train + recompress against a real in-memory database ----------------
#
# The threshold-crossing, post-training and seeded cases are specific draws
# of the generative rigs below; only the below-threshold case, which those
# rigs do not reach, is written out by hand.


async def test_below_threshold_does_not_train(
    memory_session_factory: "async_sessionmaker[AsyncSession]",
) -> None:
    sf = memory_session_factory
    db = SQLManager(sf.kw["bind"], sf)
    await _insert_responses(sf, "parse", 5)

    c = Compactor("parse", db, threshold=10)
    for _ in range(5):
        assert await c.record_request() is False

    assert c.done is False
    assert c.count == 5
    assert await get_compression_dict(db, "parse") is None


async def test_recompress_refuses_another_steps_dictionary() -> None:
    """``dict_id`` must name a dictionary trained for ``step``.

    Accepted, it rewrote every ``parse`` row against ``detail``'s
    dictionary and stamped them, so ``parse``'s own compaction later found
    them all off-dictionary and rewrote the step a second time.
    """
    async with _memory_db() as db:
        sf = db.session_factory
        await _insert_bodies(sf, "detail", [_html_body(i) for i in range(20)])
        detail_dict_id = await train_compression_dict(
            db, "detail", sample_limit=20, dict_size=32768
        )
        await _insert_bodies(sf, "parse", [_html_body(i) for i in range(5)])
        before = await _stored_rows(sf, "parse")

        with pytest.raises(ValueError, match="detail"):
            await recompress_responses(db, "parse", dict_id=detail_dict_id)

        assert await _stored_rows(sf, "parse") == before


# --- Generative rigs over a per-example in-memory database ---------------


@asynccontextmanager
async def _memory_db() -> AsyncIterator[SQLManager]:
    """A fresh initialized in-memory SQLite DB (the ``memory_session_factory``
    fixture's shape, rebuilt per Hypothesis example)."""
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn: Any, _record: Any) -> None:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield SQLManager(engine, get_session_factory(engine))
    finally:
        await engine.dispose()


def _html_body(i: int) -> bytes:
    return _HTML_TEMPLATE.replace(b"{n}", str(i).encode())


async def _insert_bodies(
    session_factory: "async_sessionmaker[AsyncSession]",
    step: str,
    bodies: list[bytes],
) -> list[int]:
    """Insert one completed row per body, dictionary-less, in order.

    Follows the storage rule (``ResponseStorage.store_response``): an empty
    body is stored as a NULL ``content_compressed`` (never ``b""``) with a
    zero compressed size, so it is never fed to zstd. Returns the row ids in
    insertion order.
    """
    ids: list[int] = []
    async with session_factory() as session:
        for i, body in enumerate(bodies):
            compressed = compress(body) if body else None
            result = await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location, response_status_code,
                        response_url, content_compressed, content_size_original,
                        content_size_compressed, compression_dict_id)
                    VALUES (:status, 9, :method, :url, :cont, '', 200,
                        :url, :compressed, :osize, :csize, NULL)
                    RETURNING id
                    """
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": f"https://example.com/{step}/{i}",
                    "cont": step,
                    "compressed": compressed,
                    "osize": len(body),
                    "csize": len(compressed) if compressed is not None else 0,
                },
            )
            ids.append(result.scalar_one())
        await session.commit()
    return ids


async def _stored_rows(
    session_factory: "async_sessionmaker[AsyncSession]", step: str
) -> list[tuple[int, bytes | None, int | None]]:
    """``(id, content_compressed, compression_dict_id)`` per row, by id."""
    async with session_factory() as session:
        rows = await session.execute(
            sa.text(
                "SELECT id, content_compressed, compression_dict_id "
                "FROM requests WHERE step = :step ORDER BY id"
            ),
            {"step": step},
        )
        return [(row[0], row[1], row[2]) for row in rows.all()]


async def _dict_count(
    session_factory: "async_sessionmaker[AsyncSession]", step: str
) -> int:
    async with session_factory() as session:
        return (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM compression_dicts WHERE step = :step"
                ),
                {"step": step},
            )
        ).scalar_one()


@st.composite
def _mixed_corpora(draw: st.DrawFn) -> tuple[list[bytes], int]:
    """(bodies, sample_limit): 2..40 bodies — arbitrary binary blobs, a few
    similar HTML pages so training has material, and always one empty body
    (the NULL-content row) — in a drawn order, with a sample limit in
    ``1..len(bodies)``."""
    n = draw(st.integers(min_value=2, max_value=40))
    html_count = draw(st.integers(min_value=0, max_value=n - 1))
    binary_count = n - 1 - html_count
    bodies = (
        [_html_body(i) for i in range(html_count)]
        + [b""]
        + draw(
            st.lists(
                st.binary(min_size=0, max_size=300),
                min_size=binary_count,
                max_size=binary_count,
            )
        )
    )
    bodies = draw(st.permutations(bodies))
    sample_limit = draw(st.integers(min_value=1, max_value=n))
    return bodies, sample_limit


@pytest.mark.generative
@given(corpus=_mixed_corpora())
def test_compaction_preserves_every_stored_body(
    corpus: tuple[list[bytes], int],
) -> None:
    """Every body round-trips through the compactor byte-for-byte.

    The threshold is the row count, so the last ``record_request`` fires
    the train+recompress. zstd cannot train on a tiny or uniform corpus;
    on that arm the call returns ``False`` rather than raising, logs why, and
    the law is: no dictionary row, every stored blob still its original
    dictionary-less encoding.
    """
    bodies, sample_limit = corpus
    n = len(bodies)

    async def drive() -> None:
        async with _memory_db() as db:
            sf = db.session_factory
            ids = await _insert_bodies(sf, "parse", bodies)
            before = await _stored_rows(sf, "parse")
            c = Compactor(
                "parse",
                db,
                threshold=n,
                sample_limit=sample_limit,
                dict_size=32768,
            )
            for _ in range(n - 1):
                assert await c.record_request() is False
            with patch.object(compaction.logger, "warning") as warned:
                fired = await c.record_request()

            latest = await get_compression_dict(db, "parse")
            trained = latest is not None
            assert fired is trained
            # A failed train is logged, never silent; a successful one is not.
            assert warned.called is not trained
            rows = await _stored_rows(sf, "parse")
            assert [row[0] for row in rows] == ids
            if trained:
                assert latest is not None
                assert await _dict_count(sf, "parse") == 1
            else:
                assert latest is None
                assert await _dict_count(sf, "parse") == 0
                assert rows == before, "a failed train must write nothing"

            for (_, blob, dict_id), body in zip(rows, bodies, strict=True):
                if not body:
                    # The NULL-content rule: never b"", never recompressed.
                    assert blob is None and dict_id is None
                    continue
                assert blob is not None
                if trained:
                    assert latest is not None and dict_id == latest.dict_id
                else:
                    assert dict_id is None
                dictionary = (
                    await get_dict_by_id(db, dict_id)
                    if dict_id is not None
                    else None
                )
                assert decompress(blob, dictionary=dictionary) == body

    asyncio.run(drive())


@st.composite
def _trainable_corpora(draw: st.DrawFn) -> list[bytes]:
    """8..40 similar HTML pages (enough for zstd to train on — 7 of the
    template is the observed floor at ``dict_size=32768``), a few arbitrary
    blobs, and one empty body, in a drawn order."""
    html_count = draw(st.integers(min_value=8, max_value=40))
    extras = draw(st.lists(st.binary(min_size=0, max_size=300), max_size=8))
    bodies = [_html_body(i) for i in range(html_count)] + extras + [b""]
    return draw(st.permutations(bodies))


@pytest.mark.generative
@given(
    bodies=_trainable_corpora(),
    chunk_size=st.integers(min_value=1, max_value=5),
)
def test_recompress_converges_then_is_idempotent(
    bodies: list[bytes], chunk_size: int
) -> None:
    """One pass moves every selected row; a second pass is a no-op.

    ``selected`` is what the pass's query picks: rows with a body that are
    not yet on the target dictionary. Chunking at 1..5 rows forces the
    multi-chunk read/recompress/write path (production uses 500).
    """
    n = len(bodies)

    async def drive() -> None:
        async with _memory_db() as db:
            sf = db.session_factory
            await _insert_bodies(sf, "parse", bodies)
            dict_id = await train_compression_dict(
                db,
                "parse",
                sample_limit=n,
                dict_size=32768,
            )
            selected = sum(1 for body in bodies if body)

            first = await recompress_responses(
                db,
                "parse",
                dict_id=dict_id,
                chunk_size=chunk_size,
            )
            assert first.recompressed_count + first.skipped_count == selected
            assert first.skipped_count == 0, "a quiescent pass fully converges"
            assert first.total_original_bytes == sum(len(b) for b in bodies)

            rows = await _stored_rows(sf, "parse")
            assert [row[2] for row in rows] == [
                dict_id if body else None
                for body in bodies  # every body now on the dictionary
            ]

            second = await recompress_responses(
                db,
                "parse",
                dict_id=dict_id,
                chunk_size=chunk_size,
            )
            assert second == RecompressStats(0, 0, 0, 0)
            assert await _stored_rows(sf, "parse") == rows

            # A compactor resumed at the threshold over the compacted step
            # goes inert without training: no second dictionary version.
            resumed = Compactor(
                "parse",
                db,
                threshold=n,
                count=n,
                sample_limit=n,
                dict_size=32768,
            )
            assert await resumed.record_request() is False
            assert resumed.done is True
            assert await _dict_count(sf, "parse") == 1
            latest = await get_compression_dict(db, "parse")
            assert latest is not None and latest.dict_id == dict_id
            assert await _stored_rows(sf, "parse") == rows

    asyncio.run(drive())


@pytest.mark.generative
@given(k=st.integers(min_value=2, max_value=6), data=st.data())
def test_concurrent_burst_across_threshold_trains_once(
    k: int, data: st.DataObject
) -> None:
    """A burst of ``k`` in-flight ``record_request`` calls straddling the
    threshold trains exactly once and exactly one call returns ``True``.

    The compactor is seeded ``j`` (1..k) short of the threshold, so the
    burst's ``j``-th call crosses; the calls before it count, the ones
    after it land while the crossing call is parked inside
    ``_train_and_compact`` — the window a second train could slip into.
    The crossing call is held there until every other call has returned.
    """
    j = data.draw(st.integers(min_value=1, max_value=k), label="crossing")
    n = 20

    async def drive() -> tuple[int, list[bool], int]:
        async with _memory_db() as db:
            sf = db.session_factory
            await _insert_bodies(
                sf, "parse", [_html_body(i) for i in range(n)]
            )
            c = Compactor(
                "parse",
                db,
                threshold=n,
                count=n - j,
                sample_limit=n,
                dict_size=32768,
            )

            real_train = c._train_and_compact
            train_calls = 0
            entered = asyncio.Event()
            release = asyncio.Event()

            async def gated_train() -> bool:
                nonlocal train_calls
                train_calls += 1
                entered.set()
                await release.wait()
                return await real_train()

            c._train_and_compact = gated_train  # type: ignore[method-assign]

            results: list[bool] = []

            async def call() -> None:
                results.append(await c.record_request())

            async def releaser() -> None:
                await entered.wait()
                while len(results) < k - 1:
                    await asyncio.sleep(0)
                release.set()

            await asyncio.gather(*(call() for _ in range(k)), releaser())
            assert c.done is True
            return train_calls, results, await _dict_count(sf, "parse")

    train_calls, results, dict_count = asyncio.run(drive())

    assert train_calls == 1
    assert results.count(True) == 1 and len(results) == k
    assert dict_count == 1
