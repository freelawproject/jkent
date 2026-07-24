"""Contract tests for ``Compactor`` (jkent.driver.unified_driver.orchestration).

``Compactor`` tracks one scraper step's response count *in memory* (so it
never queries the DB to decide when to act) and, on the call that reaches the
threshold, owns a one-shot job: train a zstd dictionary for the step from its
stored responses and recompress them.

Contract under test (see ``compactor_contract.md``):

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
"""

import asyncio
from typing import TYPE_CHECKING, cast

import pytest
import sqlalchemy as sa
from hypothesis import assume, given
from hypothesis import strategies as st

from jkent.driver.database_engine.compression import (
    compress,
    get_compression_dict,
)
from jkent.driver.unified_driver import Compactor

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

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
    session_factory: "async_sessionmaker", continuation: str, count: int
) -> None:
    """Insert ``count`` mock completed responses for ``continuation``."""
    async with session_factory() as session:
        for i in range(count):
            content = _HTML_TEMPLATE.replace(b"{n}", str(i).encode())
            compressed = compress(content)
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, queue_counter, method, url,
                        continuation, current_location, response_status_code,
                        response_url, content_compressed, content_size_original,
                        content_size_compressed, compression_dict_id)
                    VALUES ('completed', 9, :qc, 'GET', :url, :cont, '', 200,
                        :url, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "qc": i + 1,
                    "url": f"https://example.com/{continuation}/{i}",
                    "cont": continuation,
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
        # The session factory is never touched below the threshold.
        c = Compactor(
            "parse",
            cast("async_sessionmaker", None),
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


async def test_trains_and_recompresses_at_threshold(
    memory_session_factory: "async_sessionmaker",
) -> None:
    sf = memory_session_factory
    n = 20
    await _insert_responses(sf, "parse", n)

    c = Compactor("parse", sf, threshold=n, sample_limit=n, dict_size=32768)
    results = [await c.record_request() for _ in range(n)]

    assert results[:-1] == [False] * (n - 1)
    assert results[-1] is True
    assert c.done is True

    dict_result = await get_compression_dict(sf, "parse")
    assert dict_result is not None
    dict_id, _ = dict_result

    async with sf() as session:
        rows = (
            await session.execute(
                sa.text(
                    "SELECT compression_dict_id FROM requests "
                    "WHERE continuation = 'parse' "
                    "AND response_status_code IS NOT NULL"
                )
            )
        ).all()
    assert len(rows) == n
    assert all(row[0] == dict_id for row in rows)


async def test_inert_after_training(
    memory_session_factory: "async_sessionmaker",
) -> None:
    sf = memory_session_factory
    n = 20
    await _insert_responses(sf, "parse", n)

    c = Compactor("parse", sf, threshold=n, sample_limit=n, dict_size=32768)
    for _ in range(n):
        await c.record_request()
    assert c.done is True

    assert await c.record_request() is False
    async with sf() as session:
        dict_count = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM compression_dicts "
                    "WHERE continuation = 'parse'"
                )
            )
        ).scalar_one()
    assert dict_count == 1


async def test_below_threshold_does_not_train(
    memory_session_factory: "async_sessionmaker",
) -> None:
    sf = memory_session_factory
    await _insert_responses(sf, "parse", 5)

    c = Compactor("parse", sf, threshold=10)
    for _ in range(5):
        assert await c.record_request() is False

    assert c.done is False
    assert c.count == 5
    assert await get_compression_dict(sf, "parse") is None


async def test_seeded_compactor_trains_after_remaining_calls(
    memory_session_factory: "async_sessionmaker",
) -> None:
    sf = memory_session_factory
    n = 20
    await _insert_responses(sf, "parse", n)

    c = Compactor(
        "parse", sf, threshold=n, count=n - 2, sample_limit=n, dict_size=32768
    )
    assert await c.record_request() is False  # count -> n-1
    assert await c.record_request() is True  # count -> n, trains

    assert await get_compression_dict(sf, "parse") is not None


async def test_concurrent_threshold_crossing_trains_once(
    memory_session_factory: "async_sessionmaker",
) -> None:
    """Two workers crossing the threshold concurrently train exactly once.

    The Compactor is shared per step across the worker pool, so multiple
    ``record_request()`` calls can be in flight at the threshold crossing. A
    second caller that crosses while the first is awaiting the train+recompress
    must be rejected, not run a redundant train (which would mint a duplicate
    dictionary version and recompress the step again).
    """
    sf = memory_session_factory
    n = 20
    await _insert_responses(sf, "parse", n)

    c = Compactor(
        "parse", sf, threshold=n, count=n - 1, sample_limit=n, dict_size=32768
    )

    real_train = c._train_and_compact
    calls = 0
    first_entered = asyncio.Event()
    release = asyncio.Event()

    async def gated_train() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Hold the first train's await open so a second concurrent
            # record_request gets a chance to slip past the guard.
            first_entered.set()
            await release.wait()
        await real_train()

    c._train_and_compact = gated_train  # type: ignore[method-assign]

    async def first() -> bool:
        return await c.record_request()

    async def second() -> bool:
        await first_entered.wait()  # first is now inside _train_and_compact
        result = await c.record_request()
        release.set()  # let the first call finish
        return result

    r1, r2 = await asyncio.gather(first(), second())

    assert calls == 1  # trained exactly once
    assert r1 is True
    assert r2 is False

    async with sf() as session:
        dict_count = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM compression_dicts "
                    "WHERE continuation = 'parse'"
                )
            )
        ).scalar_one()
    assert dict_count == 1


async def test_seeded_over_threshold_skips_existing_dict(
    memory_session_factory: "async_sessionmaker",
) -> None:
    """A Compactor seeded over threshold won't re-train an existing dict.

    On a resumed run a step may already be compacted. The first
    ``record_request`` crosses the threshold immediately; ``_train_and_compact``
    must detect the existing dictionary and skip rather than mint a redundant
    version and recompress the step again.
    """
    sf = memory_session_factory
    n = 20
    await _insert_responses(sf, "parse", n)

    # Pre-existing compaction, as a prior run would have left it.
    first = Compactor(
        "parse", sf, threshold=n, sample_limit=n, dict_size=32768
    )
    for _ in range(n):
        await first.record_request()
    dict_result = await get_compression_dict(sf, "parse")
    assert dict_result is not None
    existing_id = dict_result[0]

    # Resume: a fresh Compactor seeded at/over the threshold fires at once.
    resumed = Compactor(
        "parse", sf, threshold=n, count=n, sample_limit=n, dict_size=32768
    )
    assert await resumed.record_request() is True
    assert resumed.done is True

    # No second dictionary version was minted; the original still wins.
    async with sf() as session:
        dict_count = (
            await session.execute(
                sa.text(
                    "SELECT COUNT(*) FROM compression_dicts "
                    "WHERE continuation = 'parse'"
                )
            )
        ).scalar_one()
    assert dict_count == 1
    latest = await get_compression_dict(sf, "parse")
    assert latest is not None and latest[0] == existing_id


def test_default_threshold_is_1000() -> None:
    assert Compactor.THRESHOLD == 1000
    c = Compactor("parse", cast("async_sessionmaker", None))
    assert c.threshold == 1000
