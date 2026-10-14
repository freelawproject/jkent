"""``Compactors.for_scraper`` — the startup check behind each step's compactor.

At run start every scraper step is classified once: already has a
dictionary → recompress any rows still off it (an interrupted pass), no
compactor; resolved-response count at/over the threshold → train now, no
compactor; otherwise → a live ``Compactor`` seeded with the current count.
A step zstd cannot train on is logged and left uncompacted — it never makes
the database un-openable. ``record`` is the worker's per-completion report
and is a no-op for steps with no compactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.compression import (
    compress,
    get_compression_dict,
)
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.unified_driver.compaction import Compactor, Compactors
from tests.driver.unified.test_run import TrivialScraper

if TYPE_CHECKING:
    from jkent.driver.database_engine.sql_manager import SQLManager


async def _insert_resolved(db: SQLManager, step_name: str, count: int) -> None:
    """Insert ``count`` resolved (response-bearing) rows for ``step_name``."""
    async with db.session_factory() as session:
        for i in range(count):
            content = (
                f"<html><body>Opinion {step_name} {i} "
                f"lorem ipsum dolor sit</body></html>"
            ).encode()
            compressed = compress(content)
            await session.execute(
                sa.text(
                    """
                    INSERT INTO requests (
                        status, priority, method, url,
                        step, current_location, response_status_code,
                        response_url, content_compressed,
                        content_size_original, content_size_compressed,
                        compression_dict_id)
                    VALUES (:status, 9, :method, :url, :cont, '', 200,
                        :url, :compressed, :osize, :csize, NULL)
                    """
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": f"https://example.com/{step_name}/{i}",
                    "cont": step_name,
                    "compressed": compressed,
                    "osize": len(content),
                    "csize": len(compressed),
                },
            )
        await session.commit()


async def test_startup_seeds_below_threshold(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 10)
    await _insert_resolved(sql_manager, "parse", 4)

    compactors = await Compactors.for_scraper(TrivialScraper(), sql_manager)

    compactor = compactors.for_step("parse")
    assert compactor is not None
    assert compactor.count == 4  # seeded with current resolved count
    assert compactor.done is False
    assert list(compactors) == ["parse"]
    # Below threshold => no dictionary trained at startup.
    assert await get_compression_dict(sql_manager, "parse") is None


async def test_startup_trains_at_threshold(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 20)
    await _insert_resolved(sql_manager, "parse", 20)

    compactors = await Compactors.for_scraper(TrivialScraper(), sql_manager)

    # At/over threshold with no dict => trained now, no live compactor seeded.
    assert compactors.for_step("parse") is None
    assert len(compactors) == 0
    assert await get_compression_dict(sql_manager, "parse") is not None


async def _dict_ids(db: SQLManager, step: str) -> list[int | None]:
    """Each stored body's ``compression_dict_id`` for ``step``, by id."""
    async with db.session_factory() as session:
        result = await session.execute(
            sa.text(
                "SELECT compression_dict_id FROM requests "
                "WHERE step = :s AND content_compressed IS NOT NULL ORDER BY id"
            ),
            {"s": step},
        )
        return [row[0] for row in result]


async def test_startup_resumes_an_interrupted_compaction(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows left off the step's dictionary are recompressed at open."""
    monkeypatch.setattr(Compactor, "THRESHOLD", 20)
    await _insert_resolved(sql_manager, "parse", 20)
    await Compactors.for_scraper(TrivialScraper(), sql_manager)
    latest = await get_compression_dict(sql_manager, "parse")
    assert latest is not None
    # A pass that died between chunks, or bodies that landed behind the
    # recompress cursor: dictionary-less rows on a step that has one.
    await _insert_resolved(sql_manager, "parse", 5)

    compactors = await Compactors.for_scraper(TrivialScraper(), sql_manager)

    assert compactors.for_step("parse") is None
    assert await _dict_ids(sql_manager, "parse") == [latest.dict_id] * 25


async def _insert_untrainable(db: SQLManager, step: str, count: int) -> None:
    """``count`` identical tiny bodies — a corpus zstd cannot train on."""
    compressed = compress(b'{"ok":true}')
    async with db.session_factory() as session:
        for i in range(count):
            await session.execute(
                sa.text(
                    "INSERT INTO requests (status, priority, method, url, "
                    "step, current_location, response_status_code, "
                    "response_url, content_compressed, content_size_original, "
                    "content_size_compressed) VALUES (:status, 9, :method, "
                    ":url, :cont, '', 200, :url, :c, 11, :cs)"
                ),
                {
                    "status": RequestStatus.COMPLETED.code,
                    "method": HttpMethod.GET.code,
                    "url": f"https://example.com/{step}/{i}",
                    "cont": step,
                    "c": compressed,
                    "cs": len(compressed),
                },
            )
        await session.commit()


async def test_startup_survives_an_untrainable_step(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 8)
    await _insert_untrainable(sql_manager, "parse", 8)

    compactors = await Compactors.for_scraper(TrivialScraper(), sql_manager)

    assert compactors.for_step("parse") is None
    assert await get_compression_dict(sql_manager, "parse") is None


async def test_record_counts_only_tracked_steps(
    sql_manager: SQLManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Compactor, "THRESHOLD", 10)
    await _insert_resolved(sql_manager, "parse", 1)
    compactors = await Compactors.for_scraper(TrivialScraper(), sql_manager)

    await compactors.record("parse")
    await compactors.record("no_such_step")  # a no-op, not an error

    compactor = compactors.for_step("parse")
    assert compactor is not None
    assert compactor.count == 2


def test_empty_registry_is_the_null_object() -> None:
    compactors = Compactors()
    assert len(compactors) == 0
    assert compactors.for_step("anything") is None
