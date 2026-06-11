"""Error paths + train/recompress flow for the zstd compression modules.

Parametrized over both duplicated copies (``database_engine.compression``
and the unified driver's verbatim migration copy) so they cannot drift:
the previously uncovered regions are exactly the unhappy paths — missing
dictionaries, empty sample sets, undecompressable samples — plus the
full train -> recompress -> retrain version flow.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa

import jkent.driver.database_engine.compression as de_compression
import jkent.driver.unified_driver.compression as ud_compression
from jkent.driver.database_engine.sql_manager import SQLManager

_HTML = (
    b"<html><body><h1>Case {n}</h1>"
    b"<p>Lorem ipsum dolor sit amet, the parties are John Doe and "
    b"Jane Smith of 123 Main Street, docket BCC-2024-{n}.</p>"
    b"</body></html>"
)


@pytest.fixture(
    params=[de_compression, ud_compression],
    ids=["database_engine", "unified_driver"],
)
def comp(request: pytest.FixtureRequest) -> Any:
    """The compression module under test (the two copies must agree)."""
    return request.param


async def _insert_responses(
    sql_manager: SQLManager,
    comp: Any,
    continuation: str,
    count: int,
    *,
    garbage: bool = False,
) -> None:
    """Insert ``count`` completed responses (optionally undecompressable)."""
    async with sql_manager._session_factory() as session:
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
                    "VALUES ('completed', 9, :qc, 'GET', :url, :cont, '', "
                    "200, :url, :compressed, :osize, :csize)"
                ),
                {
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
            sql_manager._session_factory, compressed, 12345
        )


async def test_get_dict_lookups_miss(sql_manager: SQLManager, comp) -> None:
    sf = sql_manager._session_factory
    assert await comp.get_compression_dict(sf, "no_such_step") is None
    assert await comp.get_dict_by_id(sf, 4242) is None


async def test_train_with_no_responses(sql_manager: SQLManager, comp) -> None:
    with pytest.raises(ValueError, match="No responses found"):
        await comp.train_compression_dict(
            sql_manager._session_factory, "empty_step"
        )


async def test_train_with_undecompressable_samples(
    sql_manager: SQLManager, comp
) -> None:
    await _insert_responses(sql_manager, comp, "bad_step", 5, garbage=True)
    with pytest.raises(ValueError, match="Could not decompress any samples"):
        await comp.train_compression_dict(
            sql_manager._session_factory, "bad_step"
        )


async def test_recompress_without_dictionary(
    sql_manager: SQLManager, comp
) -> None:
    sf = sql_manager._session_factory
    with pytest.raises(ValueError, match="No dictionary found for"):
        await comp.recompress_responses(sf, "no_such_step")
    with pytest.raises(ValueError, match="No dictionary found with id"):
        await comp.recompress_responses(sf, "no_such_step", dict_id=777)


async def test_train_then_recompress_round_trip(
    sql_manager: SQLManager, comp
) -> None:
    sf = sql_manager._session_factory
    await _insert_responses(sql_manager, comp, "parse", 30)

    dict_id = await comp.train_compression_dict(sf, "parse")
    found = await comp.get_compression_dict(sf, "parse")
    assert found is not None and found[0] == dict_id

    count, original, recompressed = await comp.recompress_responses(
        sf, "parse"
    )
    assert count == 30
    assert original > 0 and recompressed > 0

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

    # Retraining stores a new version; the latest wins the lookup.
    second_id = await comp.train_compression_dict(sf, "parse")
    assert second_id != dict_id
    latest = await comp.get_compression_dict(sf, "parse")
    assert latest is not None and latest[0] == second_id
