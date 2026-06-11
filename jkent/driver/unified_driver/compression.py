"""Zstd compression for unified-driver responses.

The SQLModel *table* classes are imported from ``database_engine.models``
rather than redefined here, because two classes mapping the same tables would
collide in the shared metadata.

Dictionary-based zstd compression improves ratios on similar content (HTML
from the same site); a per-continuation trained dictionary is the win.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import sqlalchemy as sa
import zstandard as zstd
from sqlmodel import col, select

from jkent.contracts import require
from jkent.driver.database_engine.models import CompressionDict, Request

if TYPE_CHECKING:
    from jkent.driver.database_engine.scoped_session import (
        ScopedSessionFactory,
    )

# Default compression level (3 is a good balance of speed/ratio)
DEFAULT_COMPRESSION_LEVEL = 3

# Default dictionary size (112640 bytes = 110KB, zstd's default)
DEFAULT_DICT_SIZE = 112640


@require(
    lambda level: 1 <= level <= 22,
    "compression level is within zstd's documented 1-22 range",
)
def compress(
    data: bytes,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    dictionary: bytes | None = None,
) -> bytes:
    """Compress data using zstd.

    Args:
        data: The data to compress.
        level: Compression level (1-22, default 3).
        dictionary: Optional pre-trained dictionary for better compression.

    Returns:
        Compressed data bytes.
    """
    if dictionary:
        dict_obj = zstd.ZstdCompressionDict(dictionary)
        compressor = zstd.ZstdCompressor(level=level, dict_data=dict_obj)
    else:
        compressor = zstd.ZstdCompressor(level=level)

    return compressor.compress(data)


def decompress(
    data: bytes,
    dictionary: bytes | None = None,
) -> bytes:
    """Decompress zstd-compressed data.

    Args:
        data: The compressed data to decompress.
        dictionary: Dictionary used for compression (must match).

    Returns:
        Decompressed data bytes.
    """
    if dictionary:
        dict_obj = zstd.ZstdCompressionDict(dictionary)
        decompressor = zstd.ZstdDecompressor(dict_data=dict_obj)
    else:
        decompressor = zstd.ZstdDecompressor()

    return decompressor.decompress(data)


async def get_compression_dict(
    session_factory: ScopedSessionFactory,
    continuation: str,
    db_lock: asyncio.Lock | None = None,
) -> tuple[int, bytes] | None:
    """Get the latest compression dictionary for a continuation.

    Returns:
        Tuple of (dict_id, dictionary_data) or None if no dictionary exists.
    """
    lock: asyncio.Lock = db_lock or asyncio.Lock()
    async with lock, session_factory() as session:
        result = await session.execute(
            select(
                col(CompressionDict.id), col(CompressionDict.dictionary_data)
            )
            .where(col(CompressionDict.continuation) == continuation)
            .order_by(col(CompressionDict.version).desc())
            .limit(1)
        )
        row = result.first()
        if row is None:
            return None
        return (row[0], row[1])


async def get_dict_by_id(
    session_factory: ScopedSessionFactory,
    dict_id: int,
    db_lock: asyncio.Lock | None = None,
) -> bytes | None:
    """Get a compression dictionary by its ID.

    Returns:
        Dictionary data bytes or None if not found.
    """
    lock: asyncio.Lock = db_lock or asyncio.Lock()
    async with lock, session_factory() as session:
        result = await session.execute(
            select(col(CompressionDict.dictionary_data)).where(
                col(CompressionDict.id) == dict_id
            )
        )
        row = result.first()
        return row[0] if row else None


async def compress_response(
    session_factory: ScopedSessionFactory,
    content: bytes,
    continuation: str,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    db_lock: asyncio.Lock | None = None,
) -> tuple[bytes, int | None]:
    """Compress response content, using a trained dictionary if available.

    Returns:
        Tuple of (compressed_data, dict_id) where dict_id is None if no
        dictionary was used.
    """
    dict_result = await get_compression_dict(
        session_factory, continuation, db_lock=db_lock
    )

    if dict_result:
        dict_id, dictionary = dict_result
        compressed = compress(content, level=level, dictionary=dictionary)
        return (compressed, dict_id)
    else:
        compressed = compress(content, level=level)
        return (compressed, None)


async def decompress_response(
    session_factory: ScopedSessionFactory,
    compressed: bytes,
    dict_id: int | None,
    db_lock: asyncio.Lock | None = None,
) -> bytes:
    """Decompress response content, using its dictionary if one was used."""
    dictionary = None
    if dict_id is not None:
        dictionary = await get_dict_by_id(
            session_factory, dict_id, db_lock=db_lock
        )
        if dictionary is None:
            raise ValueError(f"Dictionary {dict_id} not found in database")

    return decompress(compressed, dictionary=dictionary)


async def train_compression_dict(
    session_factory: ScopedSessionFactory,
    continuation: str,
    sample_limit: int = 100,
    dict_size: int = DEFAULT_DICT_SIZE,
    db_lock: asyncio.Lock | None = None,
) -> int:
    """Train a zstd compression dictionary from stored responses.

    Samples responses for the continuation, trains a zstd dictionary, and
    stores it as a new version in the compression_dicts table.

    Returns:
        The ID of the newly created dictionary.

    Raises:
        ValueError: If no responses found for continuation or training fails.
    """
    lock: asyncio.Lock = db_lock or asyncio.Lock()
    async with lock, session_factory() as session:
        result = await session.execute(
            select(
                col(Request.content_compressed),
                col(Request.compression_dict_id),
            )
            .where(
                col(Request.continuation) == continuation,
                col(Request.response_status_code).isnot(None),
                col(Request.content_compressed).isnot(None),
            )
            .order_by(sa.func.random())
            .limit(sample_limit)
        )
        rows = result.all()

    if not rows:
        raise ValueError(
            f"No responses found for continuation '{continuation}'"
        )

    samples: list[bytes | bytearray | memoryview[int]] = []
    for compressed, comp_dict_id in rows:
        try:
            content = await decompress_response(
                session_factory,
                compressed,
                comp_dict_id,
                db_lock=db_lock,
            )
            samples.append(content)
        except Exception:
            continue

    if not samples:
        raise ValueError(
            f"Could not decompress any samples for continuation '{continuation}'"
        )

    dictionary_data = zstd.train_dictionary(dict_size, samples)

    async with lock, session_factory() as session:
        result = await session.execute(  # type: ignore[assignment]
            select(
                sa.func.coalesce(sa.func.max(col(CompressionDict.version)), 0)
                + 1
            ).where(col(CompressionDict.continuation) == continuation)
        )
        next_version = result.scalar_one()

        new_dict = CompressionDict(
            continuation=continuation,
            version=next_version,  # type: ignore[arg-type]
            dictionary_data=dictionary_data.as_bytes(),
            sample_count=len(samples),
        )
        session.add(new_dict)
        await session.flush()
        dict_id = new_dict.id
        await session.commit()

        return dict_id  # type: ignore[return-value]


async def recompress_responses(
    session_factory: ScopedSessionFactory,
    continuation: str,
    level: int = DEFAULT_COMPRESSION_LEVEL,
    dict_id: int | None = None,
    db_lock: asyncio.Lock | None = None,
) -> tuple[int, int, int]:
    """Re-compress a continuation's responses using a trained dictionary.

    Returns:
        Tuple of (recompressed_count, total_original_bytes,
        total_compressed_bytes).

    Raises:
        ValueError: If no dictionary exists for this continuation or dict_id.
    """
    lock: asyncio.Lock = db_lock or asyncio.Lock()

    if dict_id is not None:
        async with lock, session_factory() as session:
            result = await session.execute(
                select(col(CompressionDict.dictionary_data)).where(
                    col(CompressionDict.id) == dict_id
                )
            )
            row = result.first()
            if row is None:
                raise ValueError(f"No dictionary found with id {dict_id}.")
            dictionary = row[0]
            target_dict_id = dict_id
    else:
        dict_result = await get_compression_dict(
            session_factory, continuation, db_lock=db_lock
        )
        if dict_result is None:
            raise ValueError(
                f"No dictionary found for continuation '{continuation}'. "
                "Train a dictionary first using train_compression_dict()."
            )
        target_dict_id, dictionary = dict_result

    async with lock, session_factory() as session:
        result = await session.execute(  # type: ignore[assignment]
            select(
                col(Request.id),
                col(Request.content_compressed),
                col(Request.compression_dict_id),
            ).where(
                col(Request.continuation) == continuation,
                col(Request.response_status_code).isnot(None),
                col(Request.content_compressed).isnot(None),
            )
        )
        rows = result.all()

    recompressed_count = 0
    total_original = 0
    total_compressed = 0

    for request_id, compressed, old_dict_id in rows:
        try:
            content = await decompress_response(
                session_factory,
                compressed,
                old_dict_id,
                db_lock=db_lock,
            )
            original_size = len(content)

            new_compressed = compress(
                content, level=level, dictionary=dictionary
            )
            new_size = len(new_compressed)

            async with lock, session_factory() as session:
                await session.execute(
                    sa.update(Request)
                    .where(col(Request.id) == request_id)
                    .values(
                        content_compressed=new_compressed,
                        content_size_original=original_size,
                        content_size_compressed=new_size,
                        compression_dict_id=target_dict_id,
                    )
                )
                await session.commit()

            recompressed_count += 1
            total_original += original_size
            total_compressed += new_size

        except Exception:
            continue

    return (recompressed_count, total_original, total_compressed)
