"""Rate limiting with jitter for LocalDevDriver.

Provides AioSQLiteBucket for pyrate_limiter integration with persistent
storage in SQLite.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import sqlalchemy as sa
from pyrate_limiter import AbstractBucket, Rate, RateItem
from sqlmodel import select

from kent.driver.persistent_driver.models import RateItem as RateItemModel

if TYPE_CHECKING:
    from kent.driver.persistent_driver.scoped_session import (
        ScopedSessionFactory,
    )


class AioSQLiteBucket(AbstractBucket):
    """Async SQLite-backed bucket for pyrate_limiter.

    Implements the AbstractBucket interface using SQLAlchemy async sessions
    for persistent rate limiting state that survives process restarts.

    The bucket stores rate items in the rate_items table, allowing the
    rate limiter to track request timestamps across restarts.

    Note: This is an async bucket - all methods return awaitables.

    Example:
        rates = [Rate(5, Duration.SECOND)]  # 5 requests per second
        bucket = AioSQLiteBucket(session_factory, rates)
        limiter = Limiter(bucket)
    """

    def __init__(
        self,
        session_factory: ScopedSessionFactory,
        rates: list[Rate],
        db_lock: asyncio.Lock,
    ) -> None:
        """Initialize the bucket.

        Args:
            session_factory: Async session factory.
            rates: List of Rate objects defining rate limits.
            db_lock: Shared asyncio lock for serializing SQLite access.
        """
        self._session_factory = session_factory
        self._rates = rates
        self._lock = threading.Lock()
        self._db_lock = db_lock

    @property
    def rates(self) -> list[Rate]:  # type: ignore[override]
        """Get the rate limits for this bucket."""
        return self._rates

    def limiter_lock(self) -> threading.Lock:
        """Get the lock for thread-safe operations."""
        return self._lock

    async def put(self, item: RateItem) -> bool:
        """Add a rate item to the bucket if within rate limits.

        Checks all rate windows before inserting.  Returns False when
        the item would exceed any configured rate, signalling pyrate_limiter
        to call ``waiting()`` and delay the request.

        Args:
            item: The rate item to add.

        Returns:
            True if item was added, False if a rate limit would be exceeded.
        """
        if item.weight == 0:
            return True

        async with self._db_lock, self._session_factory() as session:
            for rate in self._rates:
                window_start = item.timestamp - rate.interval
                result = await session.execute(
                    select(
                        sa.func.coalesce(sa.func.sum(RateItemModel.weight), 0)
                    ).where(RateItemModel.timestamp >= window_start)
                )
                current_count = result.scalar_one()
                if current_count + item.weight > rate.limit:
                    self.failing_rate = rate
                    return False

            self.failing_rate = None
            rate_item = RateItemModel(
                name=item.name,
                timestamp=item.timestamp,
                weight=item.weight,
            )
            session.add(rate_item)
            await session.commit()
        return True

    async def leak(self, current_timestamp: int | None = None) -> int:
        """Remove expired items from the bucket.

        Items older than the longest rate interval are removed.

        Args:
            current_timestamp: Current timestamp in milliseconds. If None,
                uses the current time.

        Returns:
            Number of items removed.
        """
        if current_timestamp is None:
            current_timestamp = int(time.time() * 1000)

        # Find the longest interval from all rates
        max_interval = max(rate.interval for rate in self._rates)

        # Remove items older than the longest interval
        cutoff = current_timestamp - max_interval

        async with self._db_lock, self._session_factory() as session:
            result = await session.execute(
                sa.delete(RateItemModel).where(
                    RateItemModel.timestamp < cutoff  # type: ignore[arg-type]
                )
            )
            await session.commit()
            return result.rowcount  # type: ignore[attr-defined]

    async def flush(self) -> None:
        """Remove all items from the bucket."""
        async with self._db_lock, self._session_factory() as session:
            await session.execute(sa.delete(RateItemModel))
            await session.commit()

    async def count(self) -> int:
        """Get the total weight of items in the bucket.

        Returns:
            Sum of weights of all items.
        """
        async with self._db_lock, self._session_factory() as session:
            result = await session.execute(
                select(sa.func.coalesce(sa.func.sum(RateItemModel.weight), 0))
            )
            return result.scalar_one()

    async def peek(self, index: int) -> RateItem | None:
        """Get an item at a specific index without removing it.

        Args:
            index: Zero-based index (ordered by timestamp DESC).

        Returns:
            RateItem at the index, or None if not found.
        """
        async with self._db_lock, self._session_factory() as session:
            result = await session.execute(
                select(
                    RateItemModel.name,
                    RateItemModel.timestamp,
                    RateItemModel.weight,
                )
                .order_by(RateItemModel.timestamp.desc())  # type: ignore[attr-defined]
                .limit(1)
                .offset(index)
            )
            row = result.first()

            if row is None:
                return None

            return RateItem(name=row[0], timestamp=row[1], weight=row[2])

    async def waiting(self, item: RateItem) -> int:
        """Calculate how long to wait before the item can be processed.

        Checks all rate limits and returns the maximum wait time needed.

        Args:
            item: The item wanting to be processed.

        Returns:
            Wait time in milliseconds (0 if no wait needed).
        """
        current_timestamp = item.timestamp
        max_wait = 0

        async with self._db_lock, self._session_factory() as session:
            for rate in self._rates:
                # Count items in the rate window
                window_start = current_timestamp - rate.interval

                result = await session.execute(
                    select(
                        sa.func.coalesce(sa.func.sum(RateItemModel.weight), 0),
                        sa.func.min(RateItemModel.timestamp),
                    ).where(RateItemModel.timestamp >= window_start)
                )
                row = result.first()
                current_count = row[0] if row else 0
                oldest_timestamp = (
                    row[1] if row and row[1] else current_timestamp
                )

                # Check if we've exceeded the rate limit
                if current_count + item.weight > rate.limit:
                    # Calculate wait time until oldest item expires
                    wait_until = oldest_timestamp + rate.interval
                    wait_time = wait_until - current_timestamp
                    max_wait = max(max_wait, wait_time)

        return max(0, max_wait)
