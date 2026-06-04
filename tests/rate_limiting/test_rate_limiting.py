"""Tests that rate limits declared on scrapers are respected by all drivers.

Each driver test injects a :class:`LimiterSpy` that replaces
``pyrate_limiter.Limiter.try_acquire`` / ``try_acquire_async`` with no-op
counters, then asserts on how many times the limiter was consulted.

Earlier versions of these tests asserted on wall-clock elapsed time to
infer rate-limiter behavior, which proved flaky on slow CI runners --
PlaywrightDriver browser startup alone could eat a 4 s budget.  Asserting
on the call count is deterministic and independent of runner speed.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pyrate_limiter import Duration, Limiter, Rate, RateItem

from kent.data_types import (
    ArchiveDecision,
    ArchiveResponse,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from kent.driver.persistent_driver.database import init_database
from kent.driver.persistent_driver.rate_limiter import AioSQLiteBucket
from tests.utils import collect_results, collect_results_async

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_REQUESTS = 4
# Rate value is documentation-only now -- the spy fixture stubs out
# actual delays, so tests assert on call counts rather than elapsed time.
RATE = Rate(1, Duration.SECOND)


def _make_scraper_class(
    server_url: str,
    rate: Rate = RATE,
    n_requests: int = NUM_REQUESTS,
):
    """Build a minimal scraper class that emits *n_requests* GETs."""

    class RateLimitScraper(BaseScraper[dict]):
        rate_limits = [rate]

        def get_entry(self) -> Generator[Request, None, None]:
            for i in range(n_requests):
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"{server_url}/test?i={i}",
                    ),
                    continuation="parse",
                )

        def parse(self, response: Response):
            yield ParsedData(data={"url": response.url})

    return RateLimitScraper


# ---------------------------------------------------------------------------
# Limiter spy fixture
# ---------------------------------------------------------------------------


@dataclass
class LimiterSpy:
    """Records every ``Limiter.try_acquire(_async)`` call across drivers."""

    sync_calls: list[tuple[str, int]] = field(default_factory=list)
    async_calls: list[tuple[str, int]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.sync_calls) + len(self.async_calls)


@pytest.fixture
def limiter_spy(monkeypatch: pytest.MonkeyPatch) -> LimiterSpy:
    """Replace ``Limiter.try_acquire(_async)`` with no-op counters.

    Catches calls from both the persistent-driver workers (which call
    ``try_acquire_async`` directly on ``driver.rate_limiter``) and from
    ``RateLimiterTransport`` inside the httpx-based sync/async drivers
    (which call it once per HTTP request).
    """
    spy = LimiterSpy()

    def fake_try_acquire(
        self: Limiter,
        name: str = "pyrate",
        weight: int = 1,
        blocking: bool = True,
        timeout: int | float = -1,
    ) -> bool:
        spy.sync_calls.append((name, weight))
        return True

    async def fake_try_acquire_async(
        self: Limiter,
        name: str = "pyrate",
        weight: int = 1,
        blocking: bool = True,
        timeout: int | float = -1,
    ) -> bool:
        spy.async_calls.append((name, weight))
        return True

    monkeypatch.setattr(Limiter, "try_acquire", fake_try_acquire)
    monkeypatch.setattr(Limiter, "try_acquire_async", fake_try_acquire_async)
    return spy


# ---------------------------------------------------------------------------
# SyncDriver
# ---------------------------------------------------------------------------


class TestSyncDriverRateLimiting:
    def test_rate_limit_respected(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """SyncDriver consults the limiter once per non-bypass request."""
        from kent.driver.sync_driver import SyncDriver

        Scraper = _make_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results()

        driver = SyncDriver(
            scraper=scraper,
            storage_dir=tmp_path,
            on_data=callback,
        )
        driver.run()

        assert len(results) == NUM_REQUESTS
        assert limiter_spy.count == NUM_REQUESTS, (
            f"SyncDriver consulted limiter {limiter_spy.count} times, "
            f"expected {NUM_REQUESTS}"
        )


# ---------------------------------------------------------------------------
# AsyncDriver
# ---------------------------------------------------------------------------


class TestAsyncDriverRateLimiting:
    async def test_rate_limit_respected(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """AsyncDriver consults the limiter once per non-bypass request."""
        from kent.driver.async_driver import AsyncDriver

        Scraper = _make_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        driver = AsyncDriver(
            scraper=scraper,
            storage_dir=tmp_path,
            on_data=callback,
            num_workers=1,
        )
        await driver.run()

        assert len(results) == NUM_REQUESTS
        assert limiter_spy.count == NUM_REQUESTS, (
            f"AsyncDriver consulted limiter {limiter_spy.count} times, "
            f"expected {NUM_REQUESTS}"
        )


# ---------------------------------------------------------------------------
# PersistentDriver
# ---------------------------------------------------------------------------


class TestPersistentDriverRateLimiting:
    async def test_rate_limit_respected(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """PersistentDriver consults the limiter once per non-bypass request."""
        from kent.driver.persistent_driver import PersistentDriver

        Scraper = _make_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "rate_limit_test.db"

        async with PersistentDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
        ) as driver:
            driver.on_data = callback
            await driver.run()

        assert len(results) == NUM_REQUESTS
        assert limiter_spy.count == NUM_REQUESTS, (
            f"PersistentDriver consulted limiter {limiter_spy.count} times, "
            f"expected {NUM_REQUESTS}"
        )


# ---------------------------------------------------------------------------
# PlaywrightDriver
# ---------------------------------------------------------------------------


class TestPlaywrightDriverRateLimiting:
    @pytest.mark.slow
    async def test_rate_limit_respected(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """PlaywrightDriver consults the limiter once per non-bypass request."""
        pw = pytest.importorskip("playwright")  # noqa: F841
        from kent.driver.playwright_driver import PlaywrightDriver

        Scraper = _make_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "rate_limit_pw_test.db"

        async with PlaywrightDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
            headless=True,
        ) as driver:
            driver.on_data = callback
            await driver.run()

        assert len(results) == NUM_REQUESTS
        assert limiter_spy.count == NUM_REQUESTS, (
            f"PlaywrightDriver consulted limiter {limiter_spy.count} times, "
            f"expected {NUM_REQUESTS}"
        )


# ---------------------------------------------------------------------------
# AioSQLiteBucket unit tests
# ---------------------------------------------------------------------------


class TestAioSQLiteBucketPut:
    """Tests for AioSQLiteBucket.put() rate-limit enforcement.

    These verify that put() returns False (and sets failing_rate) when
    the bucket is at capacity, which is the mechanism pyrate_limiter
    relies on to trigger delay logic.
    """

    @pytest.fixture
    async def bucket(self, tmp_path: Path) -> AsyncGenerator[AioSQLiteBucket]:
        db_path = tmp_path / "bucket_test.db"
        engine, session_factory = await init_database(db_path)
        # 2 requests per second
        rates = [Rate(2, Duration.SECOND)]
        bucket = AioSQLiteBucket(session_factory, rates, asyncio.Lock())
        yield bucket  # type: ignore[misc]
        await engine.dispose()

    async def test_put_accepts_items_within_limit(
        self, bucket: AioSQLiteBucket
    ) -> None:
        """put() returns True while under the rate limit."""
        now = int(time.time() * 1000)
        assert await bucket.put(RateItem("r", now, 1)) is True
        assert await bucket.put(RateItem("r", now, 1)) is True
        assert bucket.failing_rate is None

    async def test_put_rejects_when_limit_exceeded(
        self, bucket: AioSQLiteBucket
    ) -> None:
        """put() returns False once the rate limit is reached."""
        now = int(time.time() * 1000)
        # Fill the bucket (limit=2)
        await bucket.put(RateItem("r", now, 1))
        await bucket.put(RateItem("r", now, 1))
        # Third item should be rejected
        result = await bucket.put(RateItem("r", now, 1))
        assert result is False

    async def test_put_sets_failing_rate(
        self, bucket: AioSQLiteBucket
    ) -> None:
        """put() sets failing_rate to the exceeded Rate object."""
        now = int(time.time() * 1000)
        await bucket.put(RateItem("r", now, 1))
        await bucket.put(RateItem("r", now, 1))
        await bucket.put(RateItem("r", now, 1))
        assert bucket.failing_rate is not None
        assert bucket.failing_rate.limit == 2

    async def test_put_accepts_after_window_expires(
        self, bucket: AioSQLiteBucket
    ) -> None:
        """put() accepts items again once previous items leave the window."""
        now = int(time.time() * 1000)
        await bucket.put(RateItem("r", now, 1))
        await bucket.put(RateItem("r", now, 1))
        # Rejected within the same window
        assert await bucket.put(RateItem("r", now, 1)) is False
        # 1001 ms later (outside the 1-second window), should be accepted
        assert await bucket.put(RateItem("r", now + 1001, 1)) is True


# ---------------------------------------------------------------------------
# bypass_rate_limit tests
# ---------------------------------------------------------------------------

BYPASS_NUM_REQUESTS = 4
BYPASS_RATE = Rate(1, 2 * Duration.SECOND)


def _make_bypass_scraper_class(
    server_url: str,
    rate: Rate = BYPASS_RATE,
    n_requests: int = BYPASS_NUM_REQUESTS,
):
    """Build a scraper that emits bypass_rate_limit=True requests."""

    class BypassRateLimitScraper(BaseScraper[dict]):
        rate_limits = [rate]

        def get_entry(self) -> Generator[Request, None, None]:
            for i in range(n_requests):
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"{server_url}/test?i={i}",
                    ),
                    continuation="parse",
                    bypass_rate_limit=True,
                )

        def parse(self, response: Response):
            yield ParsedData(data={"url": response.url})

    return BypassRateLimitScraper


class TestSyncDriverBypassRateLimit:
    def test_bypass_skips_rate_limit(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """SyncDriver bypass_rate_limit requests skip rate limiting."""
        from kent.driver.sync_driver import SyncDriver

        Scraper = _make_bypass_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results()

        driver = SyncDriver(
            scraper=scraper,
            storage_dir=tmp_path,
            on_data=callback,
        )
        driver.run()

        assert len(results) == BYPASS_NUM_REQUESTS
        assert limiter_spy.count == 0, (
            f"SyncDriver consulted limiter {limiter_spy.count} times for "
            f"bypass requests, expected 0"
        )


class TestAsyncDriverBypassRateLimit:
    async def test_bypass_skips_rate_limit(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """AsyncDriver bypass_rate_limit requests skip rate limiting."""
        from kent.driver.async_driver import AsyncDriver

        Scraper = _make_bypass_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        driver = AsyncDriver(
            scraper=scraper,
            storage_dir=tmp_path,
            on_data=callback,
            num_workers=1,
        )
        await driver.run()

        assert len(results) == BYPASS_NUM_REQUESTS
        assert limiter_spy.count == 0, (
            f"AsyncDriver consulted limiter {limiter_spy.count} times for "
            f"bypass requests, expected 0"
        )


class TestPersistentDriverBypassRateLimit:
    async def test_bypass_skips_rate_limit(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """PersistentDriver bypass_rate_limit requests skip rate limiting."""
        from kent.driver.persistent_driver import PersistentDriver

        Scraper = _make_bypass_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "bypass_test.db"

        async with PersistentDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
        ) as driver:
            driver.on_data = callback
            await driver.run()

        assert len(results) == BYPASS_NUM_REQUESTS
        assert limiter_spy.count == 0, (
            f"PersistentDriver consulted limiter {limiter_spy.count} times "
            f"for bypass requests, expected 0"
        )


class TestPlaywrightDriverBypassRateLimit:
    @pytest.mark.slow
    async def test_bypass_skips_rate_limit(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """PlaywrightDriver bypass_rate_limit requests skip rate limiting."""
        pw = pytest.importorskip("playwright")  # noqa: F841
        from kent.driver.playwright_driver import PlaywrightDriver

        Scraper = _make_bypass_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "bypass_pw_test.db"

        async with PlaywrightDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
            headless=True,
        ) as driver:
            driver.on_data = callback
            await driver.run()

        assert len(results) == BYPASS_NUM_REQUESTS
        assert limiter_spy.count == 0, (
            f"PlaywrightDriver consulted limiter {limiter_spy.count} times "
            f"for bypass requests, expected 0"
        )


# ---------------------------------------------------------------------------
# Archive skip bypasses rate limiter
# ---------------------------------------------------------------------------

ARCHIVE_SKIP_NUM_ARCHIVES = 4
ARCHIVE_SKIP_RATE = Rate(1, 2 * Duration.SECOND)


class _SkipAllDownloadsAsync:
    """Async archive handler that always skips downloads."""

    async def should_download(
        self, url, deduplication_key, expected_type, hash_header_value
    ):
        return ArchiveDecision(download=False, file_url="/skipped")

    async def save(
        self, url, deduplication_key, expected_type, hash_header_value, content
    ):
        raise AssertionError(
            "save should not be called when downloads are skipped"
        )


def _make_archive_skip_scraper_class(
    server_url: str,
    rate: Rate = ARCHIVE_SKIP_RATE,
    n_archives: int = ARCHIVE_SKIP_NUM_ARCHIVES,
):
    """Build a scraper where one entry yields N archive requests (all skipped)."""

    class ArchiveSkipScraper(BaseScraper[dict]):
        rate_limits = [rate]

        def get_entry(self) -> Generator[Request, None, None]:
            # Single entry request (rate-limited HTTP)
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=f"{server_url}/test",
                ),
                continuation="archive_step",
            )

        def archive_step(self, response: Response):
            # Yield N archive requests that the handler will skip
            for i in range(n_archives):
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"{response.url}?archive={i}",
                    ),
                    continuation="parse_archive",
                    archive=True,
                    expected_type="pdf",
                )

        def parse_archive(self, response: ArchiveResponse):
            yield ParsedData(data={"file_url": response.file_url})

    return ArchiveSkipScraper


class TestPersistentDriverArchiveSkipBypassesRateLimiter:
    async def test_archive_skip_bypasses_rate_limiter(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """Skipped archive downloads should not consume rate limiter tokens."""
        from kent.driver.persistent_driver import PersistentDriver

        Scraper = _make_archive_skip_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "archive_skip_test.db"

        async with PersistentDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
        ) as driver:
            driver.archive_handler = _SkipAllDownloadsAsync()
            driver.on_data = callback
            await driver.run()

        # Only the entry request consults the limiter; the 4 archive
        # requests are bypassed because should_download() returned False.
        assert len(results) == ARCHIVE_SKIP_NUM_ARCHIVES
        assert limiter_spy.count == 1, (
            f"PersistentDriver consulted limiter {limiter_spy.count} times, "
            f"expected 1 (entry only; skipped archives must not consume tokens)"
        )


class TestPlaywrightDriverArchiveSkipBypassesRateLimiter:
    @pytest.mark.slow
    async def test_archive_skip_bypasses_rate_limiter(
        self,
        server_url: str,
        tmp_path: Path,
        limiter_spy: LimiterSpy,
    ) -> None:
        """Skipped archive downloads should not consume rate limiter tokens."""
        pw = pytest.importorskip("playwright")  # noqa: F841
        from kent.driver.playwright_driver import PlaywrightDriver

        Scraper = _make_archive_skip_scraper_class(server_url)
        scraper = Scraper()
        callback, results = collect_results_async()

        db_path = tmp_path / "archive_skip_pw_test.db"

        async with PlaywrightDriver.open(
            scraper,
            db_path,
            num_workers=1,
            resume=False,
            enable_monitor=False,
            headless=True,
        ) as driver:
            driver.archive_handler = _SkipAllDownloadsAsync()
            driver.on_data = callback
            await driver.run()

        # Only the entry request consults the limiter; the 4 archive
        # requests are bypassed because should_download() returned False.
        assert len(results) == ARCHIVE_SKIP_NUM_ARCHIVES
        assert limiter_spy.count == 1, (
            f"PlaywrightDriver consulted limiter {limiter_spy.count} times, "
            f"expected 1 (entry only; skipped archives must not consume tokens)"
        )
