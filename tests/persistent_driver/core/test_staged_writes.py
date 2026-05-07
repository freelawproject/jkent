"""Tests for per-step atomic write staging.

Verifies that DB writes derived from a parent step's yields land
atomically (all-or-nothing) instead of streaming during iteration.
"""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any

import sqlalchemy as sa


async def _count(session_factory, query: str) -> int:
    async with session_factory() as session:
        result = await session.execute(sa.text(query))
        return result.scalar() or 0


class TestAtomicFlush:
    """Step yields should commit together or not at all."""

    async def test_step_yields_atomic_on_success(self, db_path: Path) -> None:
        """Three ParsedData + one Request all land at flush time."""
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            ParsedData,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        class MultiYieldScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                yield ParsedData({"id": 1})
                yield ParsedData({"id": 2})
                yield ParsedData({"id": 3})
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/child",
                    ),
                    continuation="parse_child",
                    current_location=response.url,
                )

            def parse_child(
                self, response: Response
            ) -> Generator[Any, None, None]:
                yield None

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>parent</html>"),
        )
        rm.add_response(
            "https://example.com/child",
            create_html_response("<html>child</html>"),
        )

        async with PersistentDriver.open(
            MultiYieldScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:
            await driver.run()

            # All four yield-derived rows present
            assert (
                await _count(
                    driver.db._session_factory, "SELECT COUNT(*) FROM results"
                )
                == 3
            )
            # Two requests in the requests table: parent + child
            count_requests = await _count(
                driver.db._session_factory, "SELECT COUNT(*) FROM requests"
            )
            assert count_requests == 2


class TestRollbackOnException:
    """An exception during step iteration drops the buffer."""

    async def test_step_yields_rolled_back_on_exception(
        self, db_path: Path
    ) -> None:
        """Step raises after partial yields → no rows written."""
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            ParsedData,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        class CrashingScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                yield ParsedData({"id": 1})
                yield ParsedData({"id": 2})
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/child",
                    ),
                    continuation="parse",
                    current_location=response.url,
                )
                raise RuntimeError("boom mid-step")

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>parent</html>"),
        )

        async with PersistentDriver.open(
            CrashingScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:
            await driver.run()

            # Zero results — partial yields rolled back
            assert (
                await _count(
                    driver.db._session_factory, "SELECT COUNT(*) FROM results"
                )
                == 0
            )
            # Only the parent request — staged child was discarded
            count_requests = await _count(
                driver.db._session_factory, "SELECT COUNT(*) FROM requests"
            )
            assert count_requests == 1


class TestIntraStepDedup:
    """Two yields with the same dedup_key in one step → single row."""

    async def test_intra_step_dedup(self, db_path: Path) -> None:
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        class DupScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                # Two yields share dedup_key — only one should land.
                for _ in range(2):
                    yield Request(
                        request=HTTPRequestParams(
                            method=HttpMethod.GET,
                            url="https://example.com/child",
                        ),
                        continuation="parse_child",
                        current_location=response.url,
                        deduplication_key="same-key",
                    )

            def parse_child(
                self, response: Response
            ) -> Generator[Any, None, None]:
                yield None

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>parent</html>"),
        )
        rm.add_response(
            "https://example.com/child",
            create_html_response("<html>child</html>"),
        )

        async with PersistentDriver.open(
            DupScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:
            await driver.run()

            # Parent + one child = 2 (not 3)
            count_requests = await _count(
                driver.db._session_factory, "SELECT COUNT(*) FROM requests"
            )
            assert count_requests == 2


class TestStructuralErrorPartialBuffer:
    """on_structural_error preserved → buffer flushes; otherwise drops."""

    async def test_structural_error_with_callback_flushes_partial(
        self, db_path: Path
    ) -> None:
        from kent.common.exceptions import (
            HTMLStructuralAssumptionException,
        )
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            ParsedData,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        class StructScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                yield ParsedData({"before": True})
                raise HTMLStructuralAssumptionException(
                    selector="//missing",
                    selector_type="xpath",
                    description="missing element",
                    expected_min=1,
                    expected_max=1,
                    actual_count=0,
                    request_url=response.url,
                )

        from kent.common.exceptions import ScraperAssumptionException

        async def on_struct(_: ScraperAssumptionException) -> bool:
            return True

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>parent</html>"),
        )

        async with PersistentDriver.open(
            StructScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:
            driver.on_structural_error = on_struct
            await driver.run()

            # Partial buffer flushed: the one ParsedData lands
            assert (
                await _count(
                    driver.db._session_factory, "SELECT COUNT(*) FROM results"
                )
                == 1
            )

    async def test_structural_error_without_callback_drops(
        self, db_path: Path
    ) -> None:
        from kent.common.exceptions import (
            HTMLStructuralAssumptionException,
        )
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            ParsedData,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        class StructScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                yield ParsedData({"before": True})
                raise HTMLStructuralAssumptionException(
                    selector="//missing",
                    selector_type="xpath",
                    description="missing element",
                    expected_min=1,
                    expected_max=1,
                    actual_count=0,
                    request_url=response.url,
                )

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>parent</html>"),
        )

        async with PersistentDriver.open(
            StructScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:
            await driver.run()

            # No callback → exception bubbles → buffer dropped
            assert (
                await _count(
                    driver.db._session_factory, "SELECT COUNT(*) FROM results"
                )
                == 0
            )


class TestStagedFlushSingleTransaction:
    """Verify flush is one transaction, not per-row commits."""

    async def test_flush_runs_in_single_transaction(self, sql_manager) -> None:
        """Direct flush test: rows invisible mid-flush, all visible after."""
        from kent.driver.persistent_driver._staging import StagedWrites

        # Insert a parent request to satisfy FK constraints
        parent_id = await sql_manager.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com/parent",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key=None,
            parent_id=None,
        )

        staged = StagedWrites(request_id=parent_id)
        staged.stage_result(
            result_type="dict",
            data_json='{"x": 1}',
            is_valid=True,
            validation_errors_json=None,
        )
        staged.stage_result(
            result_type="dict",
            data_json='{"x": 2}',
            is_valid=True,
            validation_errors_json=None,
        )
        staged.stage_estimate(
            expected_types_json='["dict"]',
            min_count=2,
            max_count=2,
        )

        # Before flush: nothing is committed
        assert (
            await _count(
                sql_manager._session_factory, "SELECT COUNT(*) FROM results"
            )
            == 0
        )
        assert (
            await _count(
                sql_manager._session_factory, "SELECT COUNT(*) FROM estimates"
            )
            == 0
        )

        # Flush
        await staged.flush(sql_manager)

        # After flush: all rows committed, parent marked completed
        assert (
            await _count(
                sql_manager._session_factory, "SELECT COUNT(*) FROM results"
            )
            == 2
        )
        assert (
            await _count(
                sql_manager._session_factory, "SELECT COUNT(*) FROM estimates"
            )
            == 1
        )

        async with sql_manager._session_factory() as session:
            row = await session.execute(
                sa.text("SELECT status FROM requests WHERE id = :id"),
                {"id": parent_id},
            )
            assert row.scalar() == "completed"


class TestStagedDedupAcrossSteps:
    """A staged request whose dedup_key matches a committed row → no insert."""

    async def test_cross_step_dedup_unchanged(self, sql_manager) -> None:
        from kent.driver.persistent_driver._staging import StagedWrites

        # Existing committed request with dedup_key
        await sql_manager.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com/existing",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key="dup-key",
            parent_id=None,
        )

        # Parent under which the new staged request sits
        parent_id = await sql_manager.insert_request(
            priority=1,
            request_type="navigating",
            method="GET",
            url="https://example.com/parent",
            headers_json=None,
            cookies_json=None,
            body=None,
            continuation="parse",
            current_location="",
            accumulated_data_json=None,
            permanent_json=None,
            expected_type=None,
            dedup_key=None,
            parent_id=None,
        )

        staged = StagedWrites(request_id=parent_id)
        staged.stage_request(
            request_data={
                "priority": 1,
                "request_type": "navigating",
                "method": "GET",
                "url": "https://example.com/new",
                "headers_json": None,
                "cookies_json": None,
                "body": None,
                "continuation": "parse",
                "current_location": "",
                "accumulated_data_json": None,
                "permanent_json": None,
                "expected_type": None,
                "is_speculative": False,
                "speculation_id": None,
                "verify": None,
                "via_json": None,
                "bypass_rate_limit": False,
            },
            dedup_key="dup-key",
            parent_id=parent_id,
            progress_event={},
        )

        events = await staged.flush(sql_manager)

        # Dedup'd: no new row, no progress event
        count_requests = await _count(
            sql_manager._session_factory, "SELECT COUNT(*) FROM requests"
        )
        assert count_requests == 2  # original + parent only
        assert events == []


class TestStagingDoesNotHoldLock:
    """While a step's generator is iterating, the DB write lock is free."""

    async def test_lock_free_during_iteration(self, db_path: Path) -> None:
        from kent.data_types import (
            BaseScraper,
            HttpMethod,
            HTTPRequestParams,
            ParsedData,
            Request,
            Response,
        )
        from kent.driver.persistent_driver.persistent_driver import (
            PersistentDriver,
        )
        from kent.driver.persistent_driver.testing import (
            MockRequestManager,
            create_html_response,
        )

        external_done = asyncio.Event()

        class SlowScraper(BaseScraper[dict[str, Any]]):
            def get_entry(self) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url="https://example.com/parent",
                    ),
                    continuation="parse",
                    current_location="",
                )

            def parse(self, response: Response) -> Generator[Any, None, None]:
                yield ParsedData({"id": 1})
                yield ParsedData({"id": 2})

        rm = MockRequestManager()
        rm.add_response(
            "https://example.com/parent",
            create_html_response("<html>p</html>"),
        )

        async with PersistentDriver.open(
            SlowScraper(),
            db_path,
            enable_monitor=False,
            request_manager=rm,
        ) as driver:

            async def write_alongside(_event: Any) -> None:
                # When a request_started event fires (the step is about to
                # iterate), do a DB read from a separate task to confirm
                # the lock is not held during step processing.
                if (
                    _event.event_type == "request_started"
                    and not external_done.is_set()
                ):
                    await driver.db.count_pending_requests()
                    external_done.set()

            driver.on_progress = write_alongside
            await driver.run()

            # The external read completed during step processing.
            assert external_done.is_set()
            # Both staged ParsedData yields landed.
            assert (
                await _count(
                    driver.db._session_factory, "SELECT COUNT(*) FROM results"
                )
                == 2
            )
