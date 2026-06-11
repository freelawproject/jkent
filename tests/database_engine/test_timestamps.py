"""Millisecond wall-clock timestamps, and the duration math built on them.

``requests`` used to carry a parallel set of ``*_at_ns`` columns holding
``time.monotonic_ns()``. Those are gone: a monotonic value's epoch is the
writing machine's boot, so it was only meaningful inside the process that
wrote it — and jent's replay index compared them across source databases.
Durations now come from the text timestamps via ``unixepoch(..., 'subsec')``.

These tests hold the properties that substitution depends on: the stored
format really does carry milliseconds, it sorts chronologically as text, and
the epoch arithmetic is accurate enough to have replaced nanosecond counters.
"""

from __future__ import annotations

import random
import re
import sqlite3
from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine.database import init_database
from jkent.driver.database_engine.enums import RequestStatus, RequestType
from jkent.driver.database_engine.models import Request
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.storage import (
    DEFAULT_RETRY_JITTER,
    MIN_RETRY_DELAY_S,
)
from jkent.driver.database_engine.timestamps import (
    epoch_seconds,
    require_subsec_support,
)
from jkent.driver.unified_driver.persistence import ResponseStorage

if TYPE_CHECKING:
    from pathlib import Path

#: ``2026-08-06 22:58:11.250`` — SQLite's default shape plus milliseconds.
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")


def test_requests_carries_no_nanosecond_columns() -> None:
    """The monotonic columns are gone and should stay gone.

    ``incidental_requests`` keeps its ``*_at_ns`` pair — those are
    ``time.time_ns()``, wall clock on the Unix epoch, and a page's
    sub-requests routinely complete inside one millisecond.
    """
    assert [
        c.name for c in Request.__table__.columns if c.name.endswith("_ns")
    ] == []


def test_subsec_support_is_required_and_present() -> None:
    """This interpreter's SQLite can do sub-second epochs.

    The guard exists because an unrecognised modifier is not an error in
    SQLite — ``unixepoch(x, 'nosuch')`` returns NULL — so a build without
    ``'subsec'`` would turn every duration into NULL rather than failing.
    """
    require_subsec_support()  # raises if unsupported

    with sqlite3.connect(":memory:") as conn:
        unknown_modifier = conn.execute(
            "SELECT unixepoch('2026-01-01 00:00:00.500', 'nosuch')"
        ).fetchone()[0]
    assert unknown_modifier is None, (
        "the silent-NULL behaviour this guard defends against no longer "
        "happens; the guard may be able to go"
    )


async def test_written_timestamps_carry_milliseconds(tmp_path: Path) -> None:
    """Every server-defaulted and code-written stamp has a ``.mmm`` part."""
    async with SQLManager.open(tmp_path / "ts.db") as manager:
        request_id = await manager.insert_request(
            priority=1,
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url="https://example.com/x",
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
        await manager.dequeue_next_request()
        await manager.mark_request_completed(request_id)

        async with manager.session_factory() as session:
            row = (
                await session.execute(
                    # Cast to text: the columns map to ``DateTime``, so the ORM
                    # hands back ``datetime`` objects and the stored format —
                    # what this test exists to pin — would never be seen.
                    sa.select(
                        sa.cast(Request.created_at, sa.Text),
                        sa.cast(Request.started_at, sa.Text),
                        sa.cast(Request.completed_at, sa.Text),
                    )
                )
            ).one()

    for name, value in zip(
        ("created_at", "started_at", "completed_at"), row, strict=True
    ):
        assert value is not None, f"{name} was not stamped"
        assert _STAMP.match(value), (
            f"{name} is not millisecond format: {value!r}"
        )


async def test_timestamp_text_sorts_chronologically(tmp_path: Path) -> None:
    """Lexicographic order is chronological order for this format.

    Relied on by ``ORDER BY created_at`` and, in jent, by the replay index's
    "most recent capture wins" comparison — which does the comparison in SQL
    on the raw column, with no parsing.
    """
    engine, session_factory = await init_database(tmp_path / "sort.db")
    try:
        stamps = [
            "2026-01-01 00:00:00.000",
            "2026-01-01 00:00:00.001",
            "2026-01-01 00:00:00.999",
            "2026-01-01 00:00:01.000",
            "2026-01-01 00:01:00.000",
            "2026-12-31 23:59:59.999",
        ]
        async with session_factory() as session:
            for i, stamp in enumerate(reversed(stamps)):
                await session.execute(
                    sa.text(
                        "INSERT INTO requests (status, priority, queue_counter,"
                        " method, url, continuation, current_location,"
                        " created_at) VALUES (:s, 9, :q, :m, :u, 'parse', '',"
                        " :c)"
                    ),
                    {
                        "s": RequestStatus.PENDING.code,
                        "q": i,
                        "m": HttpMethod.GET.code,
                        "u": f"https://example.com/{i}",
                        "c": stamp,
                    },
                )
            await session.commit()
            ordered = (
                (
                    await session.execute(
                        # Select as text — the ORM would otherwise parse
                        # these back to datetimes and the stored spelling,
                        # which is what sorts, would be invisible. The
                        # ORDER BY still runs on the raw column.
                        sa.select(
                            sa.cast(Request.created_at, sa.Text)
                        ).order_by(Request.created_at)
                    )
                )
                .scalars()
                .all()
            )
    finally:
        await engine.dispose()

    assert ordered == stamps


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-01-01 00:00:00.000", "2026-01-01 00:00:00.001", 0.001),
        ("2026-01-01 00:00:00.000", "2026-01-01 00:00:02.500", 2.5),
        ("2026-01-01 00:00:00.250", "2026-01-01 00:00:01.750", 1.5),
        ("2026-01-01 23:59:59.000", "2026-01-02 00:00:01.000", 2.0),
    ],
)
async def test_epoch_seconds_differences_are_exact(
    tmp_path: Path, start: str, end: str, expected: float
) -> None:
    """``epoch_seconds`` differences are accurate to well under a microsecond.

    Not bit-exact — subtracting two ~1.7e9 doubles leaves roughly 240ns of
    representable resolution — but one to three orders of magnitude better
    than ``julianday``, which carries 6-30 microseconds of noise because it
    works in day-scale floats. Both are far finer than the millisecond the
    timestamps are quantised to; the point of the comparison is that the
    replacement for nanosecond counters did not lose meaningful precision.
    """
    engine, session_factory = await init_database(tmp_path / "dur.db")
    try:
        async with session_factory() as session:
            got = (
                await session.execute(
                    sa.select(
                        epoch_seconds(sa.literal(end))
                        - epoch_seconds(sa.literal(start))
                    )
                )
            ).scalar_one()
            julian = (
                await session.execute(
                    sa.select(
                        (
                            sa.func.julianday(sa.literal(end))
                            - sa.func.julianday(sa.literal(start))
                        )
                        * 86400.0
                    )
                )
            ).scalar_one()
    finally:
        await engine.dispose()

    # Sub-microsecond: three orders of magnitude finer than the millisecond
    # the stored timestamps resolve to.
    assert got == pytest.approx(expected, abs=1e-6)
    # And strictly better than the julianday arithmetic it replaced.
    assert abs(julian - expected) > abs(got - expected)


async def _complete(
    manager: SQLManager, url: str, start: str, end: str, *, preresolved: bool
) -> None:
    """Insert one completed request with an explicit duration."""
    async with manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, queue_counter, method,"
                " url, continuation, current_location, started_at,"
                " completed_at, preresolved)"
                " VALUES (:s, 9, :q, :m, :u, 'parse', '', :st, :ct, :p)"
            ),
            {
                "s": RequestStatus.COMPLETED.code,
                "q": abs(hash(url)) % 10_000,
                "m": HttpMethod.GET.code,
                "u": url,
                "st": start,
                "ct": end,
                "p": preresolved,
            },
        )
        await session.commit()


async def test_average_duration_ignores_preresolved_requests(
    tmp_path: Path,
) -> None:
    """Pre-resolved rows are left out of the average request duration.

    They never touch the transport — their response was attached at enqueue
    time — so they complete inside a millisecond and record a zero duration.
    Counting them would answer "how fast is this process" when the question is
    "how long does the site take to answer". With three of them beside one real
    two-second fetch, including them would report half a second.
    """
    async with SQLManager.open(tmp_path / "avg.db") as manager:
        await _complete(
            manager,
            "https://example.com/real",
            "2026-01-01 00:00:00.000",
            "2026-01-01 00:00:02.000",
            preresolved=False,
        )
        for i in range(3):
            await _complete(
                manager,
                f"https://example.com/pre/{i}",
                "2026-01-01 00:00:00.000",
                "2026-01-01 00:00:00.000",
                preresolved=True,
            )

        assert (
            await manager.avg_completed_request_duration_s()
            == pytest.approx(2.0)
        )

        async with manager.session_factory() as session:
            including = (
                await session.execute(
                    sa.select(
                        sa.func.avg(
                            epoch_seconds(Request.completed_at)
                            - epoch_seconds(Request.started_at)
                        )
                    )
                )
            ).scalar_one()
        assert including == pytest.approx(0.5), (
            "the fixture no longer demonstrates the dilution this exclusion "
            f"prevents (got {including})"
        )


async def test_average_duration_is_none_without_real_fetches(
    tmp_path: Path,
) -> None:
    """A run of nothing but pre-resolved requests reports no average.

    The honest answer: no request measured how long the site takes, so there
    is no average to give. Better than reporting zero.
    """
    async with SQLManager.open(tmp_path / "allpre.db") as manager:
        for i in range(3):
            await _complete(
                manager,
                f"https://example.com/pre/{i}",
                "2026-01-01 00:00:00.000",
                "2026-01-01 00:00:00.000",
                preresolved=True,
            )
        assert await manager.avg_completed_request_duration_s() is None


async def test_scheduled_retry_honours_a_sub_second_backoff(
    tmp_path: Path,
) -> None:
    """A fractional retry delay pushes ``started_at`` by that fraction.

    ``schedule_retry`` used to floor the delay with ``int()`` before handing
    it to SQLite, so any backoff under a second became ``+0 seconds`` — the
    row went straight back to being claimable and the retry happened with no
    wait at all. Only reachable with a sub-second ``retry_base_delay``, since
    the 1.0 default produces whole-second delays that survive flooring.
    """
    async with SQLManager.open(tmp_path / "retry.db") as manager:
        request_id = await manager.insert_request(
            priority=1,
            request_type=RequestType.NAVIGATING,
            method=HttpMethod.GET,
            url="https://example.com/x",
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

        for delay in (0.25, 1.5):
            await manager.schedule_retry(request_id, 0.0, delay, "boom")
            async with manager.session_factory() as session:
                started_at, waited = (
                    await session.execute(
                        sa.select(
                            # Text, not the mapped ``datetime`` — see above.
                            sa.cast(Request.started_at, sa.Text),
                            epoch_seconds(Request.started_at)
                            - epoch_seconds(
                                sa.func.strftime("%Y-%m-%d %H:%M:%f", "now")
                            ),
                        )
                    )
                ).one()

            assert started_at is not None
            assert _STAMP.match(started_at), (
                "schedule_retry writes started_at in a different format from "
                f"every other timestamp: {started_at!r}"
            )
            # Generous lower bound — the clock advances between the write and
            # the read — but far enough above zero to catch a floored delay.
            assert delay - 0.2 < waited <= delay, (
                f"a {delay}s backoff scheduled a {waited:.3f}s wait"
            )


async def _storage(
    manager: SQLManager, **kwargs: float
) -> tuple[ResponseStorage, int]:
    """A storage plus one enqueued request id to retry."""
    storage = ResponseStorage(manager, **kwargs)  # type: ignore[arg-type]
    request_id = await manager.insert_request(
        priority=1,
        request_type=RequestType.NAVIGATING,
        method=HttpMethod.GET,
        url="https://example.com/x",
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
    return storage, request_id


async def test_retry_delays_are_drawn_apart(tmp_path: Path) -> None:
    """Two requests failing at the same attempt count wait different amounts.

    The point of the jitter. Without it a pool that trips one site-wide
    throttle computes an identical delay everywhere and comes back as a
    synchronised burst, re-tripping the same limit.
    """
    async with SQLManager.open(tmp_path / "spread.db") as manager:
        storage, _ = await _storage(manager, retry_base_delay=10.0)
        delays = []
        for i in range(20):
            request_id = await manager.insert_request(
                priority=1,
                request_type=RequestType.NAVIGATING,
                method=HttpMethod.GET,
                url=f"https://example.com/{i}",
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
            delays.append(
                await storage.handle_retry(request_id, RuntimeError("boom"))
            )

    assert len(set(delays)) > 1, f"every retry drew the same delay: {delays}"
    # All within the configured band of the 10s nominal. Derived from the
    # constant rather than written out, so retuning the default jitter does not
    # land as a failure here — the invariant under test is "spread, and centred
    # on nominal", not any particular width.
    nominal = 10.0
    lo = nominal * (1 - DEFAULT_RETRY_JITTER)
    hi = nominal * (1 + DEFAULT_RETRY_JITTER)
    for delay in delays:
        assert delay is not None
        assert lo <= delay <= hi, delay


async def test_jitter_stays_within_the_configured_fraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extremes of the draw land exactly on +/- the configured fraction."""
    async with SQLManager.open(tmp_path / "band.db") as manager:
        storage, request_id = await _storage(
            manager, retry_base_delay=10.0, retry_jitter=0.05
        )

        monkeypatch.setattr(random, "uniform", lambda lo, hi: hi)
        assert await storage.handle_retry(
            request_id, RuntimeError("x")
        ) == pytest.approx(10.5)

        monkeypatch.setattr(random, "uniform", lambda lo, hi: lo)
        # retry_count is now 1, so nominal has doubled to 20s.
        assert await storage.handle_retry(
            request_id, RuntimeError("x")
        ) == pytest.approx(19.0)


async def test_downward_jitter_cannot_breach_the_one_second_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A maximally downward draw still waits at least a second.

    The floor is applied after the jitter precisely so this holds. With the
    default 1.0s base the first retry's nominal *is* the floor, so the
    downward half of the band is clipped and the first wait lands in
    [1.00, 1.05]; from the second attempt on the band is unclipped.
    """
    monkeypatch.setattr(random, "uniform", lambda lo, hi: lo)
    async with SQLManager.open(tmp_path / "floor.db") as manager:
        storage, request_id = await _storage(
            manager, retry_base_delay=1.0, retry_jitter=0.05
        )
        first = await storage.handle_retry(request_id, RuntimeError("x"))
        assert first == pytest.approx(MIN_RETRY_DELAY_S)

        # Second attempt: nominal 2.0s, so the downward draw is not clipped.
        second = await storage.handle_retry(request_id, RuntimeError("x"))
        assert second == pytest.approx(1.9)


async def test_run_metadata_records_the_jitter_fraction(
    tmp_path: Path,
) -> None:
    """The configured fraction is stored, so a corpus says how it was spread.

    The drawn delays themselves are not stored anywhere — they live only in
    each row's ``started_at`` — so this column is the only record of how much
    spread was configured.
    """
    async with SQLManager.open(tmp_path / "meta.db") as manager:
        await manager.init_run_metadata(
            scraper_name="S",
            scraper_version="1",
            num_workers=1,
            max_backoff_time=60.0,
            jitter=0.05,
        )
        async with manager.session_factory() as session:
            stored = (
                await session.execute(
                    sa.text("SELECT jitter FROM run_metadata")
                )
            ).scalar_one()
    assert stored == pytest.approx(0.05)
