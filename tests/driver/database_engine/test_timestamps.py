"""Millisecond wall-clock timestamps, and the duration math built on them.

``requests`` used to carry a parallel set of ``*_at_ns`` columns holding
``time.monotonic_ns()``. Those are gone: a monotonic value's epoch is the
writing machine's boot, so it was only meaningful inside the process that
wrote it — and a replay index compared them across source databases.
Durations now come from the text timestamps via ``unixepoch(..., 'subsec')``.

These tests hold the properties that substitution depends on: the stored
format really does carry milliseconds, it sorts chronologically as text, and
the epoch arithmetic is accurate enough to have replaced nanosecond counters.
"""

from __future__ import annotations

import math
import random
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest
import sqlalchemy as sa

from jkent.data_types import HttpMethod
from jkent.driver.database_engine import timestamps
from jkent.driver.database_engine.enums import RequestStatus
from jkent.driver.database_engine.models import Request
from jkent.driver.database_engine.sql_manager import SQLManager
from jkent.driver.database_engine.stats import get_stats
from jkent.driver.database_engine.storage import (
    DEFAULT_RETRY_JITTER,
    MIN_RETRY_DELAY_S,
)
from jkent.driver.database_engine.timestamps import (
    UtcDateTime,
    epoch_seconds,
    require_subsec_support,
)
from jkent.driver.unified_driver.persistence import ResponseStorage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import (
        AsyncEngine,
        AsyncSession,
        async_sessionmaker,
    )

    InsertRequest = Callable[..., Awaitable[int]]

#: ``2026-08-06 22:58:11.250`` — SQLite's default shape plus milliseconds.
_STAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}$")


def _uniform_high(lo: float, hi: float) -> float:
    """A ``random.uniform`` stand-in that always draws the upper bound."""
    return hi


def _uniform_low(lo: float, hi: float) -> float:
    """A ``random.uniform`` stand-in that always draws the lower bound."""
    return lo


def test_subsec_support_is_required_and_present() -> None:
    """This interpreter's SQLite can do sub-second epochs.

    The guard exists because an unrecognised modifier is not an error in
    SQLite — ``unixepoch(x, 'nosuch')`` returns NULL — so a build without
    ``'subsec'`` would turn every duration into NULL rather than failing.
    """
    require_subsec_support()  # raises if unsupported


def test_subsec_probe_closes_its_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The probe connection is closed, not left for the garbage collector."""
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(timestamps, "_subsec_checked", False)
    monkeypatch.setattr(timestamps.sqlite3, "connect", connect)
    require_subsec_support()
    monkeypatch.undo()

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")


@pytest.mark.parametrize(
    ("value", "error"),
    [
        (datetime(2026, 9, 19, 15, 54), ValueError),
        (date(2026, 9, 19), TypeError),
        ("2026-09-19 15:54:00.000", TypeError),
    ],
)
def test_utc_datetime_refuses_what_is_not_an_aware_instant(
    value: object, error: type[Exception]
) -> None:
    """A naive value cannot be made correct, and a date is not an instant."""
    process = UtcDateTime().bind_processor(
        sa.create_engine("sqlite://").dialect
    )
    assert process is not None
    with pytest.raises(error):
        process(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "stored"),
    [
        (
            datetime(
                2026, 9, 19, 15, 54, tzinfo=timezone(timedelta(hours=-5))
            ),
            "2026-09-19 20:54:00.000",
        ),
        # Floored to the millisecond, never six digits.
        (
            datetime(2026, 9, 19, 15, 54, 1, 250999, tzinfo=timezone.utc),
            "2026-09-19 15:54:01.250",
        ),
        (
            datetime(2026, 9, 19, 15, 54, 1, 999, tzinfo=timezone.utc),
            "2026-09-19 15:54:01.000",
        ),
    ],
)
async def test_utc_datetime_stores_utc_text_and_reads_back_aware(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    value: datetime,
    stored: str,
) -> None:
    req_id = await insert_request()
    async with sql_manager.session_factory() as session:
        await session.execute(
            sa.update(Request)
            .where(Request.id == req_id)
            .values(started_at=value)
        )
        await session.commit()
        text, read = (
            await session.execute(
                sa.select(
                    sa.cast(Request.started_at, sa.Text), Request.started_at
                ).where(Request.id == req_id)
            )
        ).one()

    assert text == stored
    assert read.tzinfo is timezone.utc
    assert read == value.replace(microsecond=value.microsecond // 1000 * 1000)


async def test_written_timestamps_carry_milliseconds(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """Every server-defaulted and code-written stamp has a ``.mmm`` part."""
    request_id = await insert_request()
    await sql_manager.dequeue_next_request()
    await sql_manager.mark_request_completed(request_id)

    async with sql_manager.session_factory() as session:
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


async def test_timestamp_text_sorts_chronologically(
    initialized_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
) -> None:
    """Lexicographic order is chronological order for this format.

    Relied on by ``ORDER BY created_at`` and, in a replay host, by the
    index's "most recent capture wins" comparison — which does the
    comparison in SQL on the raw column, with no parsing.
    """
    _, session_factory = initialized_db
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
            # Raw SQL: the insert path leaves ``created_at`` to the server
            # default, and this test needs to write the exact text.
            await session.execute(
                sa.text(
                    "INSERT INTO requests (status, priority, "
                    " method, url, step, current_location,"
                    " created_at) VALUES (:s, 9, :m, :u, 'parse', '',"
                    " :c)"
                ),
                {
                    "s": RequestStatus.PENDING.code,
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
                    sa.select(sa.cast(Request.created_at, sa.Text)).order_by(
                        Request.created_at
                    )
                )
            )
            .scalars()
            .all()
        )

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
    initialized_db: tuple[AsyncEngine, async_sessionmaker[AsyncSession]],
    start: str,
    end: str,
    expected: float,
) -> None:
    """``epoch_seconds`` differences are accurate to well under a microsecond.

    Not bit-exact — subtracting two ~1.7e9 doubles leaves roughly 240ns of
    representable resolution — but one to three orders of magnitude better
    than ``julianday``, which carries 6-30 microseconds of noise because it
    works in day-scale floats. Both are far finer than the millisecond the
    timestamps are quantised to; the point of the comparison is that the
    replacement for nanosecond counters did not lose meaningful precision.
    """
    _, session_factory = initialized_db
    async with session_factory() as session:
        got = (
            await session.execute(
                sa.select(
                    epoch_seconds(sa.literal(end))
                    - epoch_seconds(sa.literal(start))
                )
            )
        ).scalar_one()

    # Sub-microsecond: three orders of magnitude finer than the millisecond
    # the stored timestamps resolve to.
    assert got == pytest.approx(expected, abs=1e-6)


async def _complete(
    manager: SQLManager, url: str, start: str, end: str, *, preresolved: bool
) -> None:
    """Insert one completed request with an explicit duration.

    Raw SQL: the insert path can set neither a ``completed`` status nor
    explicit ``started_at``/``completed_at`` text, which is the duration.
    """
    async with manager.session_factory() as session:
        await session.execute(
            sa.text(
                "INSERT INTO requests (status, priority, method,"
                " url, step, current_location, started_at,"
                " completed_at, preresolved)"
                " VALUES (:s, 9, :m, :u, 'parse', '', :st, :ct, :p)"
            ),
            {
                "s": RequestStatus.COMPLETED.code,
                "m": HttpMethod.GET.code,
                "u": url,
                "st": start,
                "ct": end,
                "p": preresolved,
            },
        )
        await session.commit()


async def test_average_duration_ignores_preresolved_requests(
    sql_manager: SQLManager,
) -> None:
    """Pre-resolved rows are left out of the average request duration.

    They never touch the transport — their response was attached at enqueue
    time — so they complete inside a millisecond and record a zero duration.
    Counting them would answer "how fast is this process" when the question is
    "how long does the site take to answer". With three of them beside one real
    two-second fetch, including them would report half a second.
    """
    await _complete(
        sql_manager,
        "https://example.com/real",
        "2026-01-01 00:00:00.000",
        "2026-01-01 00:00:02.000",
        preresolved=False,
    )
    for i in range(3):
        await _complete(
            sql_manager,
            f"https://example.com/pre/{i}",
            "2026-01-01 00:00:00.000",
            "2026-01-01 00:00:00.000",
            preresolved=True,
        )

    stats = await get_stats(sql_manager.session_factory)
    assert stats.throughput.average_response_time_seconds == (
        pytest.approx(2.0)
    )

    async with sql_manager.session_factory() as session:
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


async def test_average_duration_is_unset_without_real_fetches(
    sql_manager: SQLManager,
) -> None:
    """A run of nothing but pre-resolved requests reports no average.

    No request measured how long the site takes, so the average stays at its
    unset value rather than a measured zero.
    """
    for i in range(3):
        await _complete(
            sql_manager,
            f"https://example.com/pre/{i}",
            "2026-01-01 00:00:00.000",
            "2026-01-01 00:00:00.000",
            preresolved=True,
        )
    stats = await get_stats(sql_manager.session_factory)
    assert stats.throughput.average_response_time_seconds == 0.0


async def test_scheduled_retry_honours_a_sub_second_backoff(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """A fractional retry delay pushes ``started_at`` by that fraction.

    ``schedule_retry`` used to floor the delay with ``int()`` before handing
    it to SQLite, so any backoff under a second became ``+0 seconds`` — the
    row went straight back to being claimable and the retry happened with no
    wait at all. Only reachable with a sub-second ``retry_base_delay``, since
    the 1.0 default produces whole-second delays that survive flooring.
    """
    request_id = await insert_request()

    for delay in (0.25, 1.5):
        await sql_manager.schedule_retry(request_id, 0.0, delay, "boom")
        async with sql_manager.session_factory() as session:
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
        # The clock only advances between the write and the read, so the
        # wait is at most the delay. A floored delay waits at most
        # ``floor(delay)``, so anything strictly above it proves the
        # fraction survived — no timing slack involved.
        assert math.floor(delay) < waited <= delay, (
            f"a {delay}s backoff scheduled a {waited:.3f}s wait"
        )


async def test_retry_delays_are_drawn_apart(
    sql_manager: SQLManager, insert_request: InsertRequest
) -> None:
    """Two requests failing at the same attempt count wait different amounts.

    The point of the jitter. Without it a pool that trips one site-wide
    throttle computes an identical delay everywhere and comes back as a
    synchronised burst, re-tripping the same limit.
    """
    storage = ResponseStorage(sql_manager, retry_base_delay=10.0)
    delays = []
    for i in range(20):
        request_id = await insert_request(url=f"https://example.com/{i}")
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
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The extremes of the draw land exactly on +/- the configured fraction."""
    storage = ResponseStorage(
        sql_manager, retry_base_delay=10.0, retry_jitter=0.05
    )
    request_id = await insert_request()

    monkeypatch.setattr(random, "uniform", _uniform_high)
    assert await storage.handle_retry(
        request_id, RuntimeError("x")
    ) == pytest.approx(10.5)

    monkeypatch.setattr(random, "uniform", _uniform_low)
    # retry_count is now 1, so nominal has doubled to 20s.
    assert await storage.handle_retry(
        request_id, RuntimeError("x")
    ) == pytest.approx(19.0)


async def test_downward_jitter_cannot_breach_the_one_second_floor(
    sql_manager: SQLManager,
    insert_request: InsertRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A maximally downward draw still waits at least a second.

    The floor is applied after the jitter precisely so this holds. With the
    default 1.0s base the first retry's nominal *is* the floor, so the
    downward half of the band is clipped and the first wait lands in
    [1.00, 1.05]; from the second attempt on the band is unclipped.
    """
    monkeypatch.setattr(random, "uniform", _uniform_low)
    storage = ResponseStorage(
        sql_manager, retry_base_delay=1.0, retry_jitter=0.05
    )
    request_id = await insert_request()
    first = await storage.handle_retry(request_id, RuntimeError("x"))
    assert first == pytest.approx(MIN_RETRY_DELAY_S)

    # Second attempt: nominal 2.0s, so the downward draw is not clipped.
    second = await storage.handle_retry(request_id, RuntimeError("x"))
    assert second == pytest.approx(1.9)


async def test_run_metadata_records_the_jitter_fraction(
    sql_manager: SQLManager,
) -> None:
    """The configured fraction is stored, so a corpus says how it was spread.

    The drawn delays themselves are not stored anywhere — they live only in
    each row's ``started_at`` — so this column is the only record of how much
    spread was configured.
    """
    await sql_manager.init_run_metadata(
        scraper_name="S",
        scraper_version="1",
        num_workers=1,
        max_backoff_time=60.0,
        jitter=0.05,
    )
    async with sql_manager.session_factory() as session:
        stored = (
            await session.execute(sa.text("SELECT jitter FROM run_metadata"))
        ).scalar_one()
    assert stored == pytest.approx(0.05)
