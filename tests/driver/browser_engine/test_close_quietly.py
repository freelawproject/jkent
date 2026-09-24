"""Engine teardown failures are visible.

``close_quietly`` is the one place a browser-teardown error is swallowed so
the rest of the teardown still runs. That swallowed error is also the only
trace of a leaked browser process, so it must surface at WARNING with the
traceback — a DEBUG record is off in every production configuration.
"""

from __future__ import annotations

import logging

import pytest

from jkent.driver.browser_engine.engines.base import close_quietly

_LOGGER = "jkent.driver.browser_engine.engines.base"


async def test_close_failure_is_logged_at_warning_with_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def boom() -> None:
        raise RuntimeError("browser already gone")

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await close_quietly("browser", boom)  # must not raise

    records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("closing browser failed" in r.getMessage() for r in records)
    assert any(r.exc_info for r in records), "traceback must travel"
