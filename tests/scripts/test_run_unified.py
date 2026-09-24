"""``scripts/run_unified.py`` — argv reaches ``RunBootstrapper`` unaltered."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from jkent.driver.database_engine.enums import RunStatus

_SCRIPT = Path(__file__).parents[2] / "scripts" / "run_unified.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_unified", _SCRIPT)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class _Scraper:
    pass


def _import_fake_scraper(_path: str) -> type:
    return _Scraper


async def test_params_reach_bootstrapper_when_db_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A launch failure leaves a schema'd, request-less database; the retry
    # with the same --params must seed them, not every no-arg entry. The
    # bootstrapper owns the existing-run guard.
    script = _load_script()
    captured: dict[str, Any] = {}

    class _Bootstrapper:
        def __init__(self, scraper: Any, db_path: Path, **kwargs: Any) -> None:
            captured.update(kwargs)
            raise SystemExit(0)

    monkeypatch.setattr(script, "RunBootstrapper", _Bootstrapper)
    monkeypatch.setattr(script, "_import_scraper", _import_fake_scraper)
    db_path = tmp_path / "run.db"
    db_path.touch()
    params = [{"fetch_page": {"page_id": 1}}]
    args = argparse.Namespace(
        scraper="x:Y",
        db=str(db_path),
        storage=str(tmp_path / "files"),
        params=json.dumps(params),
        workers=1,
        worker_ramp=0.0,
        headed=False,
        proxy=None,
    )

    with pytest.raises(SystemExit):
        await script._run(args)

    assert captured["config"].seed_params == params


def _fake_run_bootstrapper(final_status: Any) -> type:
    class _Run:
        def __init__(self, hooks: Any) -> None:
            self._hooks = hooks

        async def run(self) -> None:
            await self._hooks.on_run_complete("Y", final_status, None)

        async def status(self) -> str:
            return "in_progress"

    class _Bootstrapper:
        def __init__(self, scraper: Any, db_path: Path, **kwargs: Any) -> None:
            self._run = _Run(kwargs["hooks"])

        async def __aenter__(self) -> _Run:
            return self._run

        async def __aexit__(self, *_exc: object) -> None:
            return None

    return _Bootstrapper


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scraper="x:Y",
        db=str(tmp_path / "run.db"),
        storage=str(tmp_path / "files"),
        params=None,
        workers=1,
        worker_ramp=0.0,
        headed=False,
        proxy=None,
        verbose=False,
    )


@pytest.mark.parametrize(
    ("final_status", "code"),
    [
        (RunStatus.COMPLETED, 0),
        # SIGINT/SIGTERM stop the run gracefully and run() returns normally;
        # a supervisor (docker stop, CI) must not read that as success.
        (RunStatus.INTERRUPTED, 130),
    ],
)
def test_exit_code_reflects_final_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    final_status: RunStatus,
    code: int,
) -> None:
    script = _load_script()
    monkeypatch.setattr(
        script, "RunBootstrapper", _fake_run_bootstrapper(final_status)
    )
    monkeypatch.setattr(script, "_import_scraper", _import_fake_scraper)
    monkeypatch.setattr(script, "_parse_args", lambda: _args(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert exc_info.value.code == code


def test_interrupt_during_startup_exits_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script()

    def _interrupt(_path: str) -> type:
        raise KeyboardInterrupt

    monkeypatch.setattr(script, "_import_scraper", _interrupt)
    monkeypatch.setattr(script, "_parse_args", lambda: _args(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        script.main()

    assert exc_info.value.code == 130


@pytest.mark.parametrize(
    ("env", "argv", "headed"),
    [
        (None, [], False),
        # The Xvfb image exports JKENT_HEADED=1: headed is its default, since
        # the Cloudflare OS-click path needs a mapped window.
        ("1", [], True),
        ("1", ["--no-headed"], False),
        (None, ["--headed"], True),
    ],
)
def test_headed_defaults_from_jkent_headed(
    monkeypatch: pytest.MonkeyPatch,
    env: str | None,
    argv: list[str],
    headed: bool,
) -> None:
    script = _load_script()
    if env is None:
        monkeypatch.delenv("JKENT_HEADED", raising=False)
    else:
        monkeypatch.setenv("JKENT_HEADED", env)
    monkeypatch.setattr(
        "sys.argv",
        ["run_unified.py", "x:Y", "--db", "d", "--storage", "s", *argv],
    )

    assert script._parse_args().headed is headed
