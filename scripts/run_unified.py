#!/usr/bin/env python
"""Run a scraper through the **unified driver** (`ScrapeRun`).

This is jkent's own entry point for running a scraper — it needs nothing but `jkent[operational]` and an
importable scraper module.

    uv run python scripts/run_unified.py \
        --db runs/foo.db --storage runs/foo-files \
        --params '[{"enumerate_dockets": {...}}]' \
        juriscraper.state.new_york.nycourts_gov.scraper:Site

All the wiring (transport auto-selection from ``driver_requirements``,
browser-profile resolution from ``$JKENT_HOME/profiles``, DB pre-init for
browser transports, archive handler) lives in
:class:`jkent.driver.unified_driver.RunBootstrapper`, and STRICTLY_SERIAL
capping in :meth:`RunConfig.for_scraper`; this script is argv parsing +
progress printing around them. Ctrl-C triggers a graceful, resumable
shutdown. Re-running against the same ``--db`` resumes it. A run database
is pinned to its seed set: ``--params`` is rejected once the database has
requests, so to run different params, use a new ``--db``.

Exit status: 0 when the run completed, 130 when it was interrupted (SIGINT or
SIGTERM, which stop it gracefully and resumably, or Ctrl-C during startup),
and 1 when it raised.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from jkent.driver.database_engine.enums import RunStatus
from jkent.driver.unified_driver import RunBootstrapper, RunConfig, RunHooks

# The shell's code for a SIGINT kill; also used for a graceful SIGTERM stop,
# since both leave a resumable run a supervisor must not read as success.
EXIT_INTERRUPTED = 130


def _import_scraper(path: str) -> type:
    """Import a ``module.path:ClassName`` scraper class."""
    if ":" not in path:
        raise SystemExit(f"bad scraper path {path!r}; want 'module:Class'")
    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _parse_params(raw: str | None, flag: str) -> list[dict[str, Any]] | None:
    if not raw:
        return None
    try:
        params = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} is not valid JSON: {exc}")
    if not isinstance(params, list):
        raise SystemExit(f"{flag} must be a JSON list")
    return params


async def _run(args: argparse.Namespace) -> int:
    scraper_cls = _import_scraper(args.scraper)
    scraper = scraper_cls()
    print(f"Scraper:   {args.scraper}")

    db_path = Path(args.db)
    storage_dir = Path(args.storage)
    print(f"Database:  {db_path}")
    print(f"Storage:   {storage_dir}")

    seed_params = _parse_params(args.params, "--params")

    seen = {"data": 0}
    final: dict[str, str] = {}

    async def on_data(_data: Any) -> None:
        seen["data"] += 1
        if seen["data"] % 50 == 0:
            print(f"  … {seen['data']} records so far")

    async def on_progress(event: str, data: dict[str, Any]) -> None:
        if event in ("run_started", "run_completed"):
            print(f"[{event}] {data}")

    async def on_run_complete(
        name: str, status: str, error: Exception | None
    ) -> None:
        final["status"] = status
        print(
            f"Run complete: {name} → {status}"
            + (f" ({error})" if error else "")
        )

    bootstrapper = RunBootstrapper(
        scraper,
        db_path,
        config=RunConfig(
            seed_params=seed_params,
            num_workers=args.workers,
            worker_ramp_interval=args.worker_ramp,
        ),
        hooks=RunHooks(
            on_data=on_data,
            on_progress=on_progress,
            on_run_complete=on_run_complete,
        ),
        storage_dir=storage_dir,
        headless=not args.headed,
        proxy=args.proxy,
    )
    async with bootstrapper as run:
        await run.run()
        print(f"Status: {await run.status()} — {seen['data']} records emitted")
    status = final.get("status")
    if status == RunStatus.COMPLETED.value:
        return 0
    if status == RunStatus.INTERRUPTED.value:
        return EXIT_INTERRUPTED
    return 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "scraper",
        help="Scraper coordinates as 'module.path:ClassName'.",
    )
    p.add_argument(
        "--db",
        required=True,
        help="Run database path (created if missing; re-running resumes).",
    )
    p.add_argument(
        "--storage",
        required=True,
        help="Archive download directory.",
    )
    p.add_argument(
        "--params",
        default=None,
        help="JSON list of seed invocations for initial_seed() "
        "(rejected once the database has requests).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Worker count, pinned for the whole run. Keep at 1 against "
        "Cloudflare-protected sites.",
    )
    p.add_argument(
        "--worker-ramp",
        type=float,
        default=0.0,
        dest="worker_ramp",
        help="Seconds between worker spawns at startup (0 = all at once). "
        "Staggers arrival so --workers browser pages don't launch "
        "simultaneously; the final pool size is unchanged.",
    )
    p.add_argument("--proxy", default=None)
    p.add_argument(
        "--headed",
        action=argparse.BooleanOptionalAction,
        # The docker image exports JKENT_HEADED=1: it exists for the headed
        # Cloudflare OS-click path, so headed is its default.
        default=os.environ.get("JKENT_HEADED") == "1",
        help="Run the browser headed (browser transports only). Defaults "
        "to headed when JKENT_HEADED=1, as in the docker image.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Once the run is open, SIGINT is routed to ``run.stop()`` by the
    # bootstrapper and ``run()`` returns normally, with the exit code taken
    # from the final status; this only catches an interrupt during startup
    # (imports, transport/browser launch).
    try:
        code = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted during startup.")
        code = EXIT_INTERRUPTED
    sys.exit(code)


if __name__ == "__main__":
    main()
