#!/usr/bin/env python
"""Run a scraper through the **unified driver** (`ScrapeRun`) — a manual-test
stand-in for `jjkent run`.

Equivalent of, e.g.:

    uv run jjkent run --db runs/NYCoA-full-2026-06-08.db \
       --storage runs/NYApp-files \
       --params '[{"enumerate_dockets": {...}}]' \
       juriscraper.state.new_york.nycourts_gov.scraper:Site

but it drives the unified `ScrapeRun` instead of the legacy `PersistentDriver`,
so you can exercise the new transport/worker/run stack against a real scraper.

Run it the same way you'd run `jjkent run` (from a project whose env has both
`jjkent` and the target scraper importable), e.g. from `../juriscraper`:

    uv run python ../kent/scripts/run_unified.py \
        --db runs/foo.db --storage runs/foo-files \
        --params '[...]' my.module:Scraper

All the wiring (transport auto-selection from ``driver_requirements``,
browser-profile resolution from ``$JKENT_HOME/profiles``, DB pre-init for
browser transports, archive handler, STRICTLY_SERIAL capping) lives in
:class:`jkent.driver.unified_driver.RunBootstrapper`; this script is argv
parsing + progress printing around it. Ctrl-C triggers a graceful,
resumable shutdown. Re-running against the same ``--db`` resumes
(``--params`` is ignored on resume; use ``--add-params`` to layer new
invocations onto an existing run).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
from pathlib import Path
from typing import Any

from jkent.driver.unified_driver import RunBootstrapper


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
    params = json.loads(raw)
    if not isinstance(params, list):
        raise SystemExit(f"{flag} must be a JSON list")
    return params


async def _run(args: argparse.Namespace) -> None:
    scraper_cls = _import_scraper(args.scraper)
    scraper = scraper_cls()
    print(f"Scraper:   {args.scraper}")

    db_path = Path(args.db)
    storage_dir = Path(args.storage)
    print(f"Database:  {db_path}")
    print(f"Storage:   {storage_dir}")

    seed_params = _parse_params(args.params, "--params")
    add_params = _parse_params(args.add_params, "--add-params")
    if db_path.exists() and seed_params is not None:
        if add_params is not None:
            raise SystemExit(
                "--params and --add-params are mutually exclusive"
            )
        print(
            "Resuming an existing run — ignoring --params "
            "(use --add-params to add entries)."
        )
        seed_params = None

    seen = {"data": 0}

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
        print(
            f"Run complete: {name} → {status}"
            + (f" ({error})" if error else "")
        )

    bootstrapper = RunBootstrapper(
        scraper,
        db_path,
        storage_dir=storage_dir,
        seed_params=seed_params,
        add_params=add_params,
        resume=not args.no_resume,
        num_workers=args.workers,
        max_workers=args.max_workers,
        headless=not args.headed,
        proxy=args.proxy,
        on_data=on_data,
        on_progress=on_progress,
        on_run_complete=on_run_complete,
    )
    async with bootstrapper as run:
        await run.run()
        print(f"Status: {await run.status()} — {seen['data']} records emitted")


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
        "(fresh runs only; ignored on resume).",
    )
    p.add_argument(
        "--add-params",
        default=None,
        dest="add_params",
        help="JSON list of invocations to add to an existing run.",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=10, dest="max_workers")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--proxy", default=None)
    p.add_argument(
        "--headed",
        action="store_true",
        help="Run the browser headed (browser transports only).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nInterrupted — run is resumable (re-run the same --db).")


if __name__ == "__main__":
    main()
