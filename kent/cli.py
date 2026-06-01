"""Kent CLI — run scrapers and start the web UI.

Usage:
    kent list                               # List available scrapers
    kent inspect module.path:ScraperClass   # Show scraper metadata
    kent inspect ... --seed-params          # Output seed params JSON
    kent serve                              # Start the persistent driver web UI
    kent run module.path:ScraperClass       # Run a scraper (default: persistent driver)
    kent run module.path:ScraperClass --driver sync
"""

from __future__ import annotations

import asyncio
import importlib
import inspect as inspect_mod
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from kent.data_types import BaseScraper


def import_scraper(scraper_path: str) -> type[BaseScraper]:
    """Import a scraper class from a dotted path.

    Args:
        scraper_path: ``"module.path:ClassName"`` string.

    Returns:
        The scraper class.

    Raises:
        click.BadParameter: If the format is invalid or import fails.
    """
    if ":" not in scraper_path:
        raise click.BadParameter(
            f"Invalid scraper path '{scraper_path}'. "
            "Expected format: 'module.path:ClassName'"
        )

    module_path, class_name = scraper_path.rsplit(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise click.BadParameter(
            f"Could not import module '{module_path}': {e}"
        ) from e

    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise click.BadParameter(
            f"Module '{module_path}' has no class '{class_name}'"
        ) from e


def _example_value(param_type: type) -> Any:
    """Return a representative example value for a parameter type."""
    from datetime import date

    from pydantic import BaseModel as PydanticBaseModel

    if param_type is int:
        return 1
    if param_type is str:
        return "example"
    if param_type is date:
        return "2025-01-01"
    if isinstance(param_type, type) and issubclass(
        param_type, PydanticBaseModel
    ):
        # Build example from the model's field info
        pydantic_model: type[PydanticBaseModel] = param_type
        example: dict[str, Any] = {}
        for field_name, field_info in pydantic_model.model_fields.items():
            annotation = field_info.annotation
            if annotation is int:
                example[field_name] = 1
            elif annotation is str:
                example[field_name] = "example"
            elif annotation is date:
                example[field_name] = "2025-01-01"
            else:
                example[field_name] = None
        return example
    return None


@click.group()
@click.version_option(package_name="kent")
def cli() -> None:
    """Kent — scraper-driver framework CLI."""


@cli.command("list")
@click.option("-v", "--verbose", is_flag=True, help="Show import errors.")
def list_scrapers(verbose: bool) -> None:
    """List available scrapers in the current directory tree.

    Scans ``.py`` files under the working directory for
    BaseScraper subclasses.
    """
    from kent.discovery import discover_scrapers

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    scrapers = sorted(
        discover_scrapers(Path.cwd()),
        key=lambda t: f"{t[0]}:{t[1]}",
    )
    if not scrapers:
        click.echo("No scrapers found.")
        return

    for module_path, class_name, cls in scrapers:
        full_path = f"{module_path}:{class_name}"
        status = getattr(cls, "status", None)
        status_str = f" [{status.value}]" if status else ""
        entries = cls.list_entries()
        entry_names = ", ".join(e.name for e in entries)
        entries_str = f"  entries: {entry_names}" if entries else ""
        click.echo(f"{full_path}{status_str}{entries_str}")


@cli.command()
@click.argument("scraper")
@click.option(
    "--seed-params",
    is_flag=True,
    help=(
        "Output only a JSON seed-params list suitable for "
        "``kent run --params``."
    ),
)
def inspect(scraper: str, seed_params: bool) -> None:
    """Inspect a scraper's metadata and entry points.

    SCRAPER is a dotted import path in the form module.path:ClassName.

    \b
    Examples:
        kent inspect kent.demo.scraper:BugCourtDemoScraper
        kent inspect kent.demo.scraper:BugCourtDemoScraper --seed-params
    """
    scraper_class = import_scraper(scraper)

    entries = scraper_class.list_entries()

    # --seed-params: emit JSON and exit
    if seed_params:
        params_list: list[dict[str, dict[str, Any]]] = []
        for entry_info in entries:
            kwargs: dict[str, Any] = {}
            for pname, ptype in entry_info.param_types.items():
                kwargs[pname] = _example_value(ptype)
            params_list.append({entry_info.name: kwargs})
        click.echo(json.dumps(params_list, indent=2))
        return

    # -- Human-readable output ----------------------------------------

    click.echo(f"Class:     {scraper_class.__name__}")
    click.echo(f"Module:    {scraper_class.__module__}")

    status = getattr(scraper_class, "status", None)
    if status is not None:
        click.echo(f"Status:    {status.value}")

    version = getattr(scraper_class, "version", "")
    if version:
        click.echo(f"Version:   {version}")

    court_url = getattr(scraper_class, "court_url", "")
    if court_url:
        click.echo(f"Court URL: {court_url}")

    court_ids: set[str] = getattr(scraper_class, "court_ids", set())
    if court_ids:
        click.echo(f"Court IDs: {', '.join(sorted(court_ids))}")

    data_types: set[str] = getattr(scraper_class, "data_types", set())
    if data_types:
        click.echo(f"Data types: {', '.join(sorted(data_types))}")

    oldest = getattr(scraper_class, "oldest_record", None)
    if oldest is not None:
        click.echo(f"Oldest record: {oldest}")

    last_verified = getattr(scraper_class, "last_verified", "")
    if last_verified:
        click.echo(f"Last verified: {last_verified}")

    if getattr(scraper_class, "requires_auth", False):
        click.echo("Auth:      required")

    rate_limits = getattr(scraper_class, "rate_limits", None)
    if rate_limits:
        parts = []
        for r in rate_limits:
            parts.append(f"{r.limit}/{r.interval}ms")
        click.echo(f"Rate limits: {', '.join(parts)}")

    driver_reqs = getattr(scraper_class, "driver_requirements", [])
    if driver_reqs:
        click.echo(f"Driver reqs: {', '.join(r.value for r in driver_reqs)}")

    # Entry points
    if entries:
        click.echo(f"\nEntry points ({len(entries)}):")
        for entry_info in entries:
            spec_tag = " [speculative]" if entry_info.speculative else ""
            returns = entry_info.return_type.__name__
            click.echo(f"  {entry_info.name}{spec_tag} -> {returns}")
            if entry_info.param_types:
                for pname, ptype in entry_info.param_types.items():
                    click.echo(f"    {pname}: {ptype.__name__}")

    # Steps
    steps = scraper_class.list_steps()
    if steps:
        click.echo(f"\nSteps ({len(steps)}):")
        for step_info in steps:
            # Determine response type tags from signature
            method = getattr(scraper_class, step_info.name, None)
            tag = ""
            if method is not None:
                sig = inspect_mod.signature(method)
                param_names = set(sig.parameters) - {"self"}
                if "page" in param_names or "lxml_tree" in param_names:
                    tag = "html"
                elif "json_content" in param_names:
                    tag = "json"
                elif "response" in param_names:
                    tag = "file"

            tag_str = f" [{tag}]" if tag else ""
            click.echo(f"  {step_info.name}{tag_str}")


@cli.command()
@click.argument("db_path", type=click.Path(exists=True))
@click.option(
    "--target",
    "target_version",
    type=int,
    default=None,
    help="Target schema version. Defaults to latest.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def migrate(db_path: str, target_version: int | None, verbose: bool) -> None:
    """Apply pending database migrations.

    DB_PATH is the path to a SQLite database file.

    \b
    Examples:
        kent migrate run.db
        kent migrate run.db --target 16
    """
    from kent.driver.persistent_driver.migrations import (
        get_current_version,
        get_latest_version,
        migrate_to,
    )

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    latest = get_latest_version()
    target = target_version if target_version is not None else latest

    async def _migrate() -> None:
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=NullPool,
        )
        try:
            current = await get_current_version(engine)
            click.echo(f"Current version: {current}")
            click.echo(f"Target version:  {target}")

            if current >= target:
                click.echo("Nothing to do.")
                return

            applied = await migrate_to(engine, target=target)
            if applied:
                click.echo(
                    f"Applied {len(applied)} migration(s): "
                    f"{', '.join(str(v) for v in applied)}"
                )
            else:
                click.echo("No migrations applied.")
        finally:
            await engine.dispose()

    asyncio.run(_migrate())


@cli.command()
@click.option(
    "--runs-dir",
    default="runs",
    show_default=True,
    help="Directory containing run databases.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=int,
    help="Port to bind the server to.",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def serve(runs_dir: str, host: str, port: int, verbose: bool) -> None:
    """Start the persistent driver web UI."""
    try:
        import uvicorn  # noqa: F811

        from kent.driver.persistent_driver.web.app import create_app
    except ImportError as e:
        raise click.ClickException(
            f"Missing dependency: {e}. "
            "Install the 'web' extra: "
            "uv install kent[web]"
        ) from e

    runs_path = Path(runs_dir)
    runs_path.mkdir(parents=True, exist_ok=True)

    app = create_app(runs_dir=runs_path)

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("kent.driver.persistent_driver").setLevel(log_level)

    click.echo(f"Starting web server at http://{host}:{port}")
    click.echo(f"Runs directory: {runs_path.absolute()}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info" if verbose else "warning",
    )


@cli.command()
@click.argument("scraper")
@click.option(
    "--driver",
    "driver_name",
    type=click.Choice(["sync", "async", "persistent", "playwright"]),
    default=None,
    help="Driver to use. Auto-selected from scraper requirements if omitted.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(),
    default=None,
    help="SQLite database path (persistent/playwright).",
)
@click.option(
    "--workers",
    type=int,
    default=1,
    show_default=True,
    help="Number of concurrent workers (async/persistent/playwright).",
)
@click.option(
    "--max-workers",
    type=int,
    default=None,
    help=(
        "Maximum number of workers for dynamic scaling "
        "(persistent/playwright). Defaults to 10."
    ),
)
@click.option(
    "--storage",
    type=click.Path(),
    default=None,
    help="Directory for downloaded files.",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Start fresh instead of resuming (persistent/playwright).",
)
@click.option(
    "--params",
    "params_json",
    default=None,
    help=(
        "JSON list of seed parameters for initial_seed(). "
        "Example: '[{\"get_oral_arguments\": {}}]'. "
        "Use ``kent inspect --seed-params`` to generate a template. "
        "Rejected if the database already has a run; use --add-params "
        "to layer entries onto an existing run."
    ),
)
@click.option(
    "--add-params",
    "add_params_json",
    default=None,
    help=(
        "JSON list of seed parameters to add to an existing run. "
        "Runs initial_seed() with these params and enqueues the resulting "
        "requests before continuing the run. Mutually exclusive with "
        "--params."
    ),
)
@click.option(
    "--headed",
    is_flag=True,
    help="Show the browser window (playwright driver only).",
)
@click.option(
    "--browser-profile",
    "browser_profile_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help=(
        "Path to a browser profile directory (contains manifest.json). "
        "Configures browser launch for sites with bot protection. "
        "Only supported with --driver playwright."
    ),
)
@click.option(
    "--skip-archive",
    is_flag=True,
    help="Skip archive requests; local_filepath will be 'skipped'.",
)
@click.option(
    "--proxy",
    default=None,
    metavar="URL",
    help=(
        "Route requests through a proxy. Accepts http://, https://, "
        "socks4://, or socks5:// URLs, with optional credentials "
        "(e.g. socks5://user:pass@host:1080). Applied to HTTP drivers "
        "and the Playwright browser alike."
    ),
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def run(
    scraper: str,
    driver_name: str | None,
    db_path: str | None,
    workers: int,
    max_workers: int | None,
    storage: str | None,
    no_resume: bool,
    params_json: str | None,
    add_params_json: str | None,
    headed: bool,
    browser_profile_path: str | None,
    skip_archive: bool,
    proxy: str | None,
    verbose: bool,
) -> None:
    """Run a scraper with the chosen driver.

    SCRAPER is a dotted import path in the form module.path:ClassName.

    If --driver is omitted, the driver is auto-selected from the scraper's
    driver_requirements. JS_EVAL, FF_ALIKE, or CHROME_ALIKE all select
    the playwright driver. FF_ALIKE and CHROME_ALIKE also auto-resolve
    a browser profile from $KENT_HOME/profiles/{firefox,chrome}/.

    \b
    Examples:
        kent run kent.demo.scraper:BugCourtDemoScraper
        kent run kent.demo.scraper:BugCourtDemoScraper --driver sync
        kent run my.scraper:MyScraper --driver persistent --db run.db
        kent run my.scraper:MyScraper --params '[{"get_entry": {}}]'
    """
    import os

    from kent.data_types import DriverRequirement

    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scraper_class = import_scraper(scraper)
    scraper_instance = scraper_class()
    scraper_name = scraper_class.__name__

    storage_dir = Path(storage) if storage else None
    if storage_dir:
        storage_dir.mkdir(parents=True, exist_ok=True)

    if params_json is not None and add_params_json is not None:
        raise click.UsageError(
            "--params and --add-params are mutually exclusive. "
            "Use --params on a fresh database, --add-params on an existing one."
        )

    seed_params: list[dict[str, dict[str, Any]]] | None = None
    if params_json is not None:
        try:
            seed_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --params: {e}") from e
        if not isinstance(seed_params, list):
            raise click.BadParameter("--params must be a JSON list")

    add_seed_params: list[dict[str, dict[str, Any]]] | None = None
    if add_params_json is not None:
        try:
            add_seed_params = json.loads(add_params_json)
        except json.JSONDecodeError as e:
            raise click.BadParameter(
                f"Invalid JSON for --add-params: {e}"
            ) from e
        if not isinstance(add_seed_params, list) or not add_seed_params:
            raise click.BadParameter(
                "--add-params must be a non-empty JSON list"
            )

    # --- Driver auto-selection from scraper requirements ---
    reqs = getattr(scraper_class, "driver_requirements", [])
    user_chose_driver = driver_name is not None

    # Validate: FF_ALIKE and CHROME_ALIKE are mutually exclusive
    if (
        DriverRequirement.FF_ALIKE in reqs
        and DriverRequirement.CHROME_ALIKE in reqs
    ):
        raise click.UsageError(
            f"Scraper '{scraper_name}' declares both FF_ALIKE and "
            f"CHROME_ALIKE driver requirements. These are mutually exclusive."
        )

    needs_playwright = any(
        r in reqs
        for r in (
            DriverRequirement.JS_EVAL,
            DriverRequirement.FF_ALIKE,
            DriverRequirement.CHROME_ALIKE,
            DriverRequirement.HCAP_HANDLER,
            DriverRequirement.RCAP_HANDLER,
            DriverRequirement.STRICTLY_SERIAL,
        )
    )

    if not user_chose_driver:
        driver_name = "playwright" if needs_playwright else "persistent"
    elif needs_playwright and driver_name not in ("playwright",):
        click.echo(
            f"Warning: Scraper requires {[r.value for r in reqs]} "
            f"but --driver {driver_name} was explicitly chosen.",
            err=True,
        )

    if DriverRequirement.STRICTLY_SERIAL in reqs:
        if workers != 1 or (max_workers is not None and max_workers != 1):
            click.echo(
                f"Warning: Scraper '{scraper_name}' requires STRICTLY_SERIAL; "
                f"overriding --workers/--max-workers to 1.",
                err=True,
            )
        workers = 1
        max_workers = 1

    # Auto-resolve browser profile from $KENT_HOME/profiles/{name}/
    if not browser_profile_path and not user_chose_driver:
        profile_name: str | None = None
        if DriverRequirement.FF_ALIKE in reqs:
            profile_name = "firefox"
        elif DriverRequirement.CHROME_ALIKE in reqs:
            profile_name = "chrome"

        if profile_name is not None:
            kent_home = Path(
                os.environ.get("KENT_HOME", "~/.kent")
            ).expanduser()
            resolved_path = kent_home / "profiles" / profile_name
            if not resolved_path.is_dir():
                raise click.UsageError(
                    f"Scraper requires {profile_name} browser profile but "
                    f"no profile found at {resolved_path}.\n"
                    f"Create the directory with a manifest.json, or pass "
                    f"--browser-profile explicitly."
                )
            browser_profile_path = str(resolved_path)
            click.echo(f"Profile: {resolved_path} (auto-resolved)")

    if browser_profile_path and driver_name != "playwright":
        raise click.UsageError(
            "--browser-profile is only supported with --driver playwright"
        )

    if add_seed_params is not None and driver_name not in (
        "persistent",
        "playwright",
    ):
        raise click.UsageError(
            "--add-params is only supported with the persistent and "
            "playwright drivers (it requires a request database)."
        )

    click.echo(f"Scraper: {scraper_name}")
    click.echo(f"Driver:  {driver_name}")

    if driver_name == "sync":
        _run_sync(
            scraper_instance,
            storage_dir,
            seed_params,
            skip_archive=skip_archive,
            proxy=proxy,
        )
    elif driver_name == "async":
        _run_async(
            scraper_instance,
            storage_dir,
            workers,
            seed_params,
            skip_archive=skip_archive,
            proxy=proxy,
        )
    elif driver_name == "persistent":
        _run_persistent(
            scraper_instance,
            scraper_name,
            db_path,
            storage_dir,
            workers,
            no_resume,
            seed_params,
            max_workers=max_workers,
            skip_archive=skip_archive,
            proxy=proxy,
            add_seed_params=add_seed_params,
        )
    elif driver_name == "playwright":
        _run_playwright(
            scraper_instance,
            scraper_name,
            db_path,
            storage_dir,
            workers,
            no_resume,
            seed_params,
            headed=headed,
            browser_profile_path=browser_profile_path,
            max_workers=max_workers,
            skip_archive=skip_archive,
            proxy=proxy,
            add_seed_params=add_seed_params,
        )


# ------------------------------------------------------------------
# Driver runners
# ------------------------------------------------------------------


def _run_sync(
    scraper: Any,
    storage_dir: Path | None,
    seed_params: list[dict[str, dict[str, Any]]] | None,
    *,
    skip_archive: bool = False,
    proxy: str | None = None,
) -> None:
    from kent.driver.archive_handler import NoDownloadsSyncArchiveHandler
    from kent.driver.sync_driver import SyncDriver

    archive_handler = NoDownloadsSyncArchiveHandler() if skip_archive else None
    driver = SyncDriver(
        scraper=scraper,
        storage_dir=storage_dir,
        archive_handler=archive_handler,
        proxy=proxy,
    )
    driver.seed_params = seed_params
    driver.run()
    click.echo("Done.")


def _run_async(
    scraper: Any,
    storage_dir: Path | None,
    workers: int,
    seed_params: list[dict[str, dict[str, Any]]] | None,
    *,
    skip_archive: bool = False,
    proxy: str | None = None,
) -> None:
    from kent.driver.archive_handler import NoDownloadsAsyncArchiveHandler
    from kent.driver.async_driver import AsyncDriver

    archive_handler = (
        NoDownloadsAsyncArchiveHandler() if skip_archive else None
    )

    async def _go() -> None:
        driver = AsyncDriver(
            scraper=scraper,
            storage_dir=storage_dir,
            num_workers=workers,
            archive_handler=archive_handler,
            proxy=proxy,
        )
        driver.seed_params = seed_params
        await driver.run()

    asyncio.run(_go())
    click.echo("Done.")


async def _reject_params_on_existing_db(db_path: Path) -> None:
    """Raise if ``db_path`` is a database that already has a run.

    Called when the user passes ``--params`` so they don't silently
    clobber or be ignored by an existing run.  ``--add-params`` is the
    intended path for layering entries onto an existing database.
    """
    if not db_path.exists():
        return
    from kent.driver.persistent_driver.sql_manager import SQLManager

    async with SQLManager.open(db_path) as sql:
        existing = await sql.get_run_metadata()
    if existing is not None:
        raise click.ClickException(
            f"Database {db_path} already has a run for scraper "
            f"'{existing.get('scraper_name')}'. --params is only valid on "
            "a fresh database; use --add-params to add entries to an "
            "existing run, or delete the database to start over."
        )


def _run_persistent(
    scraper: Any,
    scraper_name: str,
    db_path: str | None,
    storage_dir: Path | None,
    workers: int,
    no_resume: bool,
    seed_params: list[dict[str, dict[str, Any]]] | None,
    *,
    max_workers: int | None = None,
    skip_archive: bool = False,
    proxy: str | None = None,
    add_seed_params: list[dict[str, dict[str, Any]]] | None = None,
) -> None:
    try:
        from kent.driver.persistent_driver import PersistentDriver
    except ImportError as e:
        raise click.ClickException(f"Missing dependency: {e}. ") from e

    resolved_db = Path(db_path) if db_path else Path(f"{scraper_name}.db")
    click.echo(f"Database: {resolved_db}")

    async def _go() -> None:
        if seed_params is not None:
            await _reject_params_on_existing_db(resolved_db)
        open_kwargs: dict[str, Any] = {
            "scraper": scraper,
            "db_path": resolved_db,
            "storage_dir": storage_dir,
            "num_workers": workers,
            "resume": not no_resume,
            "seed_params": seed_params,
        }
        if max_workers is not None:
            open_kwargs["max_workers"] = max_workers
        if proxy is not None:
            open_kwargs["proxy"] = proxy
        async with PersistentDriver.open(**open_kwargs) as driver:
            if skip_archive:
                from kent.driver.archive_handler import (
                    NoDownloadsAsyncArchiveHandler,
                )

                driver.archive_handler = NoDownloadsAsyncArchiveHandler()
            if add_seed_params is not None:
                await driver.add_seed_params(add_seed_params)
            await driver.run()

    asyncio.run(_go())
    click.echo("Done.")


def _run_playwright(
    scraper: Any,
    scraper_name: str,
    db_path: str | None,
    storage_dir: Path | None,
    workers: int,
    no_resume: bool,
    seed_params: list[dict[str, dict[str, Any]]] | None,
    *,
    headed: bool = False,
    browser_profile_path: str | None = None,
    max_workers: int | None = None,
    skip_archive: bool = False,
    proxy: str | None = None,
    add_seed_params: list[dict[str, dict[str, Any]]] | None = None,
) -> None:
    try:
        from kent.driver.playwright_driver import PlaywrightDriver
    except ImportError as e:
        raise click.ClickException(f"Missing dependency: {e}. ") from e

    # Load browser profile if provided
    browser_profile = None
    if browser_profile_path:
        from kent.driver.playwright_driver.browser_profile import (
            load_browser_profile,
        )

        try:
            browser_profile = load_browser_profile(Path(browser_profile_path))
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"Invalid browser profile: {e}") from e
        click.echo(f"Profile: {browser_profile.name}")

    resolved_db = Path(db_path) if db_path else Path(f"{scraper_name}.db")
    click.echo(f"Database: {resolved_db}")

    async def _go() -> None:
        if seed_params is not None:
            await _reject_params_on_existing_db(resolved_db)
        open_kwargs: dict[str, Any] = {
            "scraper": scraper,
            "db_path": resolved_db,
            "storage_dir": storage_dir,
            "num_workers": workers,
            "resume": not no_resume,
            "seed_params": seed_params,
            "headless": not headed,
            "browser_profile": browser_profile,
        }
        if max_workers is not None:
            open_kwargs["max_workers"] = max_workers
        if proxy is not None:
            open_kwargs["proxy"] = proxy
        async with PlaywrightDriver.open(**open_kwargs) as driver:
            if skip_archive:
                from kent.driver.archive_handler import (
                    NoDownloadsAsyncArchiveHandler,
                )

                driver.archive_handler = NoDownloadsAsyncArchiveHandler()
            if add_seed_params is not None:
                await driver.add_seed_params(add_seed_params)
            await driver.run()

    asyncio.run(_go())
    click.echo("Done.")


def main() -> None:
    """Entry point for the ``kent`` console script."""
    cli()
