"""CLI commands: ``pdd replay {strict,lenient,error-stubs}``.

Each subcommand drives :class:`LocalOnlyDriver` against one or more source
DBs, producing an output DB that is itself a valid PersistentDriver
resumable run. The three subcommands map to the three replay modes:

- ``strict``     → ``prev-error-free`` (previously-errored rows fall
  through to the miss policy).
- ``lenient``    → ``curr-error-free`` (retry previously-errored
  continuations against the stored response).
- ``error-stubs``→ ``desc-error-free`` (HATEOAS-aware re-seeding of
  errored subtrees; replaces ``pdd seed-error-patch-rerun``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click

from kent.cli import import_scraper
from kent.driver.local_only_driver import LocalOnlyDriver
from kent.driver.persistent_driver.cli import cli

F = TypeVar("F", bound=Callable[..., Any])


@cli.group(
    "replay",
    help=(
        "Replay a scraper from one or more source DBs instead of the "
        "network. Choose a subcommand: `strict` is conservative, "
        "`lenient` retries previously-errored continuations, "
        "`error-stubs` re-seeds errored subtrees at HATEOAS-safe "
        "ancestors (replaces seed-error-patch-rerun)."
    ),
)
def replay() -> None:
    """Replay a scraper from source DBs. See subcommands."""


def _shared_options(f: F) -> F:
    """Decorator that adds the options every replay subcommand needs."""
    f = click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")(
        f
    )
    f = click.option(
        "--params",
        "params_json",
        default=None,
        help=(
            "JSON list of entry-method invocations, identical to "
            "`kent run --params`. Used to seed the output DB on first run."
        ),
    )(f)
    f = click.option(
        "--index-db",
        "index_db_path",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help=(
            "Path for the on-disk source-routing index. Default is "
            "in-memory (:memory:); set this for very large consolidations."
        ),
    )(f)
    f = click.option(
        "--workers",
        type=int,
        default=4,
        show_default=True,
        help="Number of concurrent worker tasks.",
    )(f)
    f = click.option(
        "--output",
        "output_path",
        type=click.Path(dir_okay=False, path_type=Path),
        required=True,
        help="Path to the output DB (created if missing).",
    )(f)
    f = click.option(
        "--source-db",
        "source_dbs",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        multiple=True,
        required=True,
        help="Source DB(s) to read responses from. May be repeated.",
    )(f)
    f = click.option(
        "--scraper",
        "scraper_path",
        required=True,
        help="Dotted import path `module.path:ClassName`.",
    )(f)
    return f


def _miss_option(default: str) -> Callable[[F], F]:
    """`--miss` option factory; default differs per subcommand."""

    def wrap(f: F) -> F:
        return click.option(
            "--miss",
            "miss_policy",
            type=click.Choice(["raise", "skip", "stub"]),
            default=default,
            show_default=True,
            help=(
                "What to do when a yielded request has no source match. "
                "`stub` writes the request to the output as "
                "status='pending' for a downstream `kent run` to fetch."
            ),
        )(f)

    return wrap


@replay.command("strict")
@_shared_options
@_miss_option(default="stub")
def replay_strict(
    scraper_path: str,
    source_dbs: tuple[Path, ...],
    output_path: Path,
    workers: int,
    index_db_path: Path | None,
    params_json: str | None,
    verbose: bool,
    miss_policy: str,
) -> None:
    """Conservative replay: previously-errored rows fall through to --miss.

    Mode: ``prev-error-free``. The source index excludes any row whose
    unresolved error is parser-side (HTMLStructuralAssumption or
    DataFormatAssumption). The scraper still re-executes against every
    fulfillable response — only the previously-errored rows are skipped.
    Combine with ``--miss stub`` (default) to stage those rows as
    pending for a downstream ``kent run``.
    """
    _run_replay(
        scraper_path=scraper_path,
        source_dbs=list(source_dbs),
        output_path=output_path,
        mode="prev-error-free",
        miss_policy=miss_policy,
        trust_subtree_after_retry=False,
        workers=workers,
        index_db_path=index_db_path,
        params_json=params_json,
        verbose=verbose,
    )


@replay.command("lenient")
@_shared_options
@_miss_option(default="stub")
@click.option(
    "--trust-subtree-after-retry/--no-trust-subtree-after-retry",
    default=False,
    show_default=True,
    help=(
        "When a previously-errored continuation now succeeds: with "
        "--trust-subtree-after-retry, its yielded children go through "
        "the normal source-DB lookup. Default (off): children are "
        "unconditionally stubbed as pending for re-fetch."
    ),
)
def replay_lenient(
    scraper_path: str,
    source_dbs: tuple[Path, ...],
    output_path: Path,
    workers: int,
    index_db_path: Path | None,
    params_json: str | None,
    verbose: bool,
    miss_policy: str,
    trust_subtree_after_retry: bool,
) -> None:
    """Lenient replay: retry previously-errored continuations.

    Mode: ``curr-error-free``. Previously-errored rows whose error type
    indicates a parser break (HTMLStructuralAssumption /
    DataFormatAssumption) are served from source and the current
    scraper code re-executes against them. If the continuation now
    succeeds, the row is written completed in the output; its yielded
    children become pending (unless ``--trust-subtree-after-retry``).
    """
    _run_replay(
        scraper_path=scraper_path,
        source_dbs=list(source_dbs),
        output_path=output_path,
        mode="curr-error-free",
        miss_policy=miss_policy,
        trust_subtree_after_retry=trust_subtree_after_retry,
        workers=workers,
        index_db_path=index_db_path,
        params_json=params_json,
        verbose=verbose,
    )


@replay.command("error-stubs")
@_shared_options
def replay_error_stubs(
    scraper_path: str,
    source_dbs: tuple[Path, ...],
    output_path: Path,
    workers: int,
    index_db_path: Path | None,
    params_json: str | None,
    verbose: bool,
) -> None:
    """Re-seed errored subtrees at the nearest HATEOAS-safe ancestor.

    Mode: ``desc-error-free``. Pre-pass walks each errored row in the
    source(s) up the parent chain to the nearest ``hateoas=True``
    ancestor (or to the root, if none is marked); that ancestor enters
    the output as a pending entry request. The error-free portion of
    each source's request graph copies over unchanged. Replaces
    ``pdd seed-error-patch-rerun``.

    Miss policy is locked to ``stub`` — that's the whole point.
    """
    _run_replay(
        scraper_path=scraper_path,
        source_dbs=list(source_dbs),
        output_path=output_path,
        mode="desc-error-free",
        miss_policy="stub",
        trust_subtree_after_retry=False,
        workers=workers,
        index_db_path=index_db_path,
        params_json=params_json,
        verbose=verbose,
    )


def _run_replay(
    *,
    scraper_path: str,
    source_dbs: list[Path],
    output_path: Path,
    mode: str,
    miss_policy: str,
    trust_subtree_after_retry: bool,
    workers: int,
    index_db_path: Path | None,
    params_json: str | None,
    verbose: bool,
) -> None:
    """Drive a LocalOnlyDriver.open(...).run() invocation from CLI args."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    scraper_class = import_scraper(scraper_path)
    scraper_instance = scraper_class()
    scraper_name = scraper_class.__name__

    seed_params: list[dict[str, dict[str, Any]]] | None = None
    if params_json is not None:
        try:
            seed_params = json.loads(params_json)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON for --params: {e}") from e
        if not isinstance(seed_params, list):
            raise click.BadParameter("--params must be a JSON list")

    click.echo(f"Scraper:   {scraper_name}")
    click.echo(f"Mode:      {mode}")
    click.echo(f"Miss:      {miss_policy}")
    click.echo(f"Sources:   {[str(p) for p in source_dbs]}")
    click.echo(f"Output:    {output_path}")
    click.echo(f"Workers:   {workers}")

    async def _go() -> None:
        async with LocalOnlyDriver.open(
            scraper=scraper_instance,
            db_path=output_path,
            source_db_paths=source_dbs,
            miss_policy=miss_policy,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            trust_subtree_after_retry=trust_subtree_after_retry,
            index_db_path=index_db_path,
            num_workers=workers,
            seed_params=seed_params,
        ) as driver:
            await driver.run()

    try:
        asyncio.run(_go())
    except Exception as exc:
        click.echo(f"replay failed: {exc}", err=True)
        sys.exit(1)
    click.echo("Done.")
