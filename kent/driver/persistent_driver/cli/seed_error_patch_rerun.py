"""CLI command: seed-error-patch-rerun.

Given a source run database that has errors, trace every errored request back
to its oldest ancestor (the root request whose ``parent_request_id`` is NULL,
which by contract is yielded by an entry function and targets a publicly
accessible URL), and produce a new database seeded with those deduplicated
root requests. The new database can be run without seed_params to re-attempt
all of the errored work with fresh session state.

This replaces the old in-place requeue mechanism, which was unreliable when
stale session cookies / one-time tokens were bound to the errored request.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import sqlalchemy as sa

from kent.driver.persistent_driver.cli import _resolve_db_path, cli
from kent.driver.persistent_driver.cli._options import (
    db_option,
    format_options,
)
from kent.driver.persistent_driver.cli.templating import render_output
from kent.driver.persistent_driver.database import init_database
from kent.driver.persistent_driver.debugger import LocalDevDriverDebugger
from kent.driver.persistent_driver.models import Error, Request, RunMetadata
from kent.driver.persistent_driver.sql_manager import SQLManager


@cli.command("seed-error-patch-rerun")
@click.option(
    "--output-db",
    "output_db_path",
    type=click.Path(),
    default=None,
    help=(
        "Path to the new database to create, seeded with root ancestors. "
        "Required unless --report is passed."
    ),
)
@click.option(
    "--no-resolve",
    is_flag=True,
    default=False,
    help=(
        "Do not mark covered errors in the source DB as resolved. "
        "Guarantees no writes to the source DB."
    ),
)
@click.option(
    "--report",
    "report_only",
    is_flag=True,
    default=False,
    help=(
        "Only print the plan/stats. Do not create --output-db and do not "
        "modify the source DB."
    ),
)
@db_option
@format_options
@click.pass_context
def seed_error_patch_rerun(
    ctx: click.Context,
    db_path: str | None,
    output_db_path: str | None,
    no_resolve: bool,
    report_only: bool,
    format_type: str,
    template_name: str | None,
) -> None:
    """Seed a fresh DB with the root ancestors of every errored request.

    \b
    The new DB contains:
      * Run metadata copied from the source (scraper_name, num_workers, etc.),
        with seed_params cleared so the driver does not re-run initial_seed.
      * One pending request per unique root ancestor of an unresolved error,
        with URL/method/headers/cookies/body/continuation/aux/permanent data
        copied from the source. Root requests are expected (by documented
        contract) to originate from an entry function with a publicly
        accessible URL.

    \b
    By default, covered errors in the source DB are marked resolved with
    resolution_type='rerun_generated'. Use --no-resolve to skip this and
    guarantee zero writes to the source DB. Use --report to print the stats
    without creating --output-db or modifying the source DB.

    \b
    Examples:
        pdd seed-error-patch-rerun --db run.db --output-db patch.db
        pdd seed-error-patch-rerun --db run.db --output-db patch.db --no-resolve
        pdd seed-error-patch-rerun --db run.db --report
    """
    resolved_db_path = _resolve_db_path(ctx, db_path)

    if report_only and output_db_path:
        click.echo(
            "--output-db is not allowed with --report (report is read-only).",
            err=True,
        )
        sys.exit(1)
    if not report_only and not output_db_path:
        click.echo(
            "--output-db is required (or pass --report to just print stats).",
            err=True,
        )
        sys.exit(1)
    if (
        not report_only
        and output_db_path is not None
        and os.path.exists(output_db_path)
    ):
        click.echo(
            f"Refusing to overwrite existing --output-db path: "
            f"{output_db_path}",
            err=True,
        )
        sys.exit(1)

    asyncio.run(
        _run(
            Path(resolved_db_path),
            Path(output_db_path) if output_db_path else None,
            no_resolve=no_resolve,
            report_only=report_only,
            format_type=format_type,
            template_name=template_name,
        )
    )


async def _run(
    source_db: Path,
    output_db: Path | None,
    *,
    no_resolve: bool,
    report_only: bool,
    format_type: str,
    template_name: str | None,
) -> None:
    # Source DB is read-only unless we'll be marking errors resolved.
    will_mutate_source = not (report_only or no_resolve)
    async with LocalDevDriverDebugger.open(
        source_db, read_only=not will_mutate_source
    ) as debugger:
        plan = await _build_plan(debugger)

        wrote_output_db = False
        resolved_count = 0

        if plan.root_ids and not report_only:
            assert output_db is not None
            metadata_row = await _get_run_metadata_row(debugger)
            if metadata_row is None:
                click.echo(
                    "Source DB has no run_metadata row; cannot seed a new DB.",
                    err=True,
                )
                sys.exit(1)
            await _write_output_db(output_db, metadata_row, plan.root_rows)
            wrote_output_db = True

            if will_mutate_source:
                resolved_count = await _mark_covered_errors_resolved(
                    debugger, plan.covered_error_ids
                )

        output = _plan_to_dict(
            plan,
            source_db=source_db,
            output_db=output_db,
            report_only=report_only,
            no_resolve=no_resolve,
            wrote_output_db=wrote_output_db,
            resolved_count=resolved_count,
        )
        render_output(
            output,
            format_type=format_type,
            template_path="seed_error_patch_rerun",
            template_name=template_name or "default",
        )


class _Plan:
    __slots__ = (
        "root_ids",
        "root_rows",
        "covered_error_ids",
        "total_requests",
        "total_descendants",
        "errored_descendants",
        "unresolved_errors",
    )

    def __init__(self) -> None:
        self.root_ids: list[int] = []
        self.root_rows: list[sa.Row] = []
        self.covered_error_ids: list[int] = []
        self.total_requests: int = 0
        self.total_descendants: int = 0
        self.errored_descendants: int = 0
        self.unresolved_errors: int = 0


async def _build_plan(debugger: LocalDevDriverDebugger) -> _Plan:
    """Compute roots, covered errors, and blast-radius counts.

    Uses two recursive CTEs:
      1. Upward walk from every unresolved-errored request to find the root
         ancestor (parent_request_id IS NULL).
      2. Downward walk from each unique root to count all descendants (the
         potential blast radius of a re-run).
    """
    plan = _Plan()
    async with debugger._session_factory() as session:
        # Total unresolved errors (for reporting).
        res = await session.execute(
            sa.select(sa.func.count())
            .select_from(Error)
            .where(Error.is_resolved == False)  # type: ignore[arg-type]  # noqa: E712
        )
        plan.unresolved_errors = res.scalar() or 0

        # Total requests in source DB (for percentages).
        res = await session.execute(
            sa.select(sa.func.count()).select_from(Request)
        )
        plan.total_requests = res.scalar() or 0

        # Upward CTE: walk parent_request_id from errored requests to the root.
        # Each seed row carries the originating error_request_id along so we
        # can return the (root_id, error_request_id) mapping in one pass.
        anchor = (
            sa.select(
                Request.id.label("id"),  # type: ignore[union-attr]
                Request.parent_request_id.label("parent_request_id"),  # type: ignore[union-attr]
                Request.id.label("error_request_id"),  # type: ignore[union-attr]
                Error.id.label("error_id"),  # type: ignore[union-attr]
            )
            .select_from(Request)
            .join(Error, Error.request_id == Request.id)  # type: ignore[arg-type]
            .where(Error.is_resolved == False)  # type: ignore[arg-type]  # noqa: E712
        )
        ancestors_cte = anchor.cte(name="ancestors", recursive=True)
        recursive = (
            sa.select(  # type: ignore[call-overload]
                Request.id,
                Request.parent_request_id,
                ancestors_cte.c.error_request_id,
                ancestors_cte.c.error_id,
            )
            .select_from(Request)
            .join(
                ancestors_cte, Request.id == ancestors_cte.c.parent_request_id
            )
        )
        ancestors_cte = ancestors_cte.union_all(recursive)

        res = await session.execute(
            sa.select(
                ancestors_cte.c.id,
                ancestors_cte.c.error_request_id,
                ancestors_cte.c.error_id,
            ).where(ancestors_cte.c.parent_request_id.is_(None))
        )
        rows = res.all()

        root_id_set: set[int] = set()
        covered_error_ids: set[int] = set()
        for root_id, _err_req_id, err_id in rows:
            root_id_set.add(int(root_id))
            if err_id is not None:
                covered_error_ids.add(int(err_id))

        plan.root_ids = sorted(root_id_set)
        plan.covered_error_ids = sorted(covered_error_ids)

        if not plan.root_ids:
            return plan

        # Fetch full row data for each unique root (order by id for determinism).
        res = await session.execute(
            sa.select(  # type: ignore[call-overload,misc]
                Request.id,
                Request.priority,
                Request.request_type,
                Request.method,
                Request.url,
                Request.headers_json,
                Request.cookies_json,
                Request.body,
                Request.continuation,
                Request.current_location,
                Request.accumulated_data_json,
                Request.permanent_json,
                Request.expected_type,
                Request.deduplication_key,
                Request.verify,
                Request.bypass_rate_limit,
            )
            .where(Request.id.in_(plan.root_ids))  # type: ignore[union-attr]
            .order_by(Request.id.asc())  # type: ignore[union-attr]
        )
        plan.root_rows = list(res.all())

        # Downward CTE: count total and errored descendants under each root.
        root_ids_tuple = tuple(plan.root_ids)
        down_anchor = sa.select(Request.id).where(  # type: ignore[call-overload]
            Request.id.in_(root_ids_tuple)  # type: ignore[union-attr]
        )
        down_cte = down_anchor.cte(name="descendants", recursive=True)
        down_recursive = sa.select(Request.id).where(  # type: ignore[call-overload]
            Request.parent_request_id == down_cte.c.id
        )
        down_cte = down_cte.union_all(down_recursive)

        res = await session.execute(
            sa.select(sa.func.count()).select_from(down_cte)
        )
        plan.total_descendants = res.scalar() or 0

        res = await session.execute(
            sa.select(sa.func.count(sa.distinct(Request.id)))  # type: ignore[arg-type]
            .select_from(Request)
            .join(Error, Error.request_id == Request.id)  # type: ignore[arg-type]
            .where(
                Request.id.in_(sa.select(down_cte.c.id)),  # type: ignore[union-attr]
                Error.is_resolved == False,  # type: ignore[arg-type]  # noqa: E712
            )
        )
        plan.errored_descendants = res.scalar() or 0

    return plan


async def _get_run_metadata_row(
    debugger: LocalDevDriverDebugger,
) -> RunMetadata | None:
    async with debugger._session_factory() as session:
        res = await session.execute(
            sa.select(RunMetadata).where(RunMetadata.id == 1)  # type: ignore[arg-type]
        )
        return res.scalar_one_or_none()


async def _write_output_db(
    output_db: Path,
    source_metadata: RunMetadata,
    root_rows: list[sa.Row],
) -> None:
    """Create a fresh output DB and seed it with the root-ancestor requests.

    The output DB is created without running migrations — it uses the current
    schema directly via SQLModel.metadata.create_all, and schema_info is
    stamped with the latest version.
    """
    engine, session_factory = await init_database(output_db)
    try:
        sql = SQLManager(engine, session_factory)

        speculation_config: dict[str, Any] | None = None
        if source_metadata.speculation_config_json:
            speculation_config = json.loads(
                source_metadata.speculation_config_json
            )
        browser_config: dict[str, Any] | None = None
        if source_metadata.browser_config_json:
            browser_config = json.loads(source_metadata.browser_config_json)

        await sql.init_run_metadata(
            scraper_name=source_metadata.scraper_name,
            scraper_version=source_metadata.scraper_version,
            num_workers=source_metadata.num_workers,
            max_backoff_time=source_metadata.max_backoff_time,
            speculation_config=speculation_config,
            browser_config=browser_config,
            seed_params=None,
        )

        for row in root_rows:
            await sql.insert_entry_request(
                priority=row.priority or 0,
                method=row.method,
                url=row.url,
                headers_json=row.headers_json,
                cookies_json=row.cookies_json,
                body=row.body,
                continuation=row.continuation,
                current_location=row.current_location or "",
                accumulated_data_json=row.accumulated_data_json,
                permanent_json=row.permanent_json,
                dedup_key=row.deduplication_key,
                verify=row.verify,
                bypass_rate_limit=bool(row.bypass_rate_limit),
                request_type=row.request_type or "navigating",
                expected_type=row.expected_type,
            )
    finally:
        await engine.dispose()


async def _mark_covered_errors_resolved(
    debugger: LocalDevDriverDebugger,
    error_ids: list[int],
) -> int:
    """Mark the given errors resolved with resolution_type='rerun_generated'."""
    if not error_ids:
        return 0

    async with debugger._session_factory() as session:
        result = await session.execute(
            sa.update(Error)
            .where(
                Error.id.in_(error_ids),  # type: ignore[union-attr]
                Error.is_resolved == False,  # type: ignore[arg-type]  # noqa: E712
            )
            .values(
                is_resolved=True,
                resolved_at=sa.func.current_timestamp(),
                resolution_type="rerun_generated",
                resolution_notes="Resolved via pdd seed-error-patch-rerun",
            )
        )
        await session.commit()
        return result.rowcount or 0  # type: ignore[attr-defined]


def _pct(num: int, denom: int) -> float | None:
    """Percentage as a float 0..100, or None if denom <= 0."""
    if denom <= 0:
        return None
    return 100.0 * num / denom


def _plan_to_dict(
    plan: _Plan,
    *,
    source_db: Path,
    output_db: Path | None,
    report_only: bool,
    no_resolve: bool,
    wrote_output_db: bool,
    resolved_count: int,
) -> dict[str, Any]:
    total = plan.total_requests
    desc = plan.total_descendants
    errored = plan.errored_descendants
    nonerrored = max(0, desc - errored)
    n_roots = len(plan.root_ids)
    n_covered_errors = len(plan.covered_error_ids)

    return {
        "source_db": str(source_db),
        "output_db": str(output_db) if output_db else None,
        "mode": {
            "report_only": report_only,
            "no_resolve": no_resolve,
            "wrote_output_db": wrote_output_db,
        },
        "source_stats": {
            "total_requests": total,
            "unresolved_errors": plan.unresolved_errors,
        },
        "roots": {
            "count": n_roots,
            "pct_of_all_requests": _pct(n_roots, total),
            "covered_error_count": n_covered_errors,
        },
        "blast_radius": {
            "total_descendants": desc,
            "pct_of_all_requests": _pct(desc, total),
            "errored_descendants": errored,
            "errored_pct_of_descendants": _pct(errored, desc),
            "non_errored_descendants": nonerrored,
            "non_errored_pct_of_descendants": _pct(nonerrored, desc),
        },
        "resolution": {
            "resolved_count": resolved_count,
            "resolution_type": "rerun_generated" if resolved_count else None,
        },
    }
