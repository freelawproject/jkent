"""Command-line interface for LocalDevDriverDebugger.

This module provides a Click-based CLI for inspecting and manipulating
LocalDevDriver run databases.

Usage:
    pdd --db run.db info                        # Show run metadata and stats
    pdd --db run.db requests list               # List requests (with responses)
    pdd --db run.db requests show <id>          # Show request details
    pdd --db run.db requests search <query>     # Search response content
    pdd --db run.db requests cancel <id>        # Cancel a pending request
    pdd --db run.db compression stats           # Compression statistics
    pdd --db run.db errors list                 # List errors
    pdd --db run.db errors diagnose <id>        # Diagnose an error
    pdd --db run.db results list                # List results
    pdd --db run.db results validate            # Validate response structure
    pdd --db run.db results export <output>     # Export results to JSONL
    pdd --db run.db scrape health               # Run health checks
    pdd --db run.db scrape estimates            # Check estimate accuracy
    pdd --db run.db step re-evaluate <step>     # Re-evaluate a step
    pdd --db run.db step xpath-stats <step>     # XPath selector statistics

The --db option can be placed at any level:
    pdd --db run.db scrape health
    pdd scrape --db run.db health
    pdd scrape health --db run.db

All commands support:
    --format summary|json|jsonl    Output format (default: summary)
"""

from __future__ import annotations

import asyncio
from typing import Any

import click

from kent.driver.persistent_driver.cli._options import (
    db_option,
    format_options,
)
from kent.driver.persistent_driver.cli.templating import render_output
from kent.driver.persistent_driver.debugger import (
    LocalDevDriverDebugger,
)

# =========================================================================
# Data Diff Formatting
# =========================================================================


def _format_data_diff(orig: dict[str, Any], new: dict[str, Any]) -> str:
    """Format the diff between two data dicts using jsondiff.

    Returns a human-readable diff showing changed fields.
    Aggregates list item changes to show field-level summary.
    Handles type changes (e.g., ConnTrialCourtDocket -> ConnTrialCaseUnavailable).
    """
    import jsondiff

    # Detect type change by checking if the sets of keys are fundamentally different
    # This catches cases like ConnTrialCourtDocket -> ConnTrialCaseUnavailable
    orig_keys = set(orig.keys())
    new_keys = set(new.keys())

    # If there's very little overlap in keys, it's likely a type change
    common_keys = orig_keys & new_keys
    all_keys = orig_keys | new_keys

    # If less than 30% of keys are shared, treat as type change
    if all_keys and len(common_keys) / len(all_keys) < 0.3:
        return (
            f"      Result type changed:\n"
            f"      - Removed fields: {sorted(orig_keys - new_keys)}\n"
            f"      + Added fields: {sorted(new_keys - orig_keys)}"
        )

    diff = jsondiff.diff(orig, new, syntax="symmetric")
    if not diff:
        return ""

    # Format the diff with aggregation
    return _format_jsondiff_aggregated(diff, indent=6)


def _format_jsondiff_aggregated(diff: Any, indent: int = 0) -> str:
    """Format jsondiff output with aggregation for list items.

    Groups similar changes across list items to produce concise output like:
    - date_filed: str -> datetime.date (all 50 entries)
    - description: None -> various values (35 entries)
    """
    lines: list[str] = []
    prefix = " " * indent

    if not isinstance(diff, dict):
        return f"{prefix}{_truncate_repr(diff)}"

    import jsondiff

    # Separate scalar changes from list changes
    scalar_changes: list[str] = []
    list_changes: dict[
        str, dict[int, dict[str, Any]]
    ] = {}  # list_name -> {idx -> changes}

    for key, value in diff.items():
        if key == jsondiff.symbols.insert:
            # Handle inserts - can be list of tuples or dict when type changes
            if isinstance(value, dict):
                # Type change scenario - show as nested dict
                nested = _format_jsondiff_aggregated(value, indent)
                if nested:
                    scalar_changes.append("+ inserted:")
                    scalar_changes.append(nested.lstrip())
            elif isinstance(value, list):
                try:
                    for pos, val in value:
                        scalar_changes.append(
                            f"+ [{pos}]: {_truncate_repr(val)}"
                        )
                except (ValueError, TypeError):
                    # If unpacking fails, just show the value as-is
                    scalar_changes.append(f"+ {_truncate_repr(value)}")
            else:
                scalar_changes.append(f"+ {_truncate_repr(value)}")
        elif key == jsondiff.symbols.delete:
            if isinstance(value, list):
                for pos in value:
                    scalar_changes.append(f"- [{pos}]")
            else:
                scalar_changes.append(f"- deleted: {_truncate_repr(value)}")
        elif isinstance(key, str) and isinstance(value, dict):
            # Check if this is a list with indexed changes
            if all(
                isinstance(k, int)
                for k in value
                if k not in (jsondiff.symbols.insert, jsondiff.symbols.delete)
            ):
                # This is a list field with changes
                list_changes[key] = value
            elif isinstance(value, list) and len(value) == 2:
                # Scalar field change
                scalar_changes.append(
                    f"{key}: {_truncate_repr(value[0])} \u2192 {_truncate_repr(value[1])}"
                )
            else:
                # Nested dict, recurse
                nested = _format_jsondiff_aggregated(value, indent)
                if nested:
                    scalar_changes.append(f"{key}:")
                    scalar_changes.append(nested.lstrip())
        elif isinstance(value, list) and len(value) == 2:
            # Scalar field with [old, new]
            scalar_changes.append(
                f"{key}: {_truncate_repr(value[0])} \u2192 {_truncate_repr(value[1])}"
            )
        else:
            scalar_changes.append(f"{key}: {_truncate_repr(value)}")

    # Output scalar changes
    for change in scalar_changes:
        lines.append(f"{prefix}{change}")

    # Aggregate list changes by field
    for list_name, item_changes in list_changes.items():
        # Collect all field changes across items
        field_stats: dict[
            str, dict[str, Any]
        ] = {}  # field -> {count, sample_old, sample_new, all_same}

        for idx, changes in item_changes.items():
            if isinstance(idx, int) and isinstance(changes, dict):
                for field, change_val in changes.items():
                    if isinstance(change_val, list) and len(change_val) == 2:
                        old_val, new_val = change_val
                        if field not in field_stats:
                            field_stats[field] = {
                                "count": 0,
                                "sample_old": old_val,
                                "sample_new": new_val,
                                "all_same": True,
                            }
                        field_stats[field]["count"] += 1
                        # Check if all values are the same pattern
                        if _type_name(old_val) != _type_name(
                            field_stats[field]["sample_old"]
                        ) or _type_name(new_val) != _type_name(
                            field_stats[field]["sample_new"]
                        ):
                            field_stats[field]["all_same"] = False

        # Output aggregated changes
        total_items = len(
            [idx for idx in item_changes if isinstance(idx, int)]
        )
        lines.append(f"{prefix}{list_name}: {total_items} items changed")

        for field, stats in sorted(field_stats.items()):
            count = stats["count"]
            sample_old = stats["sample_old"]
            sample_new = stats["sample_new"]

            if stats["all_same"]:
                # All changes are the same pattern (e.g., str -> date)
                old_type = _type_name(sample_old)
                new_type = _type_name(sample_new)
                if old_type != new_type:
                    lines.append(
                        f"{prefix}  .{field}: {old_type} \u2192 {new_type} ({count}x)"
                    )
                else:
                    lines.append(
                        f"{prefix}  .{field}: {_truncate_repr(sample_old)} \u2192 {_truncate_repr(sample_new)} ({count}x)"
                    )
            else:
                # Mixed changes
                lines.append(f"{prefix}  .{field}: various changes ({count}x)")

    return "\n".join(lines)


def _type_name(value: Any) -> str:
    """Get a short type name for a value."""
    if value is None:
        return "None"
    return type(value).__name__


def _truncate_repr(value: Any, max_len: int = 60) -> str:
    """Get a truncated repr of a value."""
    if value is None:
        return "None"
    s = repr(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


# =========================================================================
# CLI Groups
# =========================================================================


@click.group()
@click.version_option()
@db_option
@click.pass_context
def cli(ctx: click.Context, db_path: str | None) -> None:
    """LocalDevDriver Debugger - Inspect and manipulate scraper run databases."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path


# =========================================================================
# Shared Helpers
# =========================================================================


def register_cli_group(
    name: str,
    help_text: str,
    *,
    invoke_without_command: bool = False,
) -> click.Group:
    """Create a subgroup on the top-level :data:`cli` with standard --db propagation.

    Subcommands can be attached via ``@group.command(...)`` on the returned group.
    When ``invoke_without_command`` is true and no subcommand is given, the group
    prints its own help text.
    """

    @cli.group(
        name=name,
        help=help_text,
        invoke_without_command=invoke_without_command,
    )
    @db_option
    @click.pass_context
    def group(ctx: click.Context, db_path: str | None) -> None:
        ctx.ensure_object(dict)
        if db_path:
            ctx.obj["db_path"] = db_path
        if invoke_without_command and ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

    return group


async def _run_health_report(
    db_path: str,
    format_type: str,
    template_path: str,
    template_name: str | None,
) -> None:
    """Fetch a run's health check data and render it with the given template."""
    async with LocalDevDriverDebugger.open(db_path) as debugger:
        integrity = await debugger.check_integrity()
        ghosts = await debugger.get_ghost_requests()
        status = await debugger.get_run_status()
        stats = await debugger.get_stats()
        estimates = await debugger.check_estimates()

        render_output(
            {
                "status": status,
                "integrity": integrity,
                "ghosts": ghosts,
                "error_stats": stats["errors"],
                "estimates": estimates,
            },
            format_type=format_type,
            template_path=template_path,
            template_name=template_name or "default",
        )


def _resolve_db_path(ctx: click.Context, db_path: str | None) -> str:
    """Resolve db_path from the current option or any parent group.

    Checks the subcommand's own --db first, then walks up the context
    chain checking ctx.obj["db_path"] (set by groups) and ctx.params.
    Raises UsageError if no --db was provided at any level.
    """
    if db_path:
        return db_path
    # Walk up to find --db from parent groups
    parent = ctx.parent
    while parent is not None:
        # Check ctx.obj (where groups store propagated values)
        obj = parent.ensure_object(dict)
        if obj.get("db_path"):
            return obj["db_path"]
        # Check params directly
        parent_db = parent.params.get("db_path")
        if parent_db:
            return parent_db
        parent = parent.parent
    raise click.UsageError(
        "Missing --db option. Provide a database path with --db."
    )


# =========================================================================
# Info Command
# =========================================================================


@cli.command()
@db_option
@format_options
@click.pass_context
def info(
    ctx: click.Context,
    db_path: str | None,
    format_type: str,
    template_name: str | None,
) -> None:
    """Show run metadata and statistics.

    \b
    Examples:
        pdd info --db run.db
        pdd info --db run.db --format json
    """
    db_path = _resolve_db_path(ctx, db_path)

    async def run() -> None:
        async with LocalDevDriverDebugger.open(db_path) as debugger:
            metadata = await debugger.get_run_metadata()
            stats = await debugger.get_stats()

            output = {"metadata": metadata, "stats": stats}
            render_output(
                output,
                format_type=format_type,
                template_path="info",
                template_name=template_name or "default",
            )

    asyncio.run(run())


# =========================================================================
# Main Entry Point
# =========================================================================


def _print_help_recursive(
    group: click.MultiCommand, ctx: click.Context, prefix: str = ""
) -> None:
    """Recursively print help for all commands in a group."""
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        sub_ctx = click.Context(cmd, info_name=f"{prefix}{name}", parent=ctx)
        click.echo(cmd.get_help(sub_ctx))
        click.echo("\n")
        if isinstance(cmd, click.MultiCommand):
            _print_help_recursive(cmd, sub_ctx, prefix=f"{prefix}{name} ")


@cli.command("help-all")
@click.pass_context
def help_all(ctx: click.Context) -> None:
    """Show help for all commands and subcommands."""
    parent = ctx.parent
    assert parent is not None
    click.echo(parent.command.get_help(parent))
    click.echo("\n")
    _print_help_recursive(parent.command, parent)


def main() -> None:
    """Main CLI entry point."""
    cli()


# =========================================================================
# Register all subcommand modules
# (imports trigger @cli.group/@cli.command decorators)
# =========================================================================
from kent.driver.persistent_driver.cli import (
    cancel as _cancel_mod,
)
from kent.driver.persistent_driver.cli import (
    compression as _compression_mod,
)
from kent.driver.persistent_driver.cli import (
    doctor as _doctor_mod,
)
from kent.driver.persistent_driver.cli import (
    errors as _errors_mod,
)
from kent.driver.persistent_driver.cli import (
    incidental as _incidental_mod,
)
from kent.driver.persistent_driver.cli import (
    query as _query_mod,
)
from kent.driver.persistent_driver.cli import (
    replay as _replay_mod,
)
from kent.driver.persistent_driver.cli import (
    requests as _requests_mod,
)
from kent.driver.persistent_driver.cli import (
    responses as _responses_mod,
)
from kent.driver.persistent_driver.cli import (
    results as _results_mod,
)
from kent.driver.persistent_driver.cli import (
    scrape as _scrape_mod,
)
from kent.driver.persistent_driver.cli import (
    seed_error_patch_rerun as _seed_error_patch_rerun_mod,
)
from kent.driver.persistent_driver.cli import (
    step as _step_mod,
)

if __name__ == "__main__":
    main()
