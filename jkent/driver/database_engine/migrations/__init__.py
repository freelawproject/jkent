"""Database migration runner.

The schema that ``models.py`` / ``SQLModel.metadata.create_all`` builds is the
baseline, :data:`BASELINE_VERSION`. A freshly created database is stamped at the
baseline; there is no historical migration chain to replay.

Future schema changes are added as files named
``{version:04d}-{step:02d}.{sql,py}`` in this directory, with a version greater
than the baseline. The version is the **target** schema version; the step orders
operations within a version (allowing Python and SQL to interleave).

- ``.sql`` files are executed statement-by-statement (``--`` comments stripped,
  then split on ``;``). Benign idempotency errors are skipped; anything else
  aborts the migration.
- ``.py`` files must define ``async def migrate(engine) -> bool``.
  Return ``True`` on success, ``False`` to abort the version.

Usage from code::

    from jkent.driver.database_engine.migrations import migrate_to
    applied = await migrate_to(engine)           # migrate to latest
"""

from __future__ import annotations

import importlib.util
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import sqlalchemy as sa

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = Path(__file__).parent

# The schema version that ``create_all`` (i.e. the current models) produces.
# Migration files, if any, carry versions greater than this.
BASELINE_VERSION = 1

# Matches filenames like 0002-01.sql or 0002-02.py
_FILE_PATTERN = re.compile(r"^(\d{4})-(\d{2})\.(sql|py)$")


def _scan_migrations() -> dict[int, list[tuple[int, str, Path]]]:
    """Scan the migrations directory and group files by version.

    Returns:
        Dict mapping version → sorted list of (step, extension, path).
    """
    by_version: dict[int, list[tuple[int, str, Path]]] = defaultdict(list)
    for f in _MIGRATIONS_DIR.iterdir():
        m = _FILE_PATTERN.match(f.name)
        if m:
            version = int(m.group(1))
            step = int(m.group(2))
            ext = m.group(3)
            by_version[version].append((step, ext, f))
    # Sort steps within each version
    for v in by_version:
        by_version[v].sort()
    return dict(by_version)


def get_latest_version() -> int:
    """Return the highest known schema version.

    ``create_all`` builds the schema at :data:`BASELINE_VERSION`; any migration
    files in this directory bump it further.
    """
    by_version = _scan_migrations()
    return max([BASELINE_VERSION, *by_version])


async def get_current_version(engine: AsyncEngine) -> int:
    """Read the current schema version from the database.

    Returns:
        The MAX(version) from schema_info, or 0 if the table is empty.
    """
    async with engine.begin() as conn:
        result = await conn.run_sync(
            lambda c: c.execute(
                sa.text("SELECT MAX(version) FROM schema_info")
            ).scalar()
        )
        return result or 0


# SQLite error fragments that are benign when forward migrations run over a
# freshly ``create_all()``'d database (which is already at the latest schema):
# ``ADD COLUMN`` hits an existing column, ``CREATE`` hits an existing object,
# and an idempotent ``DROP COLUMN`` hits a column ``create_all`` never made.
# Anything else (syntax errors, constraint failures, unsupported DDL) is a real
# failure that must abort the migration loudly rather than be silently recorded
# as applied.
_BENIGN_SQL_ERROR_FRAGMENTS = (
    "duplicate column name",
    "already exists",
    "no such column",
)


def _split_sql_statements(content: str) -> list[str]:
    """Split a .sql file into statements, ignoring ``--`` line comments.

    Comments are stripped first so a ``;`` inside a comment (e.g. prose like
    "stateless; OK to re-seed") does not shatter a statement into invalid
    fragments. Migration files contain no ``--`` inside string literals.
    """
    uncommented: list[str] = []
    for line in content.splitlines():
        idx = line.find("--")
        if idx != -1:
            line = line[:idx]
        uncommented.append(line)
    text = "\n".join(uncommented)
    return [s.strip() for s in text.split(";") if s.strip()]


async def _run_sql_file(engine: AsyncEngine, path: Path) -> None:
    """Execute a .sql migration file statement-by-statement.

    Benign idempotency errors (see ``_BENIGN_SQL_ERROR_FRAGMENTS``) are logged
    and skipped. Any other error aborts the migration by raising, so the version
    is never recorded as applied and the failure is surfaced.
    """
    content = path.read_text()
    statements = _split_sql_statements(content)
    async with engine.begin() as conn:
        for stmt in statements:
            try:
                await conn.execute(sa.text(stmt))
            except Exception as exc:
                msg = str(exc).lower()
                if any(frag in msg for frag in _BENIGN_SQL_ERROR_FRAGMENTS):
                    logger.debug(
                        "  Skipping benign error in %s: %s", path.name, exc
                    )
                    continue
                raise RuntimeError(
                    f"Migration {path.name} failed on statement: {stmt!r}"
                ) from exc


async def _run_py_file(engine: AsyncEngine, path: Path) -> bool:
    """Import and execute a .py migration file.

    The module must define ``async def migrate(engine) -> bool``.

    Returns:
        The return value of migrate(), or True if it returns None.
    """
    # Build a module name from the file path relative to the package
    rel = path.relative_to(_MIGRATIONS_DIR)
    module_name = f"jkent.driver.database_engine.migrations.{rel.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load migration module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    migrate_fn = getattr(mod, "migrate", None)
    if migrate_fn is None:
        raise AttributeError(
            f"Migration {path.name} must define 'async def migrate(engine) -> bool'"
        )
    result = await migrate_fn(engine)
    return result if result is not None else True


async def _record_version(engine: AsyncEngine, version: int) -> None:
    """Insert a version row into schema_info."""
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("INSERT INTO schema_info (version) VALUES (:v)"),
            {"v": version},
        )


async def migrate_to(
    engine: AsyncEngine, target: int | None = None
) -> list[int]:
    """Apply all pending migrations up to *target* (default: latest).

    Args:
        engine: An async SQLAlchemy engine connected to the database.
        target: Target schema version. Defaults to :func:`get_latest_version`.

    Returns:
        List of migration-file version numbers that were applied (the baseline
        stamp on a fresh database is not included).
    """
    by_version = _scan_migrations()
    if target is None:
        target = get_latest_version()

    current = await get_current_version(engine)

    # A freshly ``create_all``'d database already has the baseline schema; stamp
    # it so future migrations know where to start. There is no v0 -> baseline
    # migration chain to replay.
    if current == 0:
        await _record_version(engine, BASELINE_VERSION)
        current = BASELINE_VERSION

    applied: list[int] = []

    for version in sorted(by_version):
        if version <= current or version > target:
            continue

        steps = by_version[version]
        logger.info(f"Applying migration to version {version}...")
        aborted = False

        for _step_num, ext, path in steps:
            if ext == "sql":
                logger.debug(f"  Running {path.name}")
                await _run_sql_file(engine, path)
            elif ext == "py":
                logger.debug(f"  Running {path.name}")
                success = await _run_py_file(engine, path)
                if not success:
                    logger.warning(
                        f"  Migration {path.name} returned False, "
                        f"aborting version {version}"
                    )
                    aborted = True
                    break

        if aborted:
            break

        await _record_version(engine, version)
        applied.append(version)
        logger.info(f"  Version {version} applied.")

    return applied
