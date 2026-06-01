"""Scraper registry for discovering and loading scraper classes.

This module provides functionality for:
- Scanning directories for BaseScraper subclasses
- Extracting scraper metadata (courts, data types, status)
- Extracting parameter schema via BaseScraper.schema() (@entry system)
- Building initial_seed() invocation lists from web form data
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kent.data_types import BaseScraper

logger = logging.getLogger(__name__)


@dataclass
class ScraperInfo:
    """Information about a discovered scraper."""

    module_path: str  # e.g., "juriscraper.sd.state.new_york.nyscef.scraper"
    class_name: str  # e.g., "NYSCEFScraper"
    full_path: str  # module_path:class_name

    # Metadata from scraper class
    court_ids: set[str] = field(default_factory=set)
    court_url: str = ""
    data_types: set[str] = field(default_factory=set)
    status: str = "unknown"
    version: str = ""
    requires_auth: bool = False
    rate_limits: list[dict[str, int]] | None = None

    # Entry schema (from BaseScraper.schema())
    entry_schema: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "module_path": self.module_path,
            "class_name": self.class_name,
            "full_path": self.full_path,
            "court_ids": list(self.court_ids),
            "court_url": self.court_url,
            "data_types": list(self.data_types),
            "status": self.status,
            "version": self.version,
            "requires_auth": self.requires_auth,
            "rate_limits": self.rate_limits,
        }
        if self.entry_schema is not None:
            result["entry_schema"] = self.entry_schema
        return result


class ScraperRegistry:
    """Registry for discovering and managing scraper classes."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._scrapers: dict[str, ScraperInfo] = {}
        self._classes: dict[str, type[BaseScraper[Any]]] = {}

    def scan_directory(self, base_dir: Path, package_prefix: str) -> int:
        """Scan a directory for scraper classes.

        Args:
            base_dir: Directory to scan (e.g., juriscraper/sd)
            package_prefix: Python package prefix (e.g., "juriscraper.sd")

        Returns:
            Number of scrapers discovered.
        """
        count = 0

        # Find all scraper.py files
        for scraper_file in base_dir.rglob("scraper.py"):
            try:
                # Build module path from file path
                relative_path = scraper_file.relative_to(base_dir)
                parts = list(relative_path.parent.parts) + ["scraper"]
                module_path = f"{package_prefix}.{'.'.join(parts)}"

                scrapers = self._scan_module(module_path)
                count += len(scrapers)

            except Exception as e:
                logger.warning(f"Error scanning {scraper_file}: {e}")

        return count

    def _scan_module(self, module_path: str) -> list[ScraperInfo]:
        """Scan a single module for scraper classes.

        Args:
            module_path: Full module path (e.g., "juriscraper.sd.state.new_york.nyscef.scraper")

        Returns:
            List of discovered ScraperInfo objects.
        """
        from kent.data_types import BaseScraper

        scrapers: list[ScraperInfo] = []

        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            logger.warning(f"Could not import {module_path}: {e}")
            return scrapers

        # Find all BaseScraper subclasses in the module
        for name in dir(module):
            obj = getattr(module, name)

            # Check if it's a class that inherits from BaseScraper
            if not isinstance(obj, type):
                continue
            if not issubclass(obj, BaseScraper):
                continue
            if obj is BaseScraper:
                continue
            # Skip if not defined in this module
            if obj.__module__ != module_path:
                continue

            scraper_info = self._extract_scraper_info(obj, module_path, name)
            self._scrapers[scraper_info.full_path] = scraper_info
            self._classes[scraper_info.full_path] = obj
            scrapers.append(scraper_info)

            logger.info(f"Discovered scraper: {scraper_info.full_path}")

        return scrapers

    def _extract_scraper_info(
        self,
        scraper_class: type[BaseScraper[Any]],
        module_path: str,
        class_name: str,
    ) -> ScraperInfo:
        """Extract metadata and parameter schema from a scraper class.

        Args:
            scraper_class: The scraper class.
            module_path: Module path where scraper is defined.
            class_name: Name of the scraper class.

        Returns:
            ScraperInfo with metadata and schema.
        """
        full_path = f"{module_path}:{class_name}"

        # Extract class-level metadata
        court_ids = getattr(scraper_class, "court_ids", set()) or set()  # type: ignore[var-annotated]
        court_url = getattr(scraper_class, "court_url", "") or ""
        data_types = getattr(scraper_class, "data_types", set()) or set()  # type: ignore[var-annotated]
        status_enum = getattr(scraper_class, "status", None)
        status = status_enum.value if status_enum else "unknown"
        version = getattr(scraper_class, "version", "") or ""
        requires_auth = getattr(scraper_class, "requires_auth", False)
        raw_rate_limits = getattr(scraper_class, "rate_limits", None)
        rate_limits = None
        if raw_rate_limits:
            rate_limits = [
                {"limit": r.limit, "interval_ms": r.interval}
                for r in raw_rate_limits
            ]

        entry_schema = self._extract_entry_schema(scraper_class)

        return ScraperInfo(
            module_path=module_path,
            class_name=class_name,
            full_path=full_path,
            court_ids=set(court_ids),
            court_url=court_url,
            data_types=set(data_types),
            status=status,
            version=version,
            requires_auth=requires_auth,
            rate_limits=rate_limits,
            entry_schema=entry_schema,
        )

    def _extract_entry_schema(
        self, scraper_class: type[BaseScraper[Any]]
    ) -> dict[str, Any] | None:
        """Extract schema from scraper's @entry-decorated methods.

        Uses BaseScraper.schema() which generates JSON Schema from
        @entry decorated methods and their Pydantic parameter models.

        Args:
            scraper_class: The scraper class.

        Returns:
            Schema dict if scraper has @entry methods, None otherwise.
        """
        try:
            schema = scraper_class.schema()
            # schema() returns {"scraper": name, "entries": {...}}
            # If no @entry methods, entries will be empty
            if schema.get("entries"):
                return schema
        except Exception as e:
            logger.warning(
                f"Could not extract @entry schema for "
                f"{scraper_class.__name__}: {e}"
            )
        return None

    def list_scrapers(self) -> list[ScraperInfo]:
        """List all discovered scrapers.

        Returns:
            List of ScraperInfo objects.
        """
        return list(self._scrapers.values())

    def find_scrapers_by_name(self, name: str) -> list[ScraperInfo]:
        """Find scrapers matching ``name`` via module_path, full_path, or class_name.

        Used to resolve a scraper_name stored in run metadata back to a registry
        entry. Tries progressively looser matches; stops at the first non-empty
        result.

        Args:
            name: Module path (preferred), full path (``module:class``), or class name.

        Returns:
            List of matching ScraperInfo. Empty if no match found. Callers are
            responsible for handling the zero-match and multi-match cases.
        """
        scrapers = self.list_scrapers()
        for attr in ("module_path", "full_path", "class_name"):
            matching = [s for s in scrapers if getattr(s, attr) == name]
            if matching:
                return matching
        return []

    def get_scraper(self, full_path: str) -> ScraperInfo | None:
        """Get info for a specific scraper.

        Args:
            full_path: Full scraper path (module:class).

        Returns:
            ScraperInfo or None if not found.
        """
        return self._scrapers.get(full_path)

    def get_scraper_class(
        self, full_path: str
    ) -> type[BaseScraper[Any]] | None:
        """Get the actual scraper class.

        Args:
            full_path: Full scraper path (module:class).

        Returns:
            The scraper class or None if not found.
        """
        return self._classes.get(full_path)

    def instantiate_scraper(self, full_path: str) -> BaseScraper[Any] | None:
        """Instantiate a scraper.

        Parameters are passed via initial_seed() at runtime, not at
        instantiation time.

        Args:
            full_path: Full scraper path (module:class).

        Returns:
            Instantiated scraper or None if not found.
        """
        scraper_class = self.get_scraper_class(full_path)
        if scraper_class is None:
            return None

        return scraper_class()

    def scan_tree(self, root: Path) -> int:
        """Discover BaseScraper subclasses in ``.py`` files under *root*.

        Delegates to :func:`kent.discovery.discover_scrapers` for
        file-system scanning and populates the registry with full
        metadata for each discovered class.

        Args:
            root: Directory to scan.

        Returns:
            Number of scrapers discovered.
        """
        from kent.discovery import discover_scrapers

        count = 0
        for module_path, class_name, cls in discover_scrapers(root):
            info = self._extract_scraper_info(cls, module_path, class_name)
            self._scrapers[info.full_path] = info
            self._classes[info.full_path] = cls
            count += 1
            logger.info(f"Discovered scraper: {info.full_path}")
        return count

    def register_module(self, module_path: str) -> int:
        """Register scrapers from a specific module path.

        Args:
            module_path: Full dotted module path
                (e.g., ``"kent.demo.scraper"``).

        Returns:
            Number of scrapers discovered in the module.
        """
        scrapers = self._scan_module(module_path)
        return len(scrapers)

    def build_seed_from_web_data(
        self,
        full_path: str,
        form_data: dict[str, Any],
    ) -> list[dict[str, dict[str, Any]]]:
        """Build initial_seed() invocation list from web form data.

        Converts web form data into the format expected by
        BaseScraper.initial_seed(): a list of single-key dicts
        mapping entry function names to kwargs.

        Args:
            full_path: Full scraper path (module:class).
            form_data: Parameter data from web form, keyed by entry name.

        Returns:
            Invocation list for initial_seed().

        Example form_data::

            {
                "get_entry": {},
                "fetch_docket": {"crn": 12345}
            }

        Returns::

            [
                {"get_entry": {}},
                {"fetch_docket": {"crn": 12345}}
            ]
        """
        invocations: list[dict[str, dict[str, Any]]] = []
        for entry_name, kwargs in form_data.items():
            invocations.append({entry_name: kwargs if kwargs else {}})
        return invocations


# Global registry instance
_registry: ScraperRegistry | None = None


def get_registry() -> ScraperRegistry:
    """Get the global registry instance.

    Returns:
        The ScraperRegistry instance.

    Raises:
        RuntimeError: If registry not initialized.
    """
    if _registry is None:
        raise RuntimeError("Scraper registry not initialized")
    return _registry


def init_registry(
    sd_directory: Path | None = None,
    extra_modules: list[str] | None = None,
) -> ScraperRegistry:
    """Initialize the global scraper registry.

    Args:
        sd_directory: Directory to scan for scrapers.
            Defaults to scanning the current working directory
            (same discovery logic as ``kent list``).
        extra_modules: Additional module paths to scan for scrapers
            (e.g., ``["kent.demo.scraper"]``).

    Returns:
        The initialized registry.
    """
    global _registry

    _registry = ScraperRegistry()

    if sd_directory is not None:
        if sd_directory.exists():
            count = _registry.scan_directory(sd_directory, "juriscraper.sd")
            logger.info(
                f"Initialized scraper registry with {count} scrapers "
                f"from {sd_directory}"
            )
        else:
            logger.warning(f"Scraper directory not found: {sd_directory}")
    else:
        # Default: scan from CWD, same as `kent list`
        count = _registry.scan_tree(Path.cwd())
        logger.info(f"Discovered {count} scrapers from working directory")

    # Register extra modules
    for module_path in extra_modules or []:
        try:
            n = _registry.register_module(module_path)
            logger.info(f"Registered {n} scrapers from {module_path}")
        except Exception as e:
            logger.warning(f"Could not register {module_path}: {e}")

    return _registry
