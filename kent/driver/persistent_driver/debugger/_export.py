"""Export, search, and diagnose methods for LocalDevDriverDebugger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sqlmodel import select

from kent.driver.persistent_driver.models import (
    Request,
    Result,
)
from kent.driver.persistent_driver.scoped_session import ScopedSessionFactory
from kent.driver.persistent_driver.sql_manager import (
    ResponseRecord,
    SQLManager,
)


class ExportSearchMixin:
    """Export (JSONL), response search, and error diagnosis."""

    sql: SQLManager
    _session_factory: ScopedSessionFactory

    if TYPE_CHECKING:
        # Provided by InspectionMixin / DebuggerBase at runtime.
        async def get_error(self, error_id: int) -> dict[str, Any] | None: ...
        async def get_response(
            self, request_id: int
        ) -> ResponseRecord | None: ...
        async def get_response_content(
            self, request_id: int
        ) -> bytes | None: ...
        async def get_run_metadata(
            self,
        ) -> dict[str, Any] | None: ...

    # =========================================================================
    # Debugging Methods
    # =========================================================================

    async def diagnose(
        self,
        error_id: int,
        scraper_class: type | None = None,
        speculation_cap: int | None = None,
    ) -> dict[str, Any]:
        """Diagnose an error by re-running XPath observation.

        Args:
            error_id: The error ID to diagnose.
            scraper_class: Optional scraper class. If not provided, will attempt
                to discover from run metadata's scraper_name.
            speculation_cap: Optional cap for speculation during diagnosis.

        Returns:
            Dictionary with diagnosis results:
                - error: Original error details
                - response: Response metadata
                - observations: XPath observations from re-running extraction
                - scraper_info: Information about the scraper used

        Raises:
            ValueError: If error not found or scraper cannot be discovered.
            ImportError: If scraper_name cannot be imported.
        """
        error = await self.get_error(error_id)
        if not error:
            raise ValueError(f"Error {error_id} not found")

        request_id = error.get("request_id")
        if not request_id:
            raise ValueError("Error has no associated request_id")

        # Response data is now on the request row itself
        response = await self.get_response(request_id)
        if not response:
            raise ValueError(f"No response found for request {request_id}")

        content = await self.get_response_content(request_id)
        if not content:
            raise ValueError(f"No content for request {request_id}")

        # Discover scraper if not provided
        if scraper_class is None:
            metadata = await self.get_run_metadata()
            if not metadata:
                raise ValueError("No run metadata found")

            scraper_name = metadata.get("scraper_name")
            if not scraper_name:
                raise ValueError("No scraper_name in run metadata")

            try:
                import importlib

                if ":" in scraper_name:
                    module_path, class_name = scraper_name.rsplit(":", 1)
                    module = importlib.import_module(module_path)
                    scraper_class = getattr(module, class_name)
                else:
                    module = importlib.import_module(scraper_name)
                    scraper_class = module.Site
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Cannot import scraper '{scraper_name}': {e}"
                ) from e

        diagnosis = {
            "error": error,
            "response": {
                "id": response.id,
                "status_code": response.status_code,
                "url": response.url,
                "size": response.content_size_original,
                "continuation": response.continuation,
            },
            "scraper_info": {
                "class": scraper_class.__name__ if scraper_class else None,
                "module": (
                    scraper_class.__module__ if scraper_class else None
                ),
            },
            "observations": {
                "message": "Full XPath re-execution requires scraper instantiation",
                "selector": error.get("selector"),
                "selector_type": error.get("selector_type"),
                "expected_range": f"{error.get('expected_min')}-{error.get('expected_max')}",
                "actual_count": error.get("actual_count"),
            },
        }

        return diagnosis

    # =========================================================================
    # Export Methods
    # =========================================================================

    async def export_results_jsonl(
        self,
        output_path: Path | str,
        result_type: str | None = None,
        is_valid: bool | None = None,
    ) -> int:
        """Export results to JSONL (newline-delimited JSON) file.

        Args:
            output_path: Path for the output JSONL file.
            result_type: Optional filter by result type.
            is_valid: Optional filter by validation status.

        Returns:
            Number of results exported.
        """
        if isinstance(output_path, str):
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        conditions = []
        if result_type:
            conditions.append(Result.result_type == result_type)
        if is_valid is not None:
            conditions.append(Result.is_valid == is_valid)

        count = 0
        async with self._session_factory() as session:
            query = select(  # type: ignore[call-overload]
                Result.id,
                Result.request_id,
                Result.result_type,
                Result.data_json,
                Result.is_valid,
                Result.validation_errors_json,
                Result.created_at,
            ).order_by(Result.created_at.asc())  # type: ignore[union-attr]

            for cond in conditions:
                query = query.where(cond)

            result = await session.execute(query)
            rows = result.all()

        with output_path.open("w") as f:
            for row in rows:
                (
                    result_id,
                    request_id,
                    rtype,
                    data_json,
                    valid,
                    errors_json,
                    created_at,
                ) = row

                try:
                    data = json.loads(data_json) if data_json else {}
                except json.JSONDecodeError:
                    data = {}

                validation_errors = None
                if errors_json:
                    try:
                        validation_errors = json.loads(errors_json)
                    except json.JSONDecodeError:
                        pass

                record = {
                    "id": result_id,
                    "request_id": request_id,
                    "result_type": rtype,
                    "data": data,
                    "is_valid": bool(valid),
                    "validation_errors": validation_errors,
                    "created_at": created_at,
                }

                f.write(json.dumps(record) + "\n")
                count += 1

        return count

    # =========================================================================
    # Response Search Methods
    # =========================================================================

    async def search_responses(
        self,
        text: str | None = None,
        regex: str | None = None,
        xpath: str | None = None,
        continuation: str | None = None,
    ) -> list[dict[str, int]]:
        """Search response content for matching patterns.

        Exactly one of text, regex, or xpath must be provided.

        Args:
            text: Plain text to search for (case-insensitive).
            regex: Regular expression pattern to search for.
            xpath: XPath expression to evaluate.
            continuation: Optional filter by continuation (step name).

        Returns:
            List of dictionaries with request_id.

        Raises:
            ValueError: If zero or more than one search pattern is provided.
        """
        import re

        search_types = [text, regex, xpath]
        provided = sum(1 for s in search_types if s is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of text, regex, or xpath must be provided"
            )

        regex_pattern = None
        if regex is not None:
            regex_pattern = re.compile(regex)

        xpath_expr = None
        if xpath is not None:
            from lxml import etree

            xpath_expr = etree.XPath(xpath)

        query = (
            select(Request.id)
            .where(
                Request.response_status_code.isnot(None),  # type: ignore[union-attr]
            )
            .order_by(Request.id)  # type: ignore[arg-type]
        )
        if continuation:
            query = query.where(Request.continuation == continuation)

        matches: list[dict[str, int]] = []

        async with self._session_factory() as session:
            result = await session.execute(query)
            rows = result.all()

        for row in rows:
            request_id = row[0]

            content = await self.get_response_content(request_id)
            if content is None:
                continue

            try:
                content_str = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    content_str = content.decode("latin-1")
                except UnicodeDecodeError:
                    continue

            matched = False

            if text is not None:
                matched = text.lower() in content_str.lower()

            elif regex_pattern is not None:
                matched = regex_pattern.search(content_str) is not None

            elif xpath_expr is not None:
                try:
                    from lxml import html

                    tree = html.fromstring(content_str)
                    xpath_result = xpath_expr(tree)
                    matched = bool(xpath_result)
                except Exception:
                    continue

            if matched:
                matches.append({"request_id": request_id})

        return matches
