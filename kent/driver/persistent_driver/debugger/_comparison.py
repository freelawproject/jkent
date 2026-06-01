"""Comparison methods for LocalDevDriverDebugger."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlmodel import select

from kent.driver.persistent_driver.models import (
    Request,
)
from kent.driver.persistent_driver.scoped_session import ScopedSessionFactory
from kent.driver.persistent_driver.sql_manager import (
    Page,
    RequestRecord,
    ResultRecord,
    SQLManager,
)


class ComparisonMixin:
    """Comparison and dry-run methods for scraper output analysis."""

    sql: SQLManager
    _session_factory: ScopedSessionFactory

    if TYPE_CHECKING:
        # Provided by InspectionMixin at runtime via multiple inheritance.
        async def get_response_content(
            self, request_id: int
        ) -> bytes | None: ...
        async def list_errors(
            self,
            error_type: str | None = None,
            is_resolved: bool | None = None,
            continuation: str | None = None,
            limit: int = 100,
            offset: int = 0,
        ) -> Page[dict[str, Any]]: ...

    async def get_child_requests_transitive(
        self, parent_request_id: int
    ) -> list[RequestRecord]:
        """Get all child requests transitively by parent_request_id.

        Recursively fetches all requests that were generated as children
        of the given parent request, including grandchildren and beyond.

        Args:
            parent_request_id: The parent request ID.

        Returns:
            List of RequestRecord objects for all transitive children.
        """
        # Build a recursive CTE
        base = (
            select(*RequestRecord.select_columns(Request))
            .where(Request.parent_request_id == parent_request_id)
            .cte(name="children", recursive=True)
        )

        req_alias = Request.__table__.alias("r")  # type: ignore[attr-defined]
        recursive = select(*RequestRecord.select_columns(req_alias.c)).where(
            req_alias.c.parent_request_id == base.c.id
        )

        children_cte = base.union_all(recursive)

        final_query = select(children_cte).order_by(children_cte.c.id)

        async with self._session_factory() as session:
            result = await session.execute(final_query)
            rows = result.all()

        return [RequestRecord.from_row(row) for row in rows]

    async def get_results_for_request(
        self, request_id: int
    ) -> list[ResultRecord]:
        """Get all results (ParsedData) for a request.

        Args:
            request_id: The request ID.

        Returns:
            List of ResultRecord objects.
        """
        page = await self.sql.list_results(
            request_id=request_id, limit=10000, offset=0
        )
        return page.items

    async def sample_terminal_requests(
        self, continuation: str, sample_count: int
    ) -> list[int]:
        """Sample terminal requests (requests that produced no child requests).

        Args:
            continuation: The continuation (step name) to sample from.
            sample_count: Number of terminal requests to sample.

        Returns:
            List of request IDs for sampled terminal requests.
        """
        child_alias = Request.__table__.alias("child")  # type: ignore[attr-defined]
        child_exists = (
            select(sa.literal(1))
            .select_from(child_alias)
            .where(child_alias.c.parent_request_id == Request.id)
            .correlate(Request)
        )

        async with self._session_factory() as session:
            result = await session.execute(
                select(Request.id)
                .where(
                    Request.continuation == continuation,
                    Request.status == "completed",
                    ~sa.exists(child_exists),
                )
                .order_by(sa.func.random())
                .limit(sample_count)
            )
            rows = result.all()

        return [row[0] for row in rows]

    async def sample_requests(
        self, continuation: str, sample_count: int
    ) -> list[int]:
        """Sample completed requests for a continuation (including non-terminal).

        Args:
            continuation: The continuation (step name) to sample from.
            sample_count: Number of requests to sample.

        Returns:
            List of request IDs for sampled requests.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Request.id)
                .where(
                    Request.continuation == continuation,
                    Request.status == "completed",
                )
                .order_by(sa.func.random())
                .limit(sample_count)
            )
            rows = result.all()

        return [row[0] for row in rows]

    async def compare_continuation(
        self,
        request_id: int,
        scraper_class: type,
    ) -> Any:
        """Compare continuation output between stored and dry-run execution.

        Args:
            request_id: The request ID to compare.
            scraper_class: The scraper class to instantiate for dry-run.

        Returns:
            ComparisonResult with detailed diffs.

        Raises:
            ValueError: If request not found or no response available.
        """
        from kent.driver.persistent_driver.comparison import (
            ComparisonResult,
            compare_continuation_output,
        )
        from kent.driver.persistent_driver.dry_run_driver import (
            DryRunDriver,
            DryRunResult,
        )

        # Get the full request data
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Request.id,
                    Request.url,
                    Request.method,
                    Request.continuation,
                    Request.current_location,
                    Request.accumulated_data_json,
                    Request.permanent_json,
                ).where(Request.id == request_id)
            )
            request_row = result.first()
            if not request_row:
                raise ValueError(f"Request {request_id} not found")

        request_data = {
            "url": request_row[1],
            "method": request_row[2],
            "continuation": request_row[3],
            "current_location": request_row[4],
            "accumulated_data_json": request_row[5],
            "permanent_json": request_row[6],
        }

        # Get the response data from the request row (merged table)
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload,misc]
                    Request.id,
                    Request.response_status_code,
                    Request.response_headers_json,
                    Request.response_url,
                    Request.content_compressed,
                    Request.content_size_original,
                    Request.content_size_compressed,
                    Request.compression_dict_id,
                    Request.continuation,
                    Request.response_created_at,
                    Request.speculation_outcome,
                ).where(
                    Request.id == request_id,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            response_row = result.first()
            if not response_row:
                raise ValueError(f"No response found for request {request_id}")

        response_data = {
            "id": response_row[0],
            "request_id": response_row[0],
            "status_code": response_row[1],
            "headers_json": response_row[2],
            "url": response_row[3],
            "content_compressed": response_row[4],
            "content_size_original": response_row[5],
            "content_size_compressed": response_row[6],
            "compression_dict_id": response_row[7],
            "continuation": response_row[8],
            "created_at": response_row[9],
            "speculation_outcome": response_row[10],
        }

        # Get decompressed content
        response_content = await self.get_response_content(request_id)
        if response_content is None:
            raise ValueError(
                f"No content available for response {response_data['id']}"
            )

        response_data["content"] = response_content
        try:
            response_data["text"] = response_content.decode("utf-8")
        except UnicodeDecodeError:
            response_data["text"] = ""

        # Load original stored results using recursive CTE
        base_children = (
            select(  # type: ignore[call-overload,misc]
                Request.id,
                Request.request_type,
                Request.url,
                Request.method,
                Request.continuation,
                Request.current_location,
                Request.accumulated_data_json,
                Request.permanent_json,
                Request.priority,
                Request.deduplication_key,
                Request.expected_type,
            )
            .where(Request.parent_request_id == request_id)
            .cte(name="children", recursive=True)
        )

        req_alias = Request.__table__.alias("r")  # type: ignore[attr-defined]
        recursive_children = select(  # type: ignore[call-overload]
            req_alias.c.id,
            req_alias.c.request_type,
            req_alias.c.url,
            req_alias.c.method,
            req_alias.c.continuation,
            req_alias.c.current_location,
            req_alias.c.accumulated_data_json,
            req_alias.c.permanent_json,
            req_alias.c.priority,
            req_alias.c.deduplication_key,
            req_alias.c.expected_type,
        ).where(req_alias.c.parent_request_id == base_children.c.id)

        children_cte = base_children.union_all(recursive_children)

        async with self._session_factory() as session:
            child_result = await session.execute(
                select(children_cte).order_by(children_cte.c.id)
            )
            child_rows = child_result.all()

        original_results = await self.get_results_for_request(request_id)

        from kent.driver.persistent_driver.dry_run_driver import (
            CapturedData,
            CapturedRequest,
        )

        original_requests = []
        for row in child_rows:
            original_requests.append(
                CapturedRequest(
                    request_type=row[1] or "navigating",
                    url=row[2],
                    method=row[3],
                    continuation=row[4],
                    accumulated_data=(json.loads(row[6]) if row[6] else {}),
                    permanent=(json.loads(row[7]) if row[7] else {}),
                    current_location=row[5] or "",
                    priority=row[8],
                    deduplication_key=row[9],
                    is_speculative=False,
                    speculation_id=None,
                    expected_type=row[10],
                )
            )

        original_data = [
            CapturedData(
                data=(json.loads(result.data_json) if result.data_json else {})
            )
            for result in original_results
        ]

        original: DryRunResult = DryRunResult(
            requests=original_requests, data=original_data, error=None
        )

        # Check if there was an error for this request
        errors_page = await self.list_errors(
            continuation=request_data["continuation"],
            is_resolved=None,
            limit=1000,
            offset=0,
        )
        original_error = None
        for error in errors_page.items:
            if error["request_id"] == request_id and not error["is_resolved"]:
                from kent.driver.persistent_driver.dry_run_driver import (
                    CapturedError,
                )

                original_error = CapturedError(
                    error_type=error["error_type"],
                    error_message=error["message"],
                )
                break

        original.error = original_error

        # Run dry-run with new code
        scraper_instance = scraper_class()
        driver = DryRunDriver(scraper_instance)
        new = driver.run_continuation(
            request_data["continuation"], response_data, request_data
        )

        # Compare
        comparison_result: ComparisonResult = compare_continuation_output(
            request_id=request_id,
            request_url=request_data["url"],
            continuation=request_data["continuation"],
            original=original,
            new=new,
        )

        return comparison_result

    async def run_with_selector_observer(
        self,
        request_id: int,
        scraper_class: type,
    ) -> dict[str, Any]:
        """Run a continuation with SelectorObserver to capture selector queries.

        Args:
            request_id: The request ID to run.
            scraper_class: The scraper class to instantiate.

        Returns:
            Dictionary with:
                - queries: List of SelectorQuery dicts from the observer
                - error: Error string if the continuation raised, else None

        Raises:
            ValueError: If request not found or no response available.
        """
        from kent.common.selector_observer import SelectorObserver
        from kent.data_types import (
            HttpMethod,
            HTTPRequestParams,
            Response,
        )
        from kent.data_types import (
            Request as DataRequest,
        )

        # Get request data
        async with self._session_factory() as session:
            result = await session.execute(
                select(  # type: ignore[call-overload]
                    Request.url,
                    Request.method,
                    Request.continuation,
                    Request.current_location,
                    Request.accumulated_data_json,
                    Request.permanent_json,
                ).where(Request.id == request_id)
            )
            request_row = result.first()
            if not request_row:
                raise ValueError(f"Request {request_id} not found")

        (
            url,
            method,
            continuation_name,
            current_location,
            accumulated_data_json,
            permanent_json,
        ) = request_row

        # Get response data
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    Request.response_status_code,
                    Request.response_headers_json,
                    Request.response_url,
                ).where(
                    Request.id == request_id,
                    Request.response_status_code.isnot(None),  # type: ignore[union-attr]
                )
            )
            response_row = result.first()
            if not response_row:
                raise ValueError(f"No response found for request {request_id}")

        status_code, headers_json, response_url = response_row

        # Get decompressed content
        content = await self.get_response_content(request_id)
        if content is None:
            raise ValueError(f"No content available for request {request_id}")

        # Reconstruct Response object
        headers = json.loads(headers_json) if headers_json else {}
        accumulated_data = (
            json.loads(accumulated_data_json) if accumulated_data_json else {}
        )
        permanent = json.loads(permanent_json) if permanent_json else {}

        reconstructed_request = DataRequest(
            request=HTTPRequestParams(
                method=HttpMethod(method),
                url=url,
            ),
            continuation=continuation_name,
            current_location=current_location or url,
            accumulated_data=accumulated_data,
            permanent=permanent,
        )

        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")

        response = Response(
            status_code=status_code,
            url=response_url,
            content=content,
            text=text,
            headers=headers,
            request=reconstructed_request,
        )

        # Run continuation with SelectorObserver
        scraper_instance = scraper_class()
        error: str | None = None

        with SelectorObserver() as observer:
            try:
                continuation_method = scraper_instance.get_continuation(
                    continuation_name
                )
                gen = continuation_method(response)
                for _item in gen:
                    pass
            except Exception as e:
                error = f"{type(e).__name__}: {e}"

        return {
            "queries": observer.json(),
            "error": error,
        }

    async def compare_request_tree(
        self,
        request_id: int,
        scraper_class: type,
    ) -> list[Any]:
        """Compare entire request tree starting from a request.

        Args:
            request_id: The root request ID to start comparison from.
            scraper_class: The scraper class to instantiate for dry-run.

        Returns:
            List of ComparisonResult for each request in the tree.
        """
        from collections import deque

        results = []
        queue: deque[int] = deque([request_id])
        visited: set[int] = set()

        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            try:
                result = await self.compare_continuation(
                    current_id, scraper_class
                )
                results.append(result)

                async with self._session_factory() as session:
                    child_result = await session.execute(
                        select(Request.id).where(
                            Request.parent_request_id == current_id,
                            Request.status == "completed",
                        )
                    )
                    child_rows = child_result.all()

                for row in child_rows:
                    child_id = row[0]
                    if child_id not in visited:
                        queue.append(child_id)

            except ValueError:
                pass

        return results
