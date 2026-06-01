"""Integrity check methods for LocalDevDriverDebugger."""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from sqlmodel import select

from kent.driver.persistent_driver.models import (
    Estimate,
    Request,
    Result,
)
from kent.driver.persistent_driver.scoped_session import ScopedSessionFactory


class IntegrityMixin:
    """Integrity checks: orphaned requests/responses, ghost requests, estimates."""

    _session_factory: ScopedSessionFactory

    async def check_integrity(self) -> dict[str, Any]:
        """Check database integrity for orphaned requests and responses.

        Detects two types of integrity issues:
        1. Orphaned requests: completed requests with no corresponding response
        2. Orphaned responses: responses with no matching request

        Returns:
            Dictionary with integrity check results:
                - orphaned_requests: {count: int, ids: list[int]}
                - orphaned_responses: {count: int, ids: list[int]}
                - has_issues: bool (True if any orphans found)
        """
        async with self._session_factory() as session:
            # Orphaned requests: completed requests with no response
            orphaned_req_stmt = (
                select(Request.id)
                .where(
                    Request.status == "completed",
                    Request.response_status_code.is_(None),  # type: ignore[union-attr]
                )
                .order_by(Request.id)  # type: ignore[arg-type]
            )
            orphaned_req_count_stmt = (
                select(sa.func.count())
                .select_from(Request)
                .where(
                    Request.status == "completed",
                    Request.response_status_code.is_(None),  # type: ignore[union-attr]
                )
            )

            count_result = await session.execute(orphaned_req_count_stmt)
            orphaned_requests_count = count_result.scalar() or 0

            ids_result = await session.execute(orphaned_req_stmt)
            orphaned_request_ids = [row[0] for row in ids_result.all()]

            # Orphaned responses: impossible in the merged model since
            # response columns live on the Request row.
            orphaned_responses_count = 0
            orphaned_response_ids: list[int] = []

        has_issues = orphaned_requests_count > 0

        return {
            "orphaned_requests": {
                "count": orphaned_requests_count,
                "ids": orphaned_request_ids,
            },
            "orphaned_responses": {
                "count": orphaned_responses_count,
                "ids": orphaned_response_ids,
            },
            "has_issues": has_issues,
        }

    async def get_orphan_details(self) -> dict[str, Any]:
        """Get detailed information about orphaned requests and responses.

        Returns full details for each orphaned request and response, unlike
        check_integrity() which only returns counts and IDs.

        Returns:
            Dictionary with detailed orphan information:
                - orphaned_requests: List of dicts with {id, url, continuation, completed_at}
                - orphaned_responses: List of dicts with {id, request_id, url, created_at}
        """
        async with self._session_factory() as session:
            # Get orphaned request details
            orphaned_req_result = await session.execute(
                select(
                    Request.id,
                    Request.url,
                    Request.continuation,
                    Request.completed_at,
                )
                .where(
                    Request.status == "completed",
                    Request.response_status_code.is_(None),  # type: ignore[union-attr]
                )
                .order_by(Request.id)  # type: ignore[arg-type]
            )
            orphaned_requests = [
                {
                    "id": row[0],
                    "url": row[1],
                    "continuation": row[2],
                    "completed_at": row[3],
                }
                for row in orphaned_req_result.all()
            ]

            # Orphaned responses: impossible in the merged model since
            # response columns live on the Request row.
            orphaned_responses: list[dict[str, Any]] = []

        return {
            "orphaned_requests": orphaned_requests,
            "orphaned_responses": orphaned_responses,
        }

    async def get_ghost_requests(self) -> dict[str, Any]:
        """Get ghost requests (completed requests with no children and no results).

        Ghost requests are completed requests that produced no observable output:
        no child requests and no ParsedData results.

        Returns:
            Dictionary with ghost request information:
                - total_count: Total number of ghost requests
                - by_continuation: Dict mapping continuation -> count
                - ghosts: List of dicts with {id, url, continuation, completed_at}
        """
        # Subqueries for NOT EXISTS
        child = Request.__table__.alias("child")  # type: ignore[attr-defined]
        child_exists = (
            select(sa.literal(1))
            .select_from(child)
            .where(child.c.parent_request_id == Request.id)
            .correlate(Request)
        )
        result_exists = (
            select(sa.literal(1))
            .select_from(Result)
            .where(Result.request_id == Request.id)
            .correlate(Request)
        )

        # Base ghost condition
        ghost_conditions = [
            Request.status == "completed",
            ~sa.exists(child_exists),
            ~sa.exists(result_exists),
        ]

        async with self._session_factory() as session:
            # Get total count
            count_stmt = select(sa.func.count()).select_from(Request)
            for cond in ghost_conditions:
                count_stmt = count_stmt.where(cond)  # type: ignore[arg-type]
            count_result = await session.execute(count_stmt)
            total_count = count_result.scalar() or 0

            # Get counts by continuation
            by_cont_stmt = (
                select(
                    Request.continuation,
                    sa.func.count().label("ghost_count"),
                )
                .group_by(Request.continuation)
                .order_by(Request.continuation)
            )
            for cond in ghost_conditions:
                by_cont_stmt = by_cont_stmt.where(cond)  # type: ignore[arg-type]
            by_cont_result = await session.execute(by_cont_stmt)
            by_continuation: dict[str, int] = {}
            for row in by_cont_result.all():
                by_continuation[row[0]] = row[1]

            # Get detailed ghost request list
            ghost_stmt = select(
                Request.id,
                Request.url,
                Request.continuation,
                Request.completed_at,
            ).order_by(Request.continuation, Request.id)  # type: ignore[arg-type]
            for cond in ghost_conditions:
                ghost_stmt = ghost_stmt.where(cond)  # type: ignore[arg-type]
            ghost_result = await session.execute(ghost_stmt)
            ghosts = [
                {
                    "id": row[0],
                    "url": row[1],
                    "continuation": row[2],
                    "completed_at": row[3],
                }
                for row in ghost_result.all()
            ]

        return {
            "total_count": total_count,
            "by_continuation": by_continuation,
            "ghosts": ghosts,
        }

    async def check_estimates(self) -> dict[str, Any]:
        """Check EstimateData predictions against actual result counts.

        For each stored estimate, walks the request tree (via recursive CTE
        on parent_request_id) to count actual results of the expected types
        produced by descendant requests.

        Returns:
            Dictionary with estimate check results:
                - estimates: List of per-estimate results with
                  {request_id, expected_types, min_count, max_count,
                   actual_count, status}
                - summary: {total, passed, failed}
        """
        async with self._session_factory() as session:
            # Fetch all estimates
            estimate_rows = await session.execute(
                select(  # type: ignore[call-overload]
                    Estimate.id,
                    Estimate.request_id,
                    Estimate.expected_types_json,
                    Estimate.min_count,
                    Estimate.max_count,
                )
            )
            estimates = estimate_rows.all()

        results: list[dict[str, Any]] = []
        for est_id, request_id, types_json, min_count, max_count in estimates:
            expected_types: list[str] = json.loads(types_json)

            # Recursive CTE: find all descendant request IDs
            # Start with direct children of the estimate's request
            req_table = Request.__table__  # type: ignore[attr-defined]
            descendants = (
                select(req_table.c.id)
                .where(req_table.c.parent_request_id == request_id)
                .cte(name="descendants", recursive=True)
            )
            descendants = descendants.union_all(
                select(req_table.c.id).where(
                    req_table.c.parent_request_id == descendants.c.id
                )
            )

            # Count results of expected types in the descendant tree
            # Include results from the estimate's own request too
            async with self._session_factory() as session:
                count_result = await session.execute(
                    select(sa.func.count())
                    .select_from(Result)
                    .where(
                        Result.result_type.in_(expected_types),  # type: ignore[attr-defined]
                        sa.or_(
                            Result.request_id == request_id,
                            Result.request_id.in_(  # type: ignore[union-attr]
                                select(descendants.c.id)
                            ),
                        ),
                    )
                )
                actual_count = count_result.scalar() or 0

            # Determine pass/fail
            if (
                actual_count < min_count
                or max_count is not None
                and actual_count > max_count
            ):
                status = "fail"
            else:
                status = "pass"

            results.append(
                {
                    "estimate_id": est_id,
                    "request_id": request_id,
                    "expected_types": expected_types,
                    "min_count": min_count,
                    "max_count": max_count,
                    "actual_count": actual_count,
                    "status": status,
                }
            )

        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")

        return {
            "estimates": results,
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": failed,
            },
        }
