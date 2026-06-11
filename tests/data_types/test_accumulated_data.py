"""Tests for Accumulated Data - Data Flow Across Requests.

This module tests the accumulated_data feature:
1. accumulated_data field is added to BaseRequest
2. Deep copy semantics prevent mutation bugs
3. Data flows correctly through request chains
4. Sibling requests have independent accumulated_data

These tests are driver-free: they exercise the data types and the scraper's
parsing methods directly. The driver-level pipeline coverage lives in
tests/driver/unified/test_data_types_e2e.py against the unified driver.
"""

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)
from tests.mock_server import CASES
from tests.scraper.example.bug_court_accumulated_data import (
    BugCourtScraperWithAccumulatedData,
)


class TestAccumulatedDataField:
    """Tests for accumulated_data field on BaseRequest."""

    def test_base_request_has_accumulated_data_field(self):
        """BaseRequest shall have an accumulated_data field."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/test",
            ),
            continuation="parse",
        )

        assert hasattr(request, "accumulated_data")
        assert request.accumulated_data == {}

    def test_accumulated_data_can_be_set(self):
        """BaseRequest shall allow setting accumulated_data."""
        data = {"key": "value"}
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/test",
            ),
            continuation="parse",
            accumulated_data=data,
        )

        assert request.accumulated_data == {"key": "value"}

    def test_accumulated_data_is_deep_copied(self):
        """BaseRequest shall deep copy accumulated_data in __post_init__."""
        original_data: dict = {"key": "value", "nested": {"inner": "data"}}
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/test",
            ),
            continuation="parse",
            accumulated_data=original_data,
        )

        # Mutate the original
        original_data["key"] = "modified"
        original_data["nested"]["inner"] = "modified"

        # Request should have a deep copy - unchanged
        assert request.accumulated_data == {
            "key": "value",
            "nested": {"inner": "data"},
        }


class TestDeepCopySemantics:
    """Tests for deep copy semantics preventing mutation bugs."""

    def test_sibling_requests_have_independent_data(self):
        """Sibling requests shall have independent accumulated_data copies."""
        shared_data = {"case_name": "Ant v. Bee"}

        request1 = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/case1",
            ),
            continuation="parse",
            accumulated_data=shared_data,
        )

        request2 = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/case2",
            ),
            continuation="parse",
            accumulated_data=shared_data,
        )

        # Each request should have its own deep copy
        assert request1.accumulated_data is not request2.accumulated_data
        assert request1.accumulated_data == request2.accumulated_data

    def test_nested_dict_mutations_do_not_propagate(self):
        """Mutations to nested dicts shall not affect sibling requests."""
        shared_data = {"metadata": {"court": "trial", "year": 2024}}

        request1 = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/case1",
            ),
            continuation="parse",
            accumulated_data=shared_data,
        )

        request2 = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/case2",
            ),
            continuation="parse",
            accumulated_data=shared_data,
        )

        # Mutate nested dict in shared_data
        shared_data["metadata"]["court"] = "appeals"

        # Both requests should be unaffected (they have deep copies)
        assert request1.accumulated_data["metadata"]["court"] == "trial"
        assert request2.accumulated_data["metadata"]["court"] == "trial"


class TestAccumulatedDataPropagation:
    """Tests for accumulated_data propagation through resolve_from."""

    def test_navigating_request_propagates_accumulated_data(self):
        """Request.resolve_from shall propagate accumulated_data."""
        parent_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://example.com/parent",
            ),
            continuation="parse_parent",
        )

        parent_response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://example.com/parent",
            request=parent_request,
        )

        child_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/child",
            ),
            continuation="parse_child",
            accumulated_data={"key": "value"},
        )

        resolved = child_request.resolve_from(parent_response)

        assert resolved.accumulated_data == {"key": "value"}


class TestBugCourtScraperWithAccumulatedData:
    """Tests for the Bug Court scraper with accumulated_data."""

    @pytest.fixture
    def scraper(self) -> BugCourtScraperWithAccumulatedData:
        """Create a scraper instance for testing."""
        return BugCourtScraperWithAccumulatedData()

    def test_parse_appeals_list_adds_case_name_to_accumulated_data(
        self, scraper: BugCourtScraperWithAccumulatedData, server_url: str
    ):
        """The scraper shall add case_name to accumulated_data from list page."""
        # Get appeals cases
        appeals_cases = [c for c in CASES if c.court_level == "appeals"]
        assert len(appeals_cases) > 0

        # Generate list HTML
        html_parts = [
            "<html><body><table>",
            "<tr class='case-row'>"
            f"<td class='docket'>{appeals_cases[0].docket}</td>"
            f"<td><a href='/appeals/{appeals_cases[0].docket}'>{appeals_cases[0].case_name}</a></td>"
            "</tr>",
            "</table></body></html>",
        ]
        html = "\n".join(html_parts)

        response = Response(
            status_code=200,
            headers={},
            content=html.encode(),
            text=html,
            url=f"{server_url}/appeals",
            request=next(scraper.get_entry()),
        )

        results = list(scraper.parse_appeals_list(response))

        assert len(results) > 0
        request = results[0]
        assert isinstance(request, Request)
        assert "case_name" in request.accumulated_data
        assert (
            request.accumulated_data["case_name"] == appeals_cases[0].case_name
        )
