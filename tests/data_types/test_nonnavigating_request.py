"""Non-Navigating Request - API Calls.

This test module verifies the API call capabilities:

1. Scrapers can yield Request(nonnavigating=True) for API calls
2. The driver handles non-navigating requests without updating current_location
3. Navigating requests update current_location, non-navigating requests do not
4. Request provides shared URL resolution logic
5. Both request types can be used together in the same scraper

These tests are driver-free: they exercise the data types and the scraper's
parsing methods directly. The driver-level pipeline coverage lives in
tests/driver/unified/test_data_types_e2e.py against the unified driver.
"""

import json

import pytest

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from tests.mock_server import CASES, generate_case_detail_html
from tests.scraper.example.bug_court import (
    BugCourtScraperWithAPI,
)


class TestRequest:
    """Tests for the Request data type."""

    def test_base_request_stores_url(self):
        """Request shall store the URL to fetch."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases/BCC-2024-001",
            ),
            continuation="parse_api",
        )

        assert request.request.url == "/api/cases/BCC-2024-001"

    def test_base_request_stores_continuation(self):
        """Request shall store the continuation method name."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases",
            ),
            continuation="parse_api",
        )

        assert request.continuation == "parse_api"

    def test_base_request_defaults_to_get(self):
        """Request shall default to GET method."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases",
            ),
            continuation="parse_api",
        )

        assert request.request.method == HttpMethod.GET

    def test_base_request_supports_post(self):
        """Request shall support POST method."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url="/api/search",
                data={"query": "beetle"},
            ),
            continuation="parse_results",
        )

        assert request.request.method == HttpMethod.POST
        assert request.request.data == {"query": "beetle"}

    def test_base_request_resolve_url_absolute(self):
        """Request shall return absolute URLs unchanged."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://other.example.com/api/cases",
            ),
            continuation="parse_api",
        )

        resolved = request.resolve_url("http://bugcourt.example.com/")

        assert resolved == "http://other.example.com/api/cases"

    def test_base_request_resolve_url_relative(self):
        """Request shall resolve relative URLs against current_location."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases/BCC-2024-001",
            ),
            continuation="parse_api",
        )

        resolved = request.resolve_url(
            "http://bugcourt.example.com/cases/BCC-2024-001"
        )

        assert resolved == "http://bugcourt.example.com/api/cases/BCC-2024-001"


class TestNonNavigating:
    """Tests for the non-navigating Request data type."""

    def test_non_navigating_request_stores_url(self):
        """Non-navigating Request shall store the URL to fetch."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases/BCC-2024-001",
            ),
            continuation="parse_api",
            nonnavigating=True,
        )

        assert request.request.url == "/api/cases/BCC-2024-001"

    def test_non_navigating_request_resolve_from_response(self):
        """Non-navigating Request shall resolve URL from Response."""
        base_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://bugcourt.example.com/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://bugcourt.example.com/cases/BCC-2024-001",
            request=base_request,
        )
        api_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases/BCC-2024-001",
            ),
            continuation="parse_api",
            nonnavigating=True,
        )

        resolved = api_request.resolve_from(response)

        assert resolved.nonnavigating
        assert (
            resolved.request.url
            == "http://bugcourt.example.com/api/cases/BCC-2024-001"
        )
        assert resolved.continuation == "parse_api"
        assert (
            resolved.current_location
            == "http://bugcourt.example.com/cases/BCC-2024-001"
        )


class TestNavigatingRequestResolveFrom:
    """Tests for resolving a navigating Request from a Response."""

    def test_navigating_request_resolve_from_response(self):
        """Request shall resolve URL from Response."""
        base_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://bugcourt.example.com/cases",
            ),
            continuation="parse_list",
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://bugcourt.example.com/cases",
            request=base_request,
        )
        detail_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
        )

        resolved = detail_request.resolve_from(response)

        assert (
            resolved.request.url
            == "http://bugcourt.example.com/cases/BCC-2024-001"
        )
        assert resolved.current_location == "http://bugcourt.example.com/cases"


class TestRequestCurrentLocation:
    """Tests for current_location tracking in requests."""

    @pytest.fixture
    def scraper(self, server_url: str) -> BugCourtScraperWithAPI:
        """Create a BugCourtScraperWithAPI instance configured for test server."""
        scraper = BugCourtScraperWithAPI()
        scraper.BASE_URL = server_url
        return scraper

    def test_entry_request_has_no_current_location(
        self, scraper: BugCourtScraperWithAPI
    ):
        """The entry request shall have an empty current_location."""
        entry = next(scraper.get_entry())

        assert entry.current_location == ""
        assert entry.parent_request is None

    def test_navigating_request_updates_current_location(
        self, server_url: str
    ):
        """Request shall update current_location to the response URL."""
        # Create a navigating request
        entry_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server_url}/cases",
            ),
            continuation="parse_list",
        )

        # Create a response
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url=f"{server_url}/cases",
            request=entry_request,
        )

        # Create a new navigating request resolved from the response
        detail_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
        )

        resolved = detail_request.resolve_from(response)

        # current_location should be the response URL
        assert resolved.current_location == f"{server_url}/cases"
        assert resolved.request.url == f"{server_url}/cases/BCC-2024-001"

    def test_non_navigating_request_preserves_current_location(
        self, server_url: str
    ):
        """Non-navigating Request shall preserve current_location."""
        # Create a navigating request and response
        detail_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server_url}/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
            current_location=f"{server_url}/cases",
        )

        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url=f"{server_url}/cases/BCC-2024-001",
            request=detail_request,
        )

        # Create a non-navigating Request for the API
        api_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/api/cases/BCC-2024-001",
            ),
            continuation="parse_api",
            nonnavigating=True,
        )

        resolved = api_request.resolve_from(response)

        # current_location should be the response URL (from navigating Request)
        assert resolved.current_location == f"{server_url}/cases/BCC-2024-001"
        assert resolved.request.url == f"{server_url}/api/cases/BCC-2024-001"


class TestBugCourtScraperWithAPI:
    """Tests for the BugCourtScraperWithAPI class."""

    @pytest.fixture
    def scraper(self, server_url: str) -> BugCourtScraperWithAPI:
        """Create a BugCourtScraperWithAPI instance."""
        scraper = BugCourtScraperWithAPI()
        scraper.BASE_URL = server_url
        return scraper

    @pytest.fixture
    def list_response(self, cases_html: str, server_url: str) -> Response:
        """Create a Response for the case list page."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server_url}/cases",
            ),
            continuation="parse_list",
        )
        return Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=cases_html.encode("utf-8"),
            text=cases_html,
            url=f"{server_url}/cases",
            request=request,
        )

    def test_parse_list_yields_navigating_requests(
        self, scraper: BugCourtScraperWithAPI, list_response: Response
    ):
        """The scraper shall yield Request for each case."""
        results = list(scraper.parse_list(list_response))

        assert len(results) == len(CASES)
        assert all(isinstance(r, Request) for r in results)

    def test_parse_detail_yields_non_navigating_request(
        self, scraper: BugCourtScraperWithAPI, server_url: str
    ):
        """The scraper shall yield non-navigating Request for API call."""
        case = CASES[0]
        html = generate_case_detail_html(case)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server_url}/cases/{case.docket}",
            ),
            continuation="parse_detail",
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=html.encode("utf-8"),
            text=html,
            url=f"{server_url}/cases/{case.docket}",
            request=request,
        )

        results = list(scraper.parse_detail(response))

        assert len(results) == 1
        assert isinstance(results[0], Request) and results[0].nonnavigating  # type: ignore
        assert "/api/cases/" in results[0].request.url  # type: ignore

    def test_parse_api_yields_parsed_data(
        self, scraper: BugCourtScraperWithAPI, server_url: str
    ):
        """The scraper shall yield ParsedData from API response."""
        case = CASES[0]
        api_data = {
            "docket": case.docket,
            "case_name": case.case_name,
            "plaintiff": case.plaintiff,
            "defendant": case.defendant,
            "date_filed": case.date_filed.isoformat(),
            "case_type": case.case_type,
            "status": case.status,
            "judge": case.judge,
            "summary": case.summary,
            "api_metadata": {
                "last_updated": case.date_filed.isoformat(),
                "case_number_normalized": case.docket.replace("-", ""),
                "jurisdiction": "BUG",
            },
        }
        json_text = json.dumps(api_data)

        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{server_url}/api/cases/{case.docket}",
            ),
            continuation="parse_api",
            nonnavigating=True,
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "application/json"},
            content=json_text.encode("utf-8"),
            text=json_text,
            url=f"{server_url}/api/cases/{case.docket}",
            request=request,
        )

        results = list(scraper.parse_api(response))

        assert len(results) == 1
        assert isinstance(results[0], ParsedData)
        data: dict = results[  # type: ignore
            0
        ].unwrap()
        assert "api_metadata" in data
        assert data["api_metadata"]["jurisdiction"] == "BUG"
