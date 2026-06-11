"""Request - Multi-Page Scraping.

This test module verifies the multi-page scraping capabilities:

1. Scrapers can yield Request to request additional pages
2. The driver fetches URLs and calls continuation methods by name
3. current_location is tracked and updated for relative URL resolution
4. Pattern matching is used for exhaustive handling of yield types

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
    BugCourtScraper,
)


class TestRequest:
    """Tests for the Request data type."""

    def test_navigating_request_stores_url(self):
        """Request shall store the URL to fetch."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
        )

        assert request.request.url == "/cases/BCC-2024-001"

    def test_navigating_request_stores_continuation(self):
        """Request shall store the continuation method name."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )

        assert request.continuation == "parse_list"

    def test_navigating_request_defaults_to_get(self):
        """Request shall default to GET method."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )

        assert request.request.method == HttpMethod.GET

    def test_navigating_request_supports_post(self):
        """Request shall support POST method."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url="/search",
                data={"query": "beetle"},
            ),
            continuation="parse_results",
        )

        assert request.request.method == HttpMethod.POST
        assert request.request.data == {"query": "beetle"}

    def test_resolve_url_with_absolute_url(self):
        """Request shall return absolute URLs unchanged."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://other.example.com/cases",
            ),
            continuation="parse_list",
        )

        resolved = request.resolve_url("http://bugcourt.example.com/")

        assert resolved == "http://other.example.com/cases"

    def test_resolve_url_with_relative_url(self):
        """Request shall resolve relative URLs against current_location."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases/BCC-2024-001",
            ),
            continuation="parse_detail",
        )

        resolved = request.resolve_url("http://bugcourt.example.com/cases")

        assert resolved == "http://bugcourt.example.com/cases/BCC-2024-001"

    def test_resolve_url_with_relative_path(self):
        """Request shall resolve relative paths correctly."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="BCC-2024-001",
            ),
            continuation="parse_detail",
        )

        resolved = request.resolve_url("http://bugcourt.example.com/cases/")

        assert resolved == "http://bugcourt.example.com/cases/BCC-2024-001"

    def test_continuation_as_string_is_serializable(self):
        """Continuation specified as string shall be fully serializable."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases/BCC-2024-001",
                headers={"Accept": "text/html"},
            ),
            continuation="parse_detail",
        )

        # Should be JSON serializable (string continuation, not function)
        serialized = json.dumps(
            {
                "url": request.request.url,
                "continuation": request.continuation,
                "method": request.request.method.value,
                "headers": request.request.headers,
            }
        )

        # Should deserialize correctly
        deserialized = json.loads(serialized)
        assert deserialized["continuation"] == "parse_detail"


class TestResponse:
    """Tests for the Response data type."""

    def test_response_stores_status_code(self):
        """Response shall store the HTTP status code."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=b"<html></html>",
            text="<html></html>",
            url="http://example.com/cases",
            request=request,
        )

        assert response.status_code == 200

    def test_response_stores_headers(self):
        """Response shall store the response headers."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "text/html", "X-Custom": "value"},
            content=b"",
            text="",
            url="http://example.com/cases",
            request=request,
        )

        assert response.headers["Content-Type"] == "text/html"
        assert response.headers["X-Custom"] == "value"

    def test_response_stores_content_and_text(self):
        """Response shall store both raw bytes and decoded text."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )
        html = "<html><body>Hello</body></html>"
        response = Response(
            status_code=200,
            headers={},
            content=html.encode("utf-8"),
            text=html,
            url="http://example.com/cases",
            request=request,
        )

        assert response.content == html.encode("utf-8")
        assert response.text == html

    def test_response_stores_final_url(self):
        """Response shall store the final URL after redirects."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/old-cases",
            ),
            continuation="parse_list",
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://example.com/cases",  # Redirected URL
            request=request,
        )

        assert response.url == "http://example.com/cases"

    def test_response_stores_original_request(self):
        """Response shall store the original request."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://example.com/cases",
            request=request,
        )

        assert response.request is request


class TestParsedData:
    """Tests for the ParsedData data type."""

    def test_parsed_data_stores_data(self):
        """ParsedData shall store the data dictionary."""
        parsed = ParsedData({"docket": "BCC-2024-001", "case_name": "Test"})

        assert parsed.unwrap()["docket"] == "BCC-2024-001"
        assert parsed.unwrap()["case_name"] == "Test"

    def test_parsed_data_is_frozen(self):
        """ParsedData shall be immutable (frozen dataclass)."""
        parsed = ParsedData({"docket": "BCC-2024-001"})

        # The dataclass is frozen, so we can't reassign the data attribute
        with pytest.raises(AttributeError):
            parsed.data = {"new": "data"}  # type: ignore

    def test_unwrap_returns_data(self):
        """ParsedData.unwrap() shall return the wrapped data."""
        data = {"docket": "BCC-2024-001", "case_name": "Test"}
        parsed = ParsedData(data)

        assert parsed.unwrap() is data


class TestBugCourtScraper:
    """Tests for the BugCourtScraper class."""

    @pytest.fixture
    def scraper(self) -> BugCourtScraper:
        """Create a BugCourtScraper instance."""
        return BugCourtScraper()

    @pytest.fixture
    def list_response(self, cases_html: str) -> Response:
        """Create a Response for the case list page."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/cases",
            ),
            continuation="parse_list",
        )
        return Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=cases_html.encode("utf-8"),
            text=cases_html,
            url="http://bugcourt.example.com/cases",
            request=request,
        )

    def test_parse_list_yields_navigating_requests(
        self, scraper: BugCourtScraper, list_response: Response
    ):
        """The scraper shall yield Request for each case."""
        results = list(scraper.parse_list(list_response))

        assert len(results) == len(CASES)
        assert all(isinstance(r, Request) for r in results)

    def test_parse_list_requests_have_correct_urls(
        self, scraper: BugCourtScraper, list_response: Response
    ):
        """The scraper shall request the correct detail page URLs."""
        results = list(scraper.parse_list(list_response))

        expected_urls = {f"/cases/{case.docket}" for case in CASES}
        actual_urls = {
            r.request.url for r in results if isinstance(r, Request)
        }

        assert actual_urls == expected_urls

    def test_parse_list_requests_have_correct_continuation(
        self, scraper: BugCourtScraper, list_response: Response
    ):
        """The scraper shall specify parse_detail as continuation."""
        results = list(scraper.parse_list(list_response))

        assert all(
            r.continuation == "parse_detail"
            for r in results
            if isinstance(r, Request)
        )

    def test_parse_detail_yields_parsed_data(self, scraper: BugCourtScraper):
        """The scraper shall yield ParsedData from detail pages."""
        case = CASES[0]
        html = generate_case_detail_html(case)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/cases/{case.docket}",
            ),
            continuation="parse_detail",
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=html.encode("utf-8"),
            text=html,
            url=f"http://bugcourt.example.com/cases/{case.docket}",
            request=request,
        )

        results = list(scraper.parse_detail(response))

        assert len(results) == 1
        assert isinstance(results[0], ParsedData)

    def test_parse_detail_extracts_all_fields(self, scraper: BugCourtScraper):
        """The scraper shall extract all case fields from detail page."""
        case = CASES[0]
        html = generate_case_detail_html(case)
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/cases/{case.docket}",
            ),
            continuation="parse_detail",
        )
        response = Response(
            status_code=200,
            headers={"Content-Type": "text/html"},
            content=html.encode("utf-8"),
            text=html,
            url=f"http://bugcourt.example.com/cases/{case.docket}",
            request=request,
        )

        results = list(scraper.parse_detail(response))
        data: dict = (  # ty: ignore[invalid-assignment]
            results[0].unwrap() if isinstance(results[0], ParsedData) else {}  # type: ignore
        )

        assert data["docket"] == case.docket
        assert data["case_name"] == case.case_name
        assert data["plaintiff"] == case.plaintiff
        assert data["defendant"] == case.defendant
        assert data["date_filed"] == case.date_filed.isoformat()
        assert data["case_type"] == case.case_type
        assert data["status"] == case.status
        assert data["judge"] == case.judge
        assert data["summary"] == case.summary
