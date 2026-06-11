"""Archive Request - File Downloads.

This test module verifies the file download and archiving capabilities:

1. Scrapers can yield Request(archive=True) to download files
2. The driver downloads files and saves them to local storage
3. ArchiveResponse includes file_url with the local storage path
4. Files are saved with proper filenames extracted from URL or generated
5. Multiple file types (PDF, MP3) can be archived

These tests are driver-free: they exercise the data types and the scraper's
parsing methods directly. The driver-level download coverage lives in
tests/driver/unified/test_data_types_e2e.py against the unified driver; the
archive handlers have their own dedicated suites.
"""

import pytest

from jkent.data_types import (
    ArchiveResponse,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from tests.mock_server import CASES, generate_case_detail_html
from tests.scraper.example.bug_court import (
    BugCourtScraperWithArchive,
)


class TestArchive:
    """Tests for the archive Request data type."""

    def test_archive_request_stores_url(self):
        """Archive Request shall store the URL to fetch."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )

        assert request.request.url == "/opinions/BCC-2024-001.pdf"

    def test_archive_request_stores_continuation(self):
        """Archive Request shall store the continuation method name."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )

        assert request.continuation == "archive_opinion"

    def test_archive_request_stores_expected_type(self):
        """Archive Request shall store the expected file type."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
            expected_type="pdf",
        )

        assert request.expected_type == "pdf"

    def test_archive_request_expected_type_optional(self):
        """Archive Request shall allow expected_type to be None."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )

        assert request.expected_type is None

    def test_archive_request_resolve_from_response(self):
        """Archive Request shall resolve URL from Response."""
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
        archive_request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
            expected_type="pdf",
        )

        resolved = archive_request.resolve_from(response)

        assert isinstance(resolved, Request) and resolved.archive
        assert (
            resolved.request.url
            == "http://bugcourt.example.com/opinions/BCC-2024-001.pdf"
        )
        assert resolved.continuation == "archive_opinion"
        assert resolved.expected_type == "pdf"
        assert (
            resolved.current_location
            == "http://bugcourt.example.com/cases/BCC-2024-001"
        )

    def test_resolve_from_preserves_all_http_request_params_fields(self):
        """resolve_from shall carry every HTTPRequestParams field through.

        Regression test: prior to the fix in resolve_request_from, only
        url/method/headers/params/data/cookies/verify were copied across,
        which silently reset timeout (and json/files/auth/allow_redirects/
        proxies/stream/cert) to their dataclass defaults. The Nevada
        Supreme Court scraper hit this with a ``timeout=360.0`` on an
        archive request that was reverted to ``None`` before the request
        manager ever saw it, causing downloads to hang indefinitely.
        """
        original_params = HTTPRequestParams(
            method=HttpMethod.POST,
            url="/opinions/BCC-2024-001.pdf",
            params={"q": "search"},
            data={"form": "value"},
            json={"k": "v"},
            headers={"Accept": "application/pdf"},
            cookies={"session": "abc"},
            files={"upload": "file.txt"},
            auth=("user", "pass"),
            timeout=360.0,
            allow_redirects=False,
            proxies={"http": "http://proxy.example:3128"},
            verify=False,
            stream=True,
            cert="/path/to/cert.pem",
        )

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
        archive_request = Request(
            request=original_params,
            continuation="archive_opinion",
            archive=True,
            expected_type="pdf",
        )

        resolved = archive_request.resolve_from(response)

        # URL is re-resolved against the response; everything else
        # should be carried through unchanged.
        assert (
            resolved.request.url
            == "http://bugcourt.example.com/opinions/BCC-2024-001.pdf"
        )
        assert resolved.request.method == original_params.method
        assert resolved.request.params == original_params.params
        assert resolved.request.data == original_params.data
        assert resolved.request.json == original_params.json
        assert resolved.request.headers == original_params.headers
        assert resolved.request.cookies == original_params.cookies
        assert resolved.request.files == original_params.files
        assert resolved.request.auth == original_params.auth
        assert resolved.request.timeout == original_params.timeout
        assert (
            resolved.request.allow_redirects == original_params.allow_redirects
        )
        assert resolved.request.proxies == original_params.proxies
        assert resolved.request.verify == original_params.verify
        assert resolved.request.stream == original_params.stream
        assert resolved.request.cert == original_params.cert


class TestArchiveResponse:
    """Tests for the ArchiveResponse data type."""

    def test_archive_response_inherits_from_response(self):
        """ArchiveResponse shall inherit from Response."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )
        response = ArchiveResponse(
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4",
            text="",
            url="http://bugcourt.example.com/opinions/BCC-2024-001.pdf",
            request=request,
            file_url="/tmp/BCC-2024-001.pdf",
        )

        assert isinstance(response, Response)

    def test_archive_response_stores_file_url(self):
        """ArchiveResponse shall store the local file_url."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )
        response = ArchiveResponse(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="http://bugcourt.example.com/opinions/BCC-2024-001.pdf",
            request=request,
            file_url="/tmp/juriscraper_files/BCC-2024-001.pdf",
        )

        assert response.file_url == "/tmp/juriscraper_files/BCC-2024-001.pdf"

    def test_archive_response_has_all_response_fields(self):
        """ArchiveResponse shall include all Response fields."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
        )
        response = ArchiveResponse(
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            content=b"%PDF-1.4",
            text="",
            url="http://bugcourt.example.com/opinions/BCC-2024-001.pdf",
            request=request,
            file_url="/tmp/BCC-2024-001.pdf",
        )

        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/pdf"
        assert response.content == b"%PDF-1.4"
        assert (
            response.url
            == "http://bugcourt.example.com/opinions/BCC-2024-001.pdf"
        )
        assert response.request is request


class TestBugCourtScraperWithArchive:
    """Tests for the BugCourtScraperWithArchive class."""

    @pytest.fixture
    def scraper(self, server_url: str) -> BugCourtScraperWithArchive:
        """Create a BugCourtScraperWithArchive instance."""
        scraper = BugCourtScraperWithArchive()
        scraper.BASE_URL = server_url
        return scraper

    def test_parse_detail_yields_archive_requests_for_opinions(
        self, scraper: BugCourtScraperWithArchive, server_url: str
    ):
        """The scraper shall yield archive Request for PDF opinions."""
        # Use a case with opinion
        case = [c for c in CASES if c.has_opinion][0]
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

        # Should yield archive Request for opinion
        archive_requests = [
            r for r in results if isinstance(r, Request) and r.archive
        ]
        assert len(archive_requests) > 0

        # Check the opinion request
        opinion_request = [
            r for r in archive_requests if "opinions" in r.request.url
        ][0]
        assert isinstance(opinion_request, Request) and opinion_request.archive
        assert opinion_request.continuation == "archive_opinion"
        assert opinion_request.expected_type == "pdf"

    def test_parse_detail_yields_archive_requests_for_oral_arguments(
        self, scraper: BugCourtScraperWithArchive, server_url: str
    ):
        """The scraper shall yield archive Request for MP3 oral arguments."""
        # Use a case with oral argument
        case = [c for c in CASES if c.has_oral_argument][0]
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

        # Should yield archive Request for oral argument
        archive_requests = [
            r for r in results if isinstance(r, Request) and r.archive
        ]
        assert len(archive_requests) > 0

        # Check the oral argument request
        oral_arg_request = [
            r for r in archive_requests if "oral-arguments" in r.request.url
        ][0]
        assert (
            isinstance(oral_arg_request, Request) and oral_arg_request.archive
        )
        assert oral_arg_request.continuation == "archive_oral_argument"
        assert oral_arg_request.expected_type == "audio"

    def test_parse_detail_yields_parsed_data_when_no_files(
        self, scraper: BugCourtScraperWithArchive, server_url: str
    ):
        """The scraper shall yield ParsedData when no files are available."""
        # Use a case without opinion or oral argument
        case = [
            c for c in CASES if not c.has_opinion and not c.has_oral_argument
        ][0]
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

        # Should yield ParsedData directly
        parsed_data = [r for r in results if isinstance(r, ParsedData)]
        assert len(parsed_data) == 1
        data: dict = parsed_data[0].unwrap()  # ty: ignore[invalid-assignment]
        assert data["docket"] == case.docket

    def test_archive_opinion_yields_parsed_data_with_file_url(
        self, scraper: BugCourtScraperWithArchive
    ):
        """The archive_opinion method shall yield ParsedData with file_url."""
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="http://bugcourt.example.com/opinions/BCC-2024-001.pdf",
            ),
            continuation="archive_opinion",
            archive=True,
            current_location="http://bugcourt.example.com/cases/BCC-2024-001",
        )
        response = ArchiveResponse(
            status_code=200,
            headers={},
            content=b"%PDF-1.4",
            text="",
            url="http://bugcourt.example.com/opinions/BCC-2024-001.pdf",
            request=request,
            file_url="/tmp/BCC-2024-001.pdf",
        )

        results = list(scraper.archive_opinion(response))

        assert len(results) == 1
        assert isinstance(results[0], ParsedData)
        data: dict = results[  # type: ignore
            0
        ].unwrap()
        assert "opinion_file" in data
        assert data["opinion_file"] == "/tmp/BCC-2024-001.pdf"
        assert "download_url" in data
