"""Step 4: Archive Request - File Downloads.

This test module verifies the file download and archiving capabilities introduced
in Step 4 of the scraper-driver architecture:

1. Scrapers can yield Request(archive=True) to download files
2. The driver downloads files and saves them to local storage
3. ArchiveResponse includes file_url with the local storage path
4. Files are saved with proper filenames extracted from URL or generated
5. Multiple file types (PDF, MP3) can be archived

Tests use a real aiohttp server to verify actual HTTP behavior.
"""

from pathlib import Path

import pytest

from kent.data_types import (
    ArchiveResponse,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from kent.driver.sync_driver import SyncDriver
from tests.mock_server import CASES
from tests.scraper.example.bug_court import (
    BugCourtScraperWithArchive,
)
from tests.utils import collect_results


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


class TestSyncDriverArchiving:
    """Tests for SyncDriver's file archiving capabilities."""

    @pytest.fixture
    def temp_storage(self, tmp_path: Path) -> Path:
        """Create a temporary storage directory."""
        storage = tmp_path / "test_storage"
        storage.mkdir()
        return storage

    @pytest.fixture
    def scraper(self, server_url: str) -> BugCourtScraperWithArchive:
        """Create a BugCourtScraperWithArchive instance."""
        scraper = BugCourtScraperWithArchive()
        scraper.BASE_URL = server_url
        return scraper

    @pytest.fixture
    def driver(
        self, scraper: BugCourtScraperWithArchive, temp_storage: Path
    ) -> SyncDriver[dict]:
        """Create a SyncDriver instance with custom storage."""
        return SyncDriver(scraper, storage_dir=temp_storage)

    def test_driver_creates_storage_directory(self, tmp_path: Path):
        """The driver shall create storage directory if it doesn't exist."""
        from tests.scraper.example.bug_court import (
            BugCourtScraper,
        )

        storage_dir = tmp_path / "new_storage"
        assert not storage_dir.exists()

        driver = SyncDriver(BugCourtScraper(), storage_dir=storage_dir)

        assert storage_dir.exists()
        assert driver.storage_dir == storage_dir

    def test_driver_uses_temp_directory_by_default(self):
        """The driver shall use system temp directory by default."""
        from tempfile import gettempdir

        from tests.scraper.example.bug_court import (
            BugCourtScraper,
        )

        driver = SyncDriver(BugCourtScraper())

        assert str(driver.storage_dir).startswith(gettempdir())
        assert "juriscraper_files" in str(driver.storage_dir)

    def test_driver_saves_pdf_file(
        self, driver: SyncDriver[dict], temp_storage: Path
    ):
        """The driver shall save PDF files to local storage."""
        callback, results = collect_results()
        driver.on_data = callback
        driver.run()

        # Find results with opinion files
        opinion_results = [r for r in results if "opinion_file" in r]

        assert len(opinion_results) > 0

        # Verify files were saved
        for result in opinion_results:
            file_path = Path(result["opinion_file"])
            assert file_path.exists()
            assert temp_storage in file_path.parents
            assert file_path.suffix == ".pdf"
            # Verify PDF content
            content = file_path.read_bytes()
            assert content.startswith(b"%PDF")

    def test_driver_saves_mp3_file(
        self, driver: SyncDriver[dict], temp_storage: Path
    ):
        """The driver shall save MP3 files to local storage."""
        callback, results = collect_results()
        driver.on_data = callback
        driver.run()

        # Find results with oral argument files
        oral_arg_results = [r for r in results if "oral_argument_file" in r]

        assert len(oral_arg_results) > 0

        # Verify files were saved
        for result in oral_arg_results:
            file_path = Path(result["oral_argument_file"])
            assert file_path.exists()
            assert temp_storage in file_path.parents
            assert file_path.suffix == ".mp3"
            # Verify MP3 content (should have MP3 sync word)
            content = file_path.read_bytes()
            assert content.startswith(b"\xff\xfb")

    def test_driver_extracts_filename_from_url(
        self, driver: SyncDriver[dict], temp_storage: Path
    ):
        """The driver shall extract filename from URL."""
        callback, results = collect_results()
        driver.on_data = callback
        driver.run()

        # Find results with opinion files
        opinion_results = [r for r in results if "opinion_file" in r]

        for result in opinion_results:
            file_path = Path(result["opinion_file"])
            # Filename should be extracted from URL: /opinions/BCC-2024-XXX.pdf
            assert ".pdf" in file_path.name

    def test_save_file_method_with_explicit_filename(self, temp_storage: Path):
        """The LocalSyncArchiveHandler shall extract filename from URL path."""
        from kent.driver.archive_handler import LocalSyncArchiveHandler

        handler = LocalSyncArchiveHandler(temp_storage)
        content = b"test content"
        url = "http://example.com/files/test.pdf"

        file_url = handler.save(
            url=url,
            deduplication_key=None,
            expected_type="pdf",
            hash_header_value=None,
            content=content,
        )

        file_path = Path(file_url)
        assert file_path.name == "test.pdf"
        assert file_path.exists()
        assert file_path.read_bytes() == content

    def test_save_file_method_generates_filename(self, temp_storage: Path):
        """The LocalSyncArchiveHandler shall generate filename when URL has no path."""
        from kent.driver.archive_handler import LocalSyncArchiveHandler

        handler = LocalSyncArchiveHandler(temp_storage)
        content = b"test content"
        url = "http://example.com/"

        file_url = handler.save(
            url=url,
            deduplication_key=None,
            expected_type="pdf",
            hash_header_value=None,
            content=content,
        )

        file_path = Path(file_url)
        assert file_path.name.startswith("download_")
        assert file_path.suffix == ".pdf"
        assert file_path.exists()
        assert file_path.read_bytes() == content

    def test_save_file_method_handles_audio_type(self, temp_storage: Path):
        """The LocalSyncArchiveHandler shall use .mp3 extension for audio type."""
        from kent.driver.archive_handler import LocalSyncArchiveHandler

        handler = LocalSyncArchiveHandler(temp_storage)
        content = b"test audio"
        url = "http://example.com/"

        file_url = handler.save(
            url=url,
            deduplication_key=None,
            expected_type="audio",
            hash_header_value=None,
            content=content,
        )

        file_path = Path(file_url)
        assert file_path.suffix == ".mp3"


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
        from tests.mock_server import generate_case_detail_html

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
        from tests.mock_server import generate_case_detail_html

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
        from tests.mock_server import generate_case_detail_html

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
        data: dict = results[0].unwrap()  # ty: ignore[invalid-assignment]
        assert "opinion_file" in data
        assert data["opinion_file"] == "/tmp/BCC-2024-001.pdf"
        assert "download_url" in data


class TestIntegration:
    """Integration tests using real aiohttp server."""

    def test_full_scraping_pipeline_with_archive(
        self, server_url: str, tmp_path: Path
    ):
        """The complete pipeline shall scrape and archive files."""
        scraper = BugCourtScraperWithArchive()
        scraper.BASE_URL = server_url
        storage_dir = tmp_path / "archive_storage"
        callback, results = collect_results()
        driver = SyncDriver(scraper, storage_dir=storage_dir, on_data=callback)
        driver.run()

        # Should have results for cases with files
        assert len(results) > 0

        # Verify opinion files were downloaded
        opinion_results = [r for r in results if "opinion_file" in r]
        cases_with_opinions = [c for c in CASES if c.has_opinion]
        assert len(opinion_results) == len(cases_with_opinions)

        # Verify oral argument files were downloaded
        oral_arg_results = [r for r in results if "oral_argument_file" in r]
        cases_with_oral_args = [c for c in CASES if c.has_oral_argument]
        assert len(oral_arg_results) == len(cases_with_oral_args)

        # Verify all files exist in storage
        for result in opinion_results:
            file_path = Path(result["opinion_file"])
            assert file_path.exists()
            assert storage_dir in file_path.parents

        for result in oral_arg_results:
            file_path = Path(result["oral_argument_file"])
            assert file_path.exists()
            assert storage_dir in file_path.parents

    def test_archive_request_ancestry_preserved(
        self, server_url: str, tmp_path: Path
    ):
        """Archive request ancestry shall be preserved through the request chain."""
        scraper = BugCourtScraperWithArchive()
        scraper.BASE_URL = server_url
        storage_dir = tmp_path / "archive_storage"
        callback, results = collect_results()
        driver = SyncDriver(scraper, storage_dir=storage_dir, on_data=callback)
        driver.run()

        # Should have archived files
        assert len(results) > 0

        # All results should have download_url field
        archived_results = [
            r
            for r in results
            if "opinion_file" in r or "oral_argument_file" in r
        ]
        for result in archived_results:
            assert "download_url" in result
            assert server_url in result["download_url"]
