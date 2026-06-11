"""Bug Civil Court scraper example.

This module demonstrates the scraper-driver architecture through a fictional
court where insects file civil lawsuits. It exercises the framework's main
features:

- A simple function that parses HTML and yields dicts.
- A class with multiple methods, yielding ParsedData and Request.
- Request(nonnavigating=True) to fetch JSON API data without navigating.
- Request(archive=True) to download and archive PDF and MP3 files.
- accumulated_data to flow case data from appeals to trial court.
- LxmlPageElement to validate HTML structure assumptions.
- Pydantic models with deferred validation for data validation.
"""

import json
from collections.abc import Generator
from datetime import date, datetime

from lxml import html
from pydantic import Field, HttpUrl

from jkent.common.data_models import ScrapedData
from jkent.common.decorators import entry, step
from jkent.common.lxml_page_element import LxmlPageElement
from jkent.data_types import (
    ArchiveResponse,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)

# =============================================================================
# Step 2: Scraper Class with Multi-Page Support
# =============================================================================


class BugCourtScraper(BaseScraper[dict]):
    """Scraper for the Bug Civil Court.

    This Step 2 implementation demonstrates:
    - A scraper as a class (to bundle multiple methods)
    - Yielding Request to fetch detail pages
    - Yielding ParsedData with complete case information
    - Continuation methods specified by name (for serializability)

    The scraper visits two types of pages:
    1. List page (/cases) - contains basic case info and links to details
    2. Detail page (/cases/{docket}) - contains full case information
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        """Create the initial request to start scraping."""
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{self.BASE_URL}/cases",
            ),
            continuation="parse_list",
        )

    def parse_list(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse the case list page and yield requests for detail pages.

        This method extracts basic case information from the list page,
        then yields Requests to fetch each case's detail page.

        Args:
            response: The Response from fetching the list page.

        Yields:
            Request for each case detail page.
        """
        tree = html.fromstring(response.text)

        # Find all case rows in the table
        case_rows = tree.xpath("//tr[@class='case-row']")

        for row in case_rows:
            # Extract the docket number to build the detail URL
            docket = _get_text(row, ".//td[@class='docket']")

            if docket:
                # Yield a request to fetch the detail page
                # The URL is relative - driver will resolve against parent request
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"/cases/{docket}",
                    ),
                    continuation="parse_detail",
                )

    def parse_detail(
        self, response: Response
    ) -> Generator[ScraperYield, None, None]:
        """Parse a case detail page and yield the complete case data.

        This method extracts all case information from the detail page
        and yields it as ParsedData.

        Step 8: Uses LxmlPageElement to validate that the case detail
        container exists, raising HTMLStructuralAssumptionException if the
        page structure has changed.

        Args:
            response: The Response from fetching the detail page.

        Yields:
            ParsedData containing the complete case information.
        """
        tree = LxmlPageElement(html.fromstring(response.text), response.url)

        # Step 8: Validate that the case details container exists
        # This will raise HTMLStructuralAssumptionException if the structure changed
        tree.checked_xpath(
            "//div[@class='case-details']",
            "case details container",
            min_count=1,
            max_count=1,
        )

        # Extract all case details from the page
        yield ParsedData(
            {
                "docket": _get_text_by_id(tree, "docket"),
                "case_name": _get_text(tree, "//h2"),
                "plaintiff": _get_text_by_id(tree, "plaintiff"),
                "defendant": _get_text_by_id(tree, "defendant"),
                "date_filed": _get_text_by_id(tree, "date-filed"),
                "case_type": _get_text_by_id(tree, "case-type"),
                "status": _get_text_by_id(tree, "status"),
                "judge": _get_text_by_id(tree, "judge"),
                "summary": _get_text_by_id(tree, "summary"),
            }
        )


# =============================================================================
# Step 3: Scraper with Request(nonnavigating=True)
# =============================================================================


class BugCourtScraperWithAPI(BaseScraper[dict]):
    """Scraper for the Bug Civil Court with API metadata.

    This Step 3 implementation demonstrates:
    - Using Request(nonnavigating=True) to fetch API data
    - Staying at the same current_location while fetching JSON
    - Combining HTML parsing with JSON API data

    The scraper visits three types of pages:
    1. List page (/cases) - HTML list of cases
    2. Detail page (/cases/{docket}) - HTML detail page (navigation)
    3. API endpoint (/api/cases/{docket}) - JSON metadata (no navigation)
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        """Create the initial request to start scraping."""
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{self.BASE_URL}/cases",
            ),
            continuation="parse_list",
        )

    def parse_list(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse the case list page and yield requests for detail pages.

        Args:
            response: The Response from fetching the list page.

        Yields:
            Request for each case detail page.
        """
        tree = html.fromstring(response.text)
        case_rows = tree.xpath("//tr[@class='case-row']")

        for row in case_rows:
            docket = _get_text(row, ".//td[@class='docket']")
            if docket:
                # Navigate to the detail page
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"/cases/{docket}",
                    ),
                    continuation="parse_detail",
                )

    def parse_detail(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse detail page and fetch API metadata without navigating.

        This method demonstrates the key difference:
        - Request (default) updates current_location
        - Request(nonnavigating=True) keeps current_location unchanged

        Args:
            response: The Response from fetching the detail page.

        Yields:
            Request(nonnavigating=True) to fetch JSON API data.
        """
        tree = html.fromstring(response.text)
        docket = _get_text_by_id(tree, "docket")

        # Fetch API metadata without navigating away from the detail page
        # current_location remains at /cases/{docket}
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/api/cases/{docket}",
            ),
            continuation="parse_api",
            nonnavigating=True,
        )

    def parse_api(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse JSON API response and yield complete case data.

        Args:
            response: The Response from fetching the API endpoint.

        Yields:
            ParsedData with complete case information including API metadata.
        """
        data = json.loads(response.text)

        # Combine HTML data with API metadata
        yield ParsedData(
            {
                "docket": data["docket"],
                "case_name": data["case_name"],
                "plaintiff": data["plaintiff"],
                "defendant": data["defendant"],
                "date_filed": data["date_filed"],
                "case_type": data["case_type"],
                "status": data["status"],
                "judge": data["judge"],
                "summary": data["summary"],
                # Additional metadata from API
                "api_metadata": data["api_metadata"],
            }
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _get_text(element, xpath: str) -> str:
    """Extract text content from an xpath query.

    Args:
        element: The lxml element to query.
        xpath: The xpath expression.

    Returns:
        The text content, or empty string if not found.
    """
    results = element.xpath(xpath)
    if results:
        return results[0].text_content().strip()
    return ""


def _get_text_by_id(tree, element_id: str) -> str:
    """Extract text content from an element by its ID.

    Args:
        tree: The lxml tree to query.
        element_id: The ID of the element.

    Returns:
        The text content, or empty string if not found.
    """
    return _get_text(tree, f"//*[@id='{element_id}']")


def _parse_date(date_str: str):
    """Parse a date string into a date object.

    Args:
        date_str: Date string in format 'YYYY-MM-DD'.

    Returns:
        datetime.date object, or None if parsing fails.
    """
    try:
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


# =============================================================================
# Step 4: Scraper with Request(archive=True)
# =============================================================================


class BugCourtScraperWithArchive(BaseScraper[dict]):
    """Scraper for the Bug Civil Court with file archiving.

    This Step 4 implementation demonstrates:
    - Using Request(archive=True) to download PDF opinions
    - Using Request(archive=True) to download MP3 oral arguments
    - ArchiveResponse provides file_url for local storage path
    - Combining archived file paths with case metadata

    The scraper visits three types of pages:
    1. List page (/cases) - HTML list of cases
    2. Detail page (/cases/{docket}) - HTML detail page with download links
    3. File downloads (/opinions/{docket}.pdf, /oral-arguments/{docket}.mp3)
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        """Create the initial request to start scraping."""
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{self.BASE_URL}/cases",
            ),
            continuation="parse_list",
        )

    def parse_list(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse the case list page and yield requests for detail pages.

        Args:
            response: The Response from fetching the list page.

        Yields:
            Request for each case detail page.
        """
        tree = html.fromstring(response.text)
        case_rows = tree.xpath("//tr[@class='case-row']")

        for row in case_rows:
            docket = _get_text(row, ".//td[@class='docket']")
            if docket:
                # Navigate to the detail page
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"/cases/{docket}",
                    ),
                    continuation="parse_detail",
                )

    def parse_detail(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse detail page and check for downloadable files.

        This method extracts case data and yields archive Requests for
        any available PDF opinions or MP3 oral arguments.

        Args:
            response: The Response from fetching the detail page.

        Yields:
            Request(archive=True) for PDF opinions and MP3 oral arguments.
        """
        tree = html.fromstring(response.text)

        # Extract basic case data
        docket = _get_text_by_id(tree, "docket")
        case_name = _get_text(tree, "//h2")

        # Check for opinion PDF link
        opinion_links = tree.xpath('//a[contains(@href, "/opinions/")]/@href')
        if opinion_links:
            # Download and archive the PDF
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=opinion_links[0],
                ),
                continuation="archive_opinion",
                archive=True,
                expected_type="pdf",
            )

        # Check for oral argument MP3 link
        oral_arg_links = tree.xpath(
            '//a[contains(@href, "/oral-arguments/")]/@href'
        )
        if oral_arg_links:
            # Download and archive the MP3
            yield Request(
                request=HTTPRequestParams(
                    method=HttpMethod.GET,
                    url=oral_arg_links[0],
                ),
                continuation="archive_oral_argument",
                archive=True,
                expected_type="audio",
            )

        # If there are no files to download, yield the data directly
        if not opinion_links and not oral_arg_links:
            yield ParsedData(
                {
                    "docket": docket,
                    "case_name": case_name,
                    "plaintiff": _get_text_by_id(tree, "plaintiff"),
                    "defendant": _get_text_by_id(tree, "defendant"),
                    "date_filed": _get_text_by_id(tree, "date-filed"),
                    "case_type": _get_text_by_id(tree, "case-type"),
                    "status": _get_text_by_id(tree, "status"),
                    "judge": _get_text_by_id(tree, "judge"),
                    "summary": _get_text_by_id(tree, "summary"),
                }
            )

    def archive_opinion(
        self, response: ArchiveResponse
    ) -> Generator[ScraperYield[dict], None, None]:
        """Process archived opinion PDF and yield case data.

        The ArchiveResponse includes file_url with the local storage path.

        Args:
            response: The ArchiveResponse from downloading the PDF.

        Yields:
            ParsedData with case information including the PDF file path.
        """
        # The file_url contains the local path where the PDF was saved
        yield ParsedData(
            {
                "docket": response.request.current_location.split("/")[-1],
                "opinion_file": response.file_url,
                "download_url": response.url,
            }
        )

    def archive_oral_argument(
        self, response: ArchiveResponse
    ) -> Generator[ScraperYield[dict], None, None]:
        """Process archived oral argument MP3 and yield case data.

        The ArchiveResponse includes file_url with the local storage path.

        Args:
            response: The ArchiveResponse from downloading the MP3.

        Yields:
            ParsedData with case information including the MP3 file path.
        """
        # The file_url contains the local path where the MP3 was saved
        yield ParsedData(
            {
                "docket": response.request.current_location.split("/")[-1],
                "oral_argument_file": response.file_url,
                "download_url": response.url,
            }
        )


# =============================================================================
# Step 9: Scraper with Data Validation
# =============================================================================


class BugCourtCaseData(ScrapedData):
    """Data model for Bug Court case details.

    This model defines the expected schema for case data scraped from
    Bug Civil Court. It's used for validation to catch data format changes.
    """

    docket: str = Field(
        ..., description="Docket number (e.g., 'BCC-2024-001')"
    )
    case_name: str = Field(..., description="Full case name")
    plaintiff: str = Field(..., description="Plaintiff name")
    defendant: str = Field(..., description="Defendant name")
    date_filed: date = Field(..., description="Filing date")
    case_type: str = Field(..., description="Type of case")
    status: str = Field(..., description="Current case status")
    judge: str = Field(..., description="Assigned judge")
    court_reporter: str = Field(..., description="Court reporter name")
    pdf_url: HttpUrl | None = Field(
        None, description="URL to case PDF (optional)"
    )
    audio_url: HttpUrl | None = Field(
        None, description="URL to oral argument audio (optional)"
    )


class BugCourtScraperWithValidation(BaseScraper[BugCourtCaseData]):
    """Scraper for Bug Civil Court with Pydantic data validation.

    This scraper demonstrates:
    - Step 9: Using Pydantic models to validate scraped data
    - Step 9: Raising DataFormatAssumptionException on validation errors
    - Step 9: Ensuring data conforms to expected schema before yielding
    - Step 19: Using @step decorator with automatic argument injection
    - Step 19: Using Callable continuations instead of string names

    The scraper validates data against the BugCourtCaseData model to catch
    data format changes early. It also uses the @step decorator to automatically
    inject lxml_tree and response parameters based on function signatures.
    """

    BASE_URL = "http://127.0.0.1"

    @entry(BugCourtCaseData)
    def get_entry(self) -> Generator[Request, None, None]:
        """Get the entry request for the scraper."""
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{self.BASE_URL}/cases",
            ),
            continuation="parse_list",
        )

    @step
    def parse_list(
        self, lxml_tree: LxmlPageElement
    ) -> Generator[ScraperYield, None, None]:
        """Parse the case list page and yield requests for detail pages.

        Step 19: Uses @step decorator with lxml_tree automatic injection.
        """
        case_rows = lxml_tree.checked_xpath(
            "//tr[@class='case-row']",
            "case rows",
            min_count=1,
        )

        for row in case_rows:
            if isinstance(row, LxmlPageElement):
                docket_cells = row.checked_xpath(
                    ".//td[@class='docket']",
                    "docket cell",
                    min_count=1,
                    max_count=1,
                )
                docket = (
                    docket_cells[0].text_content().strip()
                    if isinstance(docket_cells[0], LxmlPageElement)
                    else None
                )

                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"/cases/{docket}",
                    ),
                    continuation=self.parse_detail,  # Step 19: Callable continuation
                )

    @step(priority=1)
    def parse_detail(
        self, lxml_tree: LxmlPageElement, response: Response
    ) -> Generator[ScraperYield, None, None]:
        """Parse a case detail page with deferred data validation.

        This method extracts case data and wraps it in a DeferredValidation
        object. The driver will call confirm() to validate the data later.

        Step 9: Demonstrates deferred validation using Pydantic models.
        Step 19: Uses @step decorator with lxml_tree and response injection.

        Args:
            lxml_tree: Parsed HTML tree (injected by @step decorator).
            response: The Response object (injected by @step decorator).

        Yields:
            ParsedData containing DeferredValidation wrapper.
        """
        # Step 8: Validate structure
        lxml_tree.checked_xpath(
            "//div[@class='case-details']",
            "case details container",
            min_count=1,
            max_count=1,
        )

        # Step 9: Yield deferred validation - driver will validate
        yield ParsedData(
            BugCourtCaseData.raw(
                request_url=response.url,
                docket=_get_text_by_id(lxml_tree, "docket"),
                case_name=_get_text(lxml_tree, "//h2"),
                plaintiff=_get_text_by_id(lxml_tree, "plaintiff"),
                defendant=_get_text_by_id(lxml_tree, "defendant"),
                date_filed=_parse_date(
                    _get_text_by_id(lxml_tree, "date-filed")
                ),
                case_type=_get_text_by_id(lxml_tree, "case-type"),
                status=_get_text_by_id(lxml_tree, "status"),
                judge=_get_text_by_id(lxml_tree, "judge"),
                court_reporter=_get_text_by_id(lxml_tree, "court-reporter"),
            )
        )
