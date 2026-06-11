"""Bug Appeals Court scraper demonstrating accumulated_data.

This Step 5 scraper demonstrates how accumulated_data flows through
the request chain, collecting information from multiple court levels.

The scraper:
1. Scrapes appeals court list page, extracting case_name
2. Navigates to appeals detail page with accumulated_data
3. Extracts trial court docket from appeals page
4. Navigates to trial court page, enriching accumulated_data
5. Yields final combined data from both court levels
"""

from collections.abc import Generator

from lxml import html

from jkent.common.decorators import entry
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)


def _get_text(element, xpath: str) -> str:
    """Extract text content from an xpath query.

    Args:
        element: The lxml element to query.
        xpath: The XPath expression.

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


class BugCourtScraperWithAccumulatedData(BaseScraper[dict]):
    """Scraper for Bug Appeals Court demonstrating accumulated_data.

    This Step 5 implementation demonstrates:
    - Using accumulated_data to flow data across request chains
    - Collecting data from appeals court list page
    - Enriching data from appeals detail page
    - Following link to trial court and combining data
    - Deep copy semantics preventing cross-contamination

    The scraper visits three types of pages:
    1. Appeals list page (/appeals) - extracts case_name
    2. Appeals detail page (/appeals/{docket}) - extracts trial court docket
    3. Trial court page (/cases/{docket}) - enriches with trial court data
    """

    BASE_URL = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        """Create the initial request to start scraping."""
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"{self.BASE_URL}/appeals",
            ),
            continuation="parse_appeals_list",
        )

    def parse_appeals_list(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse the appeals list page and extract case_name.

        This method demonstrates starting the accumulated_data flow.
        Each case gets its case_name added to accumulated_data.

        Args:
            response: The Response from fetching the appeals list page.

        Yields:
            Request for each appeal with accumulated_data.
        """
        tree = html.fromstring(response.text)
        case_rows = tree.xpath("//tr[@class='case-row']")

        for row in case_rows:
            docket = _get_text(row, ".//td[@class='docket']")
            case_name = _get_text(row, ".//td/a")

            if docket:
                # Start accumulated_data with case_name from list page
                yield Request(
                    request=HTTPRequestParams(
                        method=HttpMethod.GET,
                        url=f"/appeals/{docket}",
                    ),
                    continuation="parse_appeals_detail",
                    accumulated_data={"case_name": case_name},
                )

    def parse_appeals_detail(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse appeals detail page and extract trial court docket.

        This method enriches accumulated_data with appeals court info,
        then navigates to the trial court to gather more data.

        Args:
            response: The Response from fetching the appeals detail page.

        Yields:
            Request to trial court with enriched accumulated_data.
        """
        tree = html.fromstring(response.text)

        # Get accumulated_data from the request
        # Make a copy to avoid mutating the original
        data = response.request.accumulated_data.copy()

        # Enrich with appeals court data
        data["appeals_docket"] = _get_text_by_id(tree, "docket")
        data["appeals_judge"] = _get_text_by_id(tree, "judge")
        data["appeals_date_filed"] = _get_text_by_id(tree, "date-filed")

        # Get trial court docket link
        trial_court_docket = _get_text_by_id(tree, "trial-court-docket")
        # Extract just the docket number from "Trial Court Case: BCC-2024-XXX"
        if ":" in trial_court_docket:
            trial_court_docket = trial_court_docket.split(":")[-1].strip()

        # Navigate to trial court with accumulated data
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/cases/{trial_court_docket}",
            ),
            continuation="parse_trial_court",
            accumulated_data=data,
        )

    def parse_trial_court(
        self, response: Response
    ) -> Generator[ScraperYield[dict], None, None]:
        """Parse trial court page and yield complete combined data.

        This method receives accumulated_data from the appeals pages
        and enriches it with trial court information before yielding.

        Args:
            response: The Response from fetching the trial court page.

        Yields:
            ParsedData with combined appeals + trial court data.
        """
        tree = html.fromstring(response.text)

        # Get accumulated_data and make a copy
        data = response.request.accumulated_data.copy()

        # Enrich with trial court data
        data["trial_docket"] = _get_text_by_id(tree, "docket")
        data["trial_judge"] = _get_text_by_id(tree, "judge")
        data["trial_date_filed"] = _get_text_by_id(tree, "date-filed")
        data["plaintiff"] = _get_text_by_id(tree, "plaintiff")
        data["defendant"] = _get_text_by_id(tree, "defendant")
        data["case_type"] = _get_text_by_id(tree, "case-type")

        # Yield the combined data
        yield ParsedData(data)
