"""Unified-driver ports of the data-type integration suites.

The ``tests/data_types`` modules originally paired their data-type tests
with SyncDriver integration runs. Those driver halves were rewritten here
against ``ScrapeRun`` + ``HttpxTransport`` (the data-type halves stayed in
``tests/data_types``, driver-free). Coverage ported:

- navigating pipeline: list → detail fan-out, all cases parsed with full
  field integrity (was ``test_navigating_request.py``)
- non-navigating pipeline: detail pages fetch supplementary API data
  (was ``test_nonnavigating_request.py``)
- archive pipeline: PDF/MP3 downloads land in storage with the right
  content, results carry the local path + the download URL
  (was ``test_archive_request.py``)
- accumulated_data flows across a three-page chain
  (was ``test_accumulated_data.py``)
- permanent headers/cookies actually go out over the wire and are
  inherited down the chain — asserted via the mock server's ``/echo``
  endpoint, where the old suite only asserted chain completion
  (was ``test_permanent_data.py``)
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.archive_handler import LocalAsyncStreamingArchiveHandler
from jkent.driver.unified_driver import ScrapeRun
from tests.mock_server import CASES
from tests.scraper.example.bug_court import (
    BugCourtScraper,
    BugCourtScraperWithAPI,
    BugCourtScraperWithArchive,
)
from tests.scraper.example.bug_court_accumulated_data import (
    BugCourtScraperWithAccumulatedData,
)


async def _run_scrape(
    scraper: BaseScraper[Any], db_path: Path, **kwargs: Any
) -> list[Any]:
    """Drive a full ScrapeRun and return everything that reached on_data."""
    results: list[Any] = []

    async def on_data(data: Any) -> None:
        results.append(data)

    run = ScrapeRun(
        scraper,
        db_path,
        num_workers=2,
        on_data=on_data,
        rate_limited=False,
        **kwargs,
    )
    await run.open(setup_signal_handlers=False)
    try:
        await run.run()
    finally:
        await run.aclose()
    return results


# --- Navigating pipeline (was test_navigating_request.py) ----------------


async def test_navigating_pipeline_scrapes_all_cases(
    server_url: str, tmp_path: Path
) -> None:
    """The driver shall fan out list → detail and parse every case."""
    scraper = BugCourtScraper()
    scraper.BASE_URL = server_url

    results = await _run_scrape(scraper, tmp_path / "run.db")

    assert len(results) == len(CASES)
    assert {r["docket"] for r in results} == {c.docket for c in CASES}


async def test_navigating_pipeline_preserves_data_integrity(
    server_url: str, tmp_path: Path
) -> None:
    """Every field parsed from a detail page shall survive the pipeline."""
    scraper = BugCourtScraper()
    scraper.BASE_URL = server_url

    results = await _run_scrape(scraper, tmp_path / "run.db")
    results_by_docket = {r["docket"]: r for r in results}

    for case in CASES:
        result = results_by_docket[case.docket]
        assert result["case_name"] == case.case_name
        assert result["plaintiff"] == case.plaintiff
        assert result["defendant"] == case.defendant
        assert result["date_filed"] == case.date_filed.isoformat()
        assert result["case_type"] == case.case_type
        assert result["status"] == case.status
        assert result["judge"] == case.judge
        assert result["summary"] == case.summary


# --- Non-navigating pipeline (was test_nonnavigating_request.py) ---------


async def test_nonnavigating_pipeline_merges_api_data(
    server_url: str, tmp_path: Path
) -> None:
    """The driver shall fetch supplementary API data and merge it per case."""
    scraper = BugCourtScraperWithAPI()
    scraper.BASE_URL = server_url

    results = await _run_scrape(scraper, tmp_path / "run.db")

    assert len(results) == len(CASES)
    results_by_docket = {r["docket"]: r for r in results}
    assert set(results_by_docket) == {c.docket for c in CASES}

    for case in CASES:
        result = results_by_docket[case.docket]
        # HTML-side fields survived the merge.
        assert result["case_name"] == case.case_name
        assert result["judge"] == case.judge
        # API-only metadata merged onto the *right* case, not a constant.
        metadata = result["api_metadata"]
        assert metadata["jurisdiction"] == "BUG"
        assert metadata["case_number_normalized"] == case.docket.replace(
            "-", ""
        )
        assert metadata["last_updated"] == case.date_filed.isoformat()


# --- Archive pipeline (was test_archive_request.py) -----------------------


async def test_archive_pipeline_downloads_files(
    server_url: str, tmp_path: Path
) -> None:
    """Archive requests shall download PDF/MP3 files into local storage."""
    scraper = BugCourtScraperWithArchive()
    scraper.BASE_URL = server_url
    storage_dir = tmp_path / "archive_storage"

    results = await _run_scrape(
        scraper,
        tmp_path / "run.db",
        archive_handler=LocalAsyncStreamingArchiveHandler(storage_dir),
    )

    assert len(results) > 0

    opinion_results = [r for r in results if "opinion_file" in r]
    assert len(opinion_results) == len([c for c in CASES if c.has_opinion])
    for result in opinion_results:
        file_path = Path(result["opinion_file"])
        assert file_path.exists()
        assert storage_dir in file_path.parents
        assert file_path.read_bytes().startswith(b"%PDF")

    oral_arg_results = [r for r in results if "oral_argument_file" in r]
    assert len(oral_arg_results) == len(
        [c for c in CASES if c.has_oral_argument]
    )
    for result in oral_arg_results:
        file_path = Path(result["oral_argument_file"])
        assert file_path.exists()
        assert storage_dir in file_path.parents
        # MP3 sync word
        assert file_path.read_bytes().startswith(b"\xff\xfb")


async def test_archive_results_carry_download_url(
    server_url: str, tmp_path: Path
) -> None:
    """Archived results shall record the request-chain download URL."""
    scraper = BugCourtScraperWithArchive()
    scraper.BASE_URL = server_url

    results = await _run_scrape(
        scraper,
        tmp_path / "run.db",
        archive_handler=LocalAsyncStreamingArchiveHandler(
            tmp_path / "archive_storage"
        ),
    )

    archived = [
        r for r in results if "opinion_file" in r or "oral_argument_file" in r
    ]
    assert len(archived) > 0
    for result in archived:
        assert server_url in result["download_url"]


# --- accumulated_data flow (was test_accumulated_data.py) -----------------


async def test_accumulated_data_flows_through_three_pages(
    server_url: str, tmp_path: Path
) -> None:
    """accumulated_data shall flow appeals list → appeals detail → trial,
    carrying each page's fields through without later pages clobbering them."""
    scraper = BugCourtScraperWithAccumulatedData()
    scraper.BASE_URL = server_url

    results = await _run_scrape(scraper, tmp_path / "run.db")

    # One combined record per appeals-court case in the fixture.
    appeals = [c for c in CASES if c.court_level == "appeals"]
    assert len(results) == len(appeals)

    cases_by_docket = {c.docket: c for c in CASES}
    results_by_trial = {r["trial_docket"]: r for r in results}
    assert set(results_by_trial) == {a.trial_court_docket for a in appeals}

    for appeal in appeals:
        assert appeal.trial_court_docket is not None
        trial = cases_by_docket[appeal.trial_court_docket]
        result = results_by_trial[appeal.trial_court_docket]

        # From the appeals list page (survived two further hops).
        assert result["case_name"] == appeal.case_name
        # From the appeals detail page (label-prefixed in the fixture HTML).
        assert appeal.docket in result["appeals_docket"]
        assert appeal.judge in result["appeals_judge"]
        # From the trial court page — the correctly-linked trial case.
        assert result["trial_docket"] == trial.docket
        assert result["trial_judge"] == trial.judge
        assert result["plaintiff"] == trial.plaintiff
        assert result["defendant"] == trial.defendant
        assert result["case_type"] == trial.case_type
        # The appeals and trial judges differ, so finding both proves the
        # trial page enriched the chain without clobbering the appeals-level
        # data it inherited.
        assert appeal.judge != trial.judge


# --- permanent data over the wire (was test_permanent_data.py) ------------
#
# The data-type merge semantics are pinned driver-free in
# tests/data_types/test_permanent_data.py; here we prove the merged
# headers/cookies actually reach the server, via the /echo endpoint.


class _PermanentChainScraper(BaseScraper[dict]):
    """entry → set permanent on hop 1 → hop 2 inherits it implicitly."""

    base = "http://127.0.0.1"

    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/test"
            ),
            continuation="parse_entry",
        )

    @step
    def parse_entry(
        self, response: Response
    ) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/echo/first"
            ),
            continuation="parse_first",
            permanent={
                "headers": {"Authorization": "Bearer token123"},
                "cookies": {"session": "xyz789"},
            },
        )

    @step
    def parse_first(
        self, json_content: dict, response: Response
    ) -> Generator[Any, None, None]:
        yield ParsedData(data={"hop": "first", "echo": json_content})
        # No permanent here — the next hop must inherit the parent's.
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"{self.base}/echo/second"
            ),
            continuation="parse_second",
        )

    @step
    def parse_second(
        self, json_content: dict
    ) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={"hop": "second", "echo": json_content})


async def test_permanent_data_reaches_the_server_and_is_inherited(
    server_url: str, tmp_path: Path
) -> None:
    """permanent headers/cookies shall be sent on the request that set them
    and on descendant requests that never mention them."""
    scraper = _PermanentChainScraper()
    scraper.base = server_url

    results = await _run_scrape(scraper, tmp_path / "run.db")

    echoes = {r["hop"]: r["echo"] for r in results}
    assert set(echoes) == {"first", "second"}
    for hop in ("first", "second"):
        headers = echoes[hop]["headers"]
        cookies = echoes[hop]["cookies"]
        assert headers["authorization"] == "Bearer token123", hop
        assert cookies["session"] == "xyz789", hop
