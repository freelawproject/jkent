"""Tests for the single_page() test utility."""

from __future__ import annotations

from collections.abc import Generator

from jkent.common.checked_html import CheckedHtmlElement
from jkent.common.decorators import single_page, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)


class HTMLScraper(BaseScraper[dict]):
    @step
    def parse_page(
        self, lxml_tree: CheckedHtmlElement, response: Response
    ) -> Generator[ScraperYield, None, None]:
        titles = lxml_tree.checked_xpath("//h1", "titles")
        for title in titles:
            yield ParsedData(
                {"title": title.text_content(), "url": response.url}
            )


class JSONScraper(BaseScraper[dict]):
    @step
    def parse_api(
        self, json_content: list
    ) -> Generator[ScraperYield, None, None]:
        for item in json_content:
            yield ParsedData(item)


class AccumulatedDataScraper(BaseScraper[dict]):
    @step
    def parse_detail(
        self, text: str, accumulated_data: dict
    ) -> Generator[ScraperYield, None, None]:
        yield ParsedData(
            {"text": text.strip(), "page": accumulated_data.get("page", 0)}
        )


class MixedYieldScraper(BaseScraper[dict]):
    @step
    def parse_and_follow(
        self, lxml_tree: CheckedHtmlElement, response: Response
    ) -> Generator[ScraperYield, None, None]:
        yield ParsedData({"found": True})
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/next"
            ),
            continuation="parse_and_follow",
        )
        yield None
        yield ParsedData({"found": True, "second": True})


class TestSinglePageHTML:
    def test_parses_html(self) -> None:
        run = single_page(HTMLScraper, "parse_page")
        results = run("<html><body><h1>Hello</h1></body></html>")
        assert len(results) == 1
        assert results[0]["title"] == "Hello"
        assert results[0]["url"] == "https://test.example.com"

    def test_custom_url(self) -> None:
        run = single_page(HTMLScraper, "parse_page")
        results = run(
            "<html><body><h1>Test</h1></body></html>",
            url="https://court.gov/page",
        )
        assert results[0]["url"] == "https://court.gov/page"

    def test_multiple_results(self) -> None:
        run = single_page(HTMLScraper, "parse_page")
        results = run(
            "<html><body><h1>A</h1><h1>B</h1><h1>C</h1></body></html>"
        )
        assert len(results) == 3
        assert [r["title"] for r in results] == ["A", "B", "C"]


class TestSinglePageJSON:
    def test_parses_json(self) -> None:
        run = single_page(JSONScraper, "parse_api")
        results = run('[{"id": 1}, {"id": 2}]')
        assert results == [{"id": 1}, {"id": 2}]


class TestSinglePageAccumulatedData:
    def test_passes_accumulated_data(self) -> None:
        run = single_page(AccumulatedDataScraper, "parse_detail")
        results = run("some content", accumulated_data={"page": 3})
        assert results[0]["page"] == 3

    def test_default_empty_accumulated_data(self) -> None:
        run = single_page(AccumulatedDataScraper, "parse_detail")
        results = run("some content")
        assert results[0]["page"] == 0


class TestSinglePageMixedYields:
    def test_filters_non_parsed_data(self) -> None:
        run = single_page(MixedYieldScraper, "parse_and_follow")
        results = run("<html><body></body></html>")
        assert len(results) == 2
        assert results[0] == {"found": True}
        assert results[1] == {"found": True, "second": True}


class ParentEchoScraper(BaseScraper[dict]):
    @step
    def parse_detail(
        self, previous_request: Request | None
    ) -> Generator[ScraperYield, None, None]:
        yield ParsedData({"parent": previous_request})


class TestPreviousRequestInjection:
    @staticmethod
    def _response_for(request: Request) -> Response:
        return Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url=request.request.url,
            request=request,
        )

    def _run(self, request: Request) -> list[dict]:
        scraper = ParentEchoScraper()
        return [
            item.unwrap()
            for item in scraper.parse_detail(self._response_for(request))
            if isinstance(item, ParsedData)
        ]

    def test_injects_immediate_parent(self) -> None:
        parent = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/list"
            ),
            continuation="seed",
        )
        child = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/detail"
            ),
            continuation="parse_detail",
            parent_request=parent,
        )

        results = self._run(child)

        assert results[0]["parent"] is parent

    def test_injects_none_for_entry_request(self) -> None:
        entry = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/list"
            ),
            continuation="parse_detail",
        )

        results = self._run(entry)

        assert results[0]["parent"] is None


class Windows1252Scraper(BaseScraper[dict]):
    @step(encoding="windows-1252")
    def parse_text(self, text: str) -> Generator[ScraperYield, None, None]:
        yield ParsedData({"text": text})


class Windows1252JSONScraper(BaseScraper[dict]):
    @step(encoding="windows-1252")
    def parse_api(
        self, json_content: dict
    ) -> Generator[ScraperYield, None, None]:
        yield ParsedData(json_content)


class TestSinglePageBytesContent:
    def test_bytes_content_injects_decoded_text(self) -> None:
        """A step asking for ``text`` gets decoded bytes, not ""."""
        run = single_page(AccumulatedDataScraper, "parse_detail")
        results = run(b"some bytes content")
        assert results[0]["text"] == "some bytes content"

    def test_bytes_content_decodes_with_step_encoding(self) -> None:
        """@step(encoding=...) governs text decoding for bytes content."""
        run = single_page(Windows1252Scraper, "parse_text")
        # 0xE9 is "é" in windows-1252 and invalid as standalone UTF-8.
        results = run(b"caf\xe9")
        assert results[0]["text"] == "café"

    def test_bytes_json_decodes_with_step_encoding(self) -> None:
        """@step(encoding=...) governs JSON decoding for bytes content."""
        run = single_page(Windows1252JSONScraper, "parse_api")
        # 0xE9 is "é" in windows-1252 and invalid as standalone UTF-8.
        results = run(b'{"name": "caf\xe9"}')
        assert results[0]["name"] == "café"

    def test_empty_text_with_content_falls_back_to_decode(self) -> None:
        """text="" with non-empty content decodes the bytes.

        _get_text treated "" as present (`is not None`) while _parse_json
        treated it as missing (falsy) — a step injecting ``text`` got ""
        where ``json_content`` would have decoded the body.
        """
        scraper = AccumulatedDataScraper()
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://test.example.com"
            ),
            continuation="parse_detail",
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"body from content",
            text="",
            url="https://test.example.com",
            request=request,
        )
        results = [
            item.unwrap()
            for item in scraper.parse_detail(response)
            if isinstance(item, ParsedData)
        ]
        assert results[0]["text"] == "body from content"
