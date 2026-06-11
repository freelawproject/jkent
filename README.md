# Jkent

**Alpha** — Jkent is in early-stage development. APIs and database schemas
may change without notice.

Jkent is a scraper-driver framework for structured web scraping. It
separates parsing logic (scrapers) from I/O orchestration (drivers): a
scraper is a collection of pure parsing steps that yield data and follow-on
requests, while the driver handles HTTP, file storage, rate limiting, and
persistence.

## Installation

The base package is the scraper SDK — everything needed to *write* scrapers
(data types, decorators, parsing helpers), with no driver machinery:

```bash
pip install jkent
```

To actually *run* scrapers, install the operational extra (database engine,
HTTP/browser transports):

```bash
pip install "jkent[operational]"
playwright install chromium   # only for browser-transport scrapers
```

## Writing a scraper

Scrapers subclass `BaseScraper`, declare entry points with `@entry`, and
parse pages in `@step` methods that yield `ParsedData` and further
`Request`s:

```python
from collections.abc import Generator

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    XPath,
)


class MyCourtScraper(BaseScraper[dict]):
    @entry(dict)
    def get_entry(self) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url="https://court.example.com/cases",
            ),
            continuation="parse_list",
        )

    @step
    def parse_list(self, page) -> Generator[Request, None, None]:
        for href in page.query_strings(XPath("//a[@class='case']/@href"), "case links"):
            yield Request(
                request=HTTPRequestParams(method=HttpMethod.GET, url=href),
                continuation="parse_detail",
            )

    @step
    def parse_detail(self, page) -> Generator[ParsedData, None, None]:
        yield ParsedData(data={...})
```

## Running a scraper

`RunBootstrapper` reads the scraper's `driver_requirements`, selects and
stitches the right transport/engine/storage, and drives a resumable run
backed by a SQLite database:

```python
from pathlib import Path

from jkent.driver.unified_driver import RunBootstrapper

async with RunBootstrapper(
    MyCourtScraper(), db_path=Path("run.db")
) as run:
    await run.run()
```

## Development

Jkent uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync
uv run pytest -n auto
uvx pre-commit run --all-files
```

## License

BSD 2-Clause. See [LICENSE](LICENSE).
