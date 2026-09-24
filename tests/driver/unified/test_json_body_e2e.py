"""``HTTPRequestParams.json`` survives serialize → DB → dequeue → dispatch.

``queue.py`` stores ``json`` in the ``json_data`` column and
``httpx_transport`` passes ``json=`` through; this pins the whole path
against a live server, including the wire half: the body goes out as
``application/json``, not form-encoded.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from jkent.common.decorators import entry, step
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
)
from jkent.driver.unified_driver import ScrapeRun
from jkent.driver.unified_driver.wiring import RunConfig, RunHooks

if TYPE_CHECKING:
    from pathlib import Path

_PAYLOAD: dict[str, Any] = {
    "courtName": "STATE OF MONTANA SUPREME COURT",
    "page": 1,
    "filters": {"years": [2024, 2025], "sealed": False},
}


class _JsonPostScraper(BaseScraper[dict[str, Any]]):
    """One entry: POST a JSON body to the echo endpoint, emit what it saw."""

    base = "http://127.0.0.1"

    @entry(dict)
    def post_search(self) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.POST,
                url=f"{self.base}/echo-body/search",
                json=_PAYLOAD,
            ),
            step="parse_echo",
        )

    @step
    def parse_echo(
        self, json_content: dict[str, Any]
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data=json_content)


async def test_json_body_survives_db_and_dispatch(
    server_url: str, tmp_path: Path
) -> None:
    results: list[dict[str, Any]] = []

    async def on_data(data: Any) -> None:
        results.append(data)

    scraper = _JsonPostScraper()
    scraper.base = server_url
    run = ScrapeRun(
        scraper,
        tmp_path / "run.db",
        config=RunConfig(
            seed_params=[{"post_search": {}}], rate_limited=False
        ),
        hooks=RunHooks(on_data=on_data),
    )
    await run.open()
    try:
        await run.run()
        assert await run.status() == "done"
    finally:
        await run.aclose()

    assert len(results) == 1
    echoed = results[0]
    # httpx sent it as a JSON document, not a form encoding...
    assert echoed["content_type"].startswith("application/json")
    # ...and the body that crossed the wire is the exact payload the entry
    # yielded, after the full serialize → DB → dequeue → dispatch trip.
    assert json.loads(echoed["body"]) == _PAYLOAD
