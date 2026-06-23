"""``HTTPRequestParams.json`` survives serialize → DB → dequeue → dispatch.

The old persistent driver dropped ``json=`` (only ``data`` reached httpx),
which forced juriscraper's Montana scraper to hand-encode its JSON body with
a BOM prefix and send it via ``data=`` bytes. The unified driver propagates
``json`` end to end — ``queue.py`` stores it in the ``json_data`` column and
``httpx_transport`` passes ``json=`` through — and this pins the whole path
against a live server so the Montana workaround stays retired. (The
DB-only round-trip is covered in ``test_serialization.py``; this adds the
wire half: the body goes out as ``application/json``, not form-encoded.)
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

if TYPE_CHECKING:
    from pathlib import Path

_PAYLOAD: dict[str, Any] = {
    "courtName": "STATE OF MONTANA SUPREME COURT",
    "page": 1,
    "filters": {"years": [2024, 2025], "sealed": False},
}


class _JsonPostScraper(BaseScraper[dict]):
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
            continuation="parse_echo",
        )

    @step
    def parse_echo(
        self, json_content: dict
    ) -> Generator[ParsedData, None, None]:
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
        seed_params=[{"post_search": {}}],
        on_data=on_data,
        rate_limited=False,
        resume=False,
    )
    await run.open(setup_signal_handlers=False)
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
