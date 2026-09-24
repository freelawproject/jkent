"""``SpeculationManager`` resume against a real run database."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from typing import Any

from pydantic import BaseModel

from jkent.common.decorators import entry, step
from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
)
from jkent.driver.database_engine.sql_manager import (
    SpeculationStateInsert,
    SQLManager,
)
from jkent.driver.unified_driver.persistence import RequestQueue
from jkent.driver.unified_driver.speculation import SpeculationManager


class _Id(BaseModel, Speculative):
    n: int = 1
    should_advance: bool = True

    def seed_range(self) -> range:
        return range(self.n, self.n)

    def from_int(self, n: int) -> _Id:
        return _Id(n=n)

    def max_gap(self) -> int:
        return 2


class _TwoEntryScraper(BaseScraper[dict[str, Any]]):
    """Two speculative entries, so a seed can select one of them."""

    @entry(dict)
    def fetch_a(self, sid: _Id) -> Request:
        return self._probe("a", sid)

    @entry(dict)
    def fetch_b(self, sid: _Id) -> Request:
        return self._probe("b", sid)

    @staticmethod
    def _probe(kind: str, sid: _Id) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"http://127.0.0.1/{kind}/{sid.n}"
            ),
            step="parse",
        )

    @step
    def parse(
        self, response: Response
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={})


def _saved(func_name: str) -> SpeculationStateInsert:
    return SpeculationStateInsert(
        func_name=func_name,
        highest_successful_id=3,
        consecutive_failures=0,
        current_ceiling=5,
        stopped=False,
        param_index=0,
        template_json=_Id(n=1).model_dump_json(),
        seed_value_json=None,
    )


async def test_load_does_not_rebuild_an_entry_the_seed_leaves_out(
    sql_manager: SQLManager,
) -> None:
    """A resume narrowed by ``seed_params`` probes only what it selects.

    ``discover`` finds templates only for the entries the seed names;
    ``load`` must not add the others back by rebuilding their saved
    templates.
    """
    await sql_manager.save_speculation_state(_saved("fetch_a:0"))
    await sql_manager.save_speculation_state(_saved("fetch_b:0"))

    manager = SpeculationManager(
        _TwoEntryScraper(),
        RequestQueue(sql_manager),
        sql_manager,
        seed_params=[{"fetch_a": {"sid": {"n": 1}}}],
    )
    manager.discover()
    await manager.load()

    assert set(manager._speculation_state) == {"fetch_a:0"}


async def test_load_rebuilds_every_saved_entry_without_a_seed(
    sql_manager: SQLManager,
) -> None:
    """With no ``seed_params`` every saved state is resumed."""
    await sql_manager.save_speculation_state(_saved("fetch_a:0"))
    await sql_manager.save_speculation_state(_saved("fetch_b:0"))

    manager = SpeculationManager(
        _TwoEntryScraper(), RequestQueue(sql_manager), sql_manager
    )
    manager.discover()
    await manager.load()

    assert set(manager._speculation_state) == {"fetch_a:0", "fetch_b:0"}


async def test_persist_all_waits_for_the_speculation_lock(
    sql_manager: SQLManager,
) -> None:
    """The close-time flush can't interleave with an in-lock state update."""
    manager = SpeculationManager(
        _TwoEntryScraper(), RequestQueue(sql_manager), sql_manager
    )
    manager.discover()

    async with manager._speculation_lock:
        flush = asyncio.create_task(manager.persist_all())
        await asyncio.sleep(0)
        assert not flush.done()
    await flush
