"""Response storage once a step has a trained compression dictionary."""

from __future__ import annotations

from typing import TYPE_CHECKING

import zstandard as zstd

import jkent.driver.database_engine.compression as comp
from jkent.data_types import HttpMethod, HTTPRequestParams, Request, Response
from jkent.driver.database_engine.storage import ResponseStorageDB
from tests.driver.database_engine.test_compression_errors import (
    _HTML,
    _insert_responses,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from jkent.driver.database_engine.sql_manager import SQLManager


async def test_store_response_uses_the_steps_dictionary(
    sql_manager: SQLManager, insert_request: Callable[..., Awaitable[int]]
) -> None:
    """A response stored after training compresses against the step's dict
    and reads back byte-identical through the pre-resolved loader."""
    await _insert_responses(sql_manager, comp, "parse", 30)
    dict_id = await comp.train_compression_dict(sql_manager, "parse")
    request_id = await insert_request(url="https://comp.test/new")
    request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://comp.test/new"
        ),
        step="parse",
    )
    body = _HTML.replace(b"{n}", b"fresh")
    storage = ResponseStorageDB(sql_manager)
    await storage.store_response(
        request_id,
        Response(
            status_code=200,
            headers={},
            content=body,
            url="https://comp.test/new",
            request=request,
        ),
        "parse",
    )

    stored = await sql_manager.get_stored_response(request_id)
    assert stored is not None
    assert stored.compression_dict_id == dict_id
    # The frame itself is bound to the dictionary, not merely labelled.
    dictionary = await comp.get_dict_by_id(sql_manager, dict_id)
    assert stored.content_compressed is not None
    assert dictionary is not None
    frame = zstd.get_frame_parameters(stored.content_compressed)
    assert frame.dict_id == dictionary.dict_id() != 0
    loaded = await storage.load_preresolved_response(request_id, request)
    assert loaded is not None and loaded.content == body
