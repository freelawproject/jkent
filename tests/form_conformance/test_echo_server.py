"""The conformance rig's ``/echo`` route reports what the server received."""

from __future__ import annotations

from typing import Any

from aiohttp.test_utils import TestClient, TestServer

from tests.form_conformance.echo_server import (
    Canonical,
    create_app,
    extract_echo,
)


async def _echo(method: str, **kwargs: Any) -> Canonical:
    async with TestClient(TestServer(create_app())) as client:
        resp = await client.request(method, "/echo", **kwargs)
        return extract_echo(await resp.text())


async def test_urlencoded_post_echoes_its_content_type_and_pairs() -> None:
    assert await _echo("POST", data={"ct": "civil"}) == (
        "POST",
        "application/x-www-form-urlencoded",
        (("ct", "civil"),),
    )


async def test_json_post_is_not_read_as_form_pairs() -> None:
    # parse_qsl('{"ct": "civil"}') yields one plausible-looking pair; a
    # transport that sent JSON must diverge on the content type instead.
    assert await _echo("POST", json={"ct": "civil"}) == (
        "POST",
        "application/json",
        (),
    )


async def test_get_echoes_the_query_with_no_content_type() -> None:
    assert await _echo("GET", params={"q": "x"}) == (
        "GET",
        "",
        (("q", "x"),),
    )
