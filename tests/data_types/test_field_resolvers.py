"""Async field-resolver resolution (``Request.resolve_deferred_fields``).

A ``FieldResolver`` — a zero-arg async callable — can stand in for a form
field value so a scraper can fetch it from an external service (e.g. an
image-captcha solver). The driver awaits resolvers when the yielded request
is enqueued, so only concrete values reach serialization and the transports.
"""

from __future__ import annotations

import pytest

from jkent.common.page_element import Form, FormField
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Selector,
    SkipDeduplicationCheck,
    ViaFormSubmit,
)
from jkent.driver.unified_driver.persistence import RequestQueue


def _search_form(method: str = "POST") -> Form:
    return Form(
        action="https://example.com/search",
        method=method,
        fields=[
            FormField(name="q", field_type="text", value=""),
            FormField(name="captcha", field_type="text", value=""),
        ],
        selector=Selector.XPath("//form[@id='search']"),
    )


def _counting_resolver(value: str | list[str]):
    """A resolver that records how many times it was awaited."""
    calls = {"count": 0}

    async def resolve() -> str | list[str]:
        calls["count"] += 1
        return value

    return resolve, calls


def test_form_submit_carries_resolver_in_both_payload_and_via():
    """The same resolver object lands in the HTTP payload and the via."""
    resolver, _ = _counting_resolver("solved")
    request = _search_form().submit(
        data={"q": "smith", "captcha": resolver}, continuation="parse"
    )

    assert isinstance(request.via, ViaFormSubmit)
    assert isinstance(request.request.data, dict)
    assert request.request.data["captcha"] is resolver
    assert request.via.field_data["captcha"] is resolver


async def test_resolve_substitutes_post_data_and_via():
    resolver, calls = _counting_resolver("solved")
    request = _search_form().submit(
        data={"q": "smith", "captcha": resolver}, continuation="parse"
    )

    resolved = await request.resolve_deferred_fields()

    assert isinstance(resolved.via, ViaFormSubmit)
    assert resolved.request.data == {"q": "smith", "captcha": "solved"}
    assert resolved.via.field_data == {"q": "smith", "captcha": "solved"}
    # One external call even though the resolver appears in two places.
    assert calls["count"] == 1


async def test_resolve_substitutes_get_params():
    resolver, _ = _counting_resolver("solved")
    request = _search_form(method="GET").submit(
        data={"captcha": resolver}, continuation="parse"
    )

    resolved = await request.resolve_deferred_fields()

    assert resolved.request.params == {"q": "", "captcha": "solved"}
    assert isinstance(resolved.via, ViaFormSubmit)
    assert resolved.via.field_data == {"q": "", "captcha": "solved"}


async def test_resolver_may_return_a_list():
    resolver, _ = _counting_resolver(["a", "b"])
    request = _search_form().submit(
        data={"captcha": resolver}, continuation="parse"
    )

    resolved = await request.resolve_deferred_fields()

    assert isinstance(resolved.request.data, dict)
    assert resolved.request.data["captcha"] == ["a", "b"]


async def test_resolve_works_without_via():
    """A hand-built request (no form) can carry resolvers in data."""

    async def resolve() -> str:
        return "token"

    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/api",
            method=HttpMethod.POST,
            data={"token": resolve},
        ),
        continuation="parse",
    )

    resolved = await request.resolve_deferred_fields()

    assert resolved.request.data == {"token": "token"}
    assert resolved.via is None


async def test_no_resolvers_returns_self():
    request = _search_form().submit(data={"q": "smith"}, continuation="parse")
    assert await request.resolve_deferred_fields() is request


async def test_auto_dedup_key_regenerated_from_resolved_values():
    """The auto key must hash the resolved values, not the resolver's repr."""
    resolver, _ = _counting_resolver("solved")
    request = _search_form().submit(
        data={"q": "smith", "captcha": resolver}, continuation="parse"
    )
    concrete = _search_form().submit(
        data={"q": "smith", "captcha": "solved"}, continuation="parse"
    )

    resolved = await request.resolve_deferred_fields()

    assert resolved.deduplication_key != request.deduplication_key
    assert resolved.deduplication_key == concrete.deduplication_key


async def test_explicit_dedup_key_preserved():
    resolver, _ = _counting_resolver("solved")
    request = _search_form().submit(
        data={"captcha": resolver},
        deduplication_key="my-key",
        continuation="parse",
    )

    resolved = await request.resolve_deferred_fields()

    assert resolved.deduplication_key == "my-key"


async def test_skip_dedup_sentinel_preserved():
    resolver, _ = _counting_resolver("solved")
    request = _search_form().submit(
        data={"captcha": resolver},
        deduplication_key=SkipDeduplicationCheck(),
        continuation="parse",
    )

    resolved = await request.resolve_deferred_fields()

    assert isinstance(resolved.deduplication_key, SkipDeduplicationCheck)


async def test_sync_callable_rejected():
    request = _search_form().submit(
        data={"captcha": lambda: "solved"},  # type: ignore[dict-item]
        continuation="parse",
    )

    with pytest.raises(TypeError, match="async"):
        await request.resolve_deferred_fields()


async def test_wrong_return_type_rejected():
    async def resolve() -> int:
        return 42

    request = _search_form().submit(
        data={"captcha": resolve},  # type: ignore[dict-item]
        continuation="parse",
    )

    with pytest.raises(TypeError, match=r"str \| list\[str\]"):
        await request.resolve_deferred_fields()


async def test_resolver_exception_propagates():
    async def resolve() -> str:
        raise RuntimeError("captcha service down")

    request = _search_form().submit(
        data={"captcha": resolve}, continuation="parse"
    )

    with pytest.raises(RuntimeError, match="captcha service down"):
        await request.resolve_deferred_fields()


async def test_prepare_enqueue_serializes_resolved_values():
    """The enqueue path resolves before serialization: the row carries the
    concrete value in both the body and via_json."""
    resolver, calls = _counting_resolver("solved")
    request = _search_form().submit(
        data={"q": "smith", "captcha": resolver}, continuation="parse"
    )
    parent = Request(
        request=HTTPRequestParams(
            url="https://example.com/", method=HttpMethod.GET
        ),
        continuation="parse",
        current_location="https://example.com/",
    )

    queue = RequestQueue.__new__(RequestQueue)  # no DB needed with parent_id
    request_data, dedup_key, _, _ = await queue._prepare_enqueue(
        request, parent, parent_request_id=1
    )

    assert calls["count"] == 1
    assert b"solved" in request_data["body"]
    assert '"captcha": "solved"' in request_data["via_json"]
    assert "resolve" not in request_data["via_json"]
    assert isinstance(dedup_key, str) and len(dedup_key) == 64
