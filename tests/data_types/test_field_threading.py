"""Field-threading regression guard for Request copy methods.

Request.resolve_from() and Request.speculative() rebuild the Request by
listing every field by hand. When a new field is added to Request (or
Request) and not threaded through those constructors, it silently
resets to its default mid-chain — no error, just lost state.

These tests are driven by dataclasses.fields(Request): adding a field
without registering a non-default test value here fails loudly, and a
copy method that drops the field fails the survival comparison.
"""

import dataclasses
from typing import Any

from jkent.common.page_element import ViaLink
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    Selector,
    Singular,
)

# One non-default value per Request field. Fields the copy methods
# intentionally change are still listed (used to build the request) but
# excluded from the survival comparison per-method below.
FIELD_VALUES: dict[str, Any] = {
    "request": HTTPRequestParams(
        method=HttpMethod.POST,
        url="https://example.com/search",
        params={"page": "2"},
        data={"q": "bees"},
        headers={"X-Custom": "1"},
        cookies={"session": "abc"},
        timeout=30,
        allow_redirects=False,
        verify=False,
        stream=True,
    ),
    "continuation": "parse_detail",
    "current_location": "https://example.com/list",
    "parent_request": None,
    "accumulated_data": {"case_name": "Ant v. Bee"},
    "priority": 4,
    "deduplication_key": "explicit-key",
    "permanent": {"headers": {"Authorization": "Bearer token"}},
    "is_speculative": True,
    "speculation_id": ("by_id", 0, 7),
    "via": ViaLink(
        selector=Selector.XPath("//a[1]"), description="link: Case"
    ),
    "incidental": Singular(url="*casedetaildata*"),
    "bypass_rate_limit": True,
    "reseedable": True,
    "nonnavigating": True,
    "archive": True,
    "expected_type": "pdf",
    "archive_hash_header": "ETag",
}


def make_full_request() -> Request:
    return Request(**FIELD_VALUES)


def make_parent_response() -> Response:
    parent = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/list"
        ),
        continuation="seed",
        current_location="https://example.com/list",
    )
    return Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/list",
        request=parent,
    )


def assert_fields_survive(
    original: Request, copy: Request, intentionally_changed: set[str]
) -> None:
    for f in dataclasses.fields(Request):
        assert f.name in FIELD_VALUES, (
            f"New Request field {f.name!r} has no test value in "
            f"FIELD_VALUES — add one so resolve_from()/speculative() "
            f"threading stays covered."
        )
        if f.name in intentionally_changed:
            continue
        assert getattr(copy, f.name) == getattr(original, f.name), (
            f"Request.{f.name} did not survive the copy — was it "
            f"threaded through the constructor call?"
        )


def test_every_field_survives_resolve_from():
    original = make_full_request()
    resolved = original.resolve_from(make_parent_response())

    assert_fields_survive(
        original,
        resolved,
        intentionally_changed={
            "request",  # URL resolved against the parent
            "current_location",  # taken from the parent
            "parent_request",  # set to the immediate parent
        },
    )
    # The intentionally-changed fields changed the way they should.
    assert resolved.request.url == "https://example.com/search"
    assert resolved.request.method == HttpMethod.POST
    assert resolved.request.data == {"q": "bees"}
    assert resolved.current_location == "https://example.com/list"
    assert resolved.parent_request is not None
    assert resolved.parent_request.continuation == "seed"


def test_every_field_survives_speculative():
    original = make_full_request()
    spec = original.speculative("by_year", 1, 2024)

    assert_fields_survive(
        original,
        spec,
        intentionally_changed={"is_speculative", "speculation_id"},
    )
    assert spec.is_speculative is True
    assert spec.speculation_id == ("by_year", 1, 2024)
