"""``IncidentalMatch.matches``: the allowlist each set field applies.

Only the fields a spec names are checked; everything else on the captured
request is ignored.
"""

from __future__ import annotations

from typing import Any

import pytest

from jkent.common.incidental import IncidentalMatch, Singular

_URL = "https://court.example/api/search?case=42&page=2"


def _matches(
    spec: IncidentalMatch,
    *,
    url: str = _URL,
    method: str | None = "POST",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    resource_type: str | None = "fetch",
) -> bool:
    return spec.matches(
        url=url,
        method=method,
        headers=headers or {},
        body=body,
        resource_type=resource_type,
    )


@pytest.mark.parametrize(
    ("spec", "captured", "expected"),
    [
        # url glob
        (Singular(url="*/api/search*"), {}, True),
        (Singular(url="*/api/other*"), {}, False),
        # method, case-insensitive either side
        (Singular(method="post"), {"method": "POST"}, True),
        (Singular(method="POST"), {"method": "post"}, True),
        (Singular(method="GET"), {"method": "POST"}, False),
        (Singular(method="GET"), {"method": None}, False),
        # resource_type, case-insensitive
        (Singular(resource_type="XHR"), {"resource_type": "xhr"}, True),
        (Singular(resource_type="fetch"), {"resource_type": "Fetch"}, True),
        (Singular(resource_type="fetch"), {"resource_type": "xhr"}, False),
        (Singular(resource_type="fetch"), {"resource_type": None}, False),
        # query_contains: every named param present with that value
        (Singular(query_contains={"case": "42"}), {}, True),
        (Singular(query_contains={"case": "42", "page": "2"}), {}, True),
        (Singular(query_contains={"case": "43"}), {}, False),
        (Singular(query_contains={"missing": "1"}), {}, False),
        # header_contains: names case-insensitive, values exact
        (
            Singular(header_contains={"X-Token": "abc"}),
            {"headers": {"x-token": "abc", "accept": "*/*"}},
            True,
        ),
        (
            Singular(header_contains={"x-token": "abc"}),
            {"headers": {"X-TOKEN": "abc"}},
            True,
        ),
        (
            Singular(header_contains={"x-token": "abc"}),
            {"headers": {"x-token": "ABC"}},
            False,
        ),
        (
            Singular(header_contains={"x-token": "abc"}),
            {"headers": {}},
            False,
        ),
        # every set field must hold
        (
            Singular(method="POST", resource_type="xhr"),
            {"method": "POST", "resource_type": "fetch"},
            False,
        ),
    ],
)
def test_field_matching(
    spec: IncidentalMatch, captured: dict[str, Any], expected: bool
) -> None:
    assert _matches(spec, **captured) is expected


@pytest.mark.parametrize(
    ("contains", "body", "expected"),
    [
        # str: substring of the decoded body
        ("GetCase", b'{"operationName":"GetCase"}', True),
        ("GetDocket", b'{"operationName":"GetCase"}', False),
        # dict as JSON: extra keys ignored
        ({"op": "a"}, b'{"op": "a", "nonce": "x"}', True),
        ({"op": "a"}, b'{"op": "b"}', False),
        ({"op": "a", "missing": 1}, b'{"op": "a"}', False),
        # nested dicts
        (
            {"vars": {"id": 7}},
            b'{"vars": {"id": 7, "token": "t"}, "op": "a"}',
            True,
        ),
        ({"vars": {"id": 7}}, b'{"vars": {"id": 8}}', False),
        ({"vars": {"id": 7}}, b'{"vars": 7}', False),
        # lists: each expected item deep-contained by some actual item
        ({"ids": [2, 1]}, b'{"ids": [1, 2, 3]}', True),
        ({"ids": [4]}, b'{"ids": [1, 2, 3]}', False),
        (
            {"rows": [{"k": "b"}]},
            b'{"rows": [{"k": "a", "v": 1}, {"k": "b", "v": 2}]}',
            True,
        ),
        ({"ids": [1]}, b'{"ids": 1}', False),
        # dict falls back to form-encoded params
        ({"case": "42"}, b"case=42&token=zzz", True),
        ({"case": "42"}, b"case=41&token=zzz", False),
        ({"case": "42"}, b"<html>case 42</html>", False),
        # a JSON body that doesn't deep-contain also gets the form fallback
        ({"case": "42"}, b'{"case": 41}', False),
        # no body at all never matches
        ({"case": "42"}, None, False),
        ("anything", None, False),
        ("", None, False),
        ({}, None, False),
    ],
)
def test_body_contains(
    contains: dict[str, Any] | str, body: bytes | None, expected: bool
) -> None:
    spec = Singular(url="*", body_contains=contains)
    assert _matches(spec, body=body) is expected


def test_empty_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one match field"):
        IncidentalMatch()


def test_empty_containers_count_as_unset() -> None:
    with pytest.raises(ValueError, match="at least one match field"):
        IncidentalMatch(query_contains={}, header_contains={}, url="")
