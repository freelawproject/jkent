"""``HTTPRequestParams`` enforces its own annotations.

It is a *validating* dataclass, so the types it declares are the guarantee
rather than documentation. That matters because several of its fields are
written to ``requests`` as JSON text and read back on dequeue against those
same narrow types: a value that *writes* but will not *read* is a row no
worker can dequeue. The failure would land in ``get_next_request`` after
the row is claimed and the in-flight count raised, with the step that
produced it long gone, leaving the row ``in_progress`` and idle siblings
waiting on a request nobody is handling.

A static annotation alone would not have caught any of these: scrapers
build requests out of scraped values, which are typed ``Any``.

Every case below is a divergence the round-trip property in
``tests/contracts`` found against the real serializers (see
``request_row_round_trips``); this file pins each as a unit.
"""

from __future__ import annotations

import io
import ssl
from typing import Any

import pytest
from pydantic import ValidationError

from jkent.data_types import HttpMethod, HTTPRequestParams

_URL = "https://example.com/x"


def _params(**kwargs: Any) -> HTTPRequestParams:
    return HTTPRequestParams(HttpMethod.GET, _URL, **kwargs)


class TestHeadersAndCookies:
    """``headers_json``/``cookies_json`` read back as ``dict[str, str]``."""

    @pytest.mark.parametrize("field_name", ["headers", "cookies"])
    @pytest.mark.parametrize("value", [1, None, 1.5, ["a"], {"n": "v"}])
    def test_a_non_string_value_is_refused(self, field_name: str, value: Any):
        """JSON carries these happily; ``DequeuedRow`` then rejects them."""
        with pytest.raises(ValidationError, match=field_name):
            _params(**{field_name: {"X": value}})

    @pytest.mark.parametrize("field_name", ["headers", "cookies"])
    def test_a_non_mapping_is_refused(self, field_name: str):
        with pytest.raises(ValidationError, match=field_name):
            _params(**{field_name: [("X", "1")]})

    @pytest.mark.parametrize("field_name", ["headers", "cookies"])
    @pytest.mark.parametrize("value", [None, {}, {"X": "1"}])
    def test_string_mappings_and_none_are_accepted(
        self, field_name: str, value: Any
    ):
        assert getattr(_params(**{field_name: value}), field_name) == value


class TestTimeout:
    """``timeout_json`` reads back as a number or a ``(connect, read)`` pair."""

    @pytest.mark.parametrize(
        "value", [(1.0, 2.0, 3.0), (1.0,), {"connect": 1}, "abc"]
    )
    def test_a_shape_the_row_cannot_give_back_is_refused(self, value: Any):
        with pytest.raises(ValidationError, match="timeout"):
            _params(timeout=value)

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan"), (1.0, float("inf"))],
    )
    def test_a_non_finite_timeout_is_refused(self, value: Any):
        """``TimeoutType`` is ``FiniteFloat``-based.

        JSON has no spelling for these, so the column's ``json_valid``
        CHECK would fail the INSERT. ``None``, not infinity, is how a
        request says it has no timeout of its own.
        """
        with pytest.raises(ValidationError, match="timeout"):
            _params(timeout=value)

    @pytest.mark.parametrize("value", [None, 5, 5.0, (1.0, 2.0)])
    def test_numbers_and_pairs_are_accepted(self, value: Any):
        assert _params(timeout=value).timeout == value

    def test_a_numeric_string_is_coerced_rather_than_stored_as_text(self):
        """``"5"`` used to store as ``'"5"'`` and read back ``5.0``.

        The row stopped matching the request that produced it. Coercing at
        construction settles it the other way: the field and the row agree.
        """
        assert _params(timeout="5").timeout == 5.0


class TestFormBody:
    """A sequence body is stored as JSON and unpacked back into pairs."""

    def test_a_list_of_strings_is_refused(self):
        """The silent one: ``["ab"]`` came back as the field ``a=b``.

        The decoder unpacks each entry into ``(name, value)``, and a
        two-character string unpacks. Nothing raised anywhere — the request
        that came off the queue simply was not the one put on it.
        """
        with pytest.raises(ValidationError, match="data"):
            _params(data=["ab"])

    @pytest.mark.parametrize(
        "entry", [("a", "b", "c"), ("a",), (), ["a", "b", "c"]]
    )
    def test_a_non_pair_entry_is_refused(self, entry: Any):
        """These raised ``ValueError`` at dequeue, on an already-claimed row."""
        with pytest.raises(ValidationError, match="data"):
            _params(data=[entry])

    @pytest.mark.parametrize(
        "value",
        [
            None,
            b"raw",
            {"a": "b"},
            [("a", "b")],
            [("a", "b"), ("a", "c")],  # repeated keys are the point of a list
            [("a", 1)],  # a non-string value is fine: it stores as JSON
        ],
    )
    def test_real_bodies_are_accepted(self, value: Any):
        assert _params(data=value).data == value

    @pytest.mark.parametrize(
        "stream", [io.BytesIO(b"x"), io.BufferedReader(io.BytesIO(b"x"))]
    )
    def test_a_file_like_body_still_passes_through(self, stream: Any):
        """``encode_body`` reads these; validating must not shut them out."""
        assert _params(data=stream).data is stream

    def test_a_duck_typed_file_counts_as_one(self):
        """The arm tests for ``read``, which is what ``encode_body`` uses."""

        class Wrapper:
            def read(self) -> bytes:
                return b""

        wrapper = Wrapper()
        assert _params(data=wrapper).data is wrapper


class TestVerify:
    """``verify_json`` reads back as ``bool | str``."""

    def test_an_ssl_context_is_refused(self):
        """It used to serialize as its ``repr`` and read back as a CA path.

        ``to_json``'s ``str`` fallback turned the context into
        ``"<ssl.SSLContext object at 0x...>"``, which validates as a
        ``str`` — so the restored request asked for a CA bundle at a path
        that does not exist, rather than for the context's own trust store.
        """
        with pytest.raises(ValidationError, match="verify"):
            _params(verify=ssl.create_default_context())

    @pytest.mark.parametrize("value", [True, False, "/etc/ca.pem", "true"])
    def test_bools_and_paths_are_accepted(self, value: Any):
        assert _params(verify=value).verify == value
