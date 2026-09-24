"""Every ``@step`` property reaches a request the same way, however spelled.

A request names its target step either by string (``step="fetch_file"``)
or by Callable (``step=self.fetch_file``), and is yielded from a step, an
``@entry``, or built as a speculative probe. The target's ``StepMetadata``
must have the same observable effect in every one of those cells.

``StepMetadata`` fields come in two kinds. *Yield-time* fields are copied
onto the Request as it leaves the scraper (unset inherits, explicit wins);
*dispatch-time* fields are read by the driver from the step it resolves by
name, which is spelling-independent by construction. Every field must be
classified here, so a new one cannot be wired for one spelling only.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import pytest
from pydantic import BaseModel
from pyrate_limiter import Duration, Rate

from jkent.common.decorator_metadata import StepMetadata
from jkent.common.decorators import entry, step
from jkent.common.request import DEFAULT_TIMEOUT_S
from jkent.common.scraper import inherit_step_metadata
from jkent.common.speculative import Speculative
from jkent.data_types import (
    DEFAULT_PRIORITY,
    DEFAULT_RATE_LIMIT,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)
from jkent.driver._speculation_support import (
    SpeculationState,
    build_speculative_request,
)

#: Yield-time fields → the value the target step declares.
_PRIORITY, _LANE, _TIMEOUT = 2, "downloads", 7.5
YIELD_TIME: dict[str, object] = {
    "priority": _PRIORITY,
    "rate_limit": _LANE,
    "timeout": _TIMEOUT,
}
#: An explicit value on the request, spelled as the field's own default
#: where it has one, so an "is it the default?" check cannot pass for "is
#: it unset?".
EXPLICIT: dict[str, object] = {
    "priority": DEFAULT_PRIORITY,
    "rate_limit": DEFAULT_RATE_LIMIT,
    "timeout": DEFAULT_TIMEOUT_S,
}
#: Read by the driver from ``scraper.get_step(name)`` at dispatch.
DISPATCH_TIME = {"encoding", "await_list", "auto_await_timeout"}


def test_every_step_metadata_field_is_classified() -> None:
    assert set(vars(StepMetadata())) == set(YIELD_TIME) | DISPATCH_TIME


class _Probe(BaseModel, Speculative):
    n: int = 0
    should_advance: bool = False

    def seed_range(self) -> range:
        return range(1, 2)

    def from_int(self, n: int) -> _Probe:
        return self.model_copy(update={"n": n})

    def max_gap(self) -> int:
        return 0


def _request(target: str | Callable[..., Any], **fields: Any) -> Request:
    timeout = {"timeout": fields.pop("timeout")} if "timeout" in fields else {}
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/doc", **timeout
        ),
        step=target,
        **fields,
    )


def _field(request: Request, field: str) -> object:
    """``field`` on the request, or on its HTTP params for ``timeout``."""
    return getattr(request.request if field == "timeout" else request, field)


class _Scraper(BaseScraper[dict[str, Any]]):
    named_rate_limits = {"downloads": [Rate(1, Duration.SECOND)]}

    def __init__(self, spelling: str, **fields: Any) -> None:
        super().__init__()
        self.spelling = spelling
        self.fields = fields

    def _child(self) -> Request:
        target = "fetch_file" if self.spelling == "string" else self.fetch_file
        return _request(target, **self.fields)

    @entry(dict)
    def start(self) -> Generator[Request, None, None]:
        yield self._child()

    @entry(dict)
    def probe(self, pid: _Probe) -> Request:
        return self._child()

    @step
    def parse_list(
        self, response: Response
    ) -> Generator[ScraperYield[dict[str, Any]], None, None]:
        yield self._child()

    @step
    def parse_detail(
        self, response: Response
    ) -> Generator[ScraperYield[dict[str, Any]], None, None]:
        yield ParsedData({})

    @step(priority=_PRIORITY, rate_limit=_LANE, timeout=_TIMEOUT)
    def fetch_file(
        self, response: Response
    ) -> Generator[ScraperYield[dict[str, Any]], None, None]:
        yield ParsedData({})


def _from_step(scraper: _Scraper) -> Request:
    response = Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/list",
        request=_request("parse_list"),
    )
    (yielded,) = list(scraper.parse_list(response))
    assert isinstance(yielded, Request)
    return yielded


def _from_entry(scraper: _Scraper) -> Request:
    (yielded,) = list(scraper.initial_seed([{"start": {}}]))
    return yielded


def _from_probe(scraper: _Scraper) -> Request:
    state = SpeculationState(
        func_name="probe:0",
        template=_Probe(),
        param_index=0,
        base_func_name="probe",
        tracking_id=1,
    )
    return build_speculative_request(scraper, state, 1)


ORIGINS = {"step": _from_step, "entry": _from_entry, "probe": _from_probe}
_matrix = pytest.mark.parametrize(
    ("field", "spelling", "origin"),
    [
        (field, spelling, origin)
        for field in sorted(YIELD_TIME)
        for spelling in ("string", "callable")
        for origin in sorted(ORIGINS)
    ],
)


@_matrix
def test_unset_field_inherits_the_target_steps(
    field: str, spelling: str, origin: str
) -> None:
    request = ORIGINS[origin](_Scraper(spelling))
    assert request.step == "fetch_file"
    assert _field(request, field) == YIELD_TIME[field]


@_matrix
def test_explicit_field_wins(field: str, spelling: str, origin: str) -> None:
    scraper = _Scraper(spelling, **{field: EXPLICIT[field]})
    assert _field(ORIGINS[origin](scraper), field) == EXPLICIT[field]


def test_target_declaring_nothing_leaves_the_lane_unset() -> None:
    scraper = _Scraper("string")
    request = inherit_step_metadata(scraper, _request("parse_detail"))
    assert (request.priority, request.rate_limit) == (DEFAULT_PRIORITY, None)
    assert request.request.timeout is None


def test_unknown_step_name_is_left_for_dispatch_to_reject() -> None:
    request = inherit_step_metadata(_Scraper("string"), _request("nope"))
    assert (request.step, request.priority) == ("nope", None)
