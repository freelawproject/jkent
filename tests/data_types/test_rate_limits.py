"""Rate-limit lanes: the scraper-side vocabulary and its storage codes.

Inheritance from ``@step(rate_limit=...)`` is under test in
``tests/decorators/test_step_inheritance.py``.

A scraper declares lanes by name (``named_rate_limits``); requests and
steps select one by name; the run database stores the lane as an integer
index into the lane list the run recorded at start. Under test:

- ``RateLimitTable`` codes: ``default`` 0, ``none`` 1, declared lanes 2+ in
  order; unknown names fail to encode and out-of-range codes fail to decode,
  both as ``ScraperConfigError`` — naming a lane the scraper never declared
  is a config mistake, and no retry fixes it.
- ``stored_names`` (the lane list a run recorded) fixes those codes across a
  reorder, appends lanes declared since, and refuses a lane that is gone.
- ``BaseScraper`` rejects a bad ``named_rate_limits`` at class definition.
- ``CappedPace`` and ``ReactivePace`` refuse gaps no lane could keep, and
  survive a trip through JSON as the kind they were.
"""

from collections.abc import Generator
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from jkent.common.decorators import step
from jkent.common.exceptions import ScraperConfigError
from jkent.data_types import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    AdaptivePace,
    BaseScraper,
    CappedPace,
    HttpMethod,
    HTTPRequestParams,
    Pace,
    RateLimitTable,
    ReactivePace,
    Request,
    Response,
)


def make_request(**kwargs: Any) -> Request:
    kwargs.setdefault("step", "parse")
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/doc"
        ),
        **kwargs,
    )


class LanedScraper(BaseScraper[dict[str, Any]]):
    rate_limits = CappedPace(seconds=0.5)
    named_rate_limits = {
        "downloads": ReactivePace(seconds=5),
        "search": CappedPace(seconds=1),
    }


class BareScraper(BaseScraper[dict[str, Any]]):
    """Declares nothing: the common case must cost nothing."""


# --- RateLimitTable ---------------------------------------------------------


class TestRateLimitTable:
    def test_codes_follow_declaration_order_after_the_reserved_pair(self):
        table = RateLimitTable.for_scraper(LanedScraper)
        assert table.names == ("default", "none", "downloads", "search")

    def test_paces_per_lane(self):
        table = RateLimitTable.for_scraper(LanedScraper)
        assert table.paces == {
            DEFAULT_RATE_LIMIT: CappedPace(seconds=0.5),
            NO_RATE_LIMIT: None,
            "downloads": ReactivePace(seconds=5),
            "search": CappedPace(seconds=1),
        }

    def test_bare_scraper_has_only_the_reserved_lanes(self):
        table = RateLimitTable.for_scraper(BareScraper)
        assert table.names == ("default", "none")
        assert table.paces[DEFAULT_RATE_LIMIT] is None

    def test_instance_and_class_build_the_same_table(self):
        assert RateLimitTable.for_scraper(
            LanedScraper()
        ) == RateLimitTable.for_scraper(LanedScraper)

    def test_bare_matches_a_scraper_that_declares_nothing(self):
        assert RateLimitTable.bare() == RateLimitTable.for_scraper(BareScraper)

    def test_encode(self):
        table = RateLimitTable.for_scraper(LanedScraper)
        assert table.encode(None) == 0
        assert table.encode(DEFAULT_RATE_LIMIT) == 0
        assert table.encode(NO_RATE_LIMIT) == 1
        assert table.encode("downloads") == 2
        assert table.encode("search") == 3

    def test_decode_default_is_the_unset_value(self):
        table = RateLimitTable.for_scraper(LanedScraper)
        assert table.decode(0) is None
        assert table.decode(1) == NO_RATE_LIMIT
        assert table.decode(2) == "downloads"
        assert table.decode(3) == "search"

    def test_round_trip(self):
        table = RateLimitTable.for_scraper(LanedScraper)
        for name in (None, NO_RATE_LIMIT, "downloads", "search"):
            assert table.decode(table.encode(name)) == name

    def test_encode_unknown_name_names_the_lanes(self):
        table = RateLimitTable.for_scraper(BareScraper)
        with pytest.raises(
            ScraperConfigError, match="'downloads'.*default.*none"
        ):
            table.encode("downloads")

    def test_decode_past_the_table_is_loud(self):
        # A code from a run whose stored lanes are not this table's: never
        # silently the default rate.
        table = RateLimitTable.for_scraper(BareScraper)
        with pytest.raises(ScraperConfigError, match="has no lane"):
            table.decode(2)
        with pytest.raises(ScraperConfigError):
            table.decode(-1)


class TestStoredLaneNames:
    """Codes come from the run's stored lane list, not the class body."""

    def test_reordering_the_class_body_does_not_move_a_lane(self):
        class Reordered(BaseScraper[dict[str, Any]]):
            rate_limits = CappedPace(seconds=0.5)
            named_rate_limits = {
                "search": CappedPace(seconds=1),
                "downloads": CappedPace(seconds=1),
            }

        stored = RateLimitTable.for_scraper(LanedScraper).names
        table = RateLimitTable.for_scraper(Reordered, stored_names=stored)
        assert table.names == stored
        assert table.encode("downloads") == 2
        # The paces still come from the scraper, only the order is stored.
        assert table.paces["search"] == Reordered.named_rate_limits["search"]

    def test_a_lane_added_since_the_run_started_is_appended(self):
        stored = RateLimitTable.for_scraper(BareScraper).names
        table = RateLimitTable.for_scraper(LanedScraper, stored_names=stored)
        assert table.names == ("default", "none", "downloads", "search")

    def test_a_dropped_lane_is_refused_naming_both_lists(self):
        stored = RateLimitTable.for_scraper(LanedScraper).names
        with pytest.raises(ScraperConfigError) as excinfo:
            RateLimitTable.for_scraper(BareScraper, stored_names=stored)
        message = str(excinfo.value)
        assert "downloads" in message and "search" in message

    def test_stored_names_must_start_with_the_reserved_pair(self):
        with pytest.raises(ScraperConfigError, match="jkent wrote"):
            RateLimitTable.for_scraper(
                LanedScraper, stored_names=("downloads", "search")
            )

    def test_stored_names_matching_the_class_body_change_nothing(self):
        stored = RateLimitTable.for_scraper(LanedScraper).names
        assert (
            RateLimitTable.for_scraper(LanedScraper, stored_names=stored).names
            == stored
        )


# --- BaseScraper validation at class definition ---------------------------


class TestNamedRateLimitsValidation:
    def test_reserved_name_is_rejected(self):
        with pytest.raises(ScraperConfigError, match="reserved"):

            class Bad(BaseScraper[dict[str, Any]]):
                named_rate_limits = {"none": CappedPace(seconds=1)}

    @pytest.mark.parametrize("name", ["", 3])
    def test_a_name_that_is_not_a_non_empty_string_is_rejected(
        self, name: Any
    ):
        with pytest.raises(
            ScraperConfigError, match=rf"named_rate_limits\[{name!r}\]"
        ):

            class Bad(BaseScraper[dict[str, Any]]):
                named_rate_limits = {name: CappedPace(seconds=1)}

    @pytest.mark.parametrize(
        "not_a_pace",
        [0.5, [CappedPace(seconds=1)], None, {"kind": "capped", "seconds": 1}],
    )
    def test_a_lane_that_is_not_a_pace_is_rejected(self, not_a_pace: Any):
        with pytest.raises(
            ScraperConfigError, match=r"named_rate_limits\['downloads'\]"
        ):

            class Bad(BaseScraper[dict[str, Any]]):
                named_rate_limits = {"downloads": not_a_pace}

    @pytest.mark.parametrize(
        "not_a_pace",
        [0.5, [CappedPace(seconds=1)], {"kind": "capped", "seconds": 1}],
    )
    def test_a_default_that_is_not_a_pace_is_rejected(self, not_a_pace: Any):
        with pytest.raises(ScraperConfigError, match=r"Bad\.rate_limits"):

            class Bad(BaseScraper[dict[str, Any]]):
                rate_limits = not_a_pace

    def test_table_revalidates_a_class_built_at_runtime(self):
        # Not written in a module, so __init_subclass__ never saw it.
        built = type("Built", (), {"rate_limits": 2, "named_rate_limits": {}})

        with pytest.raises(ScraperConfigError, match="CappedPace"):
            RateLimitTable.for_scraper(cast("type[BaseScraper[Any]]", built))

    def test_step_naming_an_undeclared_lane_is_rejected(self):
        # Was caught only at the first enqueue of a request to the step.
        with pytest.raises(ScraperConfigError, match=r"Bad\.parse.*'nope'"):

            class Bad(BaseScraper[dict[str, Any]]):
                @step(rate_limit="nope")
                def parse(
                    self, response: Response
                ) -> Generator[Request, None, None]:
                    yield from ()

    def test_step_naming_a_declared_or_reserved_lane_is_accepted(self):
        class Good(BaseScraper[dict[str, Any]]):
            named_rate_limits = {"downloads": CappedPace(seconds=1)}

            @step(rate_limit="downloads")
            def fetch(
                self, response: Response
            ) -> Generator[Request, None, None]:
                yield from ()

            @step(rate_limit=NO_RATE_LIMIT)
            def presigned(
                self, response: Response
            ) -> Generator[Request, None, None]:
                yield from ()

        class Child(Good):
            pass

    def test_subclass_dropping_an_inherited_steps_lane_is_rejected(self):
        class Parent(BaseScraper[dict[str, Any]]):
            named_rate_limits = {"downloads": CappedPace(seconds=1)}

            @step(rate_limit="downloads")
            def fetch(
                self, response: Response
            ) -> Generator[Request, None, None]:
                yield from ()

        with pytest.raises(ScraperConfigError, match="'downloads'"):

            class Child(Parent):
                named_rate_limits = {"search": CappedPace(seconds=1)}

    def test_error_names_the_scraper(self):
        with pytest.raises(ScraperConfigError, match="Named"):

            class Named(BaseScraper[dict[str, Any]]):
                named_rate_limits = {"default": CappedPace(seconds=1)}


# --- Request.rate_limit ---------------------------------------------------


class TestRequestRateLimit:
    def test_unset_is_none(self):
        assert make_request().rate_limit is None

    def test_explicit_is_kept(self):
        assert make_request(rate_limit="downloads").rate_limit == "downloads"

    def test_survives_resolve_from(self):
        parent = make_request(step="seed")
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="https://example.com/list",
            request=parent,
        )
        resolved = make_request(rate_limit=NO_RATE_LIMIT).resolve_from(
            response
        )
        assert resolved.rate_limit == NO_RATE_LIMIT


# --- The pace types ---------------------------------------------------------


class TestPaces:
    @pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
    def test_capped_pace_needs_a_positive_finite_gap(self, seconds: float):
        with pytest.raises(ValidationError):
            CappedPace(seconds=seconds)

    def test_capped_pace_takes_a_gap_under_a_second(self):
        assert CappedPace(seconds=1 / 120).seconds == 1 / 120

    @pytest.mark.parametrize("seconds", [-1, float("inf"), float("nan")])
    def test_reactive_pace_needs_a_finite_rest(self, seconds: float):
        with pytest.raises(ValidationError):
            ReactivePace(seconds=seconds)

    @pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
    def test_adaptive_pace_needs_a_positive_finite_gap(self, seconds: float):
        with pytest.raises(ValidationError):
            AdaptivePace(seconds=seconds)

    def test_reactive_pace_may_rest_zero(self):
        assert ReactivePace(seconds=0).seconds == 0

    def test_a_pace_is_immutable_and_takes_no_stray_field(self):
        pace = CappedPace(seconds=1)
        with pytest.raises(ValidationError):
            pace.seconds = 2  # type: ignore[misc]
        with pytest.raises(ValidationError):
            CappedPace(seconds=1, burst=3)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "pace",
        [
            CappedPace(seconds=0.25),
            ReactivePace(seconds=5),
            AdaptivePace(seconds=0.25),
        ],
    )
    def test_a_pace_round_trips_through_json_as_its_own_kind(self, pace: Pace):
        adapter: TypeAdapter[Pace] = TypeAdapter(Pace)
        restored = adapter.validate_json(adapter.dump_json(pace))
        assert restored == pace
        assert type(restored) is type(pace)
