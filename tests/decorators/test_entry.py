"""Tests for the @entry decorator and related functionality.

This test module verifies:
- The @entry decorator attaches EntryMetadata correctly
- The get_entry_metadata() helper works
- BaseScraper.list_entries() discovers decorated methods
- EntryMetadata.validate_params() handles BaseModel and primitive types
- BaseScraper.initial_seed() dispatches correctly
- BaseScraper.schema() generates correct JSON Schema
- Speculative entries are auto-detected via the Speculative protocol
- Error cases (empty params, unknown entry, bad types)
"""

import json
from collections.abc import Generator
from datetime import date

import pytest
from pydantic import BaseModel, ValidationError

from jkent.common.decorator_metadata import (
    EntryMetadata,
    get_entry_metadata,
)
from jkent.common.decorators import entry
from jkent.common.param_models import DateRange
from jkent.common.speculative import Speculative
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    Request,
)

# ── Test fixtures ──────────────────────────────────────────────────


class FakeData(BaseModel):
    name: str


class OpinionFilters(BaseModel):
    court_id: str
    year: int


class RecordId(BaseModel, Speculative):
    """Speculative parameter model for testing."""

    record_id: int
    soft_max: int = 0
    should_advance: bool = True
    gap: int = 20

    def seed_range(self) -> range:
        return range(self.record_id, self.soft_max)

    def from_int(self, n: int) -> "RecordId":
        return RecordId(
            record_id=n,
            soft_max=self.soft_max,
            should_advance=self.should_advance,
            gap=self.gap,
        )

    def max_gap(self) -> int:
        return self.gap


class AnotherSpecParam(BaseModel, Speculative):
    """Second Speculative model; entries may declare at most one."""

    x: int = 1
    should_advance: bool = True

    def seed_range(self) -> range:
        return range(self.x, self.x)

    def from_int(self, n: int) -> "AnotherSpecParam":
        return AnotherSpecParam(x=n, should_advance=self.should_advance)

    def max_gap(self) -> int:
        return 5


class SimpleScraper(BaseScraper[FakeData]):
    @entry(FakeData)
    def search_by_name(self, name: str) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"/search?name={name}"
            ),
            continuation="parse_results",
        )

    @entry(FakeData)
    def search_by_date(
        self, date_range: DateRange
    ) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/search?start={date_range.start}&end={date_range.end}",
            ),
            continuation="parse_results",
        )

    @entry(FakeData)
    def fetch_by_id(self, rid: RecordId) -> Request:
        return Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET,
                url=f"/record/{rid.record_id}",
            ),
            continuation="parse_detail",
        )


class MultiTypeScraper(BaseScraper[FakeData]):
    @entry(FakeData)
    def search_opinions(
        self, filters: OpinionFilters
    ) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url="/opinions"),
            continuation="parse_opinions",
        )

    @entry(FakeData)
    def search_by_filing_date(
        self, filing_date: date
    ) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(method=HttpMethod.GET, url="/filings"),
            continuation="parse_filings",
        )

    @entry(FakeData)
    def search_by_count(self, count: int) -> Generator[Request, None, None]:
        yield Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url=f"/search?count={count}"
            ),
            continuation="parse_results",
        )


# ── @entry metadata attachment ─────────────────────────────────────


class TestEntryMetadata:
    def test_entry_attaches_metadata(self):
        meta = get_entry_metadata(SimpleScraper.search_by_name)
        assert meta is not None
        assert meta is not None
        assert meta.return_type is FakeData
        assert meta.func_name == "search_by_name"
        assert meta.param_types == {"name": str}
        assert meta.speculative is False

    def test_entry_with_basemodel_param(self):
        meta = get_entry_metadata(SimpleScraper.search_by_date)
        assert meta is not None
        assert meta is not None
        assert meta.param_types == {"date_range": DateRange}

    def test_speculative_entry_metadata(self):
        meta = get_entry_metadata(SimpleScraper.fetch_by_id)
        assert meta is not None
        assert meta is not None
        assert meta.speculative is True
        assert meta.speculative_param == "rid"
        assert meta.param_types == {"rid": RecordId}

    def test_date_param_type(self):
        meta = get_entry_metadata(MultiTypeScraper.search_by_filing_date)
        assert meta is not None
        assert meta is not None
        assert meta.param_types == {"filing_date": date}

    def test_complex_basemodel_param(self):
        meta = get_entry_metadata(MultiTypeScraper.search_opinions)
        assert meta is not None
        assert meta.param_types == {"filters": OpinionFilters}

    def test_speculative_subclass_detected(self):
        """Models subclassing the Speculative ABC are detected; others aren't."""
        assert issubclass(RecordId, Speculative)
        assert not issubclass(OpinionFilters, Speculative)

    def test_multiple_speculative_params_rejected(self):
        """Only one Speculative param per entry is allowed."""
        with pytest.raises(TypeError, match="multiple Speculative parameters"):

            class BadScraper(BaseScraper[FakeData]):
                @entry(FakeData)
                def bad(self, a: RecordId, b: AnotherSpecParam) -> Request:  # type: ignore[empty-body]
                    ...


# ── get_entry_metadata helper ──────────────────────────────────────


class TestEntryHelpers:
    def test_get_entry_metadata_set_for_entry(self):
        assert get_entry_metadata(SimpleScraper.search_by_name) is not None

    def test_get_entry_metadata_returns_none_for_non_entry(self):
        assert get_entry_metadata(SimpleScraper.actually_successful) is None


# ── list_entries() ─────────────────────────────────────────────────


class TestListEntries:
    def test_discovers_all_entries(self):
        entries = SimpleScraper.list_entries()
        names = {e.func_name for e in entries}
        assert names == {"search_by_name", "search_by_date", "fetch_by_id"}

    def test_entry_info_fields(self):
        entries = SimpleScraper.list_entries()
        by_name = {e.func_name: e for e in entries}

        info = by_name["search_by_name"]
        assert isinstance(info, EntryMetadata)
        assert info.return_type is FakeData
        assert info.param_types == {"name": str}
        assert info.speculative is False

    def test_speculative_entry_in_list(self):
        entries = SimpleScraper.list_entries()
        by_name = {e.func_name: e for e in entries}

        info = by_name["fetch_by_id"]
        assert info.speculative is True
        assert info.speculative_param == "rid"

    def test_list_speculative_entries(self):
        spec = SimpleScraper.list_speculative_entries()
        assert len(spec) == 1
        assert spec[0].func_name == "fetch_by_id"


# ── validate_params() ─────────────────────────────────────────────


class TestValidateParams:
    def test_validate_primitive_str(self):
        meta = get_entry_metadata(SimpleScraper.search_by_name)
        assert meta is not None
        result = meta.validate_params({"name": "alice"})
        assert result == {"name": "alice"}

    def test_validate_primitive_int(self):
        meta = get_entry_metadata(MultiTypeScraper.search_by_count)
        assert meta is not None
        result = meta.validate_params({"count": 42})
        assert result == {"count": 42}

    def test_validate_speculative_param(self):
        """Speculative params are validated via model_validate like any BaseModel."""
        meta = get_entry_metadata(SimpleScraper.fetch_by_id)
        assert meta is not None
        result = meta.validate_params({"rid": {"record_id": 42, "gap": 10}})
        assert isinstance(result["rid"], RecordId)
        assert result["rid"].record_id == 42
        assert result["rid"].gap == 10

    def test_validate_basemodel(self):
        meta = get_entry_metadata(SimpleScraper.search_by_date)
        assert meta is not None
        result = meta.validate_params(
            {"date_range": {"start": "2020-01-01", "end": "2020-12-31"}}
        )
        assert isinstance(result["date_range"], DateRange)
        assert result["date_range"].start == date(2020, 1, 1)
        assert result["date_range"].end == date(2020, 12, 31)

    def test_validate_date_from_string(self):
        meta = get_entry_metadata(MultiTypeScraper.search_by_filing_date)
        assert meta is not None
        result = meta.validate_params({"filing_date": "2024-06-15"})
        assert result["filing_date"] == date(2024, 6, 15)

    def test_validate_date_from_date_object(self):
        meta = get_entry_metadata(MultiTypeScraper.search_by_filing_date)
        assert meta is not None
        result = meta.validate_params({"filing_date": date(2024, 6, 15)})
        assert result["filing_date"] == date(2024, 6, 15)

    def test_validate_missing_param_raises(self):
        meta = get_entry_metadata(SimpleScraper.search_by_name)
        assert meta is not None
        with pytest.raises(ValidationError, match="Field required"):
            meta.validate_params({})

    def test_validate_unexpected_param_raises(self):
        meta = get_entry_metadata(SimpleScraper.search_by_name)
        assert meta is not None
        with pytest.raises(
            ValidationError, match="Extra inputs are not permitted"
        ):
            meta.validate_params({"name": "alice", "extra": "bad"})

    def test_validate_bad_date_raises(self):
        meta = get_entry_metadata(MultiTypeScraper.search_by_filing_date)
        assert meta is not None
        with pytest.raises(ValidationError):
            meta.validate_params({"filing_date": 12345})


# ── initial_seed() ─────────────────────────────────────────────────


class TestInitialSeed:
    def test_single_invocation(self):
        scraper = SimpleScraper()
        requests = list(
            scraper.initial_seed([{"search_by_name": {"name": "test"}}])
        )
        assert len(requests) == 1
        assert requests[0].request.url == "/search?name=test"

    def test_multiple_invocations(self):
        scraper = SimpleScraper()
        requests = list(
            scraper.initial_seed(
                [
                    {"search_by_name": {"name": "alice"}},
                    {"search_by_name": {"name": "bob"}},
                ]
            )
        )
        assert len(requests) == 2
        assert requests[0].request.url == "/search?name=alice"
        assert requests[1].request.url == "/search?name=bob"

    def test_speculative_initial_seed_stores_templates(self):
        scraper = SimpleScraper()
        requests = list(
            scraper.initial_seed(
                [{"fetch_by_id": {"rid": {"record_id": 99, "gap": 10}}}]
            )
        )
        # Speculative entries don't yield requests
        assert len(requests) == 0
        # Instead they store (template, raw_seed_value) pairs
        assert hasattr(scraper, "_speculation_templates")
        assert "fetch_by_id" in scraper._speculation_templates
        templates = scraper._speculation_templates["fetch_by_id"]
        assert len(templates) == 1
        template, seed_value = templates[0]
        assert isinstance(template, RecordId)
        assert template.record_id == 99
        assert template.gap == 10
        assert seed_value == {"record_id": 99, "gap": 10}

    def test_multiple_speculative_templates_same_entry(self):
        """Multiple invocations of the same speculative entry store multiple templates."""
        scraper = SimpleScraper()
        requests = list(
            scraper.initial_seed(
                [
                    {"fetch_by_id": {"rid": {"record_id": 50, "gap": 5}}},
                    {"fetch_by_id": {"rid": {"record_id": 100, "gap": 10}}},
                ]
            )
        )
        assert len(requests) == 0
        templates = scraper._speculation_templates["fetch_by_id"]
        assert len(templates) == 2
        assert isinstance(templates[0][0], RecordId)
        assert isinstance(templates[1][0], RecordId)
        assert templates[0][0].record_id == 50
        assert templates[1][0].record_id == 100

    def test_basemodel_param_dispatch(self):
        scraper = SimpleScraper()
        requests = list(
            scraper.initial_seed(
                [
                    {
                        "search_by_date": {
                            "date_range": {
                                "start": "2020-01-01",
                                "end": "2020-12-31",
                            }
                        }
                    }
                ]
            )
        )
        assert len(requests) == 1
        assert "start=2020-01-01" in requests[0].request.url

    def test_empty_params_raises(self):
        scraper = SimpleScraper()
        with pytest.raises(ValueError, match="at least one parameter"):
            list(scraper.initial_seed([]))

    def test_none_params_raises(self):
        scraper = SimpleScraper()
        with pytest.raises((ValueError, TypeError)):
            list(scraper.initial_seed(None))  # type: ignore[arg-type]

    def test_unknown_entry_raises(self):
        scraper = SimpleScraper()
        with pytest.raises(ValueError, match="Unknown entry"):
            list(scraper.initial_seed([{"nonexistent": {}}]))


# ── schema() ───────────────────────────────────────────────────────


class TestSchema:
    def test_schema_structure(self):
        schema = SimpleScraper.schema()
        assert schema["scraper"] == "SimpleScraper"
        assert "entries" in schema
        assert set(schema["entries"].keys()) == {
            "search_by_name",
            "search_by_date",
            "fetch_by_id",
        }

    def test_primitive_param_schema(self):
        schema = SimpleScraper.schema()
        entry = schema["entries"]["search_by_name"]
        assert entry["returns"] == "FakeData"
        assert entry["speculative"] is False
        props = entry["parameters"]["properties"]
        assert props["name"] == {"type": "string"}
        assert entry["parameters"]["required"] == ["name"]

    def test_basemodel_param_schema(self):
        schema = SimpleScraper.schema()
        entry = schema["entries"]["search_by_date"]
        props = entry["parameters"]["properties"]
        assert props["date_range"] == {"$ref": "#/$defs/DateRange"}
        assert "DateRange" in schema["$defs"]

    def test_speculative_schema_uses_pydantic_model(self):
        """Speculative entries emit their Pydantic model schema directly."""
        schema = SimpleScraper.schema()
        entry = schema["entries"]["fetch_by_id"]
        assert entry["speculative"] is True
        # The parameter schema should reference the RecordId model
        props = entry["parameters"]["properties"]
        assert props["rid"] == {"$ref": "#/$defs/RecordId"}
        assert "RecordId" in schema["$defs"]

    def test_integer_param_schema(self):
        schema = MultiTypeScraper.schema()
        entry = schema["entries"]["search_by_count"]
        props = entry["parameters"]["properties"]
        assert props["count"] == {"type": "integer"}

    def test_date_param_schema(self):
        schema = MultiTypeScraper.schema()
        entry = schema["entries"]["search_by_filing_date"]
        props = entry["parameters"]["properties"]
        assert props["filing_date"] == {"type": "string", "format": "date"}

    def test_schema_is_json_serializable(self):
        schema = SimpleScraper.schema()
        # Should not raise
        json.dumps(schema)


# ── Decorator error cases ──────────────────────────────────────────


class TestEntryDecoratorErrors:
    def test_bare_tuple_param_rejected(self):
        with pytest.raises(TypeError, match="bare, untyped `tuple`"):

            class BadScraper(BaseScraper[FakeData]):
                @entry(FakeData)
                def bad_entry(
                    self, pair: tuple
                ) -> Generator[Request, None, None]:
                    yield Request(
                        request=HTTPRequestParams(
                            method=HttpMethod.GET, url="/"
                        ),
                        continuation="x",
                    )

    def test_unannotated_param_rejected(self):
        with pytest.raises(TypeError, match="must have a type annotation"):

            @entry(FakeData)
            def bad(self, x):
                pass

    def test_unvalidatable_type_rejected_at_decoration(self):
        # A type pydantic can't build a schema for fails fast at decoration,
        # not later when a run is seeded.
        class Arbitrary:
            pass

        with pytest.raises(TypeError, match="pydantic cannot validate"):

            class BadScraper(BaseScraper[FakeData]):
                @entry(FakeData)
                def bad_entry(
                    self, thing: Arbitrary
                ) -> Generator[Request, None, None]:
                    yield Request(
                        request=HTTPRequestParams(
                            method=HttpMethod.GET, url="/"
                        ),
                        continuation="x",
                    )


class TestRelaxedParamTypes:
    """Typed containers are accepted without a wrapper model.

    The @entry gate validates param values through a per-entry pydantic
    model, so any annotation pydantic can validate is allowed (only a bare,
    unparameterized ``tuple`` is rejected). A ``list[str]`` param needs no
    wrapper model, so its seed value is specified at a single level.
    """

    def test_typed_container_params_accepted_and_coerced(self):
        class ContainerScraper(BaseScraper[FakeData]):
            @entry(FakeData)
            def by_ids(
                self, ids: list[str], span: tuple[int, int]
            ) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(method=HttpMethod.GET, url="/"),
                    continuation="x",
                )

        meta = get_entry_metadata(ContainerScraper.by_ids)
        assert meta is not None
        # No inner key for `ids` — single level, not a doubly-nested wrapper.
        result = meta.validate_params({"ids": ["a", "b"], "span": [1, 9]})
        assert result["ids"] == ["a", "b"]
        assert result["span"] == (1, 9)  # JSON array coerced to tuple

    def test_container_param_appears_in_schema(self):
        class ContainerScraper(BaseScraper[FakeData]):
            @entry(FakeData)
            def by_ids(self, ids: list[str]) -> Generator[Request, None, None]:
                yield Request(
                    request=HTTPRequestParams(method=HttpMethod.GET, url="/"),
                    continuation="x",
                )

        schema = ContainerScraper.schema()
        prop = schema["entries"]["by_ids"]["parameters"]["properties"]["ids"]
        assert prop == {"type": "array", "items": {"type": "string"}}
