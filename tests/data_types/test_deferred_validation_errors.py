"""Tests for DataFormatAssumptionException error summaries.

Pydantic reports field-level errors with a populated ``loc`` tuple and
model-level errors (``@model_validator``) with an empty one. The exception
must produce a readable summary for both.
"""

import pytest
from pydantic import BaseModel, model_validator

from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import DataFormatAssumptionException


class _DateRange(BaseModel):
    low: int
    high: int

    @model_validator(mode="after")
    def _check_order(self) -> "_DateRange":
        if self.low > self.high:
            raise ValueError("low must not exceed high")
        return self


def test_field_level_error_names_field():
    """Field-level validation errors name the offending field."""
    deferred = DeferredValidation(
        _DateRange, request_url="https://example.com/case", low="x", high=2
    )

    with pytest.raises(DataFormatAssumptionException) as exc_info:
        deferred.confirm()

    assert "low" in str(exc_info.value)


def test_model_level_error_does_not_crash():
    """Model-level errors have an empty loc — must not raise IndexError."""
    deferred = DeferredValidation(
        _DateRange, request_url="https://example.com/case", low=5, high=1
    )

    with pytest.raises(DataFormatAssumptionException) as exc_info:
        deferred.confirm()

    # The real validation message survives into the summary
    assert "low must not exceed high" in str(exc_info.value)
    assert exc_info.value.model_name == "_DateRange"


def test_nested_loc_reports_full_path():
    """Errors below the top level report the full dotted location path."""

    class _Outer(BaseModel):
        inner: _DateRange

    deferred = DeferredValidation(
        _Outer,
        request_url="https://example.com/case",
        inner={"low": "x", "high": 2},
    )

    with pytest.raises(DataFormatAssumptionException) as exc_info:
        deferred.confirm()

    assert "inner.low" in str(exc_info.value)
