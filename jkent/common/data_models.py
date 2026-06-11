"""Pydantic data models for scraper validation.

This module contains Pydantic models that define the expected schema for
scraped data. These models are used for validation to ensure scraped data
conforms to consumer expectations.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

from jkent.common.deferred_validation import (
    DeferredValidation,
)

T = TypeVar("T", bound="ScrapedData")


class ScrapedData(BaseModel):
    """Base class for scraped data with deferred validation support.

    This base class provides a .raw() classmethod that creates DeferredValidation
    wrappers for unvalidated data.

    Example:
        # Normal usage (validates immediately)
        data = CaseData(case_name="Test", docket="123")

        # Deferred validation
        deferred = CaseData.raw(case_name="Test", docket="123")
        validated = deferred.confirm()  # Validates later
    """

    @classmethod
    def raw(
        cls: type[T], request_url: str = "", **data: Any
    ) -> DeferredValidation[T]:
        """Create a DeferredValidation wrapper with raw, unvalidated data.

        Args:
            request_url: Optional URL for error reporting. NOTE: ``request_url``
                is a reserved parameter name — a model with its own
                ``request_url`` field cannot set it through ``.raw()`` (the
                value is consumed here and never reaches ``**data``). Set such a
                field by constructing the model directly instead.
            **data: Raw field values (not validated).

        Returns:
            DeferredValidation wrapper that validates on confirm().
        """
        return DeferredValidation(cls, request_url, **data)
