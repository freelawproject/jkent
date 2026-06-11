"""Deferred validation for scraper data.

This module provides DeferredValidation, a wrapper that delays Pydantic
validation until the driver explicitly calls confirm(). This is useful
when data needs to be collected from multiple sources before validation.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from jkent.common.exceptions import (
    DataFormatAssumptionException,
)

T = TypeVar("T", bound=BaseModel)


class DeferredValidation(Generic[T]):
    """Wrapper for unvalidated data that validates on confirm().

    This class holds unvalidated data and a Pydantic model type. Validation
    is deferred until confirm() is called by the driver, allowing data to be
    accumulated from multiple sources before validation.

    Example:
        # In scraper - defer validation
        deferred = CaseData.raw(case_name=name, docket=docket)
        yield ParsedData(deferred)

        # In driver - validate when ready
        validated_case = deferred.confirm()  # Raises if invalid
    """

    def __init__(
        self,
        model_class: type[T],
        request_url: str = "",
        **data: Any,
    ) -> None:
        """Initialize deferred validation.

        Args:
            model_class: The Pydantic model class to validate against.
            request_url: Optional URL for error reporting.
            **data: Raw field values (not validated).
        """
        self._model_class = model_class
        self._request_url = request_url
        self._data = data

    def confirm(self) -> T:
        """Validate the data and return the validated model instance.

        Returns:
            Validated instance of the model class.

        Raises:
            DataFormatAssumptionException: If validation fails.
        """
        try:
            return self._model_class.model_validate(self._data)
        except ValidationError as e:
            # Convert Pydantic ErrorDetails to dict for compatibility
            errors_list = [dict(err) for err in e.errors()]  # type: ignore
            raise DataFormatAssumptionException(
                errors=errors_list,
                failed_doc=self._data,
                model_name=self._model_class.__name__,
                request_url=self._request_url,
            ) from e

    @property
    def raw_data(self) -> dict:
        """Access the raw unvalidated data.

        Returns:
            The raw data dictionary.
        """
        return self._data.copy()

    @property
    def model_name(self) -> str:
        """Get the name of the validation model.

        Returns:
            The Pydantic model class name.
        """
        return self._model_class.__name__
