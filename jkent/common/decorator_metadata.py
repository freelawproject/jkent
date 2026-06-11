"""Metadata types and accessors for @step and @entry decorated methods.

These live apart from jkent.common.decorators (which attaches them) so that
jkent.data_types can introspect decorated scraper methods at runtime without
importing decorators — decorators imports data_types at module level, and a
top-level import in the other direction would be circular.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    # Imported from the leaf wait_conditions module, not data_types: data_types
    # imports this module to introspect decorated methods, so importing
    # WaitCondition back from data_types would be a cycle. See the docstring.
    from jkent.common.wait_conditions import WaitCondition

# Effective priority for steps/requests whose author didn't choose one
# (lower = higher priority). Lives here rather than jkent.data_types so
# both StepMetadata and data_types can share it without a circular import;
# data_types re-exports it.
DEFAULT_PRIORITY: Final = 9


class StepMetadata:
    """Metadata attached to scraper step methods by @step decorator.

    Attributes:
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for text/HTML decoding.
        xsd: Optional path to XSD schema file for structural validation hints.
        json_model: Optional dotted path to Pydantic model for JSON response validation.
        await_list: List of wait conditions for Playwright driver (WaitForSelector, etc).
        auto_await_timeout: Optional timeout in milliseconds for autowait retry logic.
    """

    def __init__(
        self,
        priority: int = DEFAULT_PRIORITY,
        encoding: str = "utf-8",
        xsd: str | None = None,
        json_model: str | None = None,
        await_list: list[WaitCondition] | None = None,
        auto_await_timeout: int | None = None,
    ):
        self.priority = priority
        self.encoding = encoding
        self.xsd = xsd
        self.json_model = json_model
        self.await_list = await_list or []
        self.auto_await_timeout = auto_await_timeout


@dataclass(frozen=True)
class EntryMetadata:
    """Metadata attached to scraper entry point methods by @entry decorator.

    Attributes:
        return_type: The data type this entry produces (e.g. Docket).
        param_types: Mapping of parameter name to type (BaseModel subclass or primitive).
        func_name: Name of the decorated function.
        speculative_param: Name of the parameter implementing the Speculative protocol,
            or None if this entry is not speculative.
    """

    return_type: type
    param_types: dict[str, type]
    func_name: str
    speculative_param: str | None = None

    @property
    def speculative(self) -> bool:
        """Whether this is a speculative entry point."""
        return self.speculative_param is not None

    def validate_params(self, kwargs_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce parameters for this entry function.

        Validates against the function signature: BaseModel parameters
        use model_validate(), primitives are coerced.

        Args:
            kwargs_dict: Raw parameter dict from JSON deserialization.

        Returns:
            Dict of validated/coerced parameter values ready for function call.

        Raises:
            pydantic.ValidationError: If a BaseModel parameter fails validation.
            TypeError: If a primitive parameter can't be coerced.
            ValueError: If unexpected parameters are provided.
        """
        validated: dict[str, Any] = {}

        # Check for unexpected parameters
        unexpected = set(kwargs_dict.keys()) - set(self.param_types.keys())
        if unexpected:
            raise ValueError(
                f"Unexpected parameters for entry '{self.func_name}': "
                f"{unexpected}. Expected: {list(self.param_types.keys())}"
            )

        for param_name, param_type in self.param_types.items():
            if param_name not in kwargs_dict:
                raise ValueError(
                    f"Missing required parameter '{param_name}' "
                    f"for entry '{self.func_name}'"
                )

            raw_value = kwargs_dict[param_name]

            if isinstance(param_type, type) and issubclass(
                param_type, BaseModel
            ):
                # Pydantic model: validate via model_validate
                pydantic_type = cast(type[BaseModel], param_type)
                validated[param_name] = pydantic_type.model_validate(raw_value)
            elif param_type is date:
                # date: accept date objects or ISO format strings
                if isinstance(raw_value, date):
                    validated[param_name] = raw_value
                elif isinstance(raw_value, str):
                    validated[param_name] = date.fromisoformat(raw_value)
                else:
                    raise TypeError(
                        f"Parameter '{param_name}' for entry "
                        f"'{self.func_name}' expected date or ISO string, "
                        f"got {type(raw_value).__name__}"
                    )
            elif param_type in (str, int):
                # Primitive: coerce
                validated[param_name] = param_type(raw_value)
            else:
                # Shouldn't happen if decorator validation is correct
                validated[param_name] = raw_value

        return validated


def get_step_metadata(func: Callable[..., Any]) -> StepMetadata | None:
    """Get step metadata from a decorated method.

    Args:
        func: A potentially decorated scraper step method.

    Returns:
        StepMetadata if the method is decorated, None otherwise.
    """
    return getattr(func, "_step_metadata", None)


def get_entry_metadata(func: Callable[..., Any]) -> EntryMetadata | None:
    """Get entry metadata from a decorated method.

    Args:
        func: A potentially decorated scraper entry method.

    Returns:
        EntryMetadata if the method is decorated with @entry, None otherwise.
    """
    return getattr(func, "_entry_metadata", None)
