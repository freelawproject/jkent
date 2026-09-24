"""Metadata types and accessors for @step and @entry decorated methods.

These live apart from jkent.common.decorators (which attaches them) so that
jkent.common.scraper can introspect decorated scraper methods without
importing decorators — decorators imports scraper at module level, and a
top-level import in the other direction would be circular.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Final

from pydantic import (
    BaseModel,
    ConfigDict,
    PydanticSchemaGenerationError,
    create_model,
)

if TYPE_CHECKING:
    from jkent.common.wait_conditions import WaitCondition

# Effective priority for steps/requests whose author didn't choose one
# (lower = higher priority). Shared by StepMetadata and Request; the
# authoring facade (jkent.data_types) re-exports it.
DEFAULT_PRIORITY: Final = 9

#: A request timeout in seconds, or a (connect, read) tuple; None is unset.
#: Shared by StepMetadata and HTTPRequestParams; the authoring facade
#: (jkent.data_types) re-exports it.
TimeoutType = float | tuple[float, float] | None


class StepMetadata:
    """Metadata attached to scraper step methods by @step decorator.

    Attributes:
        priority: Priority hint for queue ordering (lower = higher priority).
        encoding: Character encoding for text/HTML decoding.
        await_list: List of wait conditions for Playwright driver (WaitForSelector, etc).
        auto_await_timeout: Optional timeout in milliseconds for autowait retry logic.
        rate_limit: Rate-limit lane for requests routed to this step, or
            None to leave the request's own (default-lane) choice alone.
            Inherited by a yielded Request whose ``rate_limit`` is unset,
            the way ``priority`` is.
        timeout: Seconds a request routed to this step waits, or None to
            leave it to the transport. Inherited by a yielded Request whose
            ``HTTPRequestParams.timeout`` is unset.
    """

    def __init__(
        self,
        priority: int = DEFAULT_PRIORITY,
        encoding: str = "utf-8",
        await_list: list[WaitCondition] | None = None,
        auto_await_timeout: int | None = None,
        rate_limit: str | None = None,
        timeout: TimeoutType = None,
    ):
        self.priority = priority
        self.encoding = encoding
        self.await_list = await_list or []
        self.auto_await_timeout = auto_await_timeout
        self.rate_limit = rate_limit
        self.timeout = timeout


@dataclass(frozen=True)
class EntryMetadata:
    """Metadata attached to scraper entry point methods by @entry decorator.

    Attributes:
        return_type: The data type this entry produces (e.g. Docket).
        param_types: Mapping of parameter name to annotation (BaseModel
            subclass, primitive, or typed container).
        func_name: Name of the decorated function.
        speculative_param: Name of the parameter implementing the Speculative protocol,
            or None if this entry is not speculative.
    """

    return_type: type
    param_types: dict[str, Any]
    func_name: str
    speculative_param: str | None = None

    def __post_init__(self) -> None:
        # Build the per-entry validation model now so a parameter annotation
        # pydantic cannot generate a schema for fails at decoration time,
        # not later when a run is first seeded.
        try:
            _ = self._validator
        except PydanticSchemaGenerationError as e:
            raise TypeError(
                f"Entry '{self.func_name}' has a parameter type pydantic "
                f"cannot validate: {e}"
            ) from e

    @property
    def speculative(self) -> bool:
        """Whether this is a speculative entry point."""
        return self.speculative_param is not None

    @cached_property
    def _validator(self) -> type[BaseModel]:
        """Pydantic model mirroring this entry's parameter signature.

        Built once per entry from ``param_types``: every parameter is a
        required field and ``extra="forbid"`` rejects unexpected keys. The
        @entry decorator guarantees each type is a BaseModel subclass,
        ``str``, ``int``, or ``date`` — all natively validated and coerced
        by pydantic (including ISO strings to ``date``).
        """
        fields: dict[str, Any] = {
            name: (typ, ...) for name, typ in self.param_types.items()
        }
        return create_model(
            f"{self.func_name}__params",
            __config__=ConfigDict(extra="forbid"),
            **fields,
        )

    def validate_params(self, kwargs_dict: dict[str, Any]) -> dict[str, Any]:
        """Validate and coerce parameters for this entry function.

        Delegates entirely to pydantic via a per-entry model: BaseModel
        params are validated through the nested model, ISO strings are
        coerced to ``date``, primitives are coerced, and missing or
        unexpected keys are rejected.

        Args:
            kwargs_dict: Raw parameter dict from JSON deserialization.

        Returns:
            Dict of validated/coerced parameter values ready for function call.

        Raises:
            pydantic.ValidationError: If any parameter is missing, unexpected,
                or fails validation/coercion.
        """
        model = self._validator.model_validate(kwargs_dict)
        return {name: getattr(model, name) for name in self.param_types}


_STEP_METADATA_ATTR: Final = "_step_metadata"
_ENTRY_METADATA_ATTR: Final = "_entry_metadata"


def attach_step_metadata(
    func: Callable[..., Any], metadata: StepMetadata
) -> None:
    """Record *metadata* on *func* for :func:`get_step_metadata` to find."""
    setattr(func, _STEP_METADATA_ATTR, metadata)


def attach_entry_metadata(
    func: Callable[..., Any], metadata: EntryMetadata
) -> None:
    """Record *metadata* on *func* for :func:`get_entry_metadata` to find."""
    setattr(func, _ENTRY_METADATA_ATTR, metadata)


def get_step_metadata(func: Callable[..., Any]) -> StepMetadata | None:
    """Get step metadata from a decorated method.

    Args:
        func: A potentially decorated scraper step method.

    Returns:
        StepMetadata if the method is decorated, None otherwise.
    """
    return getattr(func, _STEP_METADATA_ATTR, None)


def get_entry_metadata(func: Callable[..., Any]) -> EntryMetadata | None:
    """Get entry metadata from a decorated method.

    Args:
        func: A potentially decorated scraper entry method.

    Returns:
        EntryMetadata if the method is decorated with @entry, None otherwise.
    """
    return getattr(func, _ENTRY_METADATA_ATTR, None)
