"""The ``@step`` injector registry.

A host embedding jkent (a replay worker) adds its own injection with
``register_injector``, and it can only do so *after* importing the scraper
modules — importing a scraper is how a host finds out what it needs. So the
set of injections a step asks for is resolved against the registry as it
stands when the step runs, not as it stood when the module was imported.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from typing import Any

import pytest

from jkent.common.decorators import (
    INJECTORS,
    Injector,
    register_injector,
    step,
)
from jkent.data_types import BaseScraper, ParsedData, Response


def _response() -> Response:
    return Response(
        status_code=200,
        headers={},
        content=b"<html></html>",
        text="<html></html>",
        url="http://example.test/detail",
        request=None,  # type: ignore[arg-type]
    )


class _Host(BaseScraper[dict[str, Any]]):
    # Decorated at import time, i.e. before the host below registers
    # ``replay_cursor``.
    @step
    def parse(
        self, text: str, replay_cursor: Any
    ) -> Generator[ParsedData[dict[str, Any]], None, None]:
        yield ParsedData(data={"cursor": replay_cursor, "text": text})


@pytest.fixture
def clean_registry() -> Iterator[None]:
    """Undo registrations a test makes; the registry is process-global."""
    before = dict(INJECTORS)
    yield
    INJECTORS.clear()
    INJECTORS.update(before)


def test_injection_registered_after_decoration_is_seen(
    clean_registry: None,
) -> None:
    """The failing path: register after import, and the step still gets it.

    Frozen at decoration time, ``parse`` would be called without
    ``replay_cursor`` and raise TypeError once per request for a whole run.
    """
    register_injector(
        "replay_cursor", lambda ctx: "cursor-7", doc="The replay cursor"
    )
    results = list(_Host().parse(_response()))
    assert results[0].data["cursor"] == "cursor-7"


def test_step_without_the_parameter_is_unaffected(
    clean_registry: None,
) -> None:
    register_injector("replay_cursor", lambda ctx: "x", doc="The cursor")

    class _Other(BaseScraper[dict[str, Any]]):
        @step
        def parse(
            self, text: str
        ) -> Generator[ParsedData[dict[str, Any]], None, None]:
            yield ParsedData(data={"text": text})

    assert list(_Other().parse(_response()))[0].data["text"] == "<html></html>"


def test_registering_the_same_injector_twice_is_a_noop(
    clean_registry: None,
) -> None:
    """Two imports of one host module must not blow up the second time."""

    def builder(ctx: Any) -> str:
        return "cursor-7"

    register_injector("replay_cursor", builder, doc="The replay cursor")
    register_injector("replay_cursor", builder, doc="The replay cursor")
    assert INJECTORS["replay_cursor"] == Injector(builder, "The replay cursor")


def test_registering_a_different_injector_for_one_name_raises(
    clean_registry: None,
) -> None:
    register_injector("replay_cursor", lambda ctx: "a", doc="The cursor")
    with pytest.raises(ValueError, match="already registered"):
        register_injector("replay_cursor", lambda ctx: "b", doc="The cursor")


def test_builtin_injections_are_documented_on_step() -> None:
    """Every registered name reaches ``help(step)``; the {} is rendered."""
    assert step.__doc__ is not None
    assert "{injections}" not in step.__doc__
    for name in INJECTORS:
        assert f"- {name}: " in step.__doc__
