"""Request priority semantics.

priority is tri-state: None means "the scraper author didn't choose",
which lets defaulting (archive requests, target-step inheritance, the
queue default) apply only to genuinely-unset priorities. An explicit
priority — including an explicit 9 — is always kept.
"""

from typing import Any

from jkent.data_types import (
    ARCHIVE_DEFAULT_PRIORITY,
    DEFAULT_PRIORITY,
    HttpMethod,
    HTTPRequestParams,
    Request,
)


def make_request(**kwargs: Any) -> Request:
    kwargs.setdefault("step", "parse")
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/doc"
        ),
        **kwargs,
    )


class TestRequestPriority:
    def test_unset_priority_is_none(self):
        assert make_request().priority is None

    def test_explicit_priority_is_kept(self):
        assert make_request(priority=3).priority == 3

    def test_effective_priority_defaults_when_unset(self):
        assert make_request().effective_priority == DEFAULT_PRIORITY

    def test_effective_priority_returns_explicit_value(self):
        assert make_request(priority=3).effective_priority == 3

    def test_unset_archive_priority_gets_archive_default(self):
        request = make_request(archive=True)
        assert request.priority == ARCHIVE_DEFAULT_PRIORITY

    def test_explicit_priority_9_on_archive_request_is_kept(self):
        """An author who deliberately writes priority=9 means it.

        Under the old ``priority == 9`` sentinel this was silently
        rewritten to 1 because the default was indistinguishable from
        an explicit 9.
        """
        request = make_request(archive=True, priority=9)
        assert request.priority == 9

    def test_unset_priority_survives_resolve_from_as_unset(self):
        parent = make_request(current_location="https://example.com/list")
        resolved = make_request().resolve_from(parent)
        assert resolved.priority is None

    def test_explicit_priority_survives_resolve_from(self):
        parent = make_request(current_location="https://example.com/list")
        resolved = make_request(priority=4).resolve_from(parent)
        assert resolved.priority == 4
