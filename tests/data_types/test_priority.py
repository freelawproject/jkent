"""Request priority semantics.

priority is tri-state: None means "the scraper author didn't choose",
which lets defaulting (archive requests, target-step inheritance, the
queue default) apply only to genuinely-unset priorities. An explicit
priority — including an explicit 9 — is always kept.
"""

from collections.abc import Generator

from jkent.common.decorators import step
from jkent.data_types import (
    ARCHIVE_DEFAULT_PRIORITY,
    DEFAULT_PRIORITY,
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    ScraperYield,
)


def make_request(**kwargs) -> Request:
    kwargs.setdefault("continuation", "parse")
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


class TestStepPriorityInheritance:
    """Callable continuations inherit the target step's priority."""

    @staticmethod
    def _run_step(scraper, step_name: str = "parse_listing"):
        request = Request(
            request=HTTPRequestParams(
                method=HttpMethod.GET, url="https://example.com/list"
            ),
            continuation=step_name,
        )
        response = Response(
            status_code=200,
            headers={},
            content=b"",
            text="",
            url="https://example.com/list",
            request=request,
        )
        return list(getattr(scraper, step_name)(response))

    def test_unset_priority_inherits_target_step_priority(self):
        class InheritScraper(BaseScraper[dict]):
            @step
            def parse_listing(
                self, response: Response
            ) -> Generator[ScraperYield, None, None]:
                yield make_request(continuation=self.parse_detail)

            @step(priority=2)
            def parse_detail(
                self, response: Response
            ) -> Generator[ScraperYield, None, None]:
                yield ParsedData({"ok": True})

        yields = self._run_step(InheritScraper())
        assert yields[0].priority == 2

    def test_explicit_priority_9_not_overridden_by_target_step(self):
        """An explicit 9 must not be replaced by the target's priority."""

        class ExplicitScraper(BaseScraper[dict]):
            @step
            def parse_listing(
                self, response: Response
            ) -> Generator[ScraperYield, None, None]:
                yield make_request(continuation=self.parse_detail, priority=9)

            @step(priority=2)
            def parse_detail(
                self, response: Response
            ) -> Generator[ScraperYield, None, None]:
                yield ParsedData({"ok": True})

        yields = self._run_step(ExplicitScraper())
        assert yields[0].priority == 9
