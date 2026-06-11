"""Tests for the PageElement ABC and value objects.

Tests interface conformance and value object behavior.
"""

import pytest

from jkent.common.decorators import step
from jkent.common.page_element import (
    Form,
    FormField,
    Link,
    ViaFormSubmit,
    ViaLink,
)
from jkent.data_types import (
    BaseScraper,
    HttpMethod,
    HTTPRequestParams,
    ParsedData,
    Request,
    Response,
    Selector,
)


def test_form_get_field():
    """Form.get_field should return the field by name."""
    fields = [
        FormField(name="username", field_type="input", value="john"),
        FormField(name="password", field_type="password", value="secret"),
        FormField(name="remember", field_type="checkbox", value="true"),
    ]
    form = Form(
        action="https://example.com/login",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form[@id='login']"),
    )

    # Find existing field
    username_field = form.get_field("username")
    assert username_field is not None
    assert username_field.name == "username"
    assert username_field.value == "john"

    # Non-existent field
    assert form.get_field("nonexistent") is None


def test_form_submit_post():
    """Form.submit should create a Request for POST forms."""
    fields = [
        FormField(name="username", field_type="input", value="john"),
        FormField(name="password", field_type="password", value="secret"),
    ]
    form = Form(
        action="https://example.com/login",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form[@id='login']"),
    )

    request = form.submit(continuation="test")

    assert request.request.url == "https://example.com/login"
    assert request.request.method == HttpMethod.POST
    assert request.via is not None
    assert isinstance(request.via, ViaFormSubmit)
    assert request.via.form_selector.value == "//form[@id='login']"
    assert request.via.form_selector.grammar == "xpath"
    assert request.via.field_data == {"username": "john", "password": "secret"}


def test_form_submit_with_overrides():
    """Form.submit should merge field data with overrides."""
    fields = [
        FormField(name="username", field_type="input", value="john"),
        FormField(name="password", field_type="password", value="default"),
    ]
    form = Form(
        action="https://example.com/login",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form[@id='login']"),
    )

    request = form.submit(data={"password": "newsecret"}, continuation="test")

    assert request.via is not None
    assert isinstance(request.via, ViaFormSubmit)
    # Override should take precedence
    assert request.via.field_data == {
        "username": "john",
        "password": "newsecret",
    }


def test_form_submit_with_submit_selector():
    """Form.submit should include submit_selector in ViaFormSubmit."""
    fields = [
        FormField(name="q", field_type="input", value="search term"),
    ]
    form = Form(
        action="https://example.com/search",
        method="GET",
        fields=fields,
        selector=Selector.XPath("//form[@id='search']"),
    )

    request = form.submit(
        submit_selector=".//button[@name='submit']", continuation="test"
    )

    assert request.via is not None
    assert isinstance(request.via, ViaFormSubmit)
    assert request.via.submit_selector == ".//button[@name='submit']"


def test_form_submit_with_css_submit_selector():
    """A CSS()-wrapped submit_selector is accepted, picks the matching submit,
    and is stored as a Playwright-resolvable string."""
    fields = [
        FormField(name="q", field_type="input", value="term"),
        FormField(name="btnCancel", field_type="submit", value="Cancel"),
        FormField(name="btnFind", field_type="submit", value="Find"),
    ]
    form = Form(
        action="https://example.com/search",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form"),
    )

    request = form.submit(
        submit_selector=Selector.CSS("input[name='btnFind']"),
        continuation="test",
    )

    assert isinstance(request.via, ViaFormSubmit)
    # Only the activated submit control's name/value is sent — the predicate in
    # the CSS selector must resolve to btnFind, not the first submit (btnCancel).
    assert request.via.field_data == {"q": "term", "btnFind": "Find"}
    # Stored grammar-tagged so the Playwright transport resolves it unambiguously.
    assert request.via.submit_selector == "css=input[name='btnFind']"


def test_form_submit_with_xpath_submit_selector_object():
    """An XPath()-wrapped submit_selector is accepted and resolved like a CSS one."""
    fields = [
        FormField(name="q", field_type="input", value="term"),
        FormField(name="btnCancel", field_type="submit", value="Cancel"),
        FormField(name="btnFind", field_type="submit", value="Find"),
    ]
    form = Form(
        action="https://example.com/search",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form"),
    )

    request = form.submit(
        submit_selector=Selector.XPath("//input[@name='btnFind']"),
        continuation="test",
    )

    assert isinstance(request.via, ViaFormSubmit)
    assert request.via.field_data == {"q": "term", "btnFind": "Find"}
    assert request.via.submit_selector == "xpath=//input[@name='btnFind']"


def test_form_submit_with_selector_wrapped_data_keys():
    """data keys wrapped in CSS()/XPath() are unwrapped to their field names."""
    fields = [
        FormField(
            name="ctl00$cphMain$tbPartyNames", field_type="input", value="old"
        ),
        FormField(
            name="ctl00$cphMain$ddlFindParty", field_type="input", value="A"
        ),
    ]
    form = Form(
        action="https://example.com/search",
        method="POST",
        fields=fields,
        selector=Selector.XPath("//form"),
    )

    request = form.submit(
        data={
            Selector.CSS("ctl00$cphMain$tbPartyNames"): "query",
            Selector.XPath("ctl00$cphMain$ddlFindParty"): "FindOR",
        },
        continuation="test",
    )

    assert isinstance(request.via, ViaFormSubmit)
    assert request.via.field_data == {
        "ctl00$cphMain$tbPartyNames": "query",
        "ctl00$cphMain$ddlFindParty": "FindOR",
    }


def test_link_follow():
    """Link.follow should create a Request."""
    link = Link(
        url="https://example.com/detail/123",
        text="Case Details",
        selector=Selector.XPath("//a[@class='case-link'][1]"),
    )

    request = link.follow(continuation="testing")

    assert request.request.url == "https://example.com/detail/123"
    assert request.request.method == HttpMethod.GET
    assert request.via is not None
    assert isinstance(request.via, ViaLink)
    assert request.via.selector.value == "//a[@class='case-link'][1]"
    assert request.via.selector.grammar == "xpath"
    assert request.via.description == "link: Case Details"


def test_link_follow_accepts_request_kwargs():
    """Link.follow should pass request kwargs through, like Form.submit.

    Request is frozen, so follow() is the caller's only chance to set
    the continuation — without the passthrough every follow()-built
    request enters the queue with continuation="".
    """
    link = Link(
        url="https://example.com/detail/123",
        text="Case Details",
        selector=Selector.XPath("//a[@class='case-link'][1]"),
    )

    request = link.follow(
        continuation="parse_detail",
        accumulated_data={"case_name": "Ant v. Bee"},
        priority=3,
    )

    assert request.continuation == "parse_detail"
    assert request.accumulated_data == {"case_name": "Ant v. Bee"}
    assert request.priority == 3
    assert isinstance(request.via, ViaLink)


def test_link_follow_callable_continuation_resolved_by_step():
    """A followed link yielded from a @step resolves its Callable.

    This is the real authoring pattern: yield link.follow(
    continuation=self.parse_detail) and let the step machinery resolve
    the name and inherit the target step's priority.
    """

    class LinkScraper(BaseScraper[dict]):
        @step
        def parse_listing(self, response: Response):
            link = Link(
                url="https://example.com/detail/1",
                text="Case",
                selector=Selector.XPath("//a[1]"),
            )
            yield link.follow(continuation=self.parse_detail)

        @step(priority=2)
        def parse_detail(self, response: Response):
            yield ParsedData({"ok": True})

    scraper = LinkScraper()
    listing_request = Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET, url="https://example.com/list"
        ),
        continuation="parse_listing",
    )
    response = Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/list",
        request=listing_request,
    )

    yields = list(scraper.parse_listing(response))

    assert len(yields) == 1
    assert yields[0].continuation == "parse_detail"
    assert yields[0].priority == 2  # inherited from the target step


def test_link_frozen():
    """Link should be a frozen dataclass."""
    link = Link(
        url="https://example.com/page",
        text="Test Page",
        selector=Selector.XPath("//a[@id='link1']"),
    )

    # Should not be able to modify
    with pytest.raises(AttributeError):
        link.url = "different"  # type: ignore[misc]


def test_form_frozen():
    """Form should be a frozen dataclass."""
    form = Form(
        action="https://example.com/submit",
        method="POST",
        fields=[],
        selector=Selector.XPath("//form"),
    )

    # Should not be able to modify
    with pytest.raises(AttributeError):
        form.action = "different"  # type: ignore[misc]
