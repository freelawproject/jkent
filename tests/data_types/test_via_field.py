"""Tests for via field propagation through request chains.

Tests that ViaLink and ViaFormSubmit are carried through resolve_from() calls.
"""

from jkent.common.page_element import (
    ViaFormSubmit,
    ViaLink,
)
from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
    Selector,
)


def test_via_field_on_base_request():
    """Request should have a via field that defaults to None."""
    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page", method=HttpMethod.GET
        ),
        continuation="parse_page",
    )

    assert request.via is None


def test_via_link_on_navigating_request():
    """Request should accept and store ViaLink."""
    via = ViaLink(
        selector=Selector.XPath("//a[@id='link1']"),
        description="Test link",
    )

    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page", method=HttpMethod.GET
        ),
        continuation="parse_page",
        via=via,
    )

    assert request.via == via
    assert isinstance(request.via, ViaLink)
    assert request.via.selector.value == "//a[@id='link1']"
    assert request.via.selector.grammar == "xpath"


def test_via_form_submit_on_navigating_request():
    """Request should accept and store ViaFormSubmit."""
    via = ViaFormSubmit(
        form_selector=Selector.XPath("//form[@id='search']"),
        submit_selector=".//button[@type='submit']",
        field_data={"q": "test"},
        description="Search form",
    )

    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/search", method=HttpMethod.POST
        ),
        continuation="parse_results",
        via=via,
    )

    assert request.via == via
    assert isinstance(request.via, ViaFormSubmit)
    assert request.via.form_selector.value == "//form[@id='search']"
    assert request.via.form_selector.grammar == "xpath"
    assert request.via.field_data == {"q": "test"}


def test_via_propagates_through_resolve_from_response():
    """via should be preserved when resolving from a Response."""
    via = ViaLink(
        selector=Selector.XPath("//a[@class='next']"),
        description="Next page",
    )

    original_request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page1", method=HttpMethod.GET
        ),
        continuation="parse_page",
        current_location="https://example.com/",
        via=via,
    )

    # Simulate a response from the original request
    response = Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page1",
        request=original_request,
    )

    # Create a new request that should preserve via
    new_request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page2", method=HttpMethod.GET
        ),
        continuation="parse_page",
        via=via,
    )

    resolved = new_request.resolve_from(response)

    assert resolved.via == via
    assert isinstance(resolved.via, ViaLink)
    assert resolved.via.selector.value == "//a[@class='next']"


def test_via_propagates_through_nonnavigating_request():
    """via should be preserved for nonnavigating Request."""
    via = ViaFormSubmit(
        form_selector=Selector.XPath("//form"),
        submit_selector=None,
        field_data={},
        description="API form",
    )

    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/api", method=HttpMethod.POST
        ),
        continuation="parse_api_response",
        via=via,
        nonnavigating=True,
    )

    assert request.via == via


def test_via_propagates_through_nonnavigating_resolve():
    """via should be preserved when nonnavigating Request resolves."""
    via = ViaFormSubmit(
        form_selector=Selector.XPath("//form[@id='filter']"),
        submit_selector=None,
        field_data={"filter": "active"},
        description="Filter form",
    )

    parent_request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page", method=HttpMethod.GET
        ),
        continuation="parse_page",
        current_location="https://example.com/",
    )

    response = Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
        request=parent_request,
    )

    non_nav_request = Request(
        request=HTTPRequestParams(url="/api/filter", method=HttpMethod.POST),
        continuation="parse_filter_response",
        via=via,
        nonnavigating=True,
    )

    resolved = non_nav_request.resolve_from(response)

    assert resolved.via == via
    assert isinstance(resolved.via, ViaFormSubmit)
    assert resolved.via.form_selector.value == "//form[@id='filter']"


def test_via_none_preserved():
    """When via is None, it should remain None through resolve."""
    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page", method=HttpMethod.GET
        ),
        continuation="parse_page",
        current_location="https://example.com/",
        via=None,
    )

    response = Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url="https://example.com/page",
        request=request,
    )

    new_request = Request(
        request=HTTPRequestParams(
            url="https://example.com/page2", method=HttpMethod.GET
        ),
        continuation="parse_page",
        via=None,
    )

    resolved = new_request.resolve_from(response)

    assert resolved.via is None


def test_via_different_for_different_requests():
    """Different requests can have different via values."""
    via1 = ViaLink(
        selector=Selector.XPath("//a[@id='link1']"),
        description="Link 1",
    )
    via2 = ViaLink(
        selector=Selector.XPath("//a[@id='link2']"),
        description="Link 2",
    )

    request1 = Request(
        request=HTTPRequestParams(
            url="https://example.com/page1", method=HttpMethod.GET
        ),
        continuation="parse_page",
        via=via1,
    )

    request2 = Request(
        request=HTTPRequestParams(
            url="https://example.com/page2", method=HttpMethod.GET
        ),
        continuation="parse_page",
        via=via2,
    )

    assert request1.via != request2.via
    assert isinstance(request1.via, ViaLink)
    assert isinstance(request2.via, ViaLink)
    assert request1.via.selector.value == "//a[@id='link1']"
    assert request2.via.selector.value == "//a[@id='link2']"


def test_via_preserved_in_speculative_request():
    """via should be preserved when creating speculative requests."""
    via = ViaLink(
        selector=Selector.XPath("//a[@class='detail']"),
        description="Detail link",
    )

    request = Request(
        request=HTTPRequestParams(
            url="https://example.com/detail/123", method=HttpMethod.GET
        ),
        continuation="parse_detail",
        via=via,
    )

    speculative = request.speculative("fetch_case", 0, 123)

    assert speculative.via == via
    assert speculative.is_speculative is True
    assert speculative.speculation_id == ("fetch_case", 0, 123)
