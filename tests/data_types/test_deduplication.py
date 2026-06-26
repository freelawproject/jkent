"""Deduplication key generation and resolution semantics.

This test module verifies:

1. Default keys are derived from the request params, including the HTTP
   method — a GET and a POST to the same URL must not collide.
2. Auto-generated keys are regenerated when a request's relative URL is
   resolved against its parent — two scrapers yielding "detail.aspx"
   from different pages must not dedup each other away.
3. Explicitly-set keys (and SkipDeduplicationCheck) survive resolution
   untouched.
"""

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)


def make_request(
    url: str,
    method: HttpMethod = HttpMethod.GET,
    **request_kwargs,
) -> Request:
    return Request(
        request=HTTPRequestParams(method=method, url=url),
        continuation="parse",
        **request_kwargs,
    )


def make_response(url: str) -> Response:
    parent = make_request(url, current_location=url)
    return Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url=url,
        request=parent,
    )


class TestKeyGeneration:
    """Default key generation from HTTPRequestParams."""

    def test_default_key_is_generated(self):
        req = make_request("https://example.com/list")
        assert isinstance(req.deduplication_key, str)
        assert len(req.deduplication_key) == 64

    def test_identical_requests_share_a_key(self):
        a = make_request("https://example.com/list")
        b = make_request("https://example.com/list")
        assert a.deduplication_key == b.deduplication_key

    def test_different_urls_differ(self):
        a = make_request("https://example.com/list?page=1")
        b = make_request("https://example.com/list?page=2")
        assert a.deduplication_key != b.deduplication_key

    def test_http_method_is_part_of_the_key(self):
        """GET and POST to the same URL are different requests.

        Court sites commonly serve the search page as a GET and the
        search submission as a bodyless POST to the same URL; deduping
        one against the other silently drops a request.
        """
        get_req = make_request("https://example.com/export")
        post_req = make_request(
            "https://example.com/export", method=HttpMethod.POST
        )
        assert get_req.deduplication_key != post_req.deduplication_key

    def test_explicit_key_is_preserved(self):
        req = make_request("https://example.com/list")
        explicit = make_request(
            "https://example.com/list", deduplication_key="my-key"
        )
        assert explicit.deduplication_key == "my-key"
        assert req.deduplication_key != "my-key"


class TestKeyResolution:
    """Key behavior through resolve_from()."""

    def test_relative_urls_from_different_parents_do_not_collide(self):
        """The same relative URL on two different pages is two requests.

        Yielding relative URLs is the documented normal pattern, so an
        auto-generated key hashed from the unresolved URL would make
        sibling listings dedup each other's detail pages away.
        """
        resolved_a = make_request("detail.aspx").resolve_from(
            make_response("https://example.com/court-a/list")
        )
        resolved_b = make_request("detail.aspx").resolve_from(
            make_response("https://example.com/court-b/list")
        )
        assert resolved_a.deduplication_key != resolved_b.deduplication_key

    def test_auto_key_matches_equivalent_absolute_request(self):
        """A resolved auto key equals the key of the same absolute URL.

        The whole point of dedup keys is that the same logical request
        produces the same key no matter how it was constructed.
        """
        resolved = make_request("detail.aspx?id=7").resolve_from(
            make_response("https://example.com/court/list")
        )
        direct = make_request("https://example.com/court/detail.aspx?id=7")
        assert resolved.deduplication_key == direct.deduplication_key

    def test_same_relative_url_same_parent_still_collides(self):
        """Resolution must not break genuine duplicate detection."""
        response = make_response("https://example.com/court/list")
        resolved_a = make_request("detail.aspx").resolve_from(response)
        resolved_b = make_request("detail.aspx").resolve_from(response)
        assert resolved_a.deduplication_key == resolved_b.deduplication_key

    def test_explicit_key_survives_resolve_from(self):
        resolved = make_request(
            "detail.aspx", deduplication_key="my-key"
        ).resolve_from(make_response("https://example.com/court/list"))
        assert resolved.deduplication_key == "my-key"

    def test_resolved_speculative_copy_regenerates_its_auto_key(self):
        """Auto-ness survives a speculative() copy.

        speculative() copies the request unchanged, so its key is still
        derived from the unresolved URL and must be regenerated when the
        copy is later resolved.
        """
        speculative = make_request("detail.aspx?id=7").speculative(
            "by_id", 0, 7
        )
        resolved = speculative.resolve_from(
            make_response("https://example.com/court/list")
        )
        direct = make_request("https://example.com/court/detail.aspx?id=7")
        assert resolved.deduplication_key == direct.deduplication_key
