"""Permanent Request Data.

This module tests the permanent dict on Request for persisting cookies
and headers across the request chain.

Key behaviors tested (all at the data-type level — no driver):
- permanent headers/cookies are merged into HTTPRequestParams at construction
- permanent data is inherited from parent to child through resolve_from
- the "headers" and "cookies" entries merge by inner key (child wins on
  conflicts, parent entries survive otherwise); other top-level keys are
  last-writer-wins
- permanent dicts are deep copied to isolate sibling requests

The driver-level proof that permanent headers/cookies actually go out over
the wire lives in tests/driver/unified/test_data_types_e2e.py against the
unified driver.
"""

from jkent.data_types import (
    HttpMethod,
    HTTPRequestParams,
    Request,
    Response,
)


def _request(
    url: str = "http://bugcourt.example.com/test",
    *,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    permanent: dict | None = None,
    continuation: str = "parse",
) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url=url,
            headers=headers,
            cookies=cookies,
        ),
        continuation=continuation,
        permanent=permanent or {},
    )


def _response_for(request: Request) -> Response:
    return Response(
        status_code=200,
        headers={},
        content=b"",
        text="",
        url=request.request.url,
        request=request,
    )


class TestPermanentMergeAtConstruction:
    """permanent headers/cookies are merged into the request params."""

    def test_permanent_headers_merged_into_request(self):
        """The permanent headers shall be merged into HTTPRequestParams."""
        request = _request(
            permanent={"headers": {"Authorization": "Bearer token123"}},
        )

        assert request.request.headers is not None
        assert request.request.headers["Authorization"] == "Bearer token123"

    def test_request_headers_override_permanent_headers(self):
        """An explicit per-request header shall win over the permanent one."""
        request = _request(
            headers={"Authorization": "Bearer fresh", "Accept": "text/html"},
            permanent={"headers": {"Authorization": "Bearer stale"}},
        )

        assert request.request.headers is not None
        # The per-request header wins the conflict...
        assert request.request.headers["Authorization"] == "Bearer fresh"
        # ...and the permanent base still contributes its other keys.
        assert request.request.headers["Accept"] == "text/html"

    def test_permanent_cookies_merged_into_request(self):
        """The permanent cookies shall be merged into HTTPRequestParams."""
        request = _request(
            cookies={"theme": "dark"},
            permanent={"cookies": {"session": "xyz789"}},
        )

        assert request.request.cookies == {
            "theme": "dark",
            "session": "xyz789",
        }

    def test_request_cookies_override_permanent_cookies(self):
        """An explicit per-request cookie shall win over the permanent one."""
        request = _request(
            cookies={"session": "fresh", "theme": "dark"},
            permanent={"cookies": {"session": "stale"}},
        )

        assert request.request.cookies == {
            "session": "fresh",
            "theme": "dark",
        }

    def test_permanent_supports_headers_and_cookies_simultaneously(self):
        """The permanent dict shall support both headers and cookies at once."""
        request = _request(
            permanent={
                "headers": {"Authorization": "Bearer abc"},
                "cookies": {"session": "xyz"},
            },
        )

        assert request.request.headers is not None
        assert request.request.headers["Authorization"] == "Bearer abc"
        assert request.request.cookies == {"session": "xyz"}

    def test_empty_permanent_leaves_request_untouched(self):
        """An empty permanent dict shall not alter the request params."""
        request = _request(headers={"Accept": "text/html"})

        assert request.request.headers == {"Accept": "text/html"}
        assert request.request.cookies is None


class TestPermanentInheritance:
    """permanent data flows from parent to child through resolve_from."""

    def test_child_inherits_parent_permanent(self):
        """A child without permanent shall inherit the parent's."""
        parent = _request(
            permanent={"headers": {"X-Session": "abc123"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2", continuation="parse_step2"
        ).resolve_from(_response_for(parent))

        assert child.permanent == {"headers": {"X-Session": "abc123"}}
        assert child.request.headers is not None
        assert child.request.headers["X-Session"] == "abc123"

    def test_permanent_persists_across_two_hops(self):
        """permanent set once shall reach a grandchild request unchanged."""
        login = _request(
            url="/api/data",
            continuation="parse_api",
            permanent={"headers": {"Authorization": "Bearer login-token"}},
        ).resolve_from(
            _response_for(_request(url="/login", continuation="parse_login"))
        )

        more = _request(
            url="/api/more", continuation="parse_more"
        ).resolve_from(_response_for(login))

        assert more.permanent == {
            "headers": {"Authorization": "Bearer login-token"}
        }
        assert more.request.headers is not None
        assert more.request.headers["Authorization"] == "Bearer login-token"

    def test_permanent_cookies_inherited_by_children(self):
        """The permanent cookies shall be inherited by child requests."""
        parent = _request(
            permanent={"cookies": {"user_id": "12345"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2", continuation="parse_step2"
        ).resolve_from(_response_for(parent))

        assert child.permanent == {"cookies": {"user_id": "12345"}}
        assert child.request.cookies == {"user_id": "12345"}


class TestPermanentMerging:
    """parent and child permanent dicts merge per top-level key."""

    def test_child_header_wins_over_same_named_parent_header(self):
        """On an inner-key conflict, the child's value shall win."""
        parent = _request(
            permanent={"headers": {"X-Token": "old-token"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2",
            continuation="parse_step2",
            permanent={"headers": {"X-Token": "new-token"}},
        ).resolve_from(_response_for(parent))

        assert child.permanent == {"headers": {"X-Token": "new-token"}}
        assert child.request.headers is not None
        assert child.request.headers["X-Token"] == "new-token"

    def test_parent_headers_survive_child_adding_headers(self):
        """The parent's permanent headers shall survive a child's additions.

        Establishing auth once and layering per-request headers later is
        the whole point of permanent: a child that adds X-Requested-With
        must not silently drop the chain's Authorization header (the
        symptom would be a 401 several steps deep).
        """
        parent = _request(
            permanent={"headers": {"Authorization": "Bearer abc"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2",
            continuation="parse_step2",
            permanent={"headers": {"X-Requested-With": "XMLHttpRequest"}},
        ).resolve_from(_response_for(parent))

        assert child.permanent == {
            "headers": {
                "Authorization": "Bearer abc",
                "X-Requested-With": "XMLHttpRequest",
            }
        }
        assert child.request.headers is not None
        assert child.request.headers["Authorization"] == "Bearer abc"
        assert child.request.headers["X-Requested-With"] == "XMLHttpRequest"

    def test_parent_cookies_survive_child_adding_cookies(self):
        """The parent's permanent cookies shall survive a child's additions."""
        parent = _request(
            permanent={"cookies": {"session": "xyz789"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2",
            continuation="parse_step2",
            permanent={"cookies": {"page_pref": "50"}},
        ).resolve_from(_response_for(parent))

        assert child.permanent == {
            "cookies": {"session": "xyz789", "page_pref": "50"}
        }
        assert child.request.cookies == {
            "session": "xyz789",
            "page_pref": "50",
        }

    def test_disjoint_keys_merge_across_generations(self):
        """Parent headers and child cookies shall both survive the merge."""
        parent = _request(
            permanent={"headers": {"Authorization": "Bearer abc"}},
        ).resolve_from(_response_for(_request(continuation="parse_entry")))

        child = _request(
            url="/step2",
            continuation="parse_step2",
            permanent={"cookies": {"session": "xyz"}},
        ).resolve_from(_response_for(parent))

        assert child.permanent == {
            "headers": {"Authorization": "Bearer abc"},
            "cookies": {"session": "xyz"},
        }
        assert child.request.headers is not None
        assert child.request.headers["Authorization"] == "Bearer abc"
        assert child.request.cookies == {"session": "xyz"}


class TestPermanentIsolation:
    """permanent dicts are deep copied to prevent cross-branch sharing."""

    def test_siblings_get_independent_permanent_copies(self):
        """Two requests built from one permanent dict shall not share it."""
        shared = {"headers": {"X-Branch": "initial"}}

        branch1 = _request(url="/branch1", permanent=shared)
        branch2 = _request(url="/branch2", permanent=shared)

        assert branch1.permanent is not branch2.permanent
        assert branch1.permanent == branch2.permanent

    def test_mutating_source_dict_does_not_affect_requests(self):
        """Mutations to the source permanent dict shall not propagate."""
        shared = {"headers": {"X-Branch": "initial"}}

        request = _request(permanent=shared)
        shared["headers"]["X-Branch"] = "mutated"

        assert request.permanent == {"headers": {"X-Branch": "initial"}}
