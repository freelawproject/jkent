"""Browser-side submit choreography that has no HTTP counterpart.

What a filled form *submits* is checked against a real browser by the
generative rig (``tests/form_conformance``), which fans one
``Form.submit`` request across every transport and an independent browser
oracle. What is left here is the part that rig cannot see: which DOM path
``prepare_form_submit`` takes to fire the submission. The HTTP transport
just posts ``field_data``, so a wrong branch here is invisible to a
differential comparison — it shows up as a page that never navigates, or
as a bare ``PlaywrightError`` that ``execute_via_navigation`` does not map
to a transient, burning the request with its retry budget untouched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from playwright.async_api import Page, async_playwright

from jkent.common.via import ViaFormSubmit
from jkent.data_types import CSS
from jkent.driver.via_actions import prepare_form_submit

URL = "https://example.gov/target"


@pytest_asyncio.fixture
async def page() -> AsyncIterator[Page]:
    """A blank Chromium page.

    Per-test rather than per-module: the suite pins fixtures to a
    function-scoped event loop, and a browser outliving its loop cannot be
    driven from the next one.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        yield await browser.new_page()
        await browser.close()


class TestSubmitElement:
    """``_SUBMIT_WITH`` must not assume the submitter is form-associated."""

    @pytest.mark.parametrize(
        ("markup", "selector"),
        [
            (
                "<input type='submit' name='go' value='Search'>",
                "input[name=go]",
            ),
            ("<a id='lb' href='javascript:__doPostBack()'>Search</a>", "a#lb"),
            ("<div id='lb' role='button'>Search</div>", "div#lb"),
        ],
        ids=["submit input", "LinkButton anchor", "div"],
    )
    async def test_a_non_form_element_submits_without_a_type_error(
        self, page: Page, markup: str, selector: str
    ) -> None:
        """``btn.form`` is null for an ``<a>``/``<div>`` submit selector.

        ASP.NET ``LinkButton`` s render as anchors and are plausible
        ``submit_selector`` values. Dereferencing the null raised a
        ``TypeError`` as a bare ``PlaywrightError``, which escapes
        ``execute_via_navigation``'s transient mapping and burns the request.
        """
        await page.set_content(f"<form id='f'>{markup}</form>")
        via = ViaFormSubmit(
            form_selector=CSS("form#f"),
            field_data={},
            submit_selector=selector,
            description="search form",
        )
        submit = await prepare_form_submit(page, via, URL)
        # Stub the real submit: a navigation here would leave the page.
        await page.evaluate(
            "() => { const f = document.querySelector('form#f');"
            " f.submit = () => { window.__submitted = true; };"
            " f.requestSubmit = () => { window.__submitted = true; }; }"
        )

        await submit()

        assert await page.evaluate("() => window.__submitted") is True


class TestEventTargetBranch:
    """``__EVENTTARGET`` routes to a bare ``form.submit()`` only when set."""

    _WEBFORMS = (
        "<input type='hidden' name='__VIEWSTATE' value='abc'>"
        "<input type='hidden' name='__EVENTTARGET' value=''>"
        "<input type='submit' name='btnSearch' value='Search'>"
    )

    async def _route(self, page: Page, event_target: str) -> str:
        """Return which submit path ``prepare_form_submit`` chose."""
        await page.set_content(f"<form id='f'>{self._WEBFORMS}</form>")
        via = ViaFormSubmit(
            form_selector=CSS("form#f"),
            field_data={
                "__VIEWSTATE": "abc",
                "__EVENTTARGET": event_target,
                "btnSearch": "Search",
            },
            submit_selector=None,
            description="search form",
        )
        submit = await prepare_form_submit(page, via, URL)
        await page.evaluate(
            "() => { const f = document.querySelector('form#f');"
            " f.submit = () => { window.__via = 'submit'; };"
            " f.requestSubmit = (b) => {"
            "   window.__via = 'requestSubmit:' + (b && b.name); }; }"
        )
        await submit()
        return await page.evaluate("() => window.__via")

    async def test_rendered_but_empty_uses_the_submit_button(
        self, page: Page
    ) -> None:
        """A WebForms page always renders an empty ``__EVENTTARGET``.

        ``Form.submit`` harvests every rendered field, so the hidden input is
        always in ``field_data``. Keying on its presence sent a real button
        submit down the postback path: the server got an empty event target
        and no ``btnSearch=Search`` and re-rendered the same page, while the
        HTTP transport posted the button and got results.
        """
        assert await self._route(page, "") == "requestSubmit:btnSearch"

    async def test_a_set_event_target_posts_back(self, page: Page) -> None:
        """A real ``__doPostBack`` target still gets the bare submit."""
        assert await self._route(page, "ctl00$grid") == "submit"
