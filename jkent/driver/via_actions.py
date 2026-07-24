"""Fire a request's ``via`` on a Playwright page.

The click/submit choreography that turns a stored ``via``
(:class:`~jkent.data_types.ViaLink` / :class:`~jkent.data_types.ViaFormSubmit`)
back into a browser action: wait for the element, fill the form, click/submit,
wait for the navigation. Extracted from ``PlaywrightTransport`` so hosts that
re-stage requests outside a run (jent's MCP inspector) drive the exact same
logic instead of carrying a copy — the form-submit branch priority (explicit
``submit_selector`` → ``requestSubmit(submitter)``; ``__EVENTTARGET`` → bare
``form.submit()``; default submit) is load-bearing for ASP.NET-style postbacks
and must not drift between them.

Everything here is page-level: no transport, DB, or worker state. The one
DB-adjacent helper, :func:`serve_cached_parent`, takes the parent response as
already-decompressed values; callers own the read (the transport reads the run
DB, jent reads its graph views).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.exceptions import TransientException
from jkent.common.page_element import ViaFormSubmit, ViaLink

if TYPE_CHECKING:
    from playwright.async_api import ElementHandle, Page

    from jkent.data_types import FieldResolver, FieldValue, Selector


def selector_for_playwright(selector: Selector) -> str:
    """Selector string for Playwright, forcing the xpath engine by grammar.

    ``Selector.nth`` wraps xpath as ``(...)[n]`` and css as ``:nth-match(...)``;
    the parenthesized xpath form is NOT auto-detected by Playwright as xpath,
    so prefix ``xpath=`` from the stored grammar rather than relying on
    auto-detection.
    """
    if selector.grammar == "xpath":
        return f"xpath={selector.value}"
    return selector.value


async def serve_cached_parent(
    page: Page,
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    status: int,
    timeout_ms: float | None = None,
) -> None:
    """Serve a cached response into the tab via route intercept.

    Navigates ``page`` to ``url`` while a one-shot route fulfills it with the
    given body/headers/status, so the tab sits on the real origin showing the
    cached document (ready for a via click/submit). A ``content-type`` header
    is defaulted to HTML when absent. ``timeout_ms`` bounds the staging goto;
    ``None`` leaves Playwright's default in place.
    """
    if "content-type" not in {k.lower() for k in headers}:
        headers = {**headers, "content-type": "text/html; charset=utf-8"}

    async def _intercept_handler(route: Any) -> None:
        await route.fulfill(status=status, headers=headers, body=body)

    goto_kwargs: dict[str, Any] = {}
    if timeout_ms is not None:
        goto_kwargs["timeout"] = timeout_ms
    await page.route(url, _intercept_handler)
    await page.goto(url, wait_until="domcontentloaded", **goto_kwargs)
    await page.unroute(url, _intercept_handler)


async def execute_via_navigation(
    page: Page,
    via: ViaLink | ViaFormSubmit,
    request_url: str,
    *,
    timeout_ms: float | None = None,
) -> int | None:
    """Click/submit the via element and wait for the resulting navigation.

    Expects a navigation, not a download (the download counterpart lives in
    the transport's archive path). A missing element or a navigation
    timeout/abort raises :class:`TransientException`; ``request_url`` is used
    only in error messages. ``timeout_ms`` bounds both the click and the
    navigation wait.

    Returns the navigated document's HTTP status (or ``None`` when Playwright
    surfaces no response, e.g. a same-document navigation).
    """
    expect_kwargs: dict[str, Any] = {}
    click_kwargs: dict[str, Any] = {}
    if timeout_ms is not None:
        expect_kwargs["timeout"] = timeout_ms
        click_kwargs["timeout"] = timeout_ms

    try:
        if isinstance(via, ViaLink):
            element = await wait_for_required_element(
                page, selector_for_playwright(via.selector), request_url
            )
            # Strip target=_blank: on Chromium (no open_newwindow pref) it
            # would navigate a new tab, leaking it AND hanging the
            # expect_navigation that's watching this page.
            await element.evaluate("el => el.removeAttribute('target')")
            async with page.expect_navigation(**expect_kwargs) as nav_info:  # type: ignore
                await element.click(**click_kwargs)
            response = await nav_info.value
            return response.status if response else None
        elif isinstance(via, ViaFormSubmit):
            form = await wait_for_required_element(
                page, selector_for_playwright(via.form_selector), request_url
            )
            await fill_form_fields(form, via.field_data)
            # Strip target=_blank before any submit branch below, same as
            # the ViaLink path — a new-tab submit leaks a tab and hangs
            # expect_navigation on Chromium.
            await form.evaluate("el => el.removeAttribute('target')")
            # Branch priority is load-bearing: an explicit submit_selector
            # wins over __EVENTTARGET. The page's hidden
            # __EVENTTARGET input is harvested into field_data (empty)
            # during form-field collection, so keying on its mere presence
            # wrongly routes a button submit (e.g. a grid-row Select) to a
            # bare form.submit() with an empty event target — the server
            # then re-renders the same page instead of navigating.
            if via.submit_selector:
                submit = await form.query_selector(via.submit_selector)
                if submit is None:
                    raise TransientException(
                        f"Submit selector not found: {via.submit_selector}"
                    )
                # requestSubmit(submitter) (not a bare click) so the
                # button's name/value is reliably in the POST — ASP.NET
                # uses it to identify the event source (which row's Select).
                async with page.expect_navigation(**expect_kwargs) as nav_info:
                    await submit.evaluate(
                        "(btn) => btn.form.requestSubmit(btn)"
                    )
                response = await nav_info.value
                return response.status if response else None
            elif "__EVENTTARGET" in via.field_data:
                # ASP.NET __doPostBack: __EVENTTARGET is set as a hidden
                # field, so a raw form.submit() fires that postback.
                async with page.expect_navigation(**expect_kwargs) as nav_info:
                    await form.evaluate("(form) => form.submit()")
                response = await nav_info.value
                return response.status if response else None
            else:
                submit = await form.query_selector(
                    'button[type="submit"], input[type="submit"]'
                )
                if submit is None:
                    raise TransientException(
                        f"No submit element in form {via.form_selector}"
                    )
                async with page.expect_navigation(**expect_kwargs) as nav_info:
                    await submit.evaluate(
                        "(btn) => btn.form.requestSubmit(btn)"
                    )
                response = await nav_info.value
                return response.status if response else None
        else:
            raise ValueError(
                f"via-navigation requires ViaLink or ViaFormSubmit, "
                f"got {type(via)}"
            )
    except PlaywrightTimeoutError as exc:
        raise TransientException(f"Navigation timeout: {request_url}") from exc
    except PlaywrightError as exc:
        if "NS_ERROR_ABORT" in str(exc):
            raise TransientException(
                f"Navigation aborted: {request_url}"
            ) from exc
        raise


async def wait_for_required_element(
    page: Page, selector: str, request_url: str
) -> ElementHandle:
    """Wait for a required selector; a miss/timeout is a transient."""
    try:
        element = await page.wait_for_selector(selector, timeout=5000)
    except PlaywrightTimeoutError as exc:
        raise TransientException(
            f"Selector timeout: {selector} ({request_url})"
        ) from exc
    if element is None:
        raise TransientException(
            f"Selector not found: {selector} ({request_url})"
        )
    return element


async def fill_form_fields(
    form: ElementHandle,
    field_data: dict[str, FieldValue | FieldResolver],
) -> None:
    """Populate form fields by name based on tag/type/visibility.

    ``fill`` only works on visible, editable inputs, so selects use
    ``select_option``, radios/checkboxes set ``checked`` on the matching
    value, and hidden/invisible inputs (e.g. ASP.NET ``__VIEWSTATE`` or
    Telerik 1px parents) assign ``.value`` directly via JS.

    A list value means repeated keys (checkbox groups, multi-selects): each
    member selects/checks the option with the matching value, mirroring what
    the browser POSTs as repeated names.
    """
    for name, value in field_data.items():
        if callable(value):
            # Resolvers are awaited at enqueue time
            # (Request.resolve_deferred_fields); one surviving to the
            # browser means a request bypassed that path.
            raise TypeError(
                f"unresolved field resolver for {name!r} reached the browser"
            )
        if isinstance(value, list):
            await fill_repeated_field(form, name, value)
            continue
        field = await form.query_selector(f'[name="{name}"]')
        if field is None:
            # No rendered control for this name. ViaFormSubmit can carry
            # fields the form never showed (the merged overrides a scraper
            # passed to ``Form.submit``); inject a hidden input so the
            # browser submits them too, instead of silently dropping them.
            # The HTTP transport sends these verbatim, so this keeps the
            # browser path in step with it.
            await append_hidden_input(form, name, str(value))
            continue
        tag = await field.evaluate("el => el.tagName.toLowerCase()")
        input_type = await field.get_attribute("type")
        str_value = str(value)

        if tag == "select":
            await field.select_option(value=str_value)
        elif input_type == "radio":
            radio = await form.query_selector(
                f'[name="{name}"][value="{str_value}"]'
            )
            if radio is not None:
                await radio.evaluate("(el) => el.checked = true")
        elif input_type == "checkbox":
            checkbox = await form.query_selector(
                f'[name="{name}"][value="{str_value}"]'
            )
            if checkbox is not None:
                await checkbox.evaluate("(el) => el.checked = true")
            else:
                await field.evaluate(
                    "(el, val) => el.checked = !!val", str_value
                )
        elif input_type in ("hidden", "submit") or not (
            await field.is_visible()
        ):
            await field.evaluate("(el, val) => el.value = val", str_value)
        else:
            await field.fill(str_value)


async def fill_repeated_field(
    form: ElementHandle, name: str, values: list[str]
) -> None:
    """Replay a repeated-key field (checkbox group or multi-select).

    A ``<select multiple>`` selects all matching options at once; a checkbox
    group checks each box whose value is in ``values``. The fallback covers
    repeated text/hidden inputs (rare), assigning each value positionally to
    the matching ``name=`` elements in document order.
    """
    str_values = [str(v) for v in values]
    field = await form.query_selector(f'[name="{name}"]')
    if field is None:
        # No rendered control: inject one hidden input per value so a
        # repeated key absent from the DOM still reaches the server as
        # repeated names (see ``fill_form_fields`` for the rationale).
        for str_value in str_values:
            await append_hidden_input(form, name, str_value)
        return
    tag = await field.evaluate("el => el.tagName.toLowerCase()")
    input_type = await field.get_attribute("type")

    if tag == "select":
        await field.select_option(value=str_values)
    elif input_type in ("radio", "checkbox"):
        for str_value in str_values:
            box = await form.query_selector(
                f'[name="{name}"][value="{str_value}"]'
            )
            if box is not None:
                await box.evaluate("(el) => el.checked = true")
    else:
        elements = await form.query_selector_all(f'[name="{name}"]')
        for element, str_value in zip(elements, str_values):
            await element.evaluate("(el, val) => el.value = val", str_value)


async def append_hidden_input(
    form: ElementHandle, name: str, value: str
) -> None:
    """Append a ``<input type=hidden name=value>`` to ``form``.

    Used when ``field_data`` carries a name the rendered form never showed,
    so the submitted request matches what the HTTP transport sends. Name and
    value are passed as JS arguments (not interpolated), so any characters
    are safe.
    """
    await form.evaluate(
        """(form, args) => {
            const input = document.createElement('input');
            input.type = 'hidden';
            input.name = args.name;
            input.value = args.value;
            form.appendChild(input);
        }""",
        {"name": name, "value": value},
    )
