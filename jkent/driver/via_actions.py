"""Fire a request's ``via`` on a Playwright page.

The click/submit choreography that turns a stored ``via``
(:class:`~jkent.common.via.ViaLink` / :class:`~jkent.common.via.ViaFormSubmit`)
back into a browser action: wait for the element, fill the form, click/submit,
wait for the navigation. Extracted from ``PlaywrightTransport`` so hosts that
re-stage requests outside a run (an inspector, say) drive the exact same
logic instead of carrying a copy — the form-submit branch priority (explicit
``submit_selector`` → ``requestSubmit(submitter)``; ``__EVENTTARGET`` → bare
``form.submit()``; default submit) is load-bearing for ASP.NET-style postbacks
and must not drift between them.

Everything here is page-level: no transport, DB, or worker state. The one
DB-adjacent helper, :func:`serve_cached_parent`, takes the parent response as
already-decompressed values; callers own the read (the transport reads the run
DB, a host reads whatever view it keeps).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

from jkent.common.exceptions import TransientException, TransientKind
from jkent.common.page_element import ViaFormSubmit, ViaLink

if TYPE_CHECKING:
    from playwright.async_api import ElementHandle, Page

    from jkent.data_types import FieldResolver, FieldValue, Selector


def selector_for_playwright(selector: Selector) -> str:
    """Selector string for Playwright, engine-prefixed from its grammar.

    Thin alias for :meth:`Selector.for_playwright`.
    """
    return selector.for_playwright()


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
    await page.route(url, _intercept_handler, times=1)
    try:
        await page.goto(url, wait_until="domcontentloaded", **goto_kwargs)
    finally:
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
    only in error messages. ``timeout_ms`` bounds the element wait, the
    click and the navigation wait.

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
                page,
                selector_for_playwright(via.selector),
                request_url,
                timeout_ms=timeout_ms,
            )
            # Strip target=_blank: on Chromium (no open_newwindow pref) it
            # would navigate a new tab, leaking it AND hanging the
            # expect_navigation that's watching this page.
            await element.evaluate("el => el.removeAttribute('target')")
            async with page.expect_navigation(**expect_kwargs) as nav_info:
                await element.click(**click_kwargs)
            response = await nav_info.value
            return response.status if response else None
        elif isinstance(via, ViaFormSubmit):
            submit = await prepare_form_submit(
                page, via, request_url, timeout_ms=timeout_ms
            )
            async with page.expect_navigation(**expect_kwargs) as nav_info:
                await submit()
            response = await nav_info.value
            return response.status if response else None
        else:
            raise ValueError(
                f"via-navigation requires ViaLink or ViaFormSubmit, "
                f"got {type(via)}"
            )
    except PlaywrightTimeoutError as exc:
        raise TransientException(
            f"Navigation timeout: {request_url}",
            url=request_url,
            kind=TransientKind.NAVIGATION,
        ) from exc
    except PlaywrightError as exc:
        # Firefox reports an aborted load under two distinct codes:
        # NS_ERROR_ABORT, and NS_BINDING_ABORTED ("maybe frame was detached?")
        # when the abort came from the necko binding — e.g. window.stop(), or a
        # competing navigation the page itself started. Both mean "this attempt
        # was cut short", not "this request can never succeed", so both must
        # map to a transient. Matching only NS_ERROR_ABORT let NS_BINDING_ABORTED
        # escape the whole exception taxonomy and land in the worker's generic
        # handler, which marks the request failed with no retry left to spend.
        msg = str(exc)
        if "NS_ERROR_ABORT" in msg or "NS_BINDING_ABORTED" in msg:
            raise TransientException(
                f"Navigation aborted: {request_url}",
                url=request_url,
                kind=TransientKind.NAVIGATION,
            ) from exc
        raise


#: Submit the form with ``btn`` as its submitter. ``requestSubmit`` puts
#: the button's name/value in the POST — ASP.NET uses it to identify the
#: event source (which row's Select) — but accepts only a real submit
#: button; a submit control ``fill_form_fields`` re-synthesized as a hidden
#: input already carries its name/value as a field, so a bare submit sends
#: the same data.
_SUBMIT_WITH = """(btn) => {
    if (btn.type === 'submit' || btn.type === 'image') {
        btn.form.requestSubmit(btn);
    } else {
        btn.form.submit();
    }
}"""


async def prepare_form_submit(
    page: Page,
    via: ViaFormSubmit,
    request_url: str,
    *,
    timeout_ms: float | None = None,
) -> Callable[[], Awaitable[None]]:
    """Fill ``via``'s form and return the action that submits it.

    Shared by the navigation path and the archive-download path, which
    differ only in what they wait for after submitting. The form's
    ``target`` is stripped first: a new-tab submit leaks a tab and hangs
    the navigation wait on Chromium.

    Branch priority is load-bearing: an explicit ``submit_selector`` wins
    over ``__EVENTTARGET``. The page's hidden ``__EVENTTARGET`` input is
    harvested into ``field_data`` (empty) during form-field collection, so
    keying on its mere presence would route a button submit (e.g. a
    grid-row Select) to a bare ``form.submit()`` with an empty event target
    — the server then re-renders the same page instead of navigating.
    Without a ``submit_selector`` and with ``__EVENTTARGET`` set, it is an
    ASP.NET ``__doPostBack``, which a bare ``form.submit()`` fires.

    ``timeout_ms`` bounds the wait for the form.
    """
    form = await wait_for_required_element(
        page,
        selector_for_playwright(via.form_selector),
        request_url,
        timeout_ms=timeout_ms,
    )
    await fill_form_fields(form, via.field_data)
    await form.evaluate("el => el.removeAttribute('target')")
    if via.submit_selector or "__EVENTTARGET" not in via.field_data:
        selector = via.submit_selector or (
            'button[type="submit"], input[type="submit"]'
        )
        submitter = await form.query_selector(selector)
        if submitter is None:
            raise TransientException(
                f"Submit element {selector!r} not found in form "
                f"{via.form_selector}",
                url=request_url,
                kind=TransientKind.INTERACTION,
            )

        async def submit() -> None:
            await submitter.evaluate(_SUBMIT_WITH)

        return submit

    async def post_back() -> None:
        await form.evaluate("(form) => form.submit()")

    return post_back


async def wait_for_required_element(
    page: Page,
    selector: str,
    request_url: str,
    *,
    timeout_ms: float | None = None,
) -> ElementHandle:
    """Wait for a required selector; a miss/timeout is a transient.

    ``timeout_ms`` is the request's deadline; ``None`` leaves Playwright's
    default in place.
    """
    wait_kwargs: dict[str, Any] = {}
    if timeout_ms is not None:
        wait_kwargs["timeout"] = timeout_ms
    try:
        element = await page.wait_for_selector(selector, **wait_kwargs)
    except PlaywrightTimeoutError as exc:
        raise TransientException(
            f"Selector timeout: {selector} ({request_url})",
            url=request_url,
            kind=TransientKind.INTERACTION,
        ) from exc
    if element is None:
        raise TransientException(
            f"Selector not found: {selector} ({request_url})",
            url=request_url,
            kind=TransientKind.INTERACTION,
        )
    return element


async def fill_form_fields(
    form: ElementHandle,
    field_data: dict[str, FieldValue | FieldResolver],
) -> None:
    """Populate form fields by name based on tag/type/visibility.

    ``fill`` only works on visible, editable inputs, so selects and
    radio/checkbox groups are set to exactly the given values
    (:func:`select_values`, :func:`set_checked_group`), and hidden/invisible
    inputs (e.g. ASP.NET ``__VIEWSTATE`` or Telerik 1px parents) assign
    ``.value`` directly via JS.

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
            # browser submits them too, as the HTTP transport does.
            await append_hidden_input(form, name, str(value))
            continue
        tag = await field.evaluate("el => el.tagName.toLowerCase()")
        input_type = await field.get_attribute("type")
        str_value = str(value)

        if tag == "select":
            await select_values(field, [str_value])
        elif input_type in ("radio", "checkbox"):
            await set_checked_group(form, name, [str_value])
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
    group is set to exactly ``values``. The fallback covers
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
        await select_values(field, str_values)
    elif input_type in ("radio", "checkbox"):
        await set_checked_group(form, name, str_values)
    else:
        elements = await form.query_selector_all(f'[name="{name}"]')
        for element, str_value in zip(elements, str_values):
            await element.evaluate("(el, val) => el.value = val", str_value)


_ADD_MISSING_OPTIONS = """(el, values) => {
    for (const v of values) {
        if (!Array.from(el.options).some((o) => o.value === v)) {
            el.add(new Option(v, v));
        }
    }
}"""


async def select_values(select: ElementHandle, values: list[str]) -> None:
    """Select exactly *values* on a ``<select>``.

    ``field_data`` is what the HTTP transport posts, and it posts a value no
    option carries as readily as any other. So such a value gets an
    ``<option>`` appended first, as a name with no rendered control gets a
    hidden input; ``select_option`` would otherwise wait out its timeout
    and surface as a transient, retried to budget.
    """
    await select.evaluate(_ADD_MISSING_OPTIONS, values)
    await select.select_option(value=values)


_SET_CHECKED_GROUP = """(form, args) => {
    const boxes = Array.from(form.elements).filter(
        (el) => el.name === args.name
            && (el.type === "radio" || el.type === "checkbox")
    );
    for (const el of boxes) {
        el.checked = args.values.includes(el.value);
    }
    const unsent = [...args.values];
    for (const el of boxes) {
        const i = el.checked && !el.disabled ? unsent.indexOf(el.value) : -1;
        if (i >= 0) {
            unsent.splice(i, 1);
        }
    }
    return unsent;
}"""


async def set_checked_group(
    form: ElementHandle, name: str, values: list[str]
) -> None:
    """Check exactly the ``name=`` radios/checkboxes whose value is in *values*.

    ``field_data`` is what the HTTP transport posts, so a rendered box the
    override left out is unchecked, and a value no enabled box submits (no
    box carries it, or a radio group was given two) goes as a hidden input,
    as a name with no rendered control does. Matching is on the DOM
    ``value``, so a box with no ``value`` attribute matches ``"on"``.
    """
    unsent: list[str] = await form.evaluate(
        _SET_CHECKED_GROUP, {"name": name, "values": values}
    )
    for value in unsent:
        await append_hidden_input(form, name, value)


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
