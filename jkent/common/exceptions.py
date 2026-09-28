"""Exception types for scraper errors.

Two roots, dispatched on by the worker: :class:`PersistentException` (the
scraper or the site is wrong in a way retrying cannot fix) and
:class:`TransientException` (retry it). :class:`RequestFailedHalt` is
control flow a scraper raises on purpose, and
:class:`SpeculationHTTPFailure` is the probe outcome the speculation engine
consumes; neither is an error to file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jkent.common.coded_enum import CodedEnum

if TYPE_CHECKING:
    from jkent.common.response import Response

__all__ = [
    "DataFormatAssumptionException",
    "HTMLStructuralAssumptionException",
    "HTTPResponseAssumptionException",
    "IncidentalRequestAssumptionException",
    "InterstitialUnresolved",
    "PersistentException",
    "PersistentHTTPResponseException",
    "RequestFailedHalt",
    "RequestTimeoutException",
    "ResolveTimeout",
    "ScraperAssumptionException",
    "ScraperConfigError",
    "SpeculationHTTPFailure",
    "TransientException",
    "TransientKind",
]


class TransientKind(CodedEnum):
    """What *kind* of transient failure a bare ``TransientException`` is.

    The structured transient subclasses (HTTP status, timeout) fill their own
    columns in the ``errors`` table. Everything else — a dead browser, a
    navigation that aborted, an archive that never downloaded — carries one
    of these, stored in the ``errors.kind`` column, so failures can be
    grouped by subsystem without matching on ``message``.

    Deliberately coarse: it names the *subsystem that failed*, not the
    specific failure (the message still does that). A kind earns its place
    only if you would filter or group by it.
    """

    #: Network-level failure below HTTP: connection reset, DNS, protocol.
    NETWORK = (1, "network")
    #: The browser process or its connection died, or cannot be restarted.
    BROWSER_CRASH = (2, "browser_crash")
    #: A navigation did not arrive: aborted, timed out, or never committed.
    NAVIGATION = (3, "navigation")
    #: Driving the page failed: a control the via needed was not there, a
    #: required selector never appeared.
    INTERACTION = (4, "interaction")
    #: An archive download did not produce a file.
    ARCHIVE = (5, "archive")
    #: Snapshotting the resolved page failed.
    SNAPSHOT = (6, "snapshot")
    #: An interstitial (a bot challenge) was still on the page after its
    #: handler exhausted every strategy.
    INTERSTITIAL = (7, "interstitial")
    #: The site answered, but with a page that is transient by content: an
    #: error or maintenance page, a truncated body, or one not yet rendered.
    SITE_DEGRADED = (8, "site_degraded")


class PersistentException(Exception):
    """Errors the server or site state will keep producing on retry.

    Covers three flavors of "no point retrying":

    - Our own assumptions about site structure turn out to be wrong
      (selectors don't match, data doesn't validate) — see
      :class:`ScraperAssumptionException` and its subclasses.
    - A resource that the server once advertised has become unavailable.
    - The server advertising a resource that does not actually exist.

    The worker dispatches on this base class: anything persistent skips the
    retry machinery, is marked failed on the first occurrence, and gets an
    ``errors`` row. :class:`PersistentHTTPResponseException` is caught ahead
    of it only to also persist the observed response body/headers and to
    credit the circuit breaker with the server having answered; every other
    subclass takes the plain arm.
    """

    #: The exchange that produced this failure, where one was observed, so
    #: the worker can persist it to the run db and the failure stays
    #: inspectable. See :attr:`TransientException.debug_response`.
    debug_response: Response | None = None


class ScraperAssumptionException(PersistentException):
    """Base class for scraper assumption violations.

    Scrapers make assumptions about website structure, data formats, and
    navigation patterns. When these assumptions are violated, they should
    raise clear, contextual exceptions that help diagnose the issue.

    This is the base class for all assumption violations. Subclasses should
    provide specific context about what assumption was violated.
    """

    def __init__(
        self,
        message: str,
        request_url: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable description of the assumption violation.
            request_url: The URL of the request that triggered this error.
            context: Optional dict of additional context (selector, counts, etc).
        """
        self.message = message
        self.request_url = request_url
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message with context.

        Returns:
            Formatted error message string.
        """
        parts = [self.message]
        parts.append(f"URL: {self.request_url}")

        if self.context:
            parts.append("Context:")
            for key, value in self.context.items():
                parts.append(f"  {key}: {value}")

        return "\n".join(parts)


class HTMLStructuralAssumptionException(ScraperAssumptionException):
    """Raised when HTML structure doesn't match expectations.

    This exception is raised when XPath or CSS selectors return a different
    number of elements than expected. This usually indicates that the website's
    HTML structure has changed.

    Attributes:
        selector: The XPath or CSS selector that was used.
        selector_type: Type of selector ("xpath" or "css").
        is_element_query: True if querying for elements, False for strings/attributes.
    """

    def __init__(
        self,
        selector: str,
        selector_type: str,
        description: str,
        expected_min: int,
        expected_max: int | None,
        actual_count: int,
        request_url: str,
        is_element_query: bool = True,
    ) -> None:
        """Initialize the exception.

        Args:
            selector: The XPath or CSS selector that was used.
            selector_type: Type of selector ("xpath" or "css").
            description: Human-readable description of what was being selected.
            expected_min: Minimum number of elements expected.
            expected_max: Maximum number of elements expected (None = unlimited).
            actual_count: Actual number of elements found.
            request_url: The URL of the request that triggered this error.
            is_element_query: True if querying for elements (default), False for strings.
        """
        self.selector = selector
        self.selector_type = selector_type
        self.description = description
        self.expected_min = expected_min
        self.expected_max = expected_max
        self.actual_count = actual_count
        self.is_element_query = is_element_query

        # Build expected count string
        if expected_max is None:
            expected_str = f"at least {expected_min}"
        elif expected_min == expected_max:
            expected_str = f"exactly {expected_min}"
        else:
            expected_str = f"between {expected_min} and {expected_max}"

        message = (
            f"HTML structure mismatch: Expected {expected_str} "
            f"elements for '{description}', but found {actual_count}"
        )

        context = {
            "selector": selector,
            "selector_type": selector_type,
            "expected_min": expected_min,
            "expected_max": expected_max
            if expected_max is not None
            else "unlimited",
            "actual_count": actual_count,
            "is_element_query": is_element_query,
        }

        super().__init__(message, request_url, context)


class DataFormatAssumptionException(ScraperAssumptionException):
    """Raised when scraped data doesn't match expected schema.

    This exception is raised during Pydantic validation when the scraped
    data doesn't conform to the expected data model. This indicates that
    the website's data format has changed or the scraper's extraction
    logic needs updating.
    """

    def __init__(
        self,
        errors: list[dict[str, Any]],
        failed_doc: dict[str, Any],
        model_name: str,
        request_url: str,
    ) -> None:
        """Initialize the exception.

        Args:
            errors: List of Pydantic validation errors.
            failed_doc: The document that failed validation.
            model_name: Name of the Pydantic model that was being validated against.
            request_url: The URL of the request that produced this data.
        """
        self.errors = errors
        self.failed_doc = failed_doc
        self.model_name = model_name

        # Build human-readable error summary. Field-level errors carry a
        # loc path; model-level errors (@model_validator) carry an empty
        # loc, so fall back to the model name.
        error_summary = ", ".join(
            f"{'.'.join(str(part) for part in err['loc']) or model_name}:"
            f" {err['msg']}"
            for err in errors
        )

        message = (
            f"Data validation failed for model '{model_name}': {error_summary}"
        )

        context = {
            "model": model_name,
            "error_count": len(errors),
            "errors": errors,
            "failed_doc": failed_doc,
        }

        super().__init__(message, request_url, context)


class IncidentalRequestAssumptionException(ScraperAssumptionException):
    """Raised when a captured incidental request cannot be promoted.

    A request carrying ``incidental=Singular(...)`` expects exactly one of the
    parent navigation's captured sub-requests to match its spec; ``Multiple``
    expects at least one. When the match count violates that (``Singular`` with
    zero or more than one match, ``Multiple`` with zero), the spec no longer
    describes the page's behavior — a structural assumption violation, parallel
    to :class:`HTMLStructuralAssumptionException` for the DOM. It is also
    raised when a matched sub-request got no response at all (the browser
    aborted it, or it failed), since there is nothing to promote.

    Attributes:
        match_spec: repr of the ``Singular``/``Multiple`` spec that was applied.
        candidate_urls: URLs of the incidentals that matched (for debugging an
            over-broad spec).
    """

    def __init__(
        self,
        message: str,
        request_url: str,
        *,
        match_spec: str | None = None,
        candidate_urls: list[str] | None = None,
    ) -> None:
        self.match_spec = match_spec
        self.candidate_urls = candidate_urls or []
        context: dict[str, Any] = {}
        if match_spec is not None:
            context["match_spec"] = match_spec
        context["match_count"] = len(self.candidate_urls)
        if self.candidate_urls:
            context["candidate_urls"] = self.candidate_urls
        super().__init__(message, request_url, context)


class TransientException(Exception):
    """Base class for transient errors that might resolve on retry.

    Transient exceptions represent temporary failures like network issues,
    server errors (5xx), or timeouts. Unlike assumption exceptions which
    indicate scraper code needs updating, transient exceptions suggest
    retrying the request may succeed.

    The driver is responsible for retry logic and strategy.

    Subclasses that carry real structure (an HTTP status, a deadline) fill
    their own ``errors`` columns and pass ``kind=None``; a subclass without
    such columns supplies its own kind. Raising this base directly requires
    ``kind`` — and takes ``url`` where the failure belongs to a request — so
    the row says which subsystem failed and what it was working on, rather
    than only carrying a message to grep.
    """

    def __init__(
        self,
        message: str,
        *,
        url: str | None = None,
        kind: TransientKind | None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: What went wrong, in prose.
            url: The request this failure belongs to, where there is one.
                ``None`` for failures outside a request (a restart that
                could not run, a handle checked out of band).
            kind: Which subsystem failed. Required, so a direct raise
                cannot omit it; ``None`` only from a structured subclass
                whose own columns describe the failure.
        """
        self.url = url
        self.kind = kind
        self.message = message
        super().__init__(message)

    #: The exchange that produced this failure, where one was observed —
    #: a classified HTTP error's status/headers/body, or the partial DOM a
    #: browser transport snapshotted before giving up on a timeout. The
    #: worker stores it before retrying so the *latest* failed attempt is
    #: inspectable in the run db (a later attempt overwrites it). ``None``
    #: for failures with nothing to show: a connection reset, a DNS
    #: failure, a dead browser.
    debug_response: Response | None = None

    #: Server-sent ``Retry-After`` in seconds, parsed and clamped by the
    #: transport at classification time (see
    #: ``jkent.driver.unified_driver.transport.parse_retry_after``). Two
    #: consumers: the retry scheduler floors this request's backoff at it,
    #: and the rate limiter treats it as a global pause-and-slow-down
    #: signal. ``None`` when the response carried no such header.
    retry_after: float | None = None


class HTTPResponseAssumptionException(TransientException):
    """Raised when HTTP response has unexpected status code.

    This exception indicates the server returned a status code we didn't
    expect. Server errors (5xx) are transient, but client errors (4xx)
    might indicate a permanent problem.

    Attributes:
        status_code: The actual HTTP status code received.
        expected_codes: Status codes the scraper classifies as successful.
        url: The URL that returned the unexpected status.
        debug_response: The observed exchange, where the transport had one
            to give (see :attr:`TransientException.debug_response`).
        message: Human-readable error message.
    """

    def __init__(
        self,
        status_code: int,
        expected_codes: list[int],
        url: str,
        *,
        debug_response: Response | None = None,
        retry_after: float | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            status_code: The actual status code received.
            expected_codes: Status codes the scraper treats as successful.
            url: The URL of the request.
            debug_response: The observed exchange, if the transport had one.
            retry_after: Clamped ``Retry-After`` seconds, where the server
                sent the header (see
                :attr:`TransientException.retry_after`).
        """
        self.status_code = status_code
        self.expected_codes = expected_codes
        self.url = url
        self.debug_response = debug_response
        self.retry_after = retry_after

        expected_str = ", ".join(str(code) for code in expected_codes)
        super().__init__(
            f"HTTP {status_code} from {url} (expected one of: {expected_str})",
            url=url,
            kind=None,
        )


class RequestTimeoutException(TransientException):
    """Raised when a request times out.

    This exception indicates the request took longer than the configured
    timeout. Network issues or slow servers can cause timeouts. Retrying
    may succeed.

    Attributes:
        url: The URL that timed out.
        timeout_seconds: The timeout duration in seconds.
        message: Human-readable error message.
    """

    def __init__(
        self,
        url: str,
        timeout_seconds: float,
        *,
        message: str | None = None,
        debug_response: Response | None = None,
    ) -> None:
        """Initialize the exception.

        Args:
            url: The URL that timed out.
            timeout_seconds: The timeout duration in seconds.
            message: Overrides the default message, for subclasses that can
                say something more specific about *where* it timed out.
            debug_response: Whatever was observed before giving up.
        """
        self.timeout_seconds = timeout_seconds
        self.debug_response = debug_response
        super().__init__(
            message or f"Request to {url} timed out after {timeout_seconds}s",
            url=url,
            kind=None,
        )


class InterstitialUnresolved(TransientException):
    """An interstitial handler exhausted its strategies without clearing.

    Raised by a handler's ``navigate_through`` when the challenge is still
    on the page after everything it knows how to try. Retryable — a
    challenge that beat one attempt often yields to the next. Its own class,
    so the ``errors`` row's ``error_class`` tells "the challenge never
    cleared" apart from "the scraper's own await condition timed out".

    The browser transport catches this alongside Playwright's own timeout,
    so the partial DOM is still snapshotted and persisted before the retry
    — inspecting the challenge that would not clear is the whole point.

    Its ``kind`` is always :attr:`TransientKind.INTERSTITIAL`.
    """

    def __init__(self, message: str, *, url: str | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Which challenge was left and what was tried.
            url: The request whose navigation hit the challenge, where known.
        """
        super().__init__(message, url=url, kind=TransientKind.INTERSTITIAL)


class ResolveTimeout(RequestTimeoutException):
    """A browser resolve timed out, carrying the partial DOM it snapshotted.

    A navigation or await condition ran out of time, but the page was still
    snapshotted before giving up — ``debug_response`` carries that partial
    DOM so the failed attempt is inspectable (e.g. a Cloudflare interstitial
    that never cleared) before the retry.

    A :class:`RequestTimeoutException` and not a bare
    :class:`TransientException` so that ``url``/``timeout_seconds`` land in
    the errors table's own columns, like every other timeout, instead of
    being recoverable only by substring-matching ``message``.
    """


class PersistentHTTPResponseException(PersistentException):
    """HTTP status classified as persistent per scraper policy.

    Raised by the request manager when
    ``scraper.classify(status_code, headers, content)`` returns
    ``HTTPCodeType.PERSISTENT``. Does not inherit from
    :class:`TransientException`; the worker's retry machinery is skipped
    and the request is marked failed on the first occurrence.

    Attributes:
        status_code: The HTTP status code received.
        url: The URL of the request.
        debug_response: The observed exchange, where the transport had one
            to give (see :attr:`TransientException.debug_response`) — the
            403 block page a run db should keep, not just its status code.
        retry_after: Clamped ``Retry-After`` seconds, where the server sent
            the header (see :attr:`TransientException.retry_after`). Present
            here too because a persistently-classified 429 is still rate
            feedback, even though this request will not be retried.
    """

    def __init__(
        self,
        status_code: int,
        url: str,
        *,
        debug_response: Response | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.debug_response = debug_response
        self.retry_after = retry_after
        self.message = f"HTTP {status_code} from {url} (persistent)"
        super().__init__(self.message)


class SpeculationHTTPFailure(Exception):
    """A persistent HTTP code came back for a speculative request.

    Raised by the transport as an alternative to
    :class:`PersistentHTTPResponseException` when ``request.is_speculative``
    is True. The worker records it as a speculation miss: it bumps the
    template's ``consecutive_failures``, stores the probe's response with
    ``speculation_outcome`` ``miss`` (or ``stopped``, when this miss stops
    the template), marks the request completed, skips the step, and does
    NOT write to the ``errors`` table — the "this speculative probe turned
    up nothing" signal, not an error.

    Deliberately does not inherit from :class:`PersistentException` or
    :class:`TransientException`: neither bucket fits (it's neither an
    error to log nor something to retry), and keeping it separate lets
    the worker dispatch on it without catching it in the general
    persistent / transient branches.

    Attributes:
        status_code: The HTTP status code received.
        url: The URL of the speculative request.
        debug_response: The observed exchange, where the transport had one
            to give (see :attr:`TransientException.debug_response`) — the
            miss's real body, stored with its outcome. ``None`` where the
            transport observed no body; the worker then stores the status
            alone.
        retry_after: Clamped ``Retry-After`` seconds, where the server sent
            the header — a throttled probe is rate feedback like any other
            (see :attr:`TransientException.retry_after`).
    """

    def __init__(
        self,
        status_code: int,
        url: str,
        *,
        debug_response: Response | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.debug_response = debug_response
        self.retry_after = retry_after
        self.message = (
            f"HTTP {status_code} from {url} (speculation probe failed)"
        )
        super().__init__(self.message)


class RequestFailedHalt(Exception):
    """Control-flow signal: stop processing after a request failure.

    Raised from scraper or host code — a step, or a worker subclass — to tell
    the worker to halt the run rather than retry or move on. A host's
    replay worker raises it on a replay miss. Not an assumption or config
    violation: it is the caller steering the worker, so the worker catches it
    as flow control, not as an error to classify, and it propagates out of
    ``_handle_one`` rather than being recorded.
    """


class ScraperConfigError(PersistentException):
    """Scraper or driver configuration is wrong in a way retrying won't fix.

    Raised at run-time when a scraper or driver config invariant is
    violated (e.g. a step name that doesn't resolve). Persistent so
    the worker treats it as a permanent failure for the parent request.
    """
