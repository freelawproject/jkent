"""Mock server data for the Bug Civil Court.

This module defines the case data used across the test suite.
The Bug Civil Court is a fictional court where insects file civil lawsuits
against each other.

The data structure is designed to support all steps of the tutorial,
from basic data through async drivers.
"""

import json
import secrets
import time
from collections import deque
from dataclasses import dataclass
from datetime import date

from aiohttp import web

same_url_search_count_key: web.AppKey[list[int]] = web.AppKey(
    "same_url_search_count", list
)


@dataclass
class MockCase:
    """A case in the Bug Civil Court."""

    docket: str
    case_name: str
    plaintiff: str
    defendant: str
    date_filed: date
    case_type: str
    status: str
    judge: str
    summary: str
    has_opinion: bool = False
    has_oral_argument: bool = False
    # Step 5: Appeals tracking
    trial_court_docket: str | None = (
        None  # For appeals: original trial court docket
    )
    court_level: str = "trial"  # "trial" or "appeals"


# Bug Civil Court case data - insects filing lawsuits
CASES: list[MockCase] = [
    MockCase(
        docket="BCC-2024-001",
        case_name="Beetle v. Ant Colony",
        plaintiff="Barry Beetle",
        defendant="Ant Colony #47",
        date_filed=date(2024, 1, 15),
        case_type="Property Dispute",
        status="Pending",
        judge="Hon. Mantis Green",
        summary="Plaintiff alleges defendant tunneled under his log without permission.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-002",
        case_name="Butterfly v. Caterpillar",
        plaintiff="Monarch Butterfly",
        defendant="Carl Caterpillar",
        date_filed=date(2024, 2, 1),
        case_type="Identity Theft",
        status="Closed",
        judge="Hon. Dragonfly Swift",
        summary="Plaintiff claims defendant illegally assumed their identity during metamorphosis.",
        has_opinion=True,
        has_oral_argument=True,
    ),
    MockCase(
        docket="BCC-2024-003",
        case_name="Spider v. Fly",
        plaintiff="Webster Spider",
        defendant="Freddy Fly",
        date_filed=date(2024, 2, 14),
        case_type="Contract Dispute",
        status="Pending",
        judge="Hon. Mantis Green",
        summary="Plaintiff alleges defendant breached web-visiting agreement.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-004",
        case_name="Grasshopper v. Ant",
        plaintiff="Gary Grasshopper",
        defendant="Andy Ant",
        date_filed=date(2024, 3, 1),
        case_type="Defamation",
        status="Closed",
        judge="Hon. Cricket Chirp",
        summary="Plaintiff claims defendant spread false rumors about work ethic.",
        has_opinion=True,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-005",
        case_name="Bee v. Wasp",
        plaintiff="Beatrice Bee",
        defendant="Walter Wasp",
        date_filed=date(2024, 3, 15),
        case_type="Assault",
        status="Pending",
        judge="Hon. Dragonfly Swift",
        summary="Plaintiff alleges unprovoked stinging incident at flower garden.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-006",
        case_name="Ladybug v. Aphid",
        plaintiff="Lucy Ladybug",
        defendant="Arthur Aphid",
        date_filed=date(2024, 4, 1),
        case_type="Nuisance",
        status="Closed",
        judge="Hon. Mantis Green",
        summary="Plaintiff seeks restraining order due to persistent plant damage.",
        has_opinion=True,
        has_oral_argument=True,
    ),
    MockCase(
        docket="BCC-2024-007",
        case_name="Firefly v. Moth",
        plaintiff="Flash Firefly",
        defendant="Dusty Moth",
        date_filed=date(2024, 4, 15),
        case_type="Intellectual Property",
        status="Pending",
        judge="Hon. Cricket Chirp",
        summary="Plaintiff claims defendant copied bioluminescent signaling patterns.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-008",
        case_name="Dragonfly v. Mosquito",
        plaintiff="Dana Dragonfly",
        defendant="Mike Mosquito",
        date_filed=date(2024, 5, 1),
        case_type="Trespass",
        status="Pending",
        judge="Hon. Dragonfly Swift",
        summary="Plaintiff alleges repeated unauthorized entry into pond territory.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-009",
        case_name="Cicada v. Cricket",
        plaintiff="Cecilia Cicada",
        defendant="Chris Cricket",
        date_filed=date(2024, 5, 15),
        case_type="Noise Complaint",
        status="Closed",
        judge="Hon. Mantis Green",
        summary="Counter-suit alleging excessive nighttime chirping.",
        has_opinion=True,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-010",
        case_name="Termite v. Carpenter Ant",
        plaintiff="Terry Termite",
        defendant="Carla Carpenter Ant",
        date_filed=date(2024, 6, 1),
        case_type="Unfair Competition",
        status="Pending",
        judge="Hon. Cricket Chirp",
        summary="Plaintiff alleges defendant is undercutting wood-processing rates.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-011",
        case_name="Praying Mantis v. Cockroach",
        plaintiff="Patricia Praying Mantis",
        defendant="Rocky Roach",
        date_filed=date(2024, 6, 15),
        case_type="Personal Injury",
        status="Pending",
        judge="Hon. Dragonfly Swift",
        summary="Plaintiff claims injuries from defendant's sudden appearance.",
        has_opinion=False,
        has_oral_argument=False,
    ),
    MockCase(
        docket="BCC-2024-012",
        case_name="Dung Beetle v. Fly",
        plaintiff="Douglas Dung Beetle",
        defendant="Francine Fly",
        date_filed=date(2024, 7, 1),
        case_type="Theft",
        status="Closed",
        judge="Hon. Mantis Green",
        summary="Plaintiff alleges defendant stole prized dung ball collection.",
        has_opinion=True,
        has_oral_argument=True,
    ),
    # Step 5: Appeal cases
    MockCase(
        docket="BCA-2024-001",
        case_name="Butterfly v. Caterpillar (Appeal)",
        plaintiff="Monarch Butterfly",
        defendant="Carl Caterpillar",
        date_filed=date(2024, 4, 1),
        case_type="Identity Theft",
        status="Closed",
        judge="Hon. Chief Moth",
        summary="Appeal of trial court decision. Appellant argues trial court erred in metamorphosis analysis.",
        has_opinion=True,
        has_oral_argument=False,
        trial_court_docket="BCC-2024-002",
        court_level="appeals",
    ),
    MockCase(
        docket="BCA-2024-002",
        case_name="Grasshopper v. Ant (Appeal)",
        plaintiff="Gary Grasshopper",
        defendant="Andy Ant",
        date_filed=date(2024, 5, 1),
        case_type="Defamation",
        status="Pending",
        judge="Hon. Chief Moth",
        summary="Appeal of trial court verdict. Seeking reversal of defamation finding.",
        has_opinion=False,
        has_oral_argument=True,
        trial_court_docket="BCC-2024-004",
        court_level="appeals",
    ),
]


def generate_cases_html() -> str:
    """Generate HTML for the case list page.

    Includes a hidden session_token field. The token is required for
    downloading PDF opinions.

    Returns:
        HTML string containing the list of cases.
    """
    rows = []
    for case in CASES:
        rows.append(f"""
        <tr class="case-row" data-docket="{case.docket}">
            <td class="docket">{case.docket}</td>
            <td class="case-name">{case.case_name}</td>
            <td class="date-filed">{case.date_filed.isoformat()}</td>
            <td class="case-type">{case.case_type}</td>
            <td class="status">{case.status}</td>
        </tr>""")

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Bug Civil Court - Case Search Results</title>
</head>
<body>
    <h1>Bug Civil Court</h1>
    <h2>Case Search Results</h2>

    <!-- Step 6: Hidden session token for file downloads -->
    <input type="hidden" id="session-token" value="bug-session-token-abc123" />

    <table id="cases-table">
        <thead>
            <tr>
                <th>Docket</th>
                <th>Case Name</th>
                <th>Date Filed</th>
                <th>Case Type</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</body>
</html>"""


def generate_case_detail_html(case: MockCase) -> str:
    """Generate HTML for a case detail page.

    Args:
        case: The case to generate HTML for.

    Returns:
        HTML string for the case detail page.
    """
    opinion_link = ""
    if case.has_opinion:
        opinion_link = (
            f'<a href="/opinions/{case.docket}.pdf">Download Opinion (PDF)</a>'
        )

    oral_arg_link = ""
    if case.has_oral_argument:
        oral_arg_link = f'<a href="/oral-arguments/{case.docket}.mp3">Listen to Oral Argument (MP3)</a>'

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{case.case_name} - Bug Civil Court</title>
</head>
<body>
    <h1>Bug Civil Court</h1>
    <h2>{case.case_name}</h2>

    <div class="case-details">
        <dl>
            <dt>Docket Number</dt>
            <dd id="docket">{case.docket}</dd>

            <dt>Plaintiff</dt>
            <dd id="plaintiff">{case.plaintiff}</dd>

            <dt>Defendant</dt>
            <dd id="defendant">{case.defendant}</dd>

            <dt>Date Filed</dt>
            <dd id="date-filed">{case.date_filed.isoformat()}</dd>

            <dt>Case Type</dt>
            <dd id="case-type">{case.case_type}</dd>

            <dt>Status</dt>
            <dd id="status">{case.status}</dd>

            <dt>Presiding Judge</dt>
            <dd id="judge">{case.judge}</dd>

            <dt>Case Summary</dt>
            <dd id="summary">{case.summary}</dd>
        </dl>
    </div>

    <div class="documents">
        <h3>Documents</h3>
        {opinion_link}
        {oral_arg_link}
    </div>
</body>
</html>"""


def get_case_by_docket(docket: str) -> MockCase | None:
    """Get a case by its docket number.

    Args:
        docket: The docket number to search for.

    Returns:
        The MockCase if found, None otherwise.
    """
    for case in CASES:
        if case.docket == docket:
            return case
    return None


# =============================================================================
# Step 2: aiohttp Mock Server
# =============================================================================


async def handle_cases_list(request: web.Request) -> web.Response:
    """Handle GET /cases - return the case list HTML."""
    html = generate_cases_html()
    return web.Response(text=html, content_type="text/html")


async def handle_case_detail(request: web.Request) -> web.Response:
    """Handle GET /cases/{docket} - return case detail HTML.

    Step 8: Supports ?error=true query parameter to return an error page
    with different HTML structure for testing structural assumption errors.

    Step 10: Supports ?server_error=true query parameter to return a 500
    Internal Server Error for testing transient exception handling.
    """
    docket = request.match_info["docket"]
    case = get_case_by_docket(docket)

    if case is None:
        return web.Response(
            text=f"<html><body><h1>404</h1><p>Case {docket} not found</p></body></html>",
            status=404,
            content_type="text/html",
        )

    # Step 10: Check for server_error=true query parameter
    if request.query.get("server_error") == "true":
        # Return 500 Internal Server Error
        html = f"""<!DOCTYPE html>
<html>
<head><title>500 Internal Server Error</title></head>
<body>
    <h1>500 Internal Server Error</h1>
    <p>The server encountered an error processing your request.</p>
    <p>Please try again later.</p>
    <p>Request ID: {docket}-ERROR</p>
</body>
</html>"""
        return web.Response(text=html, status=500, content_type="text/html")

    # Step 8: Check for error=true query parameter
    if request.query.get("error") == "true":
        # Return an error page with completely different structure
        html = f"""<!DOCTYPE html>
<html>
<head><title>Error - Bug Civil Court</title></head>
<body>
    <div class="error-container">
        <h1>Service Temporarily Unavailable</h1>
        <p>The case detail page is currently unavailable. Please try again later.</p>
        <p>Error code: STRUCT_CHANGE_001</p>
        <p>Reference: {docket}</p>
    </div>
</body>
</html>"""
        return web.Response(text=html, content_type="text/html")

    html = generate_case_detail_html(case)
    return web.Response(text=html, content_type="text/html")


# =============================================================================
# Step 3: JSON API Endpoint
# =============================================================================


async def handle_case_api(request: web.Request) -> web.Response:
    """Handle GET /api/cases/{docket} - return case JSON data.

    This endpoint provides supplementary case metadata as JSON, useful
    for demonstrating Request(nonnavigating=True) (fetching data without navigation).
    """
    docket = request.match_info["docket"]
    case = get_case_by_docket(docket)

    if case is None:
        return web.Response(
            text='{"error": "Case not found"}',
            status=404,
            content_type="application/json",
        )

    # Return JSON with additional metadata not in HTML
    data = {
        "docket": case.docket,
        "case_name": case.case_name,
        "plaintiff": case.plaintiff,
        "defendant": case.defendant,
        "date_filed": case.date_filed.isoformat(),
        "case_type": case.case_type,
        "status": case.status,
        "judge": case.judge,
        "summary": case.summary,
        # Additional metadata only available via API
        "api_metadata": {
            "last_updated": case.date_filed.isoformat(),
            "case_number_normalized": case.docket.replace("-", ""),
            "jurisdiction": "BUG",
        },
    }

    return web.Response(text=json.dumps(data), content_type="application/json")


# =============================================================================
# Step 4: File Download Endpoints (PDF and MP3)
# =============================================================================


async def handle_opinion_pdf(request: web.Request) -> web.Response:
    """Handle GET /opinions/{docket}.pdf - return PDF file.

    This endpoint provides downloadable opinion PDFs for cases that have
    opinions available, useful for demonstrating Request(archive=True).

    Requires X-Session-Token header for authentication when a token is
    provided; unauthenticated access is allowed for backward compatibility.
    """
    docket = request.match_info["docket"].replace(".pdf", "")
    case = get_case_by_docket(docket)

    if case is None or not case.has_opinion:
        return web.Response(
            text="<html><body><h1>404</h1><p>Opinion not found</p></body></html>",
            status=404,
            content_type="text/html",
        )

    # Optionally check for session token in header.
    # If the client provides a token, validate it. If not provided, allow access.
    session_token = request.headers.get("X-Session-Token")
    if session_token and session_token != "bug-session-token-abc123":
        return web.Response(
            text='{"error": "Invalid session token"}',
            status=403,
            content_type="application/json",
        )
    # If no token provided, allow access (backward compatibility)

    # Generate a simple PDF-like binary content
    # Real PDFs have complex structure, but for testing we just need binary data
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT\n/F1 12 Tf\n100 700 Td\n("
        + case.case_name.encode("utf-8")
        + b") Tj\nET\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n"
        b"0000000214 00000 n\ntrailer\n"
        b"<< /Size 5 /Root 1 0 R >>\nstartxref\n318\n%%EOF"
    )

    return web.Response(
        body=pdf_content,
        content_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{docket}.pdf"'
        },
    )


async def handle_oral_argument_mp3(request: web.Request) -> web.Response:
    """Handle GET /oral-arguments/{docket}.mp3 - return MP3 file.

    This endpoint provides downloadable oral argument audio files for cases
    that have oral arguments available, useful for demonstrating Request(archive=True).
    """
    docket = request.match_info["docket"].replace(".mp3", "")
    case = get_case_by_docket(docket)

    if case is None or not case.has_oral_argument:
        return web.Response(
            text="<html><body><h1>404</h1><p>Oral argument not found</p></body></html>",
            status=404,
            content_type="text/html",
        )

    # Generate a minimal MP3-like binary content
    # Real MP3s have complex structure, but for testing we just need binary data
    # This is a minimal MP3 frame header followed by some data
    mp3_content = (
        b"\xff\xfb\x90\x00"  # MP3 sync word and header
        b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"Oral argument for " + case.case_name.encode("utf-8")
    )

    return web.Response(
        body=mp3_content,
        content_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{docket}.mp3"'
        },
    )


# =============================================================================
# Step 5: Appeals Court Endpoint
# =============================================================================


async def handle_appeals_list(request: web.Request) -> web.Response:
    """Handle GET /appeals - return list of appeals court cases.

    This endpoint demonstrates accumulated_data flowing across court levels.
    Appeals cases reference their trial court dockets, allowing scrapers to
    collect data from both courts.
    """
    # Get only appeals court cases
    appeals_cases = [c for c in CASES if c.court_level == "appeals"]

    html_parts = [
        "<html><head><title>Bug Appeals Court - Case List</title></head><body>",
        "<h1>Bug Appeals Court - Case List</h1>",
        "<table>",
        "<tr><th>Docket</th><th>Case Name</th><th>Status</th><th>Trial Court Docket</th></tr>",
    ]

    for case in appeals_cases:
        html_parts.append(
            f"<tr class='case-row'>"
            f"<td class='docket'>{case.docket}</td>"
            f"<td><a href='/appeals/{case.docket}'>{case.case_name}</a></td>"
            f"<td>{case.status}</td>"
            f"<td><a href='/cases/{case.trial_court_docket}'>{case.trial_court_docket}</a></td>"
            f"</tr>"
        )

    html_parts.append("</table></body></html>")

    return web.Response(text="\n".join(html_parts), content_type="text/html")


async def handle_appeal_detail(request: web.Request) -> web.Response:
    """Handle GET /appeals/{docket} - return appeal case detail page.

    This page includes a link to the trial court case, demonstrating
    how accumulated_data can track relationships between court levels.
    """
    docket = request.match_info["docket"]
    case = get_case_by_docket(docket)

    if case is None or case.court_level != "appeals":
        return web.Response(
            text="<html><body><h1>404</h1><p>Appeal case not found</p></body></html>",
            status=404,
            content_type="text/html",
        )

    html = f"""<html>
<head><title>{case.case_name} - Bug Appeals Court</title></head>
<body>
<h1>Bug Appeals Court</h1>
<h2>{case.case_name}</h2>

<div id="docket">Docket: {case.docket}</div>
<div id="plaintiff">Appellant: {case.plaintiff}</div>
<div id="defendant">Appellee: {case.defendant}</div>
<div id="date-filed">Appeal Filed: {case.date_filed}</div>
<div id="case-type">Type: {case.case_type}</div>
<div id="status">Status: {case.status}</div>
<div id="judge">Judge: {case.judge}</div>
<div id="summary">Summary: {case.summary}</div>
<div id="trial-court-docket">Trial Court Case: <a href="/cases/{case.trial_court_docket}">{case.trial_court_docket}</a></div>

{"<div id='opinion'><a href='/opinions/" + case.docket + ".pdf'>Download Opinion</a></div>" if case.has_opinion else ""}
{"<div id='oral-argument'><a href='/oral-arguments/" + case.docket + ".mp3'>Download Oral Argument</a></div>" if case.has_oral_argument else ""}

</body>
</html>"""

    return web.Response(text=html, content_type="text/html")


# Step 12: Rate limiting state (global for testing)
# Track request times in a deque for efficient cleanup
_rate_limit_requests: deque[float] = deque()
_RATE_LIMIT_MAX_REQUESTS = 2  # Allow 2 requests per second
_RATE_LIMIT_WINDOW = 1.0  # 1 second window


async def handle_rate_limited(request: web.Request) -> web.Response:
    """Handle rate-limited endpoint that returns 429 when exceeded.

    This endpoint enforces a rate limit of 2 requests per second for testing
    purposes. It tracks request times and returns 429 if the limit is exceeded.

    Args:
        request: The aiohttp request.

    Returns:
        200 response if within rate limit, 429 if exceeded.
    """
    global _rate_limit_requests

    current_time = time.time()

    # Remove requests older than the window
    while _rate_limit_requests and (
        current_time - _rate_limit_requests[0] > _RATE_LIMIT_WINDOW
    ):
        _rate_limit_requests.popleft()

    # Check if we're over the limit
    if len(_rate_limit_requests) >= _RATE_LIMIT_MAX_REQUESTS:
        return web.Response(
            status=429,
            text="Too Many Requests",
            content_type="text/plain",
        )

    # Add this request to the tracker
    _rate_limit_requests.append(current_time)

    # Return success
    return web.Response(
        status=200,
        text="Request allowed",
        content_type="text/plain",
    )


async def handle_search_form(request: web.Request) -> web.Response:
    """Handle GET /search - return a page with a case search form."""
    case_types = sorted({c.case_type for c in CASES})
    options = "\n".join(
        f'            <option value="{ct}">{ct}</option>' for ct in case_types
    )

    html = f"""<!DOCTYPE html>
<html>
<head><title>Bug Civil Court - Case Search</title></head>
<body>
    <h1>Bug Civil Court</h1>
    <h2>Search Cases</h2>
    <form id="case-search" method="GET" action="/search/results">
        <label for="case_type">Case Type:</label>
        <select name="case_type" id="case_type">
            <option value="">All</option>
{options}
        </select>
        <label for="status">Status:</label>
        <select name="status" id="status">
            <option value="">All</option>
            <option value="Pending">Pending</option>
            <option value="Closed">Closed</option>
        </select>
        <button type="submit">Search</button>
    </form>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_search_results(request: web.Request) -> web.Response:
    """Handle GET /search/results - return filtered case results."""
    case_type = request.query.get("case_type", "")
    status = request.query.get("status", "")

    filtered = CASES
    if case_type:
        filtered = [c for c in filtered if c.case_type == case_type]
    if status:
        filtered = [c for c in filtered if c.status == status]

    rows = []
    for case in filtered:
        rows.append(
            f'<tr class="case-row"><td class="docket">{case.docket}</td>'
            f'<td class="case-name">{case.case_name}</td>'
            f'<td class="status">{case.status}</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head><title>Bug Civil Court - Search Results</title></head>
<body>
    <h1>Bug Civil Court</h1>
    <h2>Search Results</h2>
    <p id="result-count">{len(filtered)} case(s) found</p>
    <table id="results-table">
        <thead><tr><th>Docket</th><th>Case Name</th><th>Status</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_complex_form(request: web.Request) -> web.Response:
    """Handle GET /complex-search - form with hidden, radio, select, invisible fields.

    Simulates an ASP.NET WebForms page with Telerik-style date pickers:
    hidden __VIEWSTATE, radio buttons, select dropdowns, invisible parent
    inputs, and hidden ClientState fields.
    """
    html = """<!DOCTYPE html>
<html>
<head><title>Bug Civil Court - Complex Search</title></head>
<body>
    <h1>Bug Civil Court</h1>
    <form id="Form1" method="POST" action="/complex-search/results">
        <input type="hidden" name="__VIEWSTATE" value="dummyviewstate123" />
        <input type="hidden" name="__EVENTVALIDATION" value="dummyvalidation456" />

        <label>Case Category:</label>
        <input type="radio" name="category" value="civil" checked /> Civil
        <input type="radio" name="category" value="criminal" /> Criminal

        <label for="case_type">Case Type:</label>
        <select name="case_type" id="case_type">
            <option value="">All</option>
            <option value="Property Dispute">Property Dispute</option>
            <option value="Contract Dispute">Contract Dispute</option>
        </select>

        <label>Start Date:</label>
        <input type="text" name="date_start_display" value="" />
        <!-- Invisible parent input (like Telerik RadDatePicker) -->
        <input type="text" name="date_start_hidden"
               style="visibility:hidden;width:1px;height:1px;" value="" />
        <!-- Hidden ClientState (what server actually reads) -->
        <input type="hidden" name="date_start_client_state" value="" />

        <input type="submit" name="btnFind" value="Find" />
    </form>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_complex_form_results(request: web.Request) -> web.Response:
    """Handle POST /complex-search/results - return results from complex form.

    Validates that hidden fields, radio values, and select values were
    submitted correctly.  The date is read from the ``date_start_client_state``
    field (like Telerik), not the visible display field.
    """
    data = await request.post()

    viewstate = data.get("__VIEWSTATE", "")
    category = str(data.get("category", ""))
    case_type = data.get("case_type", "")
    client_state = data.get("date_start_client_state", "")

    # Filter cases
    filtered = CASES
    if category == "criminal":
        filtered = []  # No criminal cases in mock data
    if case_type:
        filtered = [c for c in filtered if c.case_type == case_type]

    # Check that hidden fields came through
    has_viewstate = bool(viewstate)
    has_client_state = bool(client_state)

    rows = []
    for case in filtered:
        rows.append(
            f'<tr class="case-row"><td class="docket">{case.docket}</td>'
            f'<td class="case-name">{case.case_name}</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html>
<head><title>Bug Civil Court - Complex Results</title></head>
<body>
    <h1>Bug Civil Court</h1>
    <p id="result-count">{len(filtered)} case(s) found</p>
    <p id="viewstate-ok">{"yes" if has_viewstate else "no"}</p>
    <p id="client-state-ok">{"yes" if has_client_state else "no"}</p>
    <p id="category">{category}</p>
    <table id="results-table">
        <tbody>{"".join(rows)}</tbody>
    </table>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


# =============================================================================
# Session tree endpoints for tab forking tests
# =============================================================================

# Global secret for session tree (set per-request to validate cookies)


async def handle_session_tree_root(request: web.Request) -> web.Response:
    """Root of session tree: sets a cookie and shows 2 branch links."""
    tree_secret = secrets.token_hex(8)
    html = f"""<html>
<head><title>Session Tree Root</title></head>
<body>
    <h1>Session Tree</h1>
    <a href="/session-tree/branch/A?secret={tree_secret}" data-needs-secret="true">Branch A</a>
    <a href="/session-tree/branch/B?secret={tree_secret}" data-needs-secret="true">Branch B</a>
</body>
</html>"""
    resp = web.Response(text=html, content_type="text/html")
    resp.set_cookie("tree_secret", tree_secret)
    return resp


async def handle_session_tree_branch(request: web.Request) -> web.Response:
    """Branch page: validates cookie matches secret param, shows 2 leaf links."""
    name = request.match_info["name"]
    query_secret = request.query.get("secret", "")
    cookie_secret = request.cookies.get("tree_secret", "")

    if not query_secret or query_secret != cookie_secret:
        return web.Response(
            text="<html><body>404 - Cookie mismatch</body></html>",
            status=404,
            content_type="text/html",
        )

    html = f"""<html>
<head><title>Branch {name}</title></head>
<body>
    <h1>Branch {name}</h1>
    <a href="/session-tree/leaf/{name}1?secret={query_secret}" data-needs-secret="true">Leaf {name}1</a>
    <a href="/session-tree/leaf/{name}2?secret={query_secret}" data-needs-secret="true">Leaf {name}2</a>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


# Leaf number mapping for deterministic test assertions
_LEAF_NUMBERS = {"A1": 1, "A2": 2, "B1": 3, "B2": 4}


async def handle_session_tree_leaf(request: web.Request) -> web.Response:
    """Leaf page: validates cookie, returns a number."""
    name = request.match_info["name"]
    query_secret = request.query.get("secret", "")
    cookie_secret = request.cookies.get("tree_secret", "")

    if not query_secret or query_secret != cookie_secret:
        return web.Response(
            text="<html><body>404 - Cookie mismatch</body></html>",
            status=404,
            content_type="text/html",
        )

    number = _LEAF_NUMBERS.get(name, 0)
    html = f"""<html>
<head><title>Leaf {name}</title></head>
<body>
    <h1>Leaf {name}</h1>
    <div class="number">{number}</div>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


# =============================================================================
# Same-URL search: GET shows form, POST shows results (tab forking unroute test)
# =============================================================================


async def handle_same_url_search_get(request: web.Request) -> web.Response:
    """GET /same-url-search — form with a submit button that POSTs to the same URL."""
    request.app[same_url_search_count_key][0] += 1
    html = """<html>
<head><title>Same URL Search</title></head>
<body>
    <h1>Search Form</h1>
    <form id="search-form" method="POST" action="/same-url-search">
        <input type="hidden" name="query" value="magic" />
        <button type="submit" id="go-btn">Get search results</button>
    </form>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def handle_same_url_search_post(request: web.Request) -> web.Response:
    """POST /same-url-search — returns search results page."""
    request.app[same_url_search_count_key][0] += 1
    html = """<html>
<head><title>Same URL Search Results</title></head>
<body>
    <h1>Results</h1>
    <div class="answer">42</div>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


def create_app() -> web.Application:
    """Create the aiohttp application with all routes.

    Returns:
        Configured aiohttp Application.
    """
    app = web.Application()
    app[same_url_search_count_key] = [0]
    app.router.add_get("/cases", handle_cases_list)
    app.router.add_get("/cases/{docket}", handle_case_detail)
    app.router.add_get("/api/cases/{docket}", handle_case_api)
    app.router.add_get("/opinions/{docket}.pdf", handle_opinion_pdf)
    app.router.add_get(
        "/oral-arguments/{docket}.mp3", handle_oral_argument_mp3
    )
    # Step 5: Appeals court routes
    app.router.add_get("/appeals", handle_appeals_list)
    app.router.add_get("/appeals/{docket}", handle_appeal_detail)
    # Step 12: Rate limit testing endpoint
    app.router.add_get("/rate-limited", handle_rate_limited)
    # Form search endpoint
    app.router.add_get("/search", handle_search_form)
    app.router.add_get("/search/results", handle_search_results)
    # Complex form with hidden/radio/select/invisible fields
    app.router.add_get("/complex-search", handle_complex_form)
    app.router.add_post("/complex-search/results", handle_complex_form_results)
    # Session tree endpoints for tab forking tests
    app.router.add_get("/session-tree", handle_session_tree_root)
    app.router.add_get(
        "/session-tree/branch/{name}", handle_session_tree_branch
    )
    app.router.add_get("/session-tree/leaf/{name}", handle_session_tree_leaf)
    # Same-URL search (GET=form, POST=results) for unroute verification
    app.router.add_get("/same-url-search", handle_same_url_search_get)
    app.router.add_post("/same-url-search", handle_same_url_search_post)

    # Speculation probe: /spec/{n} returns 200 for n <= 3, else 404. Lets
    # a @speculate scraper walk a sequential param until a persistent 404.
    async def handle_spec(request: web.Request) -> web.Response:
        n = int(request.match_info["n"])
        if n <= 3:
            return web.Response(
                text=f"<html><body>spec {n}</body></html>",
                content_type="text/html",
            )
        return web.Response(
            text="<html><body>404 not found</body></html>",
            status=404,
            content_type="text/html",
        )

    app.router.add_get("/spec/{n}", handle_spec)

    # Header/cookie echo: returns what the server received as JSON, so
    # tests can assert that request data (e.g. permanent headers/cookies)
    # actually went out over the wire. /echo/{tail} so chained scrapes can
    # hit distinct URLs (/echo/step1, /echo/step2, ...). Header names are
    # lowercased — clients differ in the casing they send.
    async def handle_echo(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "path": request.path,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "cookies": dict(request.cookies),
            }
        )

    app.router.add_get("/echo/{tail:.*}", handle_echo)

    # POST body echo: returns the raw bytes and content-type the server
    # received, so tests can assert a JSON body (``HTTPRequestParams.json``)
    # survives serialize → DB → dequeue → dispatch and goes out as
    # ``application/json``, not form-encoded.
    async def handle_echo_body(request: web.Request) -> web.Response:
        body = await request.read()
        return web.json_response(
            {
                "path": request.path,
                "content_type": request.headers.get("Content-Type", ""),
                "body": body.decode("utf-8", errors="replace"),
            }
        )

    app.router.add_post("/echo-body/{tail:.*}", handle_echo_body)

    # Catch-all: any unmatched GET returns 200 with a placeholder body so
    # tests that hit arbitrary URLs (priority/stop/lifecycle suites) don't
    # trip the new 4xx-as-persistent classification. Register LAST so it
    # doesn't shadow specific routes.
    async def _catch_all(request: web.Request) -> web.Response:
        return web.Response(
            text="<html><body>ok</body></html>",
            content_type="text/html",
        )

    app.router.add_get("/{tail:.*}", _catch_all)
    return app
