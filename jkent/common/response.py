"""What a transport hands back: :class:`Response` and its archive variants.

Below :mod:`jkent.common.request` in the layering — a request resolves
against a response at runtime (``isinstance``), while a response only
*names* its request — so this module imports the request type for type
checking alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jkent.common.request import Request
    from jkent.common.selector_observer import SelectorObserver


@dataclass
class Response:
    """HTTP response from fetching a page.

    Modeled after httpx.Response to provide a familiar interface.
    The driver creates Response objects and passes them to scraper
    continuation methods.

    Attributes:
        status_code: HTTP status code (200, 404, etc.).
        headers: Response headers.
        content: Raw response bytes.
        text: Decoded response text.
        url: Final URL after any redirects.
        request: The Request that triggered this response.
        observer: SelectorObserver recorded while a @step with ``page``
            injection executed against this response. Set by the step
            wrapper; per-execution by construction (the driver owns one
            Response per execution), so drivers read autowait/debug
            telemetry here. None until a page-injecting step runs.
    """

    status_code: int
    headers: dict[str, str]
    content: bytes
    text: str
    url: str
    request: Request
    observer: SelectorObserver | None = None


@dataclass
class ArchiveResponse(Response):
    """HTTP response for an archived file.

    Extends Response with a ``file_url``  The @step machinery injects this
    value into steps as the ``local_filepath`` parameter;
    This lets scrapers include the file location in their ParsedData.

    Attributes:
        file_url: path where the downloaded file was stored.
            Injected into steps as ``local_filepath``.
    """

    file_url: str = ""


@dataclass
class ArchiveDecision:
    """Decision from an ArchiveHandler about whether to download a file.

    Attributes:
        download: If True, the driver should proceed with downloading.
        file_url: When download=False, the location of the existing file.
            When download=True, may be empty (save() determines final path).
    """

    download: bool
    file_url: str = ""
