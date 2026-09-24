"""The scraper authoring API, in one import.

Scrapers and hosts import the contract types from here; the definitions live
in the leaf modules under :mod:`jkent.common`, layered so each imports only
what sits below it:

    selectors → via → incidental → response → request → scraper

(:mod:`~jkent.common.wait_conditions` and
:mod:`~jkent.common.decorator_metadata` sit beside them as leaves.) This
module is a facade: it defines nothing and re-exports everything in
``__all__``. Prefer it in scraper code; driver code may import a leaf
directly.
"""

from __future__ import annotations

from jkent.common.decorator_metadata import DEFAULT_PRIORITY
from jkent.common.incidental import IncidentalMatch, Multiple, Singular
from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    RateLimitTable,
)
from jkent.common.request import (
    ARCHIVE_DEFAULT_PRIORITY,
    SPECULATION_SOFT_FAILURE_STATUS,
    AuthType,
    CertType,
    CookiesType,
    FilesType,
    FileTuple,
    HeadersType,
    HttpMethod,
    HTTPRequestParams,
    ProxiesType,
    QueryParams,
    Request,
    RequestData,
    SkipDeduplicationCheck,
    TimeoutType,
    VerifyType,
)
from jkent.common.response import ArchiveDecision, ArchiveResponse, Response
from jkent.common.scraper import (
    BaseScraper,
    DriverRequirement,
    HTTPCodeType,
    ParsedData,
    ScraperReturnType,
    ScraperStatus,
    ScraperYield,
    StepInfo,
    T,
)
from jkent.common.selectors import CSS, Selector, XPath
from jkent.common.via import (
    FieldResolver,
    FieldValue,
    ViaFormSubmit,
    ViaLink,
    via_from_json,
)
from jkent.common.wait_conditions import (
    WaitCondition,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
)

__all__ = [
    "ARCHIVE_DEFAULT_PRIORITY",
    "CSS",
    "DEFAULT_PRIORITY",
    "DEFAULT_RATE_LIMIT",
    "NO_RATE_LIMIT",
    "SPECULATION_SOFT_FAILURE_STATUS",
    "ArchiveDecision",
    "ArchiveResponse",
    "AuthType",
    "BaseScraper",
    "CertType",
    "CookiesType",
    "DriverRequirement",
    "FieldResolver",
    "FieldValue",
    "FileTuple",
    "FilesType",
    "HTTPCodeType",
    "HTTPRequestParams",
    "HeadersType",
    "HttpMethod",
    "IncidentalMatch",
    "Multiple",
    "ParsedData",
    "ProxiesType",
    "QueryParams",
    "RateLimitTable",
    "Request",
    "RequestData",
    "Response",
    "ScraperReturnType",
    "ScraperStatus",
    "ScraperYield",
    "Selector",
    "Singular",
    "SkipDeduplicationCheck",
    "StepInfo",
    "T",
    "TimeoutType",
    "VerifyType",
    "ViaFormSubmit",
    "ViaLink",
    "WaitCondition",
    "WaitForLoadState",
    "WaitForSelector",
    "WaitForTimeout",
    "WaitForURL",
    "XPath",
    "via_from_json",
]
