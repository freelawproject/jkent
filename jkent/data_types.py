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

from jkent.common.data_models import ScrapedData
from jkent.common.decorator_metadata import DEFAULT_PRIORITY
from jkent.common.decorators import entry, step
from jkent.common.deferred_validation import DeferredValidation
from jkent.common.exceptions import (
    DataFormatAssumptionException,
    HTMLStructuralAssumptionException,
    HTTPResponseAssumptionException,
    IncidentalRequestAssumptionException,
    PersistentException,
    PersistentHTTPResponseException,
    RequestFailedHalt,
    RequestTimeoutException,
    ScraperAssumptionException,
    ScraperConfigError,
    TransientException,
    TransientKind,
)
from jkent.common.incidental import IncidentalMatch, Multiple, Singular
from jkent.common.lxml_page_element import LxmlPageElement
from jkent.common.page_element import Form, FormField, Link
from jkent.common.param_models import DateRange, SpeculativeRange
from jkent.common.parser import JKentParser
from jkent.common.rate_limits import (
    DEFAULT_RATE_LIMIT,
    NO_RATE_LIMIT,
    RateLimitTable,
)
from jkent.common.request import (
    ARCHIVE_DEFAULT_PRIORITY,
    CookiesType,
    HeadersType,
    HttpMethod,
    HTTPRequestParams,
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
from jkent.common.speculative import Speculative
from jkent.common.via import (
    FieldResolver,
    FieldValue,
    Via,
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
    "T",
    "ArchiveDecision",
    "ArchiveResponse",
    "BaseScraper",
    "CookiesType",
    "DataFormatAssumptionException",
    "DateRange",
    "DeferredValidation",
    "DriverRequirement",
    "FieldResolver",
    "FieldValue",
    "Form",
    "FormField",
    "HTMLStructuralAssumptionException",
    "HTTPCodeType",
    "HTTPRequestParams",
    "HTTPResponseAssumptionException",
    "HeadersType",
    "HttpMethod",
    "IncidentalMatch",
    "IncidentalRequestAssumptionException",
    "JKentParser",
    "Link",
    "LxmlPageElement",
    "Multiple",
    "ParsedData",
    "PersistentException",
    "PersistentHTTPResponseException",
    "QueryParams",
    "RateLimitTable",
    "Request",
    "RequestData",
    "RequestFailedHalt",
    "RequestTimeoutException",
    "Response",
    "ScrapedData",
    "ScraperAssumptionException",
    "ScraperConfigError",
    "ScraperReturnType",
    "ScraperStatus",
    "ScraperYield",
    "Selector",
    "Singular",
    "SkipDeduplicationCheck",
    "Speculative",
    "SpeculativeRange",
    "StepInfo",
    "TimeoutType",
    "TransientException",
    "TransientKind",
    "VerifyType",
    "Via",
    "ViaFormSubmit",
    "ViaLink",
    "WaitCondition",
    "WaitForLoadState",
    "WaitForSelector",
    "WaitForTimeout",
    "WaitForURL",
    "XPath",
    "entry",
    "step",
    "via_from_json",
]
