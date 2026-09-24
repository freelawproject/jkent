"""jkent — a scraper SDK and the driver that runs it.

A scraper is parsing plus navigation intent; a driver executes it. The
authoring API a scraper needs is re-exported here so one import covers it::

    from jkent import BaseScraper, Request, HTTPRequestParams, XPath, entry, step

Everything below is defined in :mod:`jkent.common` (leaf modules layered
selectors → via → incidental → response → request → scraper) and
re-exported from :mod:`jkent.data_types`, the public facade; this module
imports from the facade only. Driver and database
machinery live under :mod:`jkent.driver` and are never imported by this
package's ``__init__``, so ``import jkent`` pulls only the SDK dependency
group.

Package ``__init__`` policy
---------------------------
A package re-exports (and declares ``__all__``) only when it *is* a surface:
``jkent`` (authoring), :mod:`jkent.driver.unified_driver` (host wiring),
:mod:`jkent.driver.browser_engine`, :mod:`jkent.driver.database_engine.sql_manager`
and :mod:`jkent.observability`. Implementation packages — :mod:`jkent.common`,
:mod:`jkent.driver`, :mod:`jkent.driver.database_engine` — keep a
docstring-only ``__init__``; import their modules directly.
"""

from __future__ import annotations

from jkent.data_types import (
    ARCHIVE_DEFAULT_PRIORITY,
    CSS,
    DEFAULT_PRIORITY,
    ArchiveDecision,
    ArchiveResponse,
    BaseScraper,
    DataFormatAssumptionException,
    DateRange,
    DeferredValidation,
    DriverRequirement,
    FieldResolver,
    FieldValue,
    Form,
    FormField,
    HTMLStructuralAssumptionException,
    HTTPCodeType,
    HttpMethod,
    HTTPRequestParams,
    HTTPResponseAssumptionException,
    IncidentalMatch,
    IncidentalRequestAssumptionException,
    JKentParser,
    Link,
    LxmlPageElement,
    Multiple,
    ParsedData,
    PersistentException,
    PersistentHTTPResponseException,
    Request,
    RequestFailedHalt,
    RequestTimeoutException,
    Response,
    ScrapedData,
    ScraperAssumptionException,
    ScraperConfigError,
    ScraperStatus,
    ScraperYield,
    Selector,
    Singular,
    SkipDeduplicationCheck,
    Speculative,
    SpeculativeRange,
    StepInfo,
    TransientException,
    TransientKind,
    ViaFormSubmit,
    ViaLink,
    WaitCondition,
    WaitForLoadState,
    WaitForSelector,
    WaitForTimeout,
    WaitForURL,
    XPath,
    entry,
    step,
)

__all__ = [
    # Scraper definition
    "BaseScraper",
    "DriverRequirement",
    "HTTPCodeType",
    "ScraperStatus",
    "StepInfo",
    "entry",
    "step",
    # Requests and responses
    "ARCHIVE_DEFAULT_PRIORITY",
    "DEFAULT_PRIORITY",
    "ArchiveDecision",
    "ArchiveResponse",
    "HTTPRequestParams",
    "HttpMethod",
    "ParsedData",
    "Request",
    "Response",
    "ScraperYield",
    "SkipDeduplicationCheck",
    # Navigation detail: vias, incidentals, waits
    "FieldResolver",
    "FieldValue",
    "IncidentalMatch",
    "Multiple",
    "Singular",
    "ViaFormSubmit",
    "ViaLink",
    "WaitCondition",
    "WaitForLoadState",
    "WaitForSelector",
    "WaitForTimeout",
    "WaitForURL",
    # Page access
    "CSS",
    "Form",
    "FormField",
    "JKentParser",
    "Link",
    "LxmlPageElement",
    "Selector",
    "XPath",
    # Data and parameters
    "DateRange",
    "DeferredValidation",
    "ScrapedData",
    "Speculative",
    "SpeculativeRange",
    # Exceptions
    "DataFormatAssumptionException",
    "HTMLStructuralAssumptionException",
    "HTTPResponseAssumptionException",
    "IncidentalRequestAssumptionException",
    "PersistentException",
    "PersistentHTTPResponseException",
    "RequestFailedHalt",
    "RequestTimeoutException",
    "ScraperAssumptionException",
    "ScraperConfigError",
    "TransientException",
    "TransientKind",
]
