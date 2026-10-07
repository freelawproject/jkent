# Why assumptions are exceptions

A scraper is a pile of assumptions about a website: that this table has one row
per case, that this cell is a date in `MM/DD/YYYY`, that clicking "next" issues
a request matching this pattern. Sites change without telling anyone. The
design question is not *whether* those assumptions will break but what happens
in the minute they do.

jkent's answer: every violated assumption raises, loudly, with structure, and
the driver files it as a typed row. Nothing degrades quietly.

## Silent failure is the expensive failure mode

The alternative — `None` on a missed selector, `try/except: continue` around a
bad date — is cheap to write and extremely expensive to own, because the
failure it produces is *a successful run with less data in it*. A scrape that
crashes is noticed in minutes. A scrape that quietly returns 1,800 of 2,000
opinions, every night, is noticed when someone wonders why a case is missing,
possibly months later, by which time the window to re-fetch may have closed and
there is no record of which 200 went missing or why.

This asymmetry is the whole argument. In scraping, the run is not the product;
the data is, and the data's defect is invisible from the outside. So the
framework spends effort making the *absence* of data as loud as a crash, and
accepts the cost: scrapers raise more often, and a site tweak that a lenient
scraper would have shrugged off shows up as failures.

Two things keep that cost tolerable, and both are design requirements rather
than conveniences:

- **A raise is scoped to one request.** It fails that request and its subtree,
  not the run. The rest of the queue proceeds.
- **A raise is diagnosable without a re-run.** An exception that says only
  "something was wrong on page 12" has traded a silent data bug for a silent
  debugging bug. Hence the structure below.

## Two roots, and what dispatching on them means

Every failure the worker handles descends from one of two base classes, and the
choice between them is a statement about retry:

| | `PersistentException` | `TransientException` |
|---|---|---|
| Means | retrying cannot fix this | retrying might fix this |
| Worker behaviour | marked failed on first occurrence; retry machinery skipped | handed to the retry scheduler with backoff |
| Typical cause | our assumption is wrong, or the resource is really gone | the network, the server, or the browser misbehaved |

The worker dispatches on the base class, not on a list of concrete types, so a
new exception inherits its handling by choosing a parent. That makes the
parent choice the consequential decision when adding one.

**Persistent** — `ScraperAssumptionException` and its subclasses are the
scraper-authored ones:

- `HTMLStructuralAssumptionException` — a selector matched the wrong number of
  elements. Carries the selector, its grammar, the human description, the
  expected min/max and the actual count.
- `DataFormatAssumptionException` — extracted data failed its Pydantic model.
  Carries the model name, the validation errors, and the document that failed.
- `IncidentalRequestAssumptionException` — a `Singular`/`Multiple` incidental
  spec matched the wrong number of the parent navigation's sub-requests. Carries
  the spec and the URLs that did match, so an over-broad spec is visible.
- `PreprocessPageException` — a step's `preprocess=` repair hook raised. The
  hook encodes an assumption about the document's shape; if it fails, no retry
  of the same bytes will help.

Alongside them, `PersistentHTTPResponseException` is raised when
`scraper.classify()` returns `PERSISTENT` for the observed status, and
`ScraperConfigError` covers configuration that cannot work (a step name that
does not resolve).

**Transient** — `HTTPResponseAssumptionException` (unexpected status),
`RequestTimeoutException` and its `ResolveTimeout` subclass (a browser resolve
that ran out of time but still snapshotted the page), `InterstitialUnresolved`
(a bot challenge a handler could not clear). A bare `TransientException` is
also raisable, but *requires* a `TransientKind` — `NETWORK`,
`BROWSER_CRASH`, `NAVIGATION`, `INTERACTION`, `ARCHIVE`, `SNAPSHOT`,
`INTERSTITIAL`, `SITE_DEGRADED` — so the row records which subsystem failed
rather than leaving operators to grep messages. The kinds are deliberately
coarse: a kind earns its place only if you would filter or group by it.

## Structure is the point, not the message

Each exception lifts its specifics into attributes, and
`jkent.driver.database_engine.errors.describe_error` lifts those attributes into
columns of the `errors` table — `selector`, `selector_type`, `expected_min`,
`expected_max`, `actual_count` for structural failures; `model_name`,
`validation_errors`, `failed_doc` for validation ones; `status_code`,
`timeout_seconds`, `kind` for transient ones; plus `error_class`, the formatted
message, the request URL, the JSON context and the traceback for all of them.

So the operator-visible consequence of raising a *typed* exception rather than
a generic one is a query instead of a grep: "every structural failure on this
court where `actual_count` was 0", "every validation failure against this model",
"transient failures grouped by `kind`". See
[Driver error taxonomy](../../operating/reference/errors.md) and
[Debug a failure](../../operating/adjustments/debug-a-failure.md).

Where an exchange was observed, the exception carries it as `debug_response`
and the worker persists it — including on the transient path, where the latest
failed attempt is kept so the 403 block page or the interstitial that never
cleared is inspectable rather than summarised as a status code. A browser
timeout still snapshots the partial DOM before giving up, which is the entire
reason `ResolveTimeout` exists as its own class.

## Retry is not the only consequence

The persistent/transient choice feeds more than the retry scheduler, which is
worth knowing before picking a parent:

- **The circuit breaker** counts failures to decide whether the site has
  stopped answering. `PersistentHTTPResponseException` is caught ahead of the
  general persistent arm partly so the breaker is *credited* with the server
  having answered — a 404 is a working site, and should not look like an
  outage. See [Circuit breaker](../../operating/architecture/circuit-breaker.md).
- **The rate limiter** reads `retry_after` off the exception. A server-sent
  `Retry-After` floors that request's backoff *and* acts as a global
  slow-down signal. It is carried on `PersistentHTTPResponseException` too: a
  429 classified as persistent is still rate feedback even though this request
  will not be retried.

## Two things that are not errors

Not everything raised is a failure to file, and both exceptions that aren't say
so by not inheriting from either root:

- **`RequestFailedHalt`** is control flow. A step (or a host's worker
  subclass — `jent`'s replay worker raises it on a miss) uses it to tell the
  worker to stop rather than retry or continue. The worker catches it as
  steering and lets it propagate out, recording nothing.
- **`SpeculationHTTPFailure`** is a probe outcome. When a persistent status
  comes back for a speculative request, the worker records a speculation *miss*
  — bumping the template's consecutive-failure count, storing the response with
  a `miss` outcome, marking the request complete and skipping the step — and
  writes no `errors` row. "This guess turned up nothing" is information, not a
  fault. See [Speculation, from the author's side](speculation.md).

Both deliberately sit outside the hierarchy so the worker can dispatch on them
without the general persistent/transient arms catching them first.

## Rejected: optional returns and warn-and-continue

**Returning `None`.** The caller has to check, and the type system is the only
thing reminding them. In practice a scraper is a long chain of extractions
where a missed check propagates: `None` flows into a model field, the model
coerces or defaults it, and a row lands in the database that is structurally
valid and factually wrong. jkent instead makes count expectations explicit at
the query — a selector states how many elements it expects, and a violation
raises with both numbers — so there is no `None` to forget about. See
[Named selectors and observation](selector-contracts.md).

**Warn and continue.** Logging a warning and skipping the row is the same
failure mode as `None`, with a paper trail nobody reads. A warning in a run
that produced 1,800 rows and exited 0 is not an alert; it is an archaeology
exercise. If a condition is genuinely tolerable, the right expression is for the
scraper to decide so explicitly — an optional field is optional *in the model*,
and a page variant the scraper knows about is a branch, not a swallowed
exception.

**Catching broadly inside a step.** Wrapping a step body in
`except Exception: pass` recreates both. It also defeats the retry machinery:
a swallowed `TransientException` is a network blip turned into permanent data
loss, since the driver never learns there was anything to retry.

The one legitimate form of leniency is narrow and local: catch the specific
exception, for the specific known site variant, and either raise something more
descriptive or take a different branch. What the framework refuses to support
is leniency as a default posture.
