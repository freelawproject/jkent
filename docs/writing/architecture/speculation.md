# Speculation, from the author's side

Some sites have no listing. The only way in is a URL with an identifier in it,
`/case/2026-00017`, and no page anywhere tells you which identifiers exist. We
want those sites to be scrapable without either hard-coding a range and hoping
it covers the year, or fetching until something breaks and hoping the break
means what you think it means.

Speculation is how we do that. You describe the shape of the identifier space,
the driver probes it, works out where the live region ends, and stops. Every
guess that turns up nothing is stored as a guess, with its response, so you can
check afterwards whether the guessing was any good.

This page is your half: what you declare, what the driver gives back, and what
your step has to cope with. The run-side mechanics (ceiling advancement,
persistence, resume) are in [Speculation, driver
side](../../operating/architecture/speculation.md).

## What you declare

An `@entry` function takes a typed parameter model. When that model also
subclasses `Speculative`, the driver detects it and runs the speculation loop
instead of enqueueing a single request:

```python
class DocketId(BaseModel, Speculative):
    year: int
    min: int
    soft_max: int = 0
    should_advance: bool = True
    gap: int = 3

    def seed_range(self) -> range:
        return range(self.min, self.soft_max)

    def from_int(self, n: int) -> DocketId:
        return DocketId(
            year=self.year, min=n, soft_max=self.soft_max,
            should_advance=self.should_advance, gap=self.gap,
        )

    def max_gap(self) -> int:
        return self.gap
```

Four members, and each answers a question only you can answer:

| Member | What it says |
|---|---|
| `seed_range()` | The IDs to probe outright, as a `range` (`start` inclusive, `stop` exclusive). An empty range is legal and means "seed nothing, rely entirely on the advance window". |
| `from_int(n)` | A template for ID `n`, keeping every other field. The driver never builds your parameters; it asks you to. |
| `max_gap()` | How many consecutive misses past the highest success mean the live region has ended. Also the size of the advance window. `0` turns the window off. |
| `should_advance` | Whether to push past `seed_range()` on success at all. `False` means probe exactly the declared range. |

The entry itself is an ordinary `@entry`, with two restrictions:

```python
@entry(CaseData)
def fetch_case(self, case_id: DocketId) -> Request:
    return Request(
        request=HTTPRequestParams(
            method=HttpMethod.GET,
            url=f"{self.base}/case/{case_id.year}-{case_id.min:05d}",
        ),
        step="parse_case",
    )
```

The driver rebuilds that call once per probe, as
`fetch_case(case_id=template.from_int(n))`, and takes the single `Request`
back. So the speculative parameter has to be the only parameter, and the entry
has to return one request rather than yield. `@entry` checks both when the
class is defined, because the alternative is a `TypeError` on every probe at
run time:

> Speculative entry function 'fetch_case' takes parameters besides its
> Speculative one 'case_id': ['court']. The driver calls a speculative entry
> with that parameter alone, so the others could never be supplied; put them on
> the Speculative model instead.

If you need a second value, it belongs on the model, where `from_int` carries
it to every probe.

Note what jkent doesn't own here: your parameter class, its fields, or how an
ID becomes a URL. It owns a loop over integers and the bookkeeping around it.
The meaning of the identifier stays with the scraper, which is why `from_int`
is abstract rather than a `dataclasses.replace`.

## The seed range and the advance window

Probes come from two places, and the driver treats them differently. Most of
what follows depends on knowing which is which.

| | Seed range | Advance window |
|---|---|---|
| Where the IDs come from | `seed_range()`, so from you | the driver, guessing past the frontier |
| Which IDs | `[start, stop)` | up to `max_gap()` IDs past the seed range, or past the highest success if that's further |
| Always fetched? | yes, in full | no, see below |
| A miss here | recorded, and nothing else | counts toward `max_gap()` |

The window grows on success: whenever a hit lands within `max_gap()` of the
current ceiling, the driver opens another `max_gap()` IDs above it. Misses in
the window accrue, and `max_gap()` of them past the highest success stops the
template. Window probes already sitting in the queue when that happens are
completed without being fetched, as `terminated_early`.

A probe is only retired when nothing we know says it exists: its ID is past the
highest success and outside the seed range. You asked for the seed range, so a
run of misses in it says nothing about where the IDs end. And an ID below a
success sits in a region the site has already answered for, which matters
because the pool completes probes out of order: ID 7 can come back a hit, open
the window further, and have the IDs above it stop the template while 5 and 6
are still queued. Both get fetched. That also makes `should_advance=False` (and `max_gap()`
of `0`) do the obvious thing: there's no window, the template is stopped from
the moment it's seeded, and the declared range still gets fetched from end to
end.

## The bet in max_gap

`max_gap` is a wager about the density of the identifier space, and it's the
one parameter that can lose you data quietly, so it's worth being deliberate
about.

Stopping after `max_gap` consecutive misses is right when identifiers are
dense, with the occasional sealed or deleted record. It's wrong when the space
has real holes: if a court skips 20 numbers mid-year and `gap=3`, the window
closes at the hole and everything above it goes unfetched. Nothing errors, the
run completes, and there's less data in it.

- Too small: the run stops at the first real gap.
- Too large: the run spends rate-limit budget on probes that will all miss, and
  the tail of every scrape is slow.

Rather than guessing, look at a finished run: the `speculation_tracking` row
and the `miss` outcomes against identifiers you know exist tell you where the
window actually closed. That's the reason misses are stored instead of
discarded.

For a space you know to be sparse, declare the range and turn the window off.
`seed_range()` covering the year with `should_advance=False` trades "stop when
the data runs out" for "probe exactly this", and since seed-range misses don't
stop anything, a hole in the middle costs you nothing.

## What your step has to cope with

A speculative probe is a request whose target may not exist. Two correctness
requirements follow from that, and both are yours.

**A miss has to be recognisable from the response.** By default a miss is "a
status the scraper classifies as persistent", so a clean 404 on a nonexistent
case is ideal. Transient statuses (`408`, `425`, `429`, `502`, `503`, `504` out
of the box) are retried instead, which is why a flaky gateway doesn't retire a
template. Everything else non-2xx is persistent, `500` included, so a site that
answers a probe with a flaky `500` rather than a gateway code wants that code
reclassified on the scraper (`HTTP_CODE_TYPES = {500: HTTPCodeType.TRANSIENT}`)
rather than an exemption here.

Many sites don't oblige with a 404 at all: they return 200 with a "No
results found" page, a redirect to a search form, or a soft 404. For those,
override `actually_successful`:

```python
def actually_successful(self, response: Response) -> bool:
    if "No matching case" in response.text:
        return False
    if response.url.endswith("/search"):   # bounced back to the form
        return False
    return True
```

A 2xx that `actually_successful` rejects is a miss, exactly as a persistent
status would be. Getting this wrong breaks the loop in the worse direction: if
soft 404s read as successes, the gap never accrues, the window keeps opening,
and the run probes until something else stops it. Your miss detector is the
loop's termination condition.

**The step has to tolerate being wrong about existence.** A probe that misses
never reaches your step; the driver stores the response, records the outcome
and skips it. A probe that hits may hand your step a page the scraper has never
seen: a stub record, an unreleased entry, a case with none of the usual fields.
Raise an assumption exception on genuinely unexpected structure, which is the
system working, but don't treat "this record is sparse" as structural breakage
when sparse records are a real state of the site. If a field can legitimately
be absent, make it optional in the model.

There's a softer third requirement: a speculative step should be cheap to have
wasted. Anything expensive it triggers, like an archive download or a fan-out
of child requests, is expensive per guess rather than per record.

## What gets recorded

Every probe outcome is stored on the request as a `SpeculationOutcome`:

| Outcome | Meaning |
|---|---|
| `hit` | the probe found something; its step ran, and the window may have opened further |
| `miss` | a persistent status, or a 2xx `actually_successful` rejected; response stored, step skipped |
| `stopped` | the window miss that ended this template's probing; stored like a miss |
| `terminated_early` | a window probe dequeued after its template stopped, so never fetched and no response |

A miss doesn't get an `errors` row. When a persistent status comes back for a
speculative request the transport raises `SpeculationHTTPFailure` rather than
`PersistentHTTPResponseException`, and the worker stores the response with a
`miss` outcome, marks the request complete and skips the step. That's the
difference between an error table you read and one full of 404s you asked for.
See [Why assumptions are exceptions](assumption-exceptions.md).

The response body is kept, which is what lets you check your miss detector
afterwards. A "No results" page stored against a `miss` confirms the detector
fired on the page you think it did; a page that looks like a real record stored
against a `miss` is how you find out `actually_successful` is wrong. Only
`terminated_early` has no response, because nothing was fetched.

Probes are never deduplicated (`insert_root_request(..., deduplicate=False)`).
The speculation engine tracks which IDs it has tried through the template's
ceiling, and a dedup constraint silently dropping a probe would corrupt that
count.

## What the driver guarantees back

So you know which invariants not to re-implement:

- The tracking row exists before any probe is enqueued, so every probe can
  point at it, and the ceiling is written once the seeding enqueue finishes.
  Reopening a run plans past the recorded ceiling instead of re-probing the
  same IDs. A crash partway through seeding loses the ceiling write, so the
  reopened run re-enqueues that range; probes aren't deduplicated, so you get
  duplicate fetches rather than missing ones.
- Only window misses past the highest success count toward the gap, and a
  success below the highest doesn't reset the count. The count is still by
  arrival rather than by ID, so misses that a later-arriving success turns out
  to sit below do count, and `stopped` never reverts. What they can't do is
  retire a probe below that success.
- State is per template. Each parameter invocation of a speculative entry gets
  its own `SpeculationState` and tracking row, so two years probed in one run
  don't share a ceiling or a gap count.
- A probe that can't be attributed to a template is logged rather than guessed
  at. A request missing its tracking fields records nothing instead of counting
  against some other template.
