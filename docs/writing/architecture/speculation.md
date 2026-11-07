# Speculation, from the author's side

Some sites have no listing. The only way in is a URL with an identifier in it
— `/case/2026-00017` — and no page anywhere enumerates which identifiers
exist. The conventional options are both bad: fetch a fixed range and hope it
covers the year, or fetch until something breaks and hope the break is
meaningful.

Speculation makes the guessing explicit. The scraper declares the shape of the
identifier space; the driver probes it, learns where the live region ends, and
stops. Every guess that turned up nothing is recorded as a guess, not as an
error.

This page is the author's half: what you declare, what the driver guarantees in
return, and what your step must be true for it to work. The run-side mechanics
— ceiling advancement, persistence, resume — are in
[Speculation](../../operating/architecture/speculation.md).

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

Four members, and each is a question only the author can answer:

- **`seed_range()`** — the integer IDs to probe immediately, as a `range`
  (`start` inclusive, `stop` exclusive). An *empty* range is legal and means
  "seed nothing; rely entirely on the advance window".
- **`from_int(n)`** — build a template for ID `n`, preserving every other
  field. This is the method that makes the model the single description of the
  space: the driver never constructs your parameters, it asks you to.
- **`max_gap()`** — how many consecutive misses *beyond the highest success*
  mean the live region has ended. Also the size of the advance window.
  Returning `0` disables advancement entirely.
- **`should_advance`** — whether to push the ceiling past `seed_range()` on
  success at all. `False` means probe exactly the declared range and stop.

Note what jkent does **not** own: the parameter class, its fields, or how an ID
becomes a URL. It owns a loop over integers and the bookkeeping around it. The
semantics of the identifier are entirely the scraper's, which is why
`from_int` is abstract rather than a dataclass `replace`.

## The bet being made

`max_gap` is a wager about the *density* of the identifier space, and it is
worth being conscious of it, because it is the one parameter that can lose you
data silently.

The loop stops after `max_gap` consecutive misses past the highest success.
That is correct when identifiers are dense — a sequence with occasional sealed
or deleted records. It is wrong when the space has real holes: if a court skips
20 numbers mid-year and `gap=3`, the run stops at the hole and everything after
it is never fetched. Nothing errors. The run completes, with less data in it.

So the choice is:

- **Too small**: the run stops early at the first real gap. Quiet data loss.
- **Too large**: the run spends rate-limit budget on probes that will all miss,
  and the tail of every scrape is slow.

The honest way to set it is to look at a run's `speculation_tracking` row and
the `miss` outcomes against the identifiers you know exist, rather than
guessing; the whole reason misses are recorded rather than discarded is to make
that possible. For a space you know to be sparse, a declared `seed_range()`
covering the year with `should_advance=False` is the safer shape: it trades
"stop when the data runs out" for "probe exactly this, which I can reason
about".

## What your step must be true for

A speculative probe is a request whose target may not exist. Two correctness
requirements follow, and both are on the author.

**1. A miss must be recognisable from the response.** The driver's default
notion of a miss is "a persistent HTTP code" — a clean 404 on a nonexistent
case is ideal. Many sites do not oblige: they return 200 with a "No results
found" page, a redirect to a search form, or a soft 404. For those, override
`actually_successful`:

```python
def actually_successful(self, response: Response) -> bool:
    if "No matching case" in response.text:
        return False
    if response.url.endswith("/search"):   # bounced back to the form
        return False
    return True
```

A 2xx that `actually_successful` rejects counts toward the gap exactly as a
non-2xx status would. **Getting this wrong breaks the loop in the worse
direction**: if soft 404s read as successes, the gap never accrues, the ceiling
advances forever, and the run probes until something else stops it. The miss
detector is not an optimisation; it is the loop's termination condition.

**2. The step must tolerate being wrong about existence.** A probe that misses
never reaches your step — the driver stores the response, records the outcome
and skips the step. But a probe that *hits* reaches a step that may be looking
at a page the scraper has not seen before: a stub record, an unreleased entry,
a case with none of the usual fields. The step should raise an assumption
exception on genuinely unexpected structure (that is the system working), but
it should not treat "this record is sparse" as structural breakage if sparse
records are a real state of the site. Model fields that are optional for a
legitimately incomplete record belong optional in the model.

A third, softer requirement: **a speculative step should be cheap to have
wasted.** The probe is a fetch that may produce nothing, so anything expensive
the step triggers — an archive download, a fan-out of child requests — is
expensive per *guess*, not per record.

## A miss is recorded, not hidden

The thing that makes speculation auditable rather than a black box is that
every probe outcome is a stored value on the request's response, as a
`SpeculationOutcome`:

| Outcome | Meaning |
|---|---|
| `hit` | the probe found something; its step ran, and the template's ceiling rose |
| `miss` | a persistent code, or a 2xx `actually_successful` rejected; response stored, step skipped |
| `stopped` | the miss that ended this template's probing; stored like a miss |
| `terminated_early` | dequeued after its template had already stopped — never fetched, so no response |

Two consequences matter to an author:

**A miss is not an error.** When a persistent HTTP code comes back for a
speculative request, the transport raises `SpeculationHTTPFailure` rather than
`PersistentHTTPResponseException`, and the worker writes **no `errors` row**:
it bumps the template's consecutive-failure count, stores the probe's response
with a `miss` outcome, marks the request complete and skips the step. This is
deliberate and is the difference between a usable error table and one full of
404s you asked for. See [Why assumptions are
exceptions](assumption-exceptions.md).

**The response is still stored.** A miss keeps its body. That is what lets you
check afterwards whether your miss detector was right — a "No results" page
stored against a `miss` outcome confirms the detector fired on the page you
think it did, and a *successful-looking* page stored against a `miss` is the
signal that `actually_successful` is wrong. Only `terminated_early` has no
response, because it was never fetched.

Probes are also **never deduplicated** (`insert_root_request(...,
deduplicate=False)`): the speculation engine owns which IDs it has tried, via
the template's ceiling, and letting the dedup constraint silently drop a probe
would corrupt that accounting.

## What the driver guarantees back

So that the author knows which invariants not to re-implement:

- **The ceiling is persisted before any probe is enqueued**, and re-recorded
  after seeding, so a resume plans *past* the ceiling instead of re-probing the
  same IDs.
- **Outcomes are counted only for IDs past the highest success.** A late miss
  on a low ID does not accrue toward the gap, so out-of-order completion across
  the worker pool cannot stop the loop early.
- **State is per template.** Each parameter invocation of a speculative entry
  gets its own `SpeculationState` and its own tracking row, so two years
  probed in one run do not share a ceiling or a gap count.
- **A probe that cannot be attributed is logged, not guessed at.** A request
  missing its tracking fields records nothing rather than being counted against
  some other template.
