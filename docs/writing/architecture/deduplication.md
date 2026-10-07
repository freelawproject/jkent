# Request identity and deduplication

A scrape fans out. A listing page yields a request per row; the next listing
page yields some of the same rows again; a detail page links back to its
parent. Without a notion of identity, a run re-fetches the same pages
repeatedly — wasting the rate-limit budget on work already done, and in the
worst case never terminating.

jkent's answer is one column: `requests.deduplication_key`. A request carries a
key, and a key may appear at most once in a run's `requests` table.

## Enforcement is in the schema, not in memory

```sql
UNIQUE (deduplication_key) ON CONFLICT IGNORE
```

That constraint — `uq_requests_dedup_key` — is the whole mechanism. An insert
whose key is already present is a **no-op, not an error**, so an enqueue of an
already-seen request succeeds and inserts nothing, and the calling step needs
no "have I seen this?" branch.

Putting it in the schema rather than in a Python set is deliberate and
load-bearing:

- **It survives a resume.** A run stopped and restarted hours later still
  deduplicates against everything the earlier process enqueued, because the
  record of what was enqueued *is* the table. An in-memory set would start
  empty and re-fetch the world.
- **It is correct across concurrent workers** without a lock. Several workers
  stage enqueues independently; SQLite resolves the race.
- **It cannot drift from the queue.** There is no second structure to keep in
  sync with the rows.

A `NULL` key opts out: it is not covered by the constraint, so such rows never
collide. That is the mechanism behind `SkipDeduplicationCheck` below.

## The default key: the request as the queue stores it

When a scraper sets no key, `effective_deduplication_key` hashes the request
itself — SHA-256 over a JSON array of:

- the HTTP method,
- the URL **with query params folded in**,
- the encoded body as hex, and its "is this form-encoded?" flag,
- the `json` payload, if any.

Four details in that list are the design:

**Params are folded into the URL, because that is how the row is stored.** The
queue stores a request with its `params` already folded into the `url` column,
so `/search?x=1` and `/search` with `params=[("x", "1")]` are *one row's worth
of request*. Hashing the request as the author spelled it would give them two
keys and therefore two rows, and the scraper would fetch the same page twice
for no reason the author could see. The same encoding function is shared with
replay's key derivation so the two cannot drift.

**The parts are joined as a JSON array, so no part can shift another's
boundary.** Concatenating with a separator means a URL containing the separator
can impersonate the next field. This is not hypothetical: the key originally
joined parts with a bare `?`/`|` and rendered params with `str(list)`, and
`/a?[('k', 'v')]` hashed identically to `/a` with `params=[("k", "v")]`. Both
that collision and the two-spellings split are now pinned as properties — see
[Contracts and invariants](contracts.md).

**Dict params and data are taken in name order; pair lists are taken as
given, duplicates included.** A dict has no meaningful order, so sorting makes
the key independent of how the literal was typed. A list of pairs *does* have
meaningful order and meaningful duplicates (a checkbox group, a repeated query
key), so it is hashed as-is. Sorting by `repr` rather than by value keeps that
total: sorting full `(name, value)` tuples raised `TypeError` when two entries
shared a name and carried values of uncomparable types.

**The key is cached per instance, not recomputed per read.** It is a
`cached_property`. The queue reads it at least twice, and a file-like body has
to be read through to hash it. Caching is safe because the operations that
resolve a request — `resolve_from`, `resolve_deferred_fields` — return a *new*
`Request` rather than mutating one, so a request whose URL or form values are
filled in later is keyed by the resolved values, on a fresh instance with a
fresh cache. (Keying by object identity was a real bug here once: a `BytesIO`
body was keyed via `str(...)`, so two requests with identical bytes got
different keys.)

## What the scraper controls

**An explicit string key** replaces the derived one, and says "this is the same
work" in terms of the scraper's own domain rather than the wire:

```python
yield Request(
    url=f"/case?id={case_id}&session={token}",
    step=self.parse_case,
    deduplication_key=f"case:{case_id}",
)
```

Here the URL carries a session token that varies run to run, or between
branches of the same run. The derived key would treat each spelling as new
work; the declared key says the identity is the case, not the URL. This is the
common case for sites that decorate URLs with navigation state.

**`SkipDeduplicationCheck.SKIP`** turns identity off for one request:

```python
yield Request(url="/next-batch", step=self.parse_batch,
              deduplication_key=SkipDeduplicationCheck.SKIP)
```

The cases that need it are the ones where *the same request legitimately
returns different things*: a queue-draining endpoint that pops the next item, a
stateful "advance" postback, a poll against a job that is still running, a
re-fetch of a page whose content is expected to have changed. Deduplicating
those would silently stop the run partway.

The sentinel is an enum member rather than `None` or `False` for a reason worth
knowing, because it is also a trap the API closes deliberately:
`effective_deduplication_key` raises `TypeError` for a key that is neither a
string nor the sentinel. Passing an int docket id or a `date` — the natural
mistake when the key is "obviously" the case number — would otherwise fall
through to `None`, which means *never deduplicate*. A request asking for a
**narrower** identity would silently get none at all. Of the accepted values,
`SKIP` is the only one that produces `None`, so `None` in that column means one
thing and one thing only.

## `cache_key`: a second identity, with no reader

`requests` carries a second digest, `cache_key`: a raw 16-byte MD5 over the
wire request as stored — method, URL, body and its form flag, **headers**, and
the `json` payload. It is computed on every insert and indexed.

It is not the dedup key, and the difference is the interesting part:

| | `deduplication_key` | `cache_key` |
|---|---|---|
| Derived from | the scraper's declared intent, else the request | the wire request, always |
| Includes headers | no | yes |
| Purpose | work that must happen at most once per run | identifying requests whose stored response could be reused |
| Settable by a scraper | yes | no |
| Opt-out | `SKIP` | none |

Nothing in jkent reads it today. That is worth stating plainly rather than
leaving as a puzzle, along with why it stays:

- **A run database outlives the run, and is read by other tools.** The schema
  is a published artifact that hosts query directly. A column that is absent
  from databases written last month cannot be used by a tool written this
  month; a column that was always written can. The asymmetry favours writing
  it.
- **Written once at insert, indexed, and cheap to join on.** Every input to
  the digest is a column the row already has, so a consumer *could* recompute
  it — but only by scanning and hashing every row, and without an index to
  group by. Computing it on the way in costs one hash on a path that is
  already doing a write.
- **Its exact bytes are a compatibility surface.** The hash function changed
  once, which is why `jent` uses `cache_key` as a *schema-version tell*:
  `open_db` probes it to refuse a pre-migration database outright rather than
  half-reading one whose other migrated columns cannot be recovered from what
  is stored. A derived column's value is therefore not free to churn, which is
  an argument for pinning the definition now rather than inventing it later.
- **The cost is bounded and was paid down.** It is stored as a BLOB digest
  rather than hex text, because the column is indexed and written on every
  request and hex would double both the row and the index entry to record
  nothing extra. MD5 is chosen for width, not strength — it is a lookup key
  within one database, never a trust boundary.

What it is *not* is a response cache. jkent does not consult it to skip a
fetch. Response reuse across runs is a host concern, and the host that does it
([replay](scraper-driver-split.md)) matches on the dedup key, not this one.

## Rejected: URL-only identity

The simplest scheme is a `seen: set[str]` of URLs.

- **It is wrong for POST.** Two POSTs to one URL with different bodies are
  different work; a URL-keyed set collapses them and the run silently fetches
  one search and skips the rest. Any site whose search is a form submission —
  which is most of them — breaks immediately.
- **It is wrong for decorated URLs.** A session token or a cursor in the query
  string makes every request unique, so a URL-keyed set deduplicates nothing
  at all. This is the exact case the explicit-key escape hatch exists for, and
  a URL-only scheme has no escape hatch to offer.
- **It has no way to say "do this twice".** A poll or a queue-pop needs to
  bypass identity. With a set of URLs the only option is not to record the URL,
  which is a different code path rather than a value on the request.
- **It does not survive a resume** unless it is persisted — at which point it
  is a column, and the question becomes what goes in it. The answer is
  everything above.

The chosen design costs a scraper author one thing: when a site's URLs carry
state, they must notice and declare a key. That is a real failure mode — a run
that re-fetches more than it should, or a key narrower than the work it
identifies, both of which are visible in the run database as request counts
that do not match expectations. See [Inspect the
results](../../operating/walkthrough/03-inspect-the-results.md).
