# The step model

A step is a generator method decorated with `@step`. It is called with one
response and yields some mixture of data and further requests:

```python
@step
def parse_listing(self, lxml_tree: PageElement, response: Response):
    for row in lxml_tree.checked_xpath("//tr[@class='case']", "case rows", min_count=1):
        link = row.find_links(XPath(".//a"), "case link", min_count=1)[0]
        yield link.follow(step=self.parse_detail)

    next_page = lxml_tree.checked_xpath("//a[@rel='next']/@href", "next link", min_count=0, max_count=1)
    if next_page:
        yield Request(url=next_page[0], step=self.parse_listing)
```

That is the entire execution model. There is no `await`, no callback
registration, no return value the driver inspects. This page is about what
that shape makes possible and what it costs.

## A step is a pure function of a response

The signature the driver calls is always `(self, response)`. What the *author*
writes is `self` plus whatever injectable names they ask for, and the decorator
resolves the difference by inspecting the signature: a parameter named
`lxml_tree` gets a parsed `PageElement`, `json_content` gets parsed JSON,
`text` gets the decoded body, `response` and `request` get the objects
themselves, `previous_request` and `accumulated_data` get the chain's context,
`local_filepath` gets a downloaded archive's path.

Parameter *name* rather than type annotation is the dispatch key, which is
unusual enough to justify: it keeps the common case to one token
(`def parse(self, lxml_tree)`), and it keeps the injector set open — a host can
add one with `register_injector` and the `@step` docstring regenerates its own
list. The cost is that a typo in a parameter name is not an injection and the
method is called wrong, and that the set of magic names is something you have
to know.

Everything injected is a value already in hand. Nothing is a handle, nothing is
awaitable, and consequently a step cannot do I/O even by accident — see
[The scraper/driver split](scraper-driver-split.md).

## Yielding data and requests from one generator

A step yields `ParsedData` or `Request` — the `ScraperYield` union — in
whatever order and interleaving it likes. The driver pattern-matches each
yield:

- `ParsedData` is serialized and staged as a result. If it wraps a
  `DeferredValidation`, validation happens at that moment, and a failure stages
  the *failed document plus its validation errors* rather than discarding them.
- `Request` is resolved and staged for the queue — or, when it carries an
  `incidental=` spec, matched against the parent navigation's captured
  sub-requests.
- `None` is ignored, so a `yield` in a branch that found nothing is harmless.

The reason both kinds share one channel is that the alternative — return data,
call something to enqueue — would mean the enqueue call is a side effect, which
puts a driver object back in the scraper's hands. As a yielded value, a request
is inert: it is a description the scraper hands up, and the scraper has no way
to know or care when (or whether) it runs.

A yielded request is also *completed* on the way out. The `@step` wrapper
passes every `Request` through `inherit_step_metadata`, which resolves
`step=self.parse_detail` (a callable) to its name and fills in `priority`,
`rate_limit` and `timeout` from the target step's own `@step` options where the
request left them unset. An explicit value on the request always wins. So
declaring `@step(rate_limit="search", priority=3)` once governs every request
routed to it, and the author does not restate policy at each yield site. A
string naming no step is left alone for the driver to reject at dispatch, so
the error names the bad step rather than surfacing as an attribute error inside
the scraper.

## Why there is no `await fetch()`

The inverted control flow is the part people push back on, so the trade is
worth stating directly. Writing the sequential version:

```python
# Not possible in jkent, by construction.
listing = await self.fetch("/cases")
for row in rows(listing):
    detail = await self.fetch(row.href)
    yield parse(detail)
```

...would read better. What yielding buys instead:

- **The driver sees all pending work.** A yielded request goes into a durable
  priority queue that the whole worker pool draws from, so fan-out, politeness,
  a shared rate-limit budget and resume all operate over the real set of
  outstanding work. An `await` has already decided to do *this* request next.
- **A step's duration is bounded and its state is nil.** Because a step
  returns promptly and holds nothing, a worker is never parked inside scraper
  code waiting on a network, and there is no scraper-side state to lose when a
  run stops.
- **Every hop is addressable.** A request is a row with a dedup key, a parent,
  a priority and a step name. That is what makes deduplication, retry,
  inspection and replay possible at all; a continuation inside a coroutine is
  none of those things.

The cost: a logical sequence becomes several steps linked by yielded requests,
and context that would have been a local variable has to travel explicitly on
the request — `accumulated_data` and the parent chain. For a deep sequence
that is genuinely more awkward than `await`, and the framework does not pretend
otherwise.

## There is no mid-step resume

A natural reading of "generator" is that the driver can suspend a step, do
something, and resume it where it left off. It does not, and the design leans
on that:

**A step runs to exhaustion, or it is re-run from the top.** The generator's
`send` channel is typed but the driver only ever iterates. So a step that fails
partway is not resumed — the request is retried (for a transient failure) or
failed (for a persistent one), and a retry calls the step again from its first
line against a freshly fetched response.

What makes that safe is that a step's writes are not applied as they are
yielded. They accumulate in a `StagedWrites` buffer, and the results, the
queued child requests and the parent request's `completed` status all land in
**one transaction** at the end of the step. A step that raises halfway leaves
nothing behind: no half-written results, no orphan children, no parent marked
done. Idempotence comes from the buffer, not from the scraper being careful.

The one place a step *is* re-run deliberately is **autowait**, which is also the
clearest illustration of the staging discipline. Under a browser transport, a
step decorated with `@step(auto_await_timeout=...)` that raises
`HTMLStructuralAssumptionException` is not failed immediately. Instead the
driver:

1. discards the staged buffer — *including* the deferred `on_data` callbacks,
   or a dropped attempt's callbacks would fire against rows that were never
   committed;
2. looks the failing query up in the selector observer, walks its parent chain,
   and composes an absolute selector (see [Named selectors and
   observation](selector-contracts.md));
3. waits for that selector on the live page, within the remaining budget;
4. re-snapshots the DOM, stores the new response, and runs the step again from
   the top.

Three guards keep the loop honest, and each is a consequence of what a wait can
actually cure. `wait_for_selector` returns on the first match, so waiting can
only fix *too few* matches: an over-match (`actual_count > expected_max`)
raises at once. A wait that succeeds but leaves the DOM byte-identical also
raises, since the step would fail the same way on it — so every stored
re-snapshot is a changed one. And a selector Playwright cannot wait on at all
skips the mechanism rather than failing differently.

Autowait is therefore not a retry-until-it-works loop; it is "this page renders
late, and the step itself is the readiness check". The step's selector
expectations double as the wait condition, which is only possible because
those expectations are declared data rather than imperative checks.

## What the step declares beyond its body

`@step` takes options that are statements about the request, not the parse, and
they live on the step because the step is the thing a request is routed *to*:

| Option | Effect |
|---|---|
| `priority` | queue ordering hint (lower = earlier), inherited by yielded requests |
| `rate_limit` | the named lane requests to this step are paced in |
| `timeout` | seconds a request to this step waits |
| `encoding` | charset overriding the transport's decoding for this step's responses |
| `preprocess` | a `str -> str` document repair hook applied before parsing |
| `await_list` | browser wait conditions before the response is taken |
| `auto_await_timeout` | enables the autowait loop above |

Two of those are worth a note. `rate_limit` is validated at class-definition
time: `__init_subclass__` rejects a `@step(rate_limit=...)` naming a lane the
scraper does not declare, so the failure is an import error rather than
something that surfaces at the first enqueue to that step, hours in.

`preprocess` is the escape hatch for documents lxml cannot parse as sent — an
unclosed `<style>` swallowing the rest of the page, say. The hook runs on the
response text once, and the repaired text feeds the `text`, `lxml_tree` and
`page` injections, so the step still receives a normal page with its selector
observer wired up. `json_content` is unaffected. If the hook raises, that is
`PreprocessPageException` — the hook encodes an assumption about the document's
shape, and no retry of the same bytes will make it right.

## Interleaving and the observer

One subtlety follows from steps being generators rather than plain calls:
several executions of the *same* step method can be in flight at once across
the worker pool, and their generators resume interleaved. The selector observer
is therefore activated per-resume, around each `next(gen)`, rather than once
around the whole execution — otherwise concurrent executions of one step would
record their queries into each other's observers. The observer itself lives on
the per-execution `Response`, never on the shared step metadata, for the same
reason.
