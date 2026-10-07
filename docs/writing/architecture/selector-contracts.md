# Named selectors and observation

Every selector query in a jkent scraper carries two things an ordinary
`tree.xpath(...)` call does not: a **human description** and an **expected
count**.

```python
rows = tree.checked_xpath("//table[@id='results']//tr", "result rows", min_count=1)
date = row.checked_xpath(".//td[2]/text()", "filing date", min_count=1, max_count=1)
```

Both are mandatory. This page is about why the extra typing is worth it, and
what the framework does with it that it could not do otherwise.

## A selector always states its grammar

`Selector` is a value carrying an expression *and* the grammar it is written
in — `XPath("//tr")` or `CSS("table tr")`. There are exactly two grammars and
no third form: no bare strings, no prefix heuristic, no inference.

The reason is that a grammar guess has to be re-made at every layer that
touches the selector. A `via` persisted to the run database, the browser
transport deciding whether Playwright can wait on a selector, the `errors`
table's `selector_type` column, the composition of a relative chain into an
absolute one — each would otherwise re-run the same "does it start with `/`?"
guess, and they would eventually disagree. So grammar-specific behaviour is an
abstract method on `Selector` implemented by both subclasses: the lxml query,
the positional wrapper (`nth`), relative-chain composition, and the
Playwright-waitability predicate. `Selector` also carries its own pydantic
schema, so a model holding one serializes it as
`{"value": …, "grammar": …}` and validates straight back — the grammar
round-trips as data, never as a guess.

`jkent.common.selectors` is a leaf module: it imports nothing from jkent, and
it never names lxml in a signature either. `Selector.query` takes an
`ElementQuery`, a structural protocol that a `PageElement` and a raw lxml
element both satisfy, which keeps the parse-tree implementation behind
`PageElement` instead of surfacing in every selector's type.

## The count is the contract

`PageElement._checked` is the single query engine behind every selector
method. It runs the selector, filters results to the requested kind (strings or
elements), reports to the observer, and *then* enforces the bounds. A violation
raises `HTMLStructuralAssumptionException` carrying the selector, its grammar,
the description, the expected min/max and the actual count.

Three details in that order matter:

- **Filtering happens before counting**, so the count the bounds were checked
  against is the same count the observer recorded. The two can never tell
  different stories about the same query.
- **A scalar result counts as one.** A nodeset XPath returns a list, but
  `count()`/`string()`/`normalize-space()` return a bare value. It is wrapped
  before counting, because iterating a bare string would count its
  *characters* — a two-digit `count()` would report `len(str)` results and
  spuriously trip the bounds.
- **An unparseable selector is not a structural failure.** A selector that
  lxml cannot compile raises `ScraperConfigError`, not
  `HTMLStructuralAssumptionException`. That is the difference between "the
  scraper is broken" and "the website changed", and conflating them would
  pollute exactly the signal structural errors exist to provide. See
  [Why assumptions are exceptions](assumption-exceptions.md).

Stating the count also removes the `None` that a conventional API hands back.
There is no "did it match?" branch to forget: `min_count=1, max_count=1` means
the result is one element or the step raised.

## What the observer does with the descriptions

`SelectorObserver` records every query that runs. It is activated either by
direct injection into a `PageElement` or — the usual path — as a context
manager, with `PageElement` picking it up through a `contextvars` lookup
(`get_active_observer()`). The `@step` wrapper activates one around every
generator resume, and *adopts an already-active observer* when an outer `with`
block installed one. That contextvar is a token **stack**, not a single slot,
precisely so the inner activation can nest inside an outer one of the same
instance without clobbering the outer's reset.

What it records per query: the selector and its grammar, the description, the
match count, the expected bounds, up to three content samples truncated to 100
characters, and the **parent chain** — queries nest, so a `.//td` run against
each row records as a child of the `//tr` that produced the rows. Repeats of
the same selector under the same parent (the usual loop-over-rows shape)
deduplicate into one entry with aggregated counts and samples.

That structure buys three things.

**Autowait inside the driver.** When a step raises a structural exception under
a browser transport, the step executor looks up the failing query in the
observer tree — matched on selector *plus* grammar *plus* description, so a
selector reused under different parents resolves to the node that actually
failed — walks its parent chain and composes an absolute selector. It then
waits for that selector in Playwright and re-runs the step. A relative
`.//td[2]` is meaningless to `wait_for_selector`; `//table//tr//td[2]` is not.
If any matches already exist, the wait targets `>> nth=<expected_min - 1>`,
since a plain wait would return immediately. Composition returns `None` if the
chain mixes grammars, and the driver falls back to the raw selector. This is a
*behavioural* payoff: a step that would have failed on a slow-rendering page
succeeds, and it works because the queries described themselves.

**Selector coverage across a corpus.** `jent` reruns every selected request of
a step under an observer and folds the results together: per query, how many
pages it matched on, plus the per-page matched-selector sets used as input to
exemplar set-cover. Aggregation identity is the `(selector, description)` pair
— which is why the description cannot be optional. Two selectors that are
textually identical but describe different things stay distinct, and a selector
edited without changing what it means stays joined to its history. A page whose
step raises still contributes the queries that ran before the failure, which is
the only way the failing query is observable at all.

**Residual accounting.** The same machinery inverted: re-run a step under an
observer that tracks matched nodes, mark every DOM text slot the selectors
consumed, subtract chrome that is constant across the corpus, and whatever
remains is unclaimed variable content — data on the page that the scraper is
not selecting for. A conditional block appearing on 40% of pages is no
structural outlier and no error, so nothing else would surface it; it lights up
as residual.

Note what both of those have in common: they are implemented entirely outside
jkent, by a host that never modifies the scraper. The observer is installed by
the driver and read by the host, and the scraper neither knows nor can tell.
That is the same property the [scraper/driver
split](scraper-driver-split.md) is built on, applied to introspection.

## The cost of the convention

Worth naming plainly, because it is paid on every line:

- **Verbosity.** `checked_xpath("//tr", "result rows", min_count=1)` is three
  arguments where `xpath("//tr")` is one. In a scraper with fifty queries, that
  is fifty descriptions to write and keep honest.
- **Descriptions rot.** Nothing checks that "filing date" still describes the
  cell. A stale description degrades coverage reporting and error messages
  without failing anything. The partial mitigation is that descriptions appear
  in error messages, so the ones that matter get read when something breaks.
- **Bounds are a judgement call.** `min_count=1` is usually right and
  sometimes wrong — a results page legitimately showing zero rows is a real
  case, and writing `min_count=0` there means the query can no longer tell
  "empty result set" from "the table moved". The honest resolution is a
  separate query that asserts the *empty-state* marker is present, which is
  more work than a lenient bound.
- **An aggregation identity that includes prose.** Because coverage keys on
  `(selector, description)`, rewording a description splits its history in
  reporting. That is the correct trade for distinguishing same-text selectors,
  but it means descriptions are closer to identifiers than to comments.

The convention pays for itself when a site changes: the error names what was
being looked for in the author's own words, with both counts, and the tooling
can say which pages that selector stopped matching on. That is the difference
between "a selector broke" and "the filing-date cell moved on the 12% of pages
that use the new layout".
