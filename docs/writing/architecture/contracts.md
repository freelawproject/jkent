# Contracts and invariants

Some invariants in jkent are not obvious from a function's types and are not
cheap to assert on the hot path — "this dedup key is a function of the stored
request, not of how the author spelled it", "a returned element list always
satisfies the caller's count bounds". These are written as
[icontract](https://github.com/Parquery/icontract) pre- and postconditions
through `jkent.contracts`, and they are **off in every production install**.

## The gate

`jkent.contracts` exports two decorators, `require` and `ensure`, which stand
in for icontract's. The module reads one environment variable at import:

```python
ENFORCE_CONTRACTS: bool = os.environ.get(
    "JKENT_ENFORCE_CONTRACTS", "0"
) not in ("", "0")
```

When it is unset or `0`, `require` and `ensure` return the decorated function
*untouched* — `lambda fn: fn`. There is no wrapper, the condition lambda is
never called, and `icontract` is never imported. When it is set to anything
else, the real icontract decorators are applied and a violation raises.

Two consequences follow from that shape, and both are the point:

- **`icontract` is a dev dependency, not an SDK one.** It appears in the `dev`
  dependency group and nowhere in `[project] dependencies`. A scraper package
  installing plain `jkent` does not pull it. This matters because the SDK's
  dependency list is deliberately short (see
  [The scraper/driver split](scraper-driver-split.md)) and a contract library
  is not something a parsing-only consumer should inherit.
- **Decorated functions cost nothing when contracts are off.** Not "a cheap
  boolean" — nothing. The decorator is the identity, so the resulting function
  object *is* the original, and there is no extra frame per call. Conditions can
  therefore be as expensive as they need to be to say something true.

The gate is evaluated **at decoration time**, which is import time. Setting
`JKENT_ENFORCE_CONTRACTS=1` after a `jkent` module has been imported does
nothing — the decorators have already resolved to the identity. The test suite
handles this by setting it at the very top of `tests/conftest.py`, before any
jkent import.

## Rejected: always-on runtime enforcement

The tempting alternative is to leave contracts enabled in production, on the
theory that an invariant worth stating is worth checking.

Rejected, for three reasons:

- **The conditions are not production-priced.** A postcondition that recomputes
  a result a second way to compare against it — which several of these do — is
  reasonable as a test oracle and absurd per request in a run doing hundreds of
  thousands of them.
- **A violation has no useful production behaviour.** In a long scrape, raising
  on a broken internal invariant converts a subtly wrong row into a dead run.
  The driver already has a designed failure taxonomy for things that go wrong
  at runtime (see [Why assumptions are exceptions](assumption-exceptions.md));
  an internal invariant breach is a bug to fix in development, not a condition
  to handle in the field.
- **It would make the library's dependency surface a function of a debugging
  tool.** `icontract` would have to become a runtime dependency of the SDK for
  the benefit of a check nobody in production wants.

The inverse — expressing these as plain `assert` statements — was also
rejected: `assert` carries no description, cannot see the return value, and is
stripped under `-O` anyway, so it gets the worst of both ends. What icontract
buys over a bare assert is that a condition is a lambda over named parameters
(and, for `ensure`, over `result`), which is what makes the next section
possible.

## The property suite

Contracts stated as lambdas over named arguments can be explored
*symbolically*. `tests/contracts/properties.py` is a harness of property
functions: each one takes inputs, calls production code, and states the
expected relationship as an icontract postcondition. CrossHair then searches
the input space for a counterexample:

```bash
uv run crosshair check --analysis_kind=icontract tests/contracts/properties.py
```

The harness uses `icontract` directly rather than going through the gate, so
that command needs no environment setup. To CrossHair-check the contracts on a
*production* module instead, the gate has to be on:

```bash
JKENT_ENFORCE_CONTRACTS=1 uv run crosshair check \
    --analysis_kind=icontract <module>
```

Properties that cannot be written as a postcondition on a production function
— permutation invariance, or two implementations that must agree — are why the
harness exists at all: a postcondition sees a single call, so the harness makes
both calls and the postcondition compares the two results it returns.

**Every property in that file was once falsifiable.** The suite is not
decoration; it found bugs, and each function's docstring records the one its
counterexample exposed. A sample from the deduplication key alone:

- `sorted()` over full `(name, value)` tuples raised `TypeError` when two
  params shared a name and carried values of uncomparable types —
  counterexample `[("k", 0), ("k", "")]`. Params now sort by `repr`, which is
  total.
- A file-like body was keyed via `str(request_params.data)`, so a `BytesIO`
  hashed its memory address: two requests with identical bytes got different
  keys.
- The key hashed the request *as spelled* and joined parts with an unescaped
  `?`/`|`, so `/a?x=1` and `/a` with `params=[("x", "1")]` — one row's worth of
  request — got two keys, and a URL containing another request's rendered
  params text collided with it.

Because CrossHair is slow and not something to run per commit, each
counterexample is pinned as a Hypothesis `@example` in
`tests/contracts/test_properties.py`, which drives the same property functions
with generated inputs on every test run. So the properties act as ordinary
regression guards in CI, and can be re-explored symbolically on demand.

One caveat worth knowing before chasing a report: CrossHair's symbolic models
of the standard library are not perfect, and the harness docstring lists the
known artifacts — a spurious `ValueError` from `unquote` on non-BMP characters,
and `RecursionError`/`NotDeterministic` on long runs, which are the symbolic
interpreter blowing its own stack rather than the code under test. Re-run a
reported input concretely before treating it as a bug.

## What this means for a scraper author

Nothing, during a normal run — which is the design. Two things are worth
knowing:

- If you are running jkent's own test suite or developing against the
  framework, contracts are on, and a violation you see is a framework-level
  invariant breach worth reporting rather than something to work around.
- Contracts are not input validation and are not a substitute for it.
  Everything a scraper can get wrong about a *site* — a missing element, an
  unparseable date, an unexpected status — is handled by the assumption
  exception taxonomy, which is always on. Contracts guard jkent's internals
  against jkent.
