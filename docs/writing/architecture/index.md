# Architecture

Why the authoring API is shaped the way it is. These pages are background, not
instructions — nothing here is needed to write a working scraper, and none of
it is a how-to. Read it when you want to know why the framework asked you to do
something the way it did, or when you are about to argue with one of its
constraints.

Each page states the alternative that was rejected and why, so that the
arguments are available to be re-examined rather than rediscovered.

## Reading order

Everything else follows from the first page, so start there. After that the
order is roughly "most likely to affect how you write a scraper" first.

1. **[The scraper/driver split](scraper-driver-split.md)** — the one decision
   the rest is downstream of: a scraper is parsing plus navigation intent, and
   performs no I/O. What that buys, where the boundary leaks, and why a scraper
   that fetches its own pages forecloses replay, resume and scheduling.
2. **[The step model](the-step-model.md)** — steps as generators, navigation as
   a yielded value, and why there is no `await fetch()`. Also what "the driver
   re-runs a step" means, and the staging discipline that makes it safe.
3. **[Why assumptions are exceptions](assumption-exceptions.md)** — the
   taxonomy, and the argument that silent failure is the expensive failure mode
   in scraping. Read this before deciding to catch something.
4. **[Named selectors and observation](selector-contracts.md)** — why every
   query carries a description and an expected count, what the observer does
   with them, and the cost of the convention.
5. **[Declarative navigation](via-and-form-transport.md)** — why form
   submission is described rather than constructed, and how the two transports
   are held to a real browser's behaviour by a generative conformance rig.
6. **[Request identity and deduplication](deduplication.md)** — how the driver
   decides two requests are the same work, and the two cases where you have to
   tell it.
7. **[Speculation, from the author's side](speculation.md)** — what it means to
   yield a request you are not sure you need, and what your step has to be true
   for.
8. **[Contracts and invariants](contracts.md)** — the dev-time-only contract
   gate and the property suite behind it. Mostly of interest if you are working
   on jkent itself.

## What belongs here, and what does not

This section covers the **authoring** half of the framework: the SDK, the
contract between a scraper and whatever runs it, and the reasoning behind the
API surface you write against.

The mechanics of a run live in the operator half —
[Architecture (operating)](../../operating/architecture/index.md) — and the two
are deliberately not duplicated. Where a decision has both an authoring face
and a runtime face, each side gets the half it owns and links across:

| Topic | Here | There |
|---|---|---|
| Transports | why a scraper never knows which one it is on | [Transports](../../operating/architecture/transports.md) — the ABC, the conformance suite, where the browser cannot match httpx |
| Speculation | what you declare and what your step must tolerate | [Speculation](../../operating/architecture/speculation.md) — ceiling advancement, persistence, resume |
| Failure | the exception taxonomy and when to raise | [Circuit breaker](../../operating/architecture/circuit-breaker.md), [Rate limiting](../../operating/architecture/rate-limiting.md) — what the driver does with a failure |
| Requests | identity, and what a scraper controls | [Request queue](../../operating/architecture/request-queue.md), [Resumability](../../operating/architecture/resumability.md) |
| Storage | nothing: a scraper never touches it | [Persistence](../../operating/architecture/persistence.md), [Compression](../../operating/architecture/compression.md), [Compaction](../../operating/architecture/compaction.md) |

If you are looking for **how to do something**, the [Cookbook](../cookbook/index.md)
is task-sized and this section is not. For exact signatures and parameters, see
[Reference](../reference/index.md).
