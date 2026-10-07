# jkent

Jkent is a scraper-driver framework for structured web scraping. It separates
parsing logic from I/O orchestration: a **scraper** is a collection of pure
parsing steps that yield data and follow-on requests; a **driver** handles
HTTP, browsers, storage, rate limiting and persistence.

The documentation is split by what you are doing.

## [For scraper writers](writing/index.md)

You are writing a scraper against a site. You care about the authoring API —
decorators, selectors, requests, exceptions — and never about worker pools.

## [For jkent operators](operating/index.md)

You are running a scraper someone wrote and getting a clean run database out of
it. You care about transports, concurrency, storage and failure handling.

Each half is organised the same way: **tutorials** to learn by doing, a
**cookbook** of task-sized recipes, **reference** for exact detail, and
**architecture** for why things are the way they are.

!!! warning "Alpha"
    APIs and database schemas may change without notice.
