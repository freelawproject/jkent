# Your first scraper

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/common/scraper.py, jkent/common/decorators.py, README.md
     Delete this comment when the page has real content. -->

Build a one-page scraper that fetches a list and yields structured data.

## Planned contents

- Install the SDK and the operational extra
- Subclass `BaseScraper[T]` and declare an `@entry`
- Write one `@step` that yields `ParsedData`
- Run it with `RunBootstrapper` and look at the rows
- What the run database now contains
