# The scraper/driver split

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/__init__.py, jkent/common/scraper.py
     Delete this comment when the page has real content. -->

Why a scraper is pure parsing plus navigation intent, and the driver owns all I/O.

## Planned contents

- What a scraper may and may not do
- What the split buys: testability, replay, transport substitution, resumability
- A **Rejected** section: scrapers that fetch their own pages, and why that forecloses replay
- Where the boundary leaks and how it is policed
