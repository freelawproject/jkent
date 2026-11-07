# Following links across pages

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/common/request.py, jkent/driver/unified_driver/steps.py
     Delete this comment when the page has real content. -->

Turn a single page into a crawl by yielding further Requests from a step.

## Planned contents

- Yield `Request(..., step="...")` to queue follow-on work
- How steps chain without the scraper doing any I/O
- `accumulated_data` and carrying context between steps
- Watching the queue drain
