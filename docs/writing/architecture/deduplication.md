# Request identity and deduplication

<!-- STUB — not yet written.
     Lands with: feat(database_engine): add staged writes and request queue
     Sources: jkent/common/request.py, jkent/driver/database_engine/queue.py
     Delete this comment when the page has real content. -->

How the driver decides two requests are the same, and what the scraper controls.

## Planned contents

- The dedup key and what goes into it
- `SkipDeduplicationCheck` and the cases that need it
- Why `requests.cache_key` is retained with no current reader
- A **Rejected** section: URL-only identity
