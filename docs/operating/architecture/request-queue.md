# The request queue

<!-- STUB — not yet written.
     Lands with: feat(database_engine): add staged writes and request queue
     Sources: jkent/driver/database_engine/queue.py
     Delete this comment when the page has real content. -->

Ordering, priority and deduplication inside the driver.

## Planned contents

- Priority semantics and `ARCHIVE_DEFAULT_PRIORITY`
- Dedup at enqueue time
- Seeding and re-seeding on resume
