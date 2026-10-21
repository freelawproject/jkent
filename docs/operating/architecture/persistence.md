# Persistence and staged writes

<!-- STUB — not yet written.
     Lands with: feat(database_engine): add staged writes and request queue
     Sources: jkent/driver/database_engine/storage.py, jkent/driver/unified_driver/persistence.py
     Delete this comment when the page has real content. -->

How a run gets durable without serialising on the database.

## Planned contents

- Staged writes and the flush boundary
- What a crash can lose
- Write amplification and the instrumented lock
- A **Rejected** section: write-through on every step
