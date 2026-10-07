# Manage run database size

<!-- STUB — not yet written.
     Lands with: feat(database_engine): add zstd compression with trained dictionaries
     Sources: jkent/driver/database_engine/compression.py, jkent/driver/database_engine/sql_manager/_compression.py, jkent/driver/unified_driver/compaction.py
     Delete this comment when the page has real content. -->

Keep a long run's database from growing past what you can handle.

## Planned contents

- What actually consumes space, measured
- Compression dictionaries and when they are trained
- Running the compactor
- What is safe to delete
