# Resume an interrupted run

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add ScrapeRun
     Sources: jkent/driver/unified_driver/run.py
     Delete this comment when the page has real content. -->

Pick a run back up after a crash, a kill, or a deliberate stop.

## Planned contents

- Point the bootstrapper at the existing db
- What is replayed and what is re-fetched
- WAL files and when a run is safe to move
- Never move or rename a db before the run has observably finished
