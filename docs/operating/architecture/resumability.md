# Resumability

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add ScrapeRun
     Sources: jkent/driver/unified_driver/run.py
     Delete this comment when the page has real content. -->

What makes a run safe to stop and restart.

## Planned contents

- The invariant: every durable fact is derivable from the db
- Where idempotence is required and how it is obtained
- Known gaps — what a resume will re-fetch
- Why a run db must not be moved before observed completion
