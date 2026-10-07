# The worker pool

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add lifecycle, worker pool and circuit breaker
     Sources: jkent/driver/unified_driver/pool.py, jkent/driver/unified_driver/worker.py
     Delete this comment when the page has real content. -->

Why workers are pinned, and what that decision costs.

## Planned contents

- Pool and worker responsibilities
- Pinning: what a worker owns for its lifetime
- Idle accounting and why `worker.idle` exists
- A **Rejected** section: a free pool with per-request resource checkout
