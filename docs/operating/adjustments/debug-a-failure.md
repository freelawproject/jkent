# Debug a failed run

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add lifecycle, worker pool and circuit breaker
     Sources: jkent/driver/database_engine/errors.py, jkent/driver/unified_driver/circuit_breaker.py
     Delete this comment when the page has real content. -->

Work out why a run stopped early or collected nothing.

## Planned contents

- Reading the errors table and the exception taxonomy
- Circuit breaker opened — what tripped it
- Retry exhaustion vs a halt
- Driving `jent` against the db
