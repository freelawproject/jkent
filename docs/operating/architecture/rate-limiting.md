# Rate limiting and lanes

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add rate limiters and driver requirements
     Sources: jkent/driver/unified_driver/rate_limiter.py
     Delete this comment when the page has real content. -->

How declared limits become enforced pacing.

## Planned contents

- Lane resolution from scraper to run
- The limiter algorithm
- Interaction with retries, the breaker and the pinned pool
- A **Rejected** section: the boolean `bypass_rate_limit` flag it replaced
