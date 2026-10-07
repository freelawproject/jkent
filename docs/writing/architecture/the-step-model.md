# The step model

<!-- STUB — not yet written.
     Lands with: feat(unified_driver): add step executor and speculation
     Sources: jkent/driver/unified_driver/steps.py, jkent/common/decorators.py
     Delete this comment when the page has real content. -->

Steps as generators, and why navigation is a yielded value rather than a call.

## Planned contents

- A step is a pure function of a response
- Yielding data and yielding requests from the same generator
- How the driver resumes a half-finished step
- Why there is no `await fetch()` in scraper code
