# Issue speculative requests

<!-- STUB — not yet written.
     Lands with: feat(driver): add speculation support
     Sources: jkent/common/speculative.py, jkent/driver/_speculation_support.py
     Delete this comment when the page has real content. -->

Fan out requests before you know they are needed, and let the driver discard the misses.

## Planned contents

- `Speculative` and `SpeculativeRange`
- What the driver does with a speculation that misses
- When speculation costs more than it saves
