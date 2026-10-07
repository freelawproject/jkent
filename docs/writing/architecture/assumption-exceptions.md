# Why assumptions are exceptions

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/common/exceptions.py
     Delete this comment when the page has real content. -->

The reasoning behind a taxonomy instead of returning None.

## Planned contents

- Silent failure is the expensive failure mode in scraping
- Transient vs persistent, and the retry/circuit consequences of each
- How the taxonomy maps onto operator-visible error rows
- A **Rejected** section: optional returns and warn-and-continue
