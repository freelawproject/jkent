# Signal a broken assumption

<!-- STUB — not yet written.
     Lands with: the documentation commit (no earlier PR owns it)
     Sources: jkent/common/exceptions.py
     Delete this comment when the page has real content. -->

Pick the right exception when the site is not what the scraper expected.

## Planned contents

- `HTMLStructuralAssumptionException` vs `DataFormatAssumptionException`
- `HTTPResponseAssumptionException` and `IncidentalRequestAssumptionException`
- Transient vs persistent, and what each does to the run
- `RequestFailedHalt` — stopping the whole run on purpose
