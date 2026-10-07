# Enable metrics and tracing

<!-- STUB — not yet written.
     Lands with: refactor(observability): rename continuation phase to step, add circuit reopens
     Sources: jkent/observability/, jkent/observability/METRICS.md
     Delete this comment when the page has real content. -->

Wire the run to an OpenTelemetry collector.

## Planned contents

- Enabling the SDK and what is gated behind it
- Exporter configuration
- The queue sampler and its cost
- Reading `worker.idle` and `request.retries`
