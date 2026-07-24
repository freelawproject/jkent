# jkent telemetry catalog

Everything jkent measures, with some notes about reading the signals. We only
require the API within jkent, so all of the metrics are noops unless the consuming
app sets up an sdk properly.

---

## Conventions

**Dimensions (dynamic attributes).** :

| Attribute | Values | On |
|---|---|---|
| `scraper` | scraper class name | most metrics, request span |
| `step` | continuation/step name | request metrics, request span; on lock metrics only when the lock is taken inside a request |
| `phase` | see [phases](#phases) | `request.duration`, `request.cpu_time` |
| `kind` | `compress` / `train` / `recompress` | compression + compaction metrics |
| `outcome` | see [outcomes](#outcomes) | request span only (**not** a metric dimension) |
| `run_inst_id` | host-provided id (from baggage) | request span + the two per-run gauges only |

`run_inst_id` is high-cardinality, so we don't chuck this in histograms.

**Resource attributes** (`service.name`, `worker.pool`, `worker.concurrency`,
`worker.host`) are set once by the host on the providers and apply to *every*
metric and span automatically — they are never attached per-instrument here.
These are tracked here to track cross-run contention so we can judge if we need
to adjust our concurrency strategy (async/threads/subprocesses).

**Units.** `s` = seconds (wall unless noted), `1` = dimensionless ratio/count.

---

## Metrics

### Event loop

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.event_loop.lag` | histogram | s | *(resource only)* | `LoopLagMonitor` (`loop_monitor.py`) |

Fairly standard concurrency scheduling check. Gives us an idea of async contention for everything
sharing the same scheduler set (async loop and cpu core).

### Per-request timing

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.request.duration` | histogram | s | `scraper`, `step`, `phase` | `request_span` (`phase=total`) + `phase()` (`tracing.py`), wired in `worker.py` |
| `jkent.request.cpu_time` | histogram | s | `scraper`, `step`, `phase` | `compress_response` (`compression.py`) |

`request.duration` is recorded once per [phase](#phases) per request, plus a
`phase=total` covering the whole request. The `rate_limiter.gate` slice is the
rate-limiter token wait — a large one means rate-limit-bound, not
throughput-bound.

`request.cpu_time` is on-loop CPU (`time.thread_time`) for synchronous phases
we can measure without pollution from co-scheduled tasks. Today that is
`phase=compress` only (a sync leaf with no `await` inside). A phase whose
`cpu_time` ≈ its `duration` is hogging the loop; one whose `cpu_time` ≪
`duration` is awaiting I/O.

### Database lock

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.db.lock.wait` | histogram | s | `scraper`, `step`? | `InstrumentedLock.acquire` (`instrumented_lock.py`) |
| `jkent.db.lock.hold` | histogram | s | `scraper`, `step`? | `InstrumentedLock.release` |

The run holds a **single** `asyncio.Lock` serializing all SQLite access
(dequeue, restamp, store, staged flush, dedup, counts). `lock.wait` is nonzero
only under contention — a rising `wait` is the "lock getting fought over".
`lock.hold` shows how long each holder keeps it. (`step?`: present only when
the lock is taken inside a request; absent for dequeue / sampler / seed paths.)

### Worker pool

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.worker.idle` | histogram | s | `scraper` | `PoolWorker._run_loop` (`worker.py`) |
| `jkent.request.retries` | counter | 1 | `scraper`, `step` | `PoolWorker._handle_transient` (`worker.py`) |
| `jkent.circuit.opens` | counter | 1 | `scraper`, `step` | `CircuitBreaker.record_failure` (`circuit_breaker.py`) |

`worker.idle` measures time when workers are starved for work, or on the other end
if we might benefit from adding more. This measures when there's nothing in the queue
for them to work on.

`request.retries` counts scheduled transient retries (a retry that gave up at
max backoff is not counted; its request span gets `outcome=transient`). If this
climbs when we add workers, we should read it as server backpressure.

`circuit.opens` is the run-wide escalation of `request.retries`: a trip means
the pool saw `failure_threshold` *consecutive* transient failures and paused
all request traffic for the breaker's recovery window. This is a strong signal
of server backpressure, and a steady increase can be used to help disambiguate between
a range of requests that are slow for ther server to reply to, and general worker
driven server contention.

### Compression

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.compression.duration` | histogram | s | `scraper`, `step`, `kind=compress` | `compress_response` (`compression.py`) |
| `jkent.compression.ratio` | histogram | 1 | `scraper`, `step` | `compress_response` |
| `jkent.compaction.duration` | histogram | s | `scraper`, `step`, `kind` | `train_compression_dict` / `recompress_responses` |

This is all for sanity. In practice I expect these signals to be mostly ignorable, but if compression gets bad
suddenly it's worth looking into (something weird is happening!).

### Per-run state (gauges)

| Metric | Type | Unit | Attrs | Emitted from |
|---|---|---|---|---|
| `jkent.worker.active` | gauge | 1 | `scraper`, `run_inst_id` | `ScrapeRun._publish_worker_active` (`run.py`), at spawn/retire |
| `jkent.queue.pending` | gauge | 1 | `scraper`, `run_inst_id` | `ScrapeRun._sample_queue_gauge` (`run.py`), every 5 s |

`worker.active` is the live worker count. This is largely included as a quick reference
for metrics, and in the future if we decide to introduce dynamic worker pool sizing.

---

## Spans

I don't think we're really using any sort of tracing yet, but these are lightweight
and simple to add, and may be helpful if we see very odd behavior in prod and can turn
on sampling.

One trace tree per request (sampled by the host). httpx / SQLAlchemy /
botocore auto-instrumentation spans nest under the relevant phase span once the
host enables those instrumentors.

| Span | Parent | Attributes / notes |
|---|---|---|
| `jkent.request` | root | `jkent.scraper`, `jkent.step`, `jkent.run_inst_id`, `jkent.outcome` |
| `jkent.rate_limiter.gate` | `jkent.request` | rate-limiter token wait |
| `jkent.transport.resolve` | `jkent.request` | the fetch; httpx client spans nest here |
| `jkent.continuation` | `jkent.request` | response store + scraper continuation; SQLAlchemy spans nest here |

---

## Glossary

### Phases

Values of the `phase` attribute (on `request.duration` / `request.cpu_time`),
defined by the `Phase` enum in `metrics.py`:

| `phase` | Meaning |
|---|---|
| `total` | whole request, dequeue-to-done (duration only) |
| `circuit_breaker.gate` | waiting for an open circuit breaker to close |
| `rate_limiter.gate` | waiting for a rate-limiter token |
| `transport.resolve` | fetching the response (network / browser) |
| `continuation` | storing the response + running the scraper continuation |
| `compress` | the synchronous zstd compress leaf (`cpu_time` only) |

### Outcomes

Values of `jkent.outcome` on the request span, defined by the `Outcome` enum
in `metrics.py`:

| `outcome` | Meaning |
|---|---|
| `ok` | request completed and continuation ran |
| `halt` | `RequestFailedHalt` — propagated, stops the run |
| `skip` | skipped by an `on_transient_exception` callback |
| `transient` | transient failure → retried (or failed after max backoff) |
| `speculation_http` | persistent HTTP on a speculative probe (recorded as a speculation outcome) |
| `persistent_http` | classifier said the status is persistent → no retry |
| `error` | unexpected exception → marked failed, error row stored |

---

## Reading them together

These signals together should help us understand the perf in prod, and help us adjust
our run concurrency strategy (async loop/threads/subprocesses), worker concurrency,
rate limits, etc..

- **High `event_loop.lag` + high `request.cpu_time`** → GIL is blocking
concurrency. Adding workers to the same loop makes it worse.
- **Low lag, high `db.lock.wait`** → the single per-run DB lock is the ceiling;
  more workers or run concurrency won't help a per-run lock.
- **Low lag, low lock wait, large `request.duration{phase=rate_limiter.gate}`**
  → rate-limit-bound; more workers are pointless.
- **All low, `worker.idle` share near zero, `queue.pending` growing** →
  genuinely I/O-bound → raise `num_workers`. Watch for retry backpressure.
- **High `worker.idle` share, `queue.pending` ~0** → oversized; lower
  `num_workers` (frees loop capacity for co-resident runs). Idle that is
  *backoff* (retries pending) rather than surplus shows up as idle with a
  nonzero `queue.pending` and `seconds_until_next_pending` gaps — server
  backpressure.

---

## Toggles

- `OTEL_SDK_DISABLED=true` (host) — whole SDK no-op; jkent's API calls stay but
  cost nothing.
- `JKENT_OTEL_LOOP_MONITOR=0` — disable just the event-loop-lag sampler.
