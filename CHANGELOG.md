# Changelog

## 0.1.0 (unreleased)

Initial PyPI release.

- Scraper SDK: `BaseScraper`, `@entry` / `@step` decorators, typed
  request/response data types, speculation primitives, HTML parsing and
  selector-observation helpers.
- Unified driver: resumable SQLite-backed runs, httpx / Playwright /
  Camoufox transports, archive downloads, rate limiting, interstitial
  handling, replay.
- `RunBootstrapper`: requirement-driven component selection and run wiring.
- Driver runtime dependencies are isolated in the `operational` extra; the
  base install is the dependency-light scraper SDK.
