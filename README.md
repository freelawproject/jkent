# Kent

**Alpha** - Kent is in early-stage development. APIs, database schemas, and CLI interfaces may change without notice.

Kent is a scraper-driver framework for structured web scraping. It separates parsing logic (scrapers) from I/O orchestration (drivers), so that scrapers are pure functions that parse HTML and yield data while drivers handle HTTP requests, file storage, rate limiting, and persistence.

## Installation

Kent uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

For web support (kent serve)

```bash
uv sync --extra web                  # Web UI for inspecting runs
```

For development (includes all extras plus testing/linting tools):

```bash
uv sync --group dev
uv run playwright install
```

## CLI Tools

### `kent`

The main CLI for discovering, inspecting, and running scrapers.

```bash
kent list                # Discover scrapers in the current directory tree
kent list -v             # Verbose listing with entry points and status
kent inspect MyModule:MyScraper          # Show scraper metadata and steps
kent inspect MyModule:MyScraper --seed-params  # Output seed parameters as JSON
kent run MyModule:MyScraper              # Run with the default (persistent) driver
kent run MyModule:MyScraper --driver sync       # Run with a specific driver
kent run MyModule:MyScraper --headed            # Run Playwright in headed mode
kent serve               # Launch the persistent driver web UI
```

### `pdd`

The Persistent Driver Debugger. Inspects and manipulates scraper run databases.

```bash
pdd --db run.db info                 # Run metadata and statistics
pdd --db run.db requests list        # Browse queued/completed requests
pdd --db run.db responses search     # Search stored responses
pdd --db run.db results list         # View parsed results
pdd --db run.db errors diagnose      # Structured error diagnosis
pdd --db run.db compression stats    # Compression statistics
pdd --db run.db doctor health        # Run health checks
```

## BugCivilCourt Demo

Kent ships with a demo scraper and a local mock court website called BugCivilCourt -- a whimsical court where insects file lawsuits. It demonstrates the full feature set (speculative requests, form submission, file archiving, JSON APIs, accumulated data) and serves as a reference for how to write scrapers.

```bash
uv sync --group demo                      # installs uvicorn for the web server
uv run kent/demorun_demo.py                # Start the demo web server
kent run kent.demo.scraper:BugCourtDemoScraper   # Run the demo scraper
```

## Documentation

Documentation is built with Sphinx and lives in the `docs/` directory. It covers the scraper-driver architecture through 19 incremental design steps -- from basic parsed data and navigating requests through to speculative entry points and async drivers. The demo section provides a walkthrough of the BugCivilCourt scraper and instructions for using the web UI and `pdd` debugger.

To build:

```bash
cd docs
make html           # Build HTML docs to docs/build/html/
make livehtml       # Auto-rebuilding dev server on port 8001
```

## I just want to parse one page

Use `single_page` to run a `@step` method without a driver or HTTP server:

```python
from kent.common.decorators import single_page
from my_scraper import MyScraper

run = single_page(MyScraper, "parse_results")
results = run("<html><body>...</body></html>")

# With accumulated_data from an earlier step:
results = run(html, accumulated_data={"case_id": "12345"})

# With JSON content:
run = single_page(MyScraper, "parse_api")
results = run('[{"id": 1}, {"id": 2}]')
```

`single_page` constructs a synthetic `Response`, feeds it through the `@step` wrapper (so all argument injection — `lxml_tree`, `json_content`, `page`, `text`, `accumulated_data`, etc. — works normally), and returns the unwrapped `ParsedData` items as a list.

## Claude Code Integration

Kent ships a [Claude Code skill](https://code.claude.com/docs/en/skills.md) for debugging scrapers. To use it in a consuming project, symlink the skill directory into your project's `.claude/skills/`:

```bash
# From your project root (adjust the path to your kent clone)
mkdir -p .claude/skills
ln -s /path/to/kent/.claude/skills/debug-scraper .claude/skills/debug-scraper
```

Then invoke it in Claude Code with `/debug-scraper`.

The skill gives Claude knowledge of all `pdd` and `kent` CLI commands and a structured debugging workflow. After each debugging session it should write a brief incident report to `.claude/debug-incidents/` noting what worked and where `pdd` fell short. If you're comfortable sharing these, we can use them to improve the pdd tool.

## Stability

### Mostly settled

- Sync / Async / Persistent Driver
- Basic `@entry` and `@step` decorators
- Core scraper-driver features (navigating/nonnavigating/archive requests, accumulated data, callbacks, data validation, transient exceptions, deduplication, priority queue)

### Moving Target/Active development

- Playwright Driver
- Kent WebUI
- `pdd` feature set
- - Specifically the doctor/health/scrape subcommands