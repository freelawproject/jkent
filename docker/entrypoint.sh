#!/usr/bin/env bash
# Bring up Xvfb, install the mounted repos editable, then exec the command.
#
# Xvfb is not optional decoration: CloudflareHandler's OS-click step needs a
# real X display with a mapped browser window, so scrapers must run --headed
# in here. Headless is a broken path for CF, not just a slower one.
set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

: "${DISPLAY:=:99}"
: "${SCREEN_GEOMETRY:=1920x1080x24}"
export DISPLAY

if ! xdpyinfo >/dev/null 2>&1; then
  log "starting Xvfb on ${DISPLAY} (${SCREEN_GEOMETRY})"
  Xvfb "${DISPLAY}" -screen 0 "${SCREEN_GEOMETRY}" -nolisten tcp \
    >/tmp/xvfb.log 2>&1 &
  for _ in $(seq 60); do
    xdpyinfo >/dev/null 2>&1 && break
    sleep 0.25
  done
  if ! xdpyinfo >/dev/null 2>&1; then
    log "ERROR: Xvfb never came up; see /tmp/xvfb.log"
    tail -20 /tmp/xvfb.log >&2 || true
    exit 1
  fi
fi
log "X display ${DISPLAY} ready; xdotool $(command -v xdotool >/dev/null && echo present || echo MISSING)"

# Install the bind-mounted projects editable, so the running code is your
# working tree. Done here rather than at build time because the sources do not
# exist until the mounts are in place.
#
# Order matters: westlean and jkent are named dependencies of jent, and passing
# them as local editables in the same resolution makes uv use the mounts instead
# of trying (and for westlean, failing) to find them on an index.
# Always run: /opt/venv lives in the image, so every container starts without the
# mounted projects installed. Measured at ~2s total container start, which is not
# worth a skip flag (an earlier JKENT_SKIP_INSTALL just produced containers with
# nothing installed).
if true; then
  install_args=()
  [ -d /work/westlean ] && install_args+=(-e /work/westlean)
  [ -d /work/jkent ] && install_args+=(-e "/work/jkent[operational]")
  [ -d /work/jent ] && install_args+=(-e /work/jent)
  # juriscraper is imported by scrapers rather than installed as a dep here;
  # install it editable when present so `import juriscraper` resolves.
  [ -d /work/juriscraper ] && [ -f /work/juriscraper/pyproject.toml ] \
    && install_args+=(-e /work/juriscraper)

  if [ ${#install_args[@]} -gt 0 ]; then
    log "installing editable: ${install_args[*]}"
    # --no-deps would be faster but would silently miss a new dependency added
    # to a pyproject since the image was built; let uv resolve.
    uv pip install --python /opt/venv --quiet "${install_args[@]}" \
      || { log "ERROR: editable install failed"; exit 1; }
  else
    log "WARNING: no repos found under /work — are the mounts configured?"
  fi
fi

# juriscraper-prs is a working checkout that may not be pip-installable; make it
# importable regardless so scraper modules under it can be loaded.
if [ -d /work/juriscraper ]; then
  export PYTHONPATH="/work/juriscraper${PYTHONPATH:+:$PYTHONPATH}"
fi

log "jent: $(command -v jent || echo 'NOT INSTALLED')"
log "runs dir: /work/runs ($(ls -1 /work/runs 2>/dev/null | wc -l | tr -d ' ') entries)"
log "reminder: scrapers need --headed for the Cloudflare OS-click path"

exec "$@"
