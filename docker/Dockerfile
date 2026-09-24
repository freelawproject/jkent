# Linux + Xvfb + xdotool: the substrate where an OS-level click clears the
# Cloudflare interstitial. See CloudflareHandler's _os_click_widget_checkbox
# in jkent/driver/unified_driver/interstitials.py.
#
# The image carries only the environment — Python, X, the browser and the
# third-party deps. jkent and juriscraper are bind-mounted (compose.yaml) and
# installed editable at container start (see entrypoint.sh), so your working
# tree is what runs, and rebuilds are not needed to test a code change.
#
# Build:  docker compose -f docker/compose.yaml build
# Shell:  docker compose -f docker/compose.yaml run --rm jkent

# Debian bookworm for the Firefox runtime libs (its package names are the ones
# this image is known to work with), but NOT its Python: bookworm links SQLite
# 3.40.1 while jkent needs >= 3.42 for unixepoch(..., 'subsec'), so `jent run`
# fails at the first database open. uv's managed CPython (python-build-standalone)
# bundles a current SQLite, which fixes that without a distro upgrade.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Chicago \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    # The venv the editable installs land in; on PATH so `jent` just works.
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:/root/.local/bin:$PATH" \
    # Fixed cache root so the browser fetched below lands in a known place and
    # is baked into the image rather than re-downloaded per container.
    XDG_CACHE_HOME=/cache

# xvfb + xdotool are the point of this image: xvfb gives Firefox a real X
# server to map a window into, xdotool posts genuine XTEST pointer events.
# x11-utils supplies xdpyinfo, which the entrypoint uses to wait for Xvfb.
# The lib* set is Firefox's runtime; libgl1-mesa-dri matters specifically —
# without software GL the WebGL fingerprint degrades to a detectable state.
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb xauth xdotool x11-utils \
      libgtk-3-0 libx11-xcb1 libxt6 libasound2 libdbus-glib-1-2 \
      libgbm1 libnss3 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
      libxkbcommon0 libxtst6 libpci3 libegl1 libgl1 libglx-mesa0 \
      libgl1-mesa-dri \
      fonts-liberation fontconfig ca-certificates tzdata \
      curl git sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Pinned: uv resolves everything else in this image, and must read uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.17 /uv /uvx /usr/local/bin/

# uv-managed Python, for the SQLite version (see the FROM comment).
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.13

# Pre-install the heavy third-party dependency set so the editable installs at
# startup are near-instant. jkent[operational] comes from jkent's own uv.lock
# (build context is the repo root), so it is exactly what the lock pins and
# cannot drift from pyproject.toml. The editable install at start still
# re-resolves against the mounted pyprojects, keeping what is installed here
# wherever it satisfies them.
COPY pyproject.toml uv.lock /opt/jkent-lock/
RUN uv venv --python 3.13 /opt/venv \
    && uv export --project /opt/jkent-lock --frozen --no-dev \
         --extra operational --no-emit-project -o /opt/jkent-lock/requirements.txt \
    && uv pip install --python /opt/venv -r /opt/jkent-lock/requirements.txt

# jent's runtime deps, for a mounted /work/jent (see entrypoint.sh). jent's lock
# is not in this build context, so this layer is only a warm cache: versions
# are re-resolved against the mounted pyproject at start.
RUN uv pip install --python /opt/venv \
      "typer>=0.26.7" "fastapi[standard-no-fastapi-cloud-cli]>=0.138.0" \
      "uvicorn>=0.49.0" "fastmcp>=2.0" "pyyaml>=6.0.3" \
      "genson>=1.4.0" "jsonschema>=4.26.0" "apted>=1.0.3" "lark>=1.3.1"

# Fetch the camoufox browser at build time — it is a ~200MB download and has no
# business happening on every container start. The browser matches the
# camoufox package installed above, so pin that package for the editable
# install at start (entrypoint.sh): a mounted tree that needs a newer camoufox
# fails there, asking for a rebuild, instead of drifting from the baked browser.
RUN /opt/venv/bin/python -m camoufox fetch \
    && grep -o '^camoufox==[^ ;]*' /opt/jkent-lock/requirements.txt \
         > /opt/jkent-lock/constraints.txt

# ENV PATH covers non-login shells, but `bash -l` re-reads /etc/profile and would
# drop the venv — leaving `python` as the system interpreter with none of the
# deps installed. Anyone shelling in with a login shell must still get the venv.
RUN printf 'export VIRTUAL_ENV=/opt/venv\nexport PATH="/opt/venv/bin:$PATH"\n' \
      > /etc/profile.d/10-jkent-venv.sh

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work/jkent
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
